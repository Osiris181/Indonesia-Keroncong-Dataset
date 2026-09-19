"""Load cached framewise MusicFM and optional pitch features by complete track."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from five_label_multilabel.musicfm import EMBEDDING_DIMENSION, MODEL_TOKEN_STEP_SEC


LABEL_NAMES = ("luk", "cengkok", "gregel", "embat", "nggandhul")
PITCH_FEATURE_NAMES = (
    "pitch_track_relative_semitone",
    "pitch_periodicity",
    "pitch_reliable",
)
VALID_SPLITS = ("train", "validation", "test")
MUSICFM_LAYER_NORM_EPSILON = 1e-5
PERIODICITY_THRESHOLD = 0.50


@dataclass(frozen=True)
class FeatureSelection:
    include_pitch: bool

    @property
    def names(self) -> tuple[str, ...]:
        musicfm = tuple(f"musicfm_{index:04d}" for index in range(EMBEDDING_DIMENSION))
        return musicfm + (PITCH_FEATURE_NAMES if self.include_pitch else ())


@dataclass(frozen=True)
class TemporalTrack:
    track_id: str
    features: np.ndarray
    syllable_frame_starts: np.ndarray
    syllable_frame_ends: np.ndarray
    targets: np.ndarray
    row_indices: np.ndarray
    syllable_ids: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def syllable_count(self) -> int:
        return int(self.targets.shape[0])


@dataclass(frozen=True)
class TemporalTrackStore:
    metadata: pd.DataFrame
    tracks: tuple[TemporalTrack, ...]
    feature_names: tuple[str, ...]
    selection: FeatureSelection

    @property
    def input_dimension(self) -> int:
        return len(self.feature_names)

    @property
    def syllable_count(self) -> int:
        return sum(track.syllable_count for track in self.tracks)


class TemporalTrackDataset(Dataset[dict[str, Any]]):
    def __init__(self, store: TemporalTrackStore, track_indices: np.ndarray) -> None:
        self.store = store
        self.track_indices = np.asarray(track_indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.track_indices.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        track_index = int(self.track_indices[index])
        track = self.store.tracks[track_index]
        return {
            "features": torch.from_numpy(track.features),
            "targets": torch.from_numpy(track.targets),
            "syllable_frame_starts": torch.from_numpy(track.syllable_frame_starts),
            "syllable_frame_ends": torch.from_numpy(track.syllable_frame_ends),
            "row_indices": torch.from_numpy(track.row_indices),
            "syllable_ids": track.syllable_ids,
            "track_id": track.track_id,
            "track_index": track_index,
        }


@dataclass(frozen=True)
class FoldDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    feature_names: tuple[str, ...]
    run: int
    seed: int

    @property
    def input_dimension(self) -> int:
        return len(self.feature_names)


def normalize_musicfm_per_frame(embeddings: np.ndarray) -> np.ndarray:
    """Apply the original non-learned LayerNorm to every frame vector."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected MusicFM [frame, 1024], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("MusicFM features contain non-finite values")
    centered = values - np.mean(values, axis=1, keepdims=True)
    variances = np.mean(centered * centered, axis=1, keepdims=True)
    normalized = centered / np.sqrt(variances + MUSICFM_LAYER_NORM_EPSILON)
    return normalized.astype(np.float32, copy=False)


def track_relative_pitch_features(
    f0_hz: np.ndarray,
    periodicity: np.ndarray,
    reliable: np.ndarray,
) -> np.ndarray:
    """Construct the exact three-channel pitch representation."""

    f0 = np.asarray(f0_hz, dtype=np.float64)
    confidence = np.asarray(periodicity, dtype=np.float64)
    mask = np.asarray(reliable, dtype=bool)
    if not (f0.ndim == confidence.ndim == mask.ndim == 1):
        raise ValueError("Pitch arrays must be one-dimensional")
    if not (f0.size == confidence.size == mask.size):
        raise ValueError("Pitch array lengths differ")
    if not np.isfinite(f0).all() or np.any(f0 <= 0.0):
        raise ValueError("F0 must be finite and positive")
    if not np.isfinite(confidence).all() or np.any(
        (confidence < 0.0) | (confidence > 1.0)
    ):
        raise ValueError("Periodicity must lie within [0, 1]")
    if not np.array_equal(mask, confidence >= PERIODICITY_THRESHOLD):
        raise ValueError("Pitch reliability must equal periodicity >= 0.50")
    if not mask.any():
        raise ValueError("A track contains no reliable pitch frames")

    reference_hz = float(np.median(f0[mask]))
    relative_semitone = np.zeros(f0.size, dtype=np.float64)
    relative_semitone[mask] = 12.0 * np.log2(f0[mask] / reference_hz)
    features = np.column_stack(
        (relative_semitone, confidence, mask.astype(np.float64))
    )
    if not np.isfinite(features).all():
        raise ValueError("Constructed pitch features contain non-finite values")
    return features.astype(np.float32, copy=False)


def load_track_store(
    metadata: pd.DataFrame,
    musicfm_dir: Path,
    pitch_dir: Path,
    selection: FeatureSelection,
) -> TemporalTrackStore:
    names = selection.names
    tracks: list[TemporalTrack] = []
    track_values = metadata["track_id"].astype(str).to_numpy()

    for track_id in metadata["track_id"].drop_duplicates().astype(str):
        row_indices = np.flatnonzero(track_values == track_id).astype(np.int64)
        musicfm_path = musicfm_dir / f"{track_id}_musicfm_msd_framewise.npz"
        if not musicfm_path.is_file():
            raise FileNotFoundError(musicfm_path)
        with np.load(musicfm_path, allow_pickle=False) as archive:
            required = {"track_id", "frame_index", "frame_time_sec", "embeddings"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{musicfm_path} is missing arrays: {sorted(missing)}")
            observed_track = str(archive["track_id"].item())
            frame_index = archive["frame_index"].astype(np.int32, copy=True)
            frame_time_sec = archive["frame_time_sec"].astype(np.float64, copy=True)
            musicfm = normalize_musicfm_per_frame(archive["embeddings"])
        if observed_track != track_id:
            raise ValueError(f"{musicfm_path} stores {observed_track}, expected {track_id}")
        if not np.array_equal(frame_index, np.arange(frame_index.size, dtype=np.int32)):
            raise ValueError(f"{track_id}: MusicFM frame index is not contiguous")
        if not np.array_equal(
            frame_time_sec, frame_index.astype(np.float64) * MODEL_TOKEN_STEP_SEC
        ):
            raise ValueError(f"{track_id}: MusicFM grid is not exactly 25 Hz")
        blocks = [musicfm]

        if selection.include_pitch:
            pitch_path = pitch_dir / f"{track_id}_pitch_25hz.npz"
            if not pitch_path.is_file():
                raise FileNotFoundError(pitch_path)
            with np.load(pitch_path, allow_pickle=False) as archive:
                required = {
                    "track_id",
                    "frame_index",
                    "frame_time_sec",
                    "pitch_f0_hz",
                    "pitch_periodicity",
                    "pitch_reliable",
                }
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(f"{pitch_path} is missing arrays: {sorted(missing)}")
                if str(archive["track_id"].item()) != track_id:
                    raise ValueError(f"{pitch_path} stores the wrong track ID")
                if not np.array_equal(archive["frame_index"], frame_index):
                    raise ValueError(f"{track_id}: pitch frame index differs from MusicFM")
                if not np.array_equal(archive["frame_time_sec"], frame_time_sec):
                    raise ValueError(f"{track_id}: pitch timestamps differ from MusicFM")
                pitch = track_relative_pitch_features(
                    archive["pitch_f0_hz"],
                    archive["pitch_periodicity"],
                    archive["pitch_reliable"],
                )
            blocks.append(pitch)

        features = np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)
        if features.shape != (frame_index.size, len(names)):
            raise ValueError(
                f"{track_id}: constructed {features.shape}, expected "
                f"{(frame_index.size, len(names))}"
            )

        track_metadata = metadata.iloc[row_indices]
        starts_sec = track_metadata["start_sec"].to_numpy(dtype=np.float64)
        ends_sec = track_metadata["end_sec"].to_numpy(dtype=np.float64)
        starts = np.searchsorted(frame_time_sec, starts_sec, side="left").astype(np.int32)
        ends = np.searchsorted(frame_time_sec, ends_sec, side="left").astype(np.int32)
        if np.any(starts < 0) or np.any(starts >= ends) or np.any(ends > frame_index.size):
            raise ValueError(f"{track_id}: invalid syllable interval on the 25 Hz grid")
        targets = track_metadata.loc[:, LABEL_NAMES].to_numpy(dtype=np.float32)
        tracks.append(
            TemporalTrack(
                track_id=track_id,
                features=features,
                syllable_frame_starts=starts,
                syllable_frame_ends=ends,
                targets=np.ascontiguousarray(targets),
                row_indices=row_indices,
                syllable_ids=tuple(track_metadata["syllable_id"].astype(str)),
            )
        )

    store = TemporalTrackStore(metadata, tuple(tracks), names, selection)
    if store.syllable_count != len(metadata):
        raise RuntimeError("Track store does not cover every syllable")
    return store


def collate_tracks(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(items)
    feature_dimension = int(items[0]["features"].shape[1])
    frame_lengths = torch.tensor(
        [item["features"].shape[0] for item in items], dtype=torch.long
    )
    syllable_lengths = torch.tensor(
        [item["targets"].shape[0] for item in items], dtype=torch.long
    )
    maximum_frames = int(frame_lengths.max())
    maximum_syllables = int(syllable_lengths.max())
    features = torch.zeros((batch_size, feature_dimension, maximum_frames))
    frame_mask = torch.zeros((batch_size, maximum_frames), dtype=torch.bool)
    targets = torch.zeros((batch_size, maximum_syllables, len(LABEL_NAMES)))
    syllable_mask = torch.zeros((batch_size, maximum_syllables), dtype=torch.bool)
    starts = torch.zeros((batch_size, maximum_syllables), dtype=torch.long)
    ends = torch.zeros((batch_size, maximum_syllables), dtype=torch.long)
    row_indices = torch.full((batch_size, maximum_syllables), -1, dtype=torch.long)

    for index, item in enumerate(items):
        frame_count = int(frame_lengths[index])
        syllable_count = int(syllable_lengths[index])
        features[index, :, :frame_count] = item["features"].transpose(0, 1)
        frame_mask[index, :frame_count] = True
        targets[index, :syllable_count] = item["targets"]
        syllable_mask[index, :syllable_count] = True
        starts[index, :syllable_count] = item["syllable_frame_starts"]
        ends[index, :syllable_count] = item["syllable_frame_ends"]
        row_indices[index, :syllable_count] = item["row_indices"]

    return {
        "features": features,
        "frame_mask": frame_mask,
        "frame_lengths": frame_lengths,
        "targets": targets,
        "syllable_mask": syllable_mask,
        "syllable_lengths": syllable_lengths,
        "syllable_frame_starts": starts,
        "syllable_frame_ends": ends,
        "row_indices": row_indices,
        "syllable_ids": [item["syllable_ids"] for item in items],
        "track_ids": [item["track_id"] for item in items],
        "track_indices": torch.tensor([item["track_index"] for item in items]),
    }


def make_fold_loaders(
    store: TemporalTrackStore,
    run: int,
    seed: int,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> FoldDataLoaders:
    if run not in range(1, 6):
        raise ValueError("run must be from 1 through 5")
    split_column = f"fold_{run}_split"
    split_indices: dict[str, list[int]] = {name: [] for name in VALID_SPLITS}
    for track_index, track in enumerate(store.tracks):
        roles = store.metadata.iloc[track.row_indices][split_column].unique()
        if roles.size != 1 or str(roles[0]) not in VALID_SPLITS:
            raise ValueError(f"Run {run} has an invalid role for {track.track_id}")
        split_indices[str(roles[0])].append(track_index)
    datasets = {
        split: TemporalTrackDataset(
            store, np.asarray(indices, dtype=np.int64)
        )
        for split, indices in split_indices.items()
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError(f"Run {run} contains an empty partition")
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_tracks,
    }
    return FoldDataLoaders(
        train=DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        validation=DataLoader(datasets["validation"], shuffle=False, **common),
        test=DataLoader(datasets["test"], shuffle=False, **common),
        feature_names=store.feature_names,
        run=run,
        seed=seed,
    )
