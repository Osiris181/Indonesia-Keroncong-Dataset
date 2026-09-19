#!/usr/bin/env python3
"""Align 100 Hz TorchCREPE contours to the 25 Hz MusicFM grid."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

TASK_ROOT = Path(__file__).resolve().parent
BASELINES_ROOT = TASK_ROOT.parent
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from common.syllable_index import load_syllable_index
from five_label_multilabel.musicfm import MODEL_TOKEN_STEP_SEC

DEFAULT_DATASET = BASELINES_ROOT / "data" / "syllable_dataset.csv"
DEFAULT_MUSICFM = TASK_ROOT / "outputs" / "features" / "musicfm"
DEFAULT_PITCH = TASK_ROOT / "outputs" / "features" / "pitch_contours"
DEFAULT_OUTPUT = TASK_ROOT / "outputs" / "features" / "pitch_25hz"
PITCH_INDEX_RATIO = 4
PERIODICITY_THRESHOLD = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--musicfm-dir", type=Path, default=DEFAULT_MUSICFM)
    parser.add_argument("--pitch-dir", type=Path, default=DEFAULT_PITCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--track-id", action="append")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_tracks(metadata, requested: list[str] | None) -> list[str]:
    available = metadata["track_id"].drop_duplicates().astype(str).tolist()
    if requested is None:
        return available
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate --track-id values")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown track IDs: {unknown}")
    selected = set(requested)
    return [track for track in available if track in selected]


def align_track(track_id: str, musicfm_path: Path, pitch_path: Path, output: Path) -> None:
    if not musicfm_path.is_file():
        raise FileNotFoundError(musicfm_path)
    if not pitch_path.is_file():
        raise FileNotFoundError(pitch_path)
    with np.load(musicfm_path, allow_pickle=False) as archive:
        frame_index = archive["frame_index"].astype(np.int32, copy=True)
        frame_time_sec = archive["frame_time_sec"].astype(np.float64, copy=True)
        observed_track = str(archive["track_id"].item())
    if observed_track != track_id:
        raise ValueError(f"{musicfm_path} stores {observed_track}, expected {track_id}")
    if not np.array_equal(
        frame_time_sec, frame_index.astype(np.float64) * MODEL_TOKEN_STEP_SEC
    ):
        raise ValueError(f"{track_id}: unexpected MusicFM frame grid")
    source_index = frame_index.astype(np.int64) * PITCH_INDEX_RATIO
    with np.load(pitch_path, allow_pickle=False) as archive:
        required = {"time_sec", "f0_hz", "periodicity"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{pitch_path} is missing arrays: {sorted(missing)}")
        if "track_id" in archive.files:
            observed_pitch_track = str(archive["track_id"].item())
            if observed_pitch_track != track_id:
                raise ValueError(
                    f"{pitch_path} stores {observed_pitch_track}, expected {track_id}"
                )
        if source_index[-1] >= archive["time_sec"].size:
            raise ValueError(f"{track_id}: pitch contour ends before MusicFM grid")
        if not np.array_equal(frame_time_sec, archive["time_sec"][source_index]):
            raise ValueError(f"{track_id}: pitch and MusicFM timestamps do not align")
        f0_hz = archive["f0_hz"][source_index].astype(np.float32, copy=True)
        periodicity = archive["periodicity"][source_index].astype(np.float32, copy=True)
    reliable = (periodicity >= PERIODICITY_THRESHOLD).astype(np.uint8)
    if not reliable.any():
        raise ValueError(f"{track_id}: no reliable pitch frame")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray("IKD_pitch_25hz_v1"),
            track_id=np.asarray(track_id),
            frame_index=frame_index,
            frame_time_sec=frame_time_sec,
            pitch_source_frame_index=source_index.astype(np.int32),
            pitch_f0_hz=f0_hz,
            pitch_periodicity=periodicity,
            pitch_reliable=reliable,
        )
    os.replace(temporary, output)


def write_definition(output_dir: Path) -> None:
    definition = {
        "schema_version": "IKD_pitch_25hz_definition_v1",
        "target_grid": "MusicFM 25 Hz frame grid",
        "source_grid_hz": 100,
        "mapping": "pitch_source_frame_index = 4 * musicfm_frame_index",
        "interpolation": False,
        "periodicity_threshold": PERIODICITY_THRESHOLD,
        "model_channels": [
            "pitch_track_relative_semitone",
            "pitch_periodicity",
            "pitch_reliable",
        ],
        "target_information_used": False,
    }
    path = output_dir / "pitch_25hz_definition.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    metadata = load_syllable_index(args.dataset.resolve())
    tracks = choose_tracks(metadata, args.track_id)
    musicfm_dir = args.musicfm_dir.resolve()
    pitch_dir = args.pitch_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_definition(output_dir)
    for track_id in tqdm(tracks, desc="Pitch alignment", unit="track", dynamic_ncols=True):
        output = output_dir / f"{track_id}_pitch_25hz.npz"
        if output.is_file() and not args.overwrite:
            tqdm.write(f"Skipped existing {output}")
            continue
        align_track(
            track_id,
            musicfm_dir / f"{track_id}_musicfm_msd_framewise.npz",
            pitch_dir / f"{track_id}_pitch_contour.npz",
            output,
        )
        tqdm.write(f"Wrote {output}")


if __name__ == "__main__":
    main()
