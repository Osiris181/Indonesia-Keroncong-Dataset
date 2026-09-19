#!/usr/bin/env python3
"""Extract cached framewise MusicFM-MSD layer-7 representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm.auto import tqdm

TASK_ROOT = Path(__file__).resolve().parent
BASELINES_ROOT = TASK_ROOT.parent
RELEASE_ROOT = BASELINES_ROOT.parent
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from common.syllable_index import load_syllable_index
from five_label_multilabel.musicfm import (
    CONTEXT_DURATION_SEC,
    CORE_DURATION_SEC,
    EMBEDDING_DIMENSION,
    MODEL_TOKEN_STEP_SAMPLES,
    MODEL_TOKEN_STEP_SEC,
    OUTPUT_LAYER_INDEX,
    SAMPLE_RATE_HZ,
    chunk_plan,
    expected_token_count,
    extract_frozen_layer,
    load_musicfm,
    validate_musicfm_root,
)

DEFAULT_DATASET = BASELINES_ROOT / "data" / "syllable_dataset.csv"
DEFAULT_AUDIO_ROOT = RELEASE_ROOT / "IKD_Dataset" / "Audio"
DEFAULT_OUTPUT = TASK_ROOT / "outputs" / "features" / "musicfm"
RANDOM_SEED = 20260816
EXPECTED_COMMIT = "b83ebedb401bcef639b26b05c0c8bee1dc2dfe71"
EXPECTED_CHECKPOINT = "218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded"
EXPECTED_STATISTICS = "c36c61ab10ca4d2e7fdfefc3fcc15205316bec276a06a47baa3641a62c546f22"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--musicfm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--track-id", action="append")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def resolve_audio(audio_root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe vocal path: {relative_value}")
    path = (audio_root / relative).resolve()
    try:
        path.relative_to(audio_root)
    except ValueError as error:
        raise ValueError(f"Vocal path escapes --audio-root: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_audio(path: Path) -> np.ndarray:
    audio, rate = librosa.load(
        path, sr=SAMPLE_RATE_HZ, mono=True, dtype=np.float32, res_type="soxr_hq"
    )
    if rate != SAMPLE_RATE_HZ:
        raise RuntimeError(f"Expected {SAMPLE_RATE_HZ} Hz, got {rate}")
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid vocal audio: {path}")
    return np.ascontiguousarray(audio)


def extract_track(
    track_id: str,
    audio_path: Path,
    audio_root: Path,
    model: torch.nn.Module,
    device: torch.device,
    output: Path,
) -> None:
    audio = load_audio(audio_path)
    total_frames = expected_token_count(int(audio.size))
    plan = chunk_plan(total_frames)
    embeddings = np.empty((total_frames, EMBEDDING_DIMENSION), dtype=np.float32)
    for core_start, core_end, input_start, input_end in tqdm(
        plan, desc=f"{track_id} MusicFM", unit="chunk", leave=False, dynamic_ncols=True
    ):
        sample_start = input_start * MODEL_TOKEN_STEP_SAMPLES
        sample_end = min(int(audio.size), input_end * MODEL_TOKEN_STEP_SAMPLES)
        waveform = torch.from_numpy(audio[sample_start:sample_end]).unsqueeze(0).to(device)
        hidden = extract_frozen_layer(model, waveform).squeeze(0).cpu().numpy()
        expected = (input_end - input_start, EMBEDDING_DIMENSION)
        if hidden.shape != expected:
            raise RuntimeError(f"{track_id}: expected {expected}, got {hidden.shape}")
        retain_start = core_start - input_start
        retain_end = retain_start + core_end - core_start
        embeddings[core_start:core_end] = hidden[retain_start:retain_end]
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"{track_id}: non-finite MusicFM values")
    frame_index = np.arange(total_frames, dtype=np.int32)
    frame_time_sec = frame_index.astype(np.float64) * MODEL_TOKEN_STEP_SEC
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            track_id=np.asarray(track_id),
            source_audio_relative_path=np.asarray(
                audio_path.relative_to(audio_root).as_posix()
            ),
            source_num_samples=np.asarray(audio.size, dtype=np.int64),
            sample_rate_hz=np.asarray(SAMPLE_RATE_HZ, dtype=np.int32),
            token_step_samples=np.asarray(MODEL_TOKEN_STEP_SAMPLES, dtype=np.int32),
            layer_index=np.asarray(OUTPUT_LAYER_INDEX, dtype=np.int16),
            frame_index=frame_index,
            frame_time_sec=frame_time_sec,
            embeddings=embeddings,
            chunk_core_start_frame=np.asarray([value[0] for value in plan], dtype=np.int32),
            chunk_core_end_frame=np.asarray([value[1] for value in plan], dtype=np.int32),
            chunk_input_start_frame=np.asarray([value[2] for value in plan], dtype=np.int32),
            chunk_input_end_frame=np.asarray([value[3] for value in plan], dtype=np.int32),
        )
    os.replace(temporary, output)


def write_definition(output_dir: Path, checkpoint: Path, statistics: Path) -> None:
    checkpoint_hash = sha256_file(checkpoint)
    statistics_hash = sha256_file(statistics)
    if checkpoint_hash != EXPECTED_CHECKPOINT:
        raise ValueError("MusicFM checkpoint hash differs from the published model")
    if statistics_hash != EXPECTED_STATISTICS:
        raise ValueError("MusicFM statistics hash differs from the published model")
    definition = {
        "schema_version": "IKD_framewise_musicfm_v1",
        "expected_musicfm_commit": EXPECTED_COMMIT,
        "checkpoint_sha256": checkpoint_hash,
        "statistics_sha256": statistics_hash,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frame_rate_hz": 25,
        "token_step_samples": MODEL_TOKEN_STEP_SAMPLES,
        "hidden_layer_index": OUTPUT_LAYER_INDEX,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_core_seconds": CORE_DURATION_SEC,
        "chunk_context_seconds_each_side": CONTEXT_DURATION_SEC,
        "boundary_padding": False,
        "random_seed": RANDOM_SEED,
        "target_information_used": False,
    }
    path = output_dir / "framewise_musicfm_definition.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    metadata = load_syllable_index(args.dataset.resolve())
    tracks = choose_tracks(metadata, args.track_id)
    audio_root = args.audio_root.resolve()
    if not audio_root.is_dir():
        raise FileNotFoundError(audio_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, statistics = validate_musicfm_root(args.musicfm_root)
    write_definition(output_dir, checkpoint, statistics)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = resolve_device(args.device)
    print(f"Loading frozen MusicFM-MSD on {device}", flush=True)
    model = load_musicfm(args.musicfm_root, device)
    for track_id in tqdm(tracks, desc="MusicFM tracks", unit="track", dynamic_ncols=True):
        rows = metadata.loc[metadata["track_id"].astype(str) == track_id]
        paths = set(rows["vocal_audio_relative_path"].dropna().astype(str))
        if len(paths) != 1:
            raise ValueError(f"{track_id}: inconsistent vocal paths")
        audio_path = resolve_audio(audio_root, paths.pop())
        output = output_dir / f"{track_id}_musicfm_msd_framewise.npz"
        if output.is_file() and not args.overwrite:
            tqdm.write(f"Skipped existing {output}")
            continue
        extract_track(track_id, audio_path, audio_root, model, device, output)
        tqdm.write(f"Wrote {output}")


if __name__ == "__main__":
    main()
