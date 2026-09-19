#!/usr/bin/env python3
"""Extract deterministic whole-track TorchCREPE pitch contours."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchcrepe
from tqdm.auto import tqdm

TASK_ROOT = Path(__file__).resolve().parent
BASELINES_ROOT = TASK_ROOT.parent
RELEASE_ROOT = BASELINES_ROOT.parent
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from common.syllable_index import load_syllable_index

DEFAULT_DATASET = BASELINES_ROOT / "data" / "syllable_dataset.csv"
DEFAULT_AUDIO_ROOT = RELEASE_ROOT / "IKD_Dataset" / "Audio"
DEFAULT_OUTPUT = TASK_ROOT / "outputs" / "features" / "pitch_contours"
SAMPLE_RATE_HZ = 16_000
HOP_LENGTH_SAMPLES = 160
HOP_SEC = HOP_LENGTH_SAMPLES / SAMPLE_RATE_HZ
FMIN_HZ = 50.0
FMAX_HZ = 1000.0
MODEL_CAPACITY = "full"
BATCH_SIZE_FRAMES = 2048
RANDOM_SEED = 20260815
TORCHCREPE_CENTS_OFFSET = 1997.3794084376191


def deterministic_viterbi(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map Viterbi bins to fixed centers without frequency dither."""

    bins, _ = torchcrepe.decode.viterbi(logits)
    cents = torchcrepe.CENTS_PER_BIN * bins + TORCHCREPE_CENTS_OFFSET
    return bins, torchcrepe.convert.cents_to_frequency(cents)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
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


def set_seed(device: torch.device) -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def extract_track(track_id: str, audio_path: Path, device: torch.device, output: Path) -> None:
    set_seed(device)
    audio, rate = librosa.load(
        audio_path, sr=SAMPLE_RATE_HZ, mono=True, dtype=np.float32, res_type="soxr_hq"
    )
    if rate != SAMPLE_RATE_HZ:
        raise RuntimeError(f"Expected {SAMPLE_RATE_HZ} Hz, got {rate}")
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid vocal audio: {audio_path}")
    with torch.inference_mode():
        pitch_tensor, periodicity_tensor = torchcrepe.predict(
            torch.from_numpy(audio).unsqueeze(0).to(device),
            sample_rate=SAMPLE_RATE_HZ,
            hop_length=HOP_LENGTH_SAMPLES,
            fmin=FMIN_HZ,
            fmax=FMAX_HZ,
            model=MODEL_CAPACITY,
            decoder=deterministic_viterbi,
            return_periodicity=True,
            batch_size=BATCH_SIZE_FRAMES,
            device=str(device),
            pad=True,
        )
    f0_hz = pitch_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    periodicity = periodicity_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    expected = 1 + int(audio.size // HOP_LENGTH_SAMPLES)
    if f0_hz.shape != (expected,) or periodicity.shape != (expected,):
        raise RuntimeError(
            f"{track_id}: expected {expected} frames, got "
            f"F0={f0_hz.shape}, periodicity={periodicity.shape}"
        )
    if not np.isfinite(f0_hz).all() or not np.isfinite(periodicity).all():
        raise RuntimeError(f"{track_id}: non-finite TorchCREPE output")
    if np.any((f0_hz < FMIN_HZ - 1e-3) | (f0_hz > FMAX_HZ + 1e-3)):
        raise RuntimeError(f"{track_id}: F0 outside configured range")
    if np.any((periodicity < -1e-6) | (periodicity > 1.0 + 1e-6)):
        raise RuntimeError(f"{track_id}: periodicity outside [0, 1]")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            track_id=np.asarray(track_id),
            time_sec=np.arange(expected, dtype=np.float64) * HOP_SEC,
            f0_hz=f0_hz,
            periodicity=periodicity,
        )
    os.replace(temporary, output)


def write_definition(output_dir: Path) -> None:
    try:
        version = importlib.metadata.version("torchcrepe")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    definition = {
        "schema_version": "IKD_torchcrepe_pitch_contours_v1",
        "source": "HT-Demucs vocals.wav",
        "torchcrepe_version": version,
        "model_capacity": MODEL_CAPACITY,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "hop_length_samples": HOP_LENGTH_SAMPLES,
        "frame_rate_hz": 100,
        "fmin_hz": FMIN_HZ,
        "fmax_hz": FMAX_HZ,
        "batch_size_frames": BATCH_SIZE_FRAMES,
        "decoder": "viterbi_bin_center",
        "frequency_dither": False,
        "pad": True,
        "random_seed": RANDOM_SEED,
        "target_information_used": False,
    }
    path = output_dir / "pitch_contour_definition.json"
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
    write_definition(output_dir)
    device = resolve_device(args.device)
    for track_id in tqdm(tracks, desc="TorchCREPE tracks", unit="track", dynamic_ncols=True):
        rows = metadata.loc[metadata["track_id"].astype(str) == track_id]
        paths = set(rows["vocal_audio_relative_path"].dropna().astype(str))
        if len(paths) != 1:
            raise ValueError(f"{track_id}: inconsistent vocal paths")
        audio_path = resolve_audio(audio_root, paths.pop())
        output = output_dir / f"{track_id}_pitch_contour.npz"
        if output.is_file() and not args.overwrite:
            tqdm.write(f"Skipped existing {output}")
            continue
        extract_track(track_id, audio_path, device, output)
        tqdm.write(f"Wrote {output}")


if __name__ == "__main__":
    main()
