#!/usr/bin/env python3
"""Run the frozen WhisperX/Faster-Whisper lyrics-transcription baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


SCRIPT_PATH = Path(__file__).resolve()
TASK_ROOT = SCRIPT_PATH.parent
RELEASE_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = TASK_ROOT / "config.json"
DEFAULT_TRACK_INDEX = (
    RELEASE_ROOT / "IKD_Dataset" / "Indonesian_Keroncong_Dataset.csv"
)
DEFAULT_OUTPUT_ROOT = TASK_ROOT / "outputs" / "predictions"
OUTPUT_SCHEMA_VERSION = "IKD_whisperx_transcription_output_v1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return data


def config_fingerprint(config: dict[str, Any]) -> str:
    serialized = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def portable_path(path: Path, anchor: Path) -> str:
    try:
        return path.resolve().relative_to(anchor.resolve()).as_posix()
    except ValueError:
        return path.name


def load_tracks(track_index: Path) -> list[dict[str, str]]:
    with track_index.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"track_id", "lyrics_language"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{track_index}: required columns are {sorted(required_columns)}"
            )
        tracks = [
            {
                "track_id": str(row["track_id"]).strip(),
                "lyrics_language": str(row["lyrics_language"]).strip(),
            }
            for row in reader
        ]

    if any(not track["track_id"] for track in tracks):
        raise ValueError(f"{track_index}: empty track_id")
    track_ids = [track["track_id"] for track in tracks]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError(f"{track_index}: duplicate track_id")
    return sorted(tracks, key=lambda track: track["track_id"])


def select_tracks(
    tracks: list[dict[str, str]], requested_track_ids: list[str] | None
) -> list[dict[str, str]]:
    if not requested_track_ids:
        return tracks
    requested = set(requested_track_ids)
    selected = [track for track in tracks if track["track_id"] in requested]
    missing = sorted(requested - {track["track_id"] for track in selected})
    if missing:
        raise ValueError(f"Unknown track IDs: {missing}")
    return selected


def select_audio_sources(
    config: dict[str, Any], requested_sources: list[str] | None
) -> list[str]:
    configured = config.get("audio_sources")
    if not isinstance(configured, dict) or not configured:
        raise ValueError("The config must define at least one audio source")
    if not requested_sources:
        return list(configured)
    unknown = sorted(set(requested_sources) - set(configured))
    if unknown:
        raise ValueError(f"Unknown audio sources: {unknown}")
    return requested_sources


def requested_language(
    lyrics_language: str, language_routing: dict[str, str]
) -> str | None:
    if lyrics_language not in language_routing:
        raise ValueError(f"No language route configured for {lyrics_language!r}")
    language_code = language_routing[lyrics_language]
    return None if language_code == "auto" else language_code


def output_is_reusable(
    path: Path, fingerprint: str, track_id: str, audio_source: str
) -> bool:
    if not path.is_file():
        return False
    existing = load_json(path)
    return (
        existing.get("config_sha256") == fingerprint
        and existing.get("track_id") == track_id
        and existing.get("audio_source") == audio_source
    )


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(json_compatible(data), output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen WhisperX/Faster-Whisper transcription on IKD audio."
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Root containing <track_id>/original_mix.wav and vocals.wav.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--track-index", type=Path, default=DEFAULT_TRACK_INDEX)
    parser.add_argument(
        "--track-id",
        action="append",
        dest="track_ids",
        help="Process one track; repeat for multiple tracks.",
    )
    parser.add_argument(
        "--audio-source",
        action="append",
        dest="audio_sources",
        help="Process original_mix or vocal_stem; repeat for both.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Prediction root (default: outputs/predictions).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs instead of resuming them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    fingerprint = config_fingerprint(config)
    model_config = config["model"]
    vad_config = config["vad"]
    alignment_config = config["hypothesis_word_alignment"]
    language_routing = config["language_routing"]

    audio_root = args.audio_root.resolve()
    tracks = select_tracks(load_tracks(args.track_index.resolve()), args.track_ids)
    audio_sources = select_audio_sources(config, args.audio_sources)
    output_root = args.output_root.resolve()

    jobs: list[tuple[dict[str, str], str, Path, Path]] = []
    for track in tracks:
        track_id = track["track_id"]
        for audio_source in audio_sources:
            audio_filename = str(config["audio_sources"][audio_source])
            audio_path = audio_root / track_id / audio_filename
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            output_path = (
                output_root
                / audio_source
                / f"{track_id}_{config['experiment_name']}.json"
            )
            if output_path.exists() and not args.overwrite:
                if output_is_reusable(
                    output_path, fingerprint, track_id, audio_source
                ):
                    print(f"Skipping completed output: {output_path}")
                    continue
                raise FileExistsError(
                    f"{output_path} does not match this job and configuration; "
                    "use --overwrite or another --output-root"
                )
            jobs.append((track, audio_source, audio_path, output_path))

    if not jobs:
        print("No transcription jobs remain.")
        return

    import torch
    import whisperx

    device = str(model_config["device"])
    device_index = int(model_config["device_index"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The config requests CUDA, but CUDA is unavailable")

    model = whisperx.load_model(
        model_config["name"],
        device,
        device_index=device_index,
        compute_type=model_config["compute_type"],
        asr_options=dict(config["decoding"]),
        language=None,
        vad_method=vad_config["method"],
        vad_options={
            "chunk_size": int(vad_config["chunk_size"]),
            "vad_onset": float(vad_config["vad_onset"]),
            "vad_offset": float(vad_config["vad_offset"]),
        },
        task=model_config["task"],
        local_files_only=bool(model_config["local_files_only"]),
    )

    align_model = align_metadata = None
    if bool(alignment_config["enabled"]):
        align_model, align_metadata = whisperx.load_align_model(
            language_code=alignment_config["language_code"],
            device=device,
            model_name=alignment_config["model_name"],
        )

    software = {
        "python": platform.python_version(),
        "whisperx": package_version("whisperx"),
        "faster_whisper": package_version("faster-whisper"),
        "torch": package_version("torch"),
        "ctranslate2": package_version("ctranslate2"),
        "cuda_device": (
            torch.cuda.get_device_name(device_index) if device == "cuda" else None
        ),
    }

    progress = tqdm(
        jobs,
        desc="Lyrics transcription",
        unit="job",
        dynamic_ncols=True,
    )
    for track, audio_source, audio_path, output_path in progress:
        track_id = track["track_id"]
        progress.set_postfix_str(f"{track_id} {audio_source}", refresh=True)
        language_code = requested_language(
            track["lyrics_language"], language_routing
        )
        audio = whisperx.load_audio(str(audio_path))
        start_time = time.perf_counter()
        transcription = model.transcribe(
            audio,
            batch_size=int(model_config["batch_size"]),
            num_workers=int(model_config["num_workers"]),
            language=language_code,
            task=model_config["task"],
            chunk_size=int(vad_config["chunk_size"]),
            print_progress=False,
            verbose=False,
        )

        hypothesis_alignment = None
        if align_model is not None and align_metadata is not None:
            hypothesis_alignment = whisperx.align(
                transcription["segments"],
                align_model,
                align_metadata,
                audio,
                device,
                return_char_alignments=bool(
                    alignment_config["return_char_alignments"]
                ),
            )

        output = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "experiment_name": config["experiment_name"],
            "config_file": portable_path(config_path, TASK_ROOT),
            "config_sha256": fingerprint,
            "track_id": track_id,
            "lyrics_language_metadata": track["lyrics_language"],
            "requested_whisper_language": language_code,
            "returned_whisper_language": transcription.get("language"),
            "audio_source": audio_source,
            "audio_file": portable_path(audio_path, audio_root),
            "audio_duration_sec": len(audio) / whisperx.audio.SAMPLE_RATE,
            "inference_seconds": time.perf_counter() - start_time,
            "software": software,
            "model_revisions": {
                "faster_whisper": config["reported_environment"].get(
                    "faster_whisper_model_revision"
                ),
                "hypothesis_ctc_alignment": config["reported_environment"].get(
                    "ctc_alignment_model_revision"
                ),
            },
            "asr_segments": transcription["segments"],
            "hypothesis_word_alignment": hypothesis_alignment,
            "ground_truth_access": {
                "reference_lyrics": False,
                "word_or_syllable_boundaries": False,
                "non_lexical_intervals": False,
            },
        }
        write_json_atomic(output_path, output)

    print(f"Completed {len(jobs)} track-source transcription job(s).")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
