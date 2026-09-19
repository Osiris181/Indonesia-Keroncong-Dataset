#!/usr/bin/env python3
"""Align provided IKD lyrics to complete tracks with a frozen CTC model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_INPUT = TASK_ROOT / "data" / "alignment_inputs.json"
DEFAULT_OUTPUT_ROOT = TASK_ROOT / "outputs" / "predictions"
OUTPUT_SCHEMA_VERSION = "IKD_whisperx_reference_text_alignment_output_v1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    serialized = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(RELEASE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


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


def select_tracks(
    tracks: list[dict[str, Any]], requested_track_ids: list[str] | None
) -> list[dict[str, Any]]:
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


def output_is_reusable(
    path: Path,
    config_sha256: str,
    input_sha256: str,
    audio_sha256: str,
    track_id: str,
    audio_source: str,
) -> bool:
    if not path.is_file():
        return False
    existing = load_json(path)
    return (
        existing.get("config_sha256") == config_sha256
        and existing.get("alignment_input_sha256") == input_sha256
        and existing.get("audio_sha256") == audio_sha256
        and existing.get("track_id") == track_id
        and existing.get("audio_source") == audio_source
    )


def character_spans(
    track: dict[str, Any],
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    reference_text = str(track["reference_text_normalized"])
    word_units = sorted(track["word_units"], key=lambda unit: unit["unit_index"])
    syllable_units = sorted(
        track["syllable_units"], key=lambda unit: unit["unit_index"]
    )
    syllables_by_word: dict[int, list[dict[str, Any]]] = {}
    for syllable in syllable_units:
        syllables_by_word.setdefault(int(syllable["parent_word_index"]), []).append(
            syllable
        )

    word_spans: dict[int, tuple[int, int]] = {}
    syllable_spans: dict[int, tuple[int, int]] = {}
    cursor = 0
    for position, word in enumerate(word_units):
        word_index = int(word["unit_index"])
        word_text = str(word["normalized_text"])
        word_start = cursor
        word_end = word_start + len(word_text)
        if reference_text[word_start:word_end] != word_text:
            raise ValueError(
                f"{track['track_id']}: reference text does not match word "
                f"{word_index}"
            )
        word_spans[word_index] = (word_start, word_end)

        local_cursor = word_start
        word_syllables = sorted(
            syllables_by_word.get(word_index, []),
            key=lambda unit: unit["index_in_word"],
        )
        if not word_syllables:
            raise ValueError(f"{track['track_id']}: word {word_index} has no syllables")
        for syllable in word_syllables:
            syllable_index = int(syllable["unit_index"])
            syllable_text = str(syllable["normalized_text"])
            syllable_end = local_cursor + len(syllable_text)
            if reference_text[local_cursor:syllable_end] != syllable_text:
                raise ValueError(
                    f"{track['track_id']}: reference text does not match syllable "
                    f"{syllable_index}"
                )
            syllable_spans[syllable_index] = (local_cursor, syllable_end)
            local_cursor = syllable_end
        if local_cursor != word_end:
            raise ValueError(
                f"{track['track_id']}: syllable spellings do not cover word "
                f"{word_index}"
            )

        cursor = word_end
        if position < len(word_units) - 1:
            if cursor >= len(reference_text) or reference_text[cursor] != " ":
                raise ValueError(f"{track['track_id']}: missing word separator")
            cursor += 1

    if cursor != len(reference_text):
        raise ValueError(f"{track['track_id']}: unused reference-text characters")
    return word_spans, syllable_spans


def flatten_characters(alignment: dict[str, Any]) -> list[dict[str, Any]]:
    characters: list[dict[str, Any]] = []
    for segment in alignment.get("segments", []):
        segment_characters = segment.get("chars")
        if not isinstance(segment_characters, list):
            raise ValueError("WhisperX did not return character alignments")
        characters.extend(segment_characters)
    return characters


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def derive_units(
    units: list[dict[str, Any]],
    spans: dict[int, tuple[int, int]],
    characters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: item["unit_index"]):
        unit_index = int(unit["unit_index"])
        start_index, end_index = spans[unit_index]
        unit_characters = characters[start_index:end_index]
        starts = [
            float(character["start"])
            for character in unit_characters
            if finite_number(character.get("start"))
        ]
        ends = [
            float(character["end"])
            for character in unit_characters
            if finite_number(character.get("end"))
        ]
        scores = [
            float(character["score"])
            for character in unit_characters
            if finite_number(character.get("score"))
        ]
        aligned_count = sum(
            finite_number(character.get("start"))
            and finite_number(character.get("end"))
            for character in unit_characters
        )
        predictions.append(
            {
                **unit,
                "start_sec": starts[0] if starts else None,
                "end_sec": ends[-1] if ends else None,
                "score": sum(scores) / len(scores) if scores else None,
                "character_count": len(unit_characters),
                "aligned_character_count": aligned_count,
            }
        )
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align provided reference lyrics to complete IKD tracks."
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Root containing <track_id>/original_mix.wav and vocals.wav.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    input_path = args.input.resolve()
    config = load_json(config_path)
    config_sha256 = config_fingerprint(config)
    alignment_inputs = load_json(input_path)
    input_sha256 = file_sha256(input_path)
    tracks = select_tracks(alignment_inputs["tracks"], args.track_ids)
    audio_sources = select_audio_sources(config, args.audio_sources)
    audio_root = args.audio_root.resolve()
    output_root = args.output_root.resolve()

    jobs: list[tuple[dict[str, Any], str, Path, Path, str, str]] = []
    for track in tracks:
        track_id = str(track["track_id"])
        for audio_source in audio_sources:
            audio_filename = str(config["audio_sources"][audio_source])
            audio_path = audio_root / track_id / audio_filename
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            audio_sha256 = file_sha256(audio_path)
            output_path = (
                output_root
                / audio_source
                / f"{track_id}_{config['experiment_name']}.json"
            )
            if output_path.exists() and not args.overwrite:
                if output_is_reusable(
                    output_path,
                    config_sha256,
                    input_sha256,
                    audio_sha256,
                    track_id,
                    audio_source,
                ):
                    print(f"Skipping completed output: {output_path}")
                    continue
                raise FileExistsError(
                    f"{output_path} does not match this job and configuration; "
                    "use --overwrite or another --output-root"
                )
            jobs.append(
                (
                    track,
                    audio_source,
                    audio_path,
                    output_path,
                    audio_filename,
                    audio_sha256,
                )
            )

    if not jobs:
        print("No alignment jobs remain.")
        return

    import torch
    import whisperx

    model_config = config["model"]
    device = str(model_config["device"])
    device_index = int(model_config["device_index"])
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("The config requests CUDA, but CUDA is unavailable")
        torch.cuda.set_device(device_index)

    align_model, align_metadata = whisperx.load_align_model(
        language_code=str(model_config["language_code"]),
        device=device,
        model_name=str(model_config["name"]),
    )
    software = {
        "python": platform.python_version(),
        "whisperx": package_version("whisperx"),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "cuda_device": (
            torch.cuda.get_device_name(device_index) if device == "cuda" else None
        ),
    }

    progress = tqdm(
        jobs,
        desc="Lyrics alignment",
        unit="job",
        dynamic_ncols=True,
    )
    for (
        track,
        audio_source,
        audio_path,
        output_path,
        audio_filename,
        audio_sha256,
    ) in progress:
        progress.set_postfix_str(
            f"{track['track_id']} {audio_source}", refresh=True
        )
        audio = whisperx.load_audio(str(audio_path))
        duration_sec = len(audio) / whisperx.audio.SAMPLE_RATE
        transcript = [
            {
                "text": track["reference_text_normalized"],
                "start": 0.0,
                "end": duration_sec,
            }
        ]
        start_time = time.perf_counter()
        alignment = whisperx.align(
            transcript,
            align_model,
            align_metadata,
            audio,
            device,
            interpolate_method=str(model_config["interpolate_method"]),
            return_char_alignments=bool(model_config["return_char_alignments"]),
            print_progress=False,
        )
        elapsed_seconds = time.perf_counter() - start_time

        characters = flatten_characters(alignment)
        reconstructed_text = "".join(str(character["char"]) for character in characters)
        if reconstructed_text != track["reference_text_normalized"]:
            raise ValueError(
                f"{track['track_id']}: WhisperX character output does not "
                "reconstruct the provided reference text"
            )
        indexed_characters = [
            {"character_index": index, **character}
            for index, character in enumerate(characters)
        ]
        word_spans, syllable_spans = character_spans(track)
        predicted_words = derive_units(
            track["word_units"], word_spans, indexed_characters
        )
        predicted_syllables = derive_units(
            track["syllable_units"], syllable_spans, indexed_characters
        )

        output = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "experiment_name": config["experiment_name"],
            "config_file": portable_path(config_path),
            "config_sha256": config_sha256,
            "alignment_input_file": portable_path(input_path),
            "alignment_input_sha256": input_sha256,
            "track_id": track["track_id"],
            "lyrics_language": track["lyrics_language"],
            "audio_source": audio_source,
            "audio_file": f"{track['track_id']}/{audio_filename}",
            "audio_sha256": audio_sha256,
            "audio_duration_sec": duration_sec,
            "inference_seconds": elapsed_seconds,
            "software": software,
            "model": model_config,
            "character_alignment": indexed_characters,
            "predicted_word_units": predicted_words,
            "predicted_syllable_units": predicted_syllables,
            "reference_information_available": {
                "lexical_lyrics": True,
                "orthographic_syllabification_without_timestamps": True,
                "word_or_syllable_timestamps": False,
                "non_lexical_intervals": False,
            },
            "postprocessing": "none",
        }
        write_json_atomic(output_path, output)

    print(f"Completed {len(jobs)} track-source alignment job(s).")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
