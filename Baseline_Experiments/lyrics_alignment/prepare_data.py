#!/usr/bin/env python3
"""Prepare timing-free lyrics-alignment inputs and evaluation ground truth."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
TASK_ROOT = SCRIPT_PATH.parent
RELEASE_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_ANNOTATION_DIR = RELEASE_ROOT / "IKD_Dataset" / "Song_Annotations"
DEFAULT_INPUT_OUTPUT = TASK_ROOT / "data" / "alignment_inputs.json"
DEFAULT_GROUND_TRUTH_OUTPUT = TASK_ROOT / "data" / "alignment_ground_truth.json"

INPUT_SCHEMA_VERSION = "IKD_reference_text_alignment_inputs_v1"
GROUND_TRUTH_SCHEMA_VERSION = "IKD_alignment_ground_truth_v1"
NON_LEXICAL_MARKER = "[non-lexical]"
APOSTROPHE_CHARACTERS = {"'", "’", "‘", "ʼ", "＇"}
EXPECTED_RELEASE_TOTALS = {
    "tracks": 30,
    "words": 2943,
    "syllables": 6772,
    "non_lexical_intervals": 9,
}


def normalize_lexical_text(text: str) -> str:
    """Apply the fixed text normalization shared with transcription scoring."""

    normalized = unicodedata.normalize("NFC", text).lower()
    output_characters: list[str] = []
    for character in normalized:
        if character in APOSTROPHE_CHARACTERS:
            continue
        if unicodedata.category(character).startswith("P"):
            output_characters.append(" ")
        else:
            output_characters.append(character)
    return re.sub(r"\s+", " ", "".join(output_characters)).strip()


def is_non_lexical(text: str) -> bool:
    return unicodedata.normalize("NFC", text).strip().lower() == NON_LEXICAL_MARKER


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return data


def portable_path(path: Path) -> str:
    """Return a release-relative path rather than a machine-specific path."""

    try:
        return path.resolve().relative_to(RELEASE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def normalized_unit(text: str, description: str) -> str:
    normalized = normalize_lexical_text(text)
    if not normalized:
        raise ValueError(f"{description}: text normalizes to empty")
    if len(normalized.split()) != 1:
        raise ValueError(
            f"{description}: alignment unit must normalize to one token: "
            f"{normalized!r}"
        )
    return normalized


def prepare_track(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    annotation = load_json(path)
    track_id = str(annotation.get("track_id", "")).strip()
    if not track_id:
        raise ValueError(f"{path}: missing track_id")
    if annotation.get("annotation_status") != "completed":
        raise ValueError(f"{track_id}: annotation_status is not completed")

    metadata = annotation.get("metadata")
    words = annotation.get("words")
    if not isinstance(metadata, dict) or not isinstance(words, list):
        raise TypeError(f"{track_id}: missing metadata or words")

    ordered_words = sorted(
        words,
        key=lambda word: (
            float(word["start_sec"]),
            float(word["end_sec"]),
            str(word.get("source_region_id", "")),
        ),
    )
    input_words: list[dict[str, Any]] = []
    input_syllables: list[dict[str, Any]] = []
    ground_truth_words: list[dict[str, Any]] = []
    ground_truth_syllables: list[dict[str, Any]] = []
    non_lexical_intervals: list[dict[str, Any]] = []

    for word in ordered_words:
        word_text = str(word.get("text", "")).strip()
        word_start = float(word["start_sec"])
        word_end = float(word["end_sec"])
        if word_end <= word_start:
            raise ValueError(f"{track_id}: invalid word interval for {word_text!r}")

        if is_non_lexical(word_text):
            non_lexical_intervals.append(
                {
                    "source_region_id": str(word.get("source_region_id", "")),
                    "text": word_text,
                    "start_sec": word_start,
                    "end_sec": word_end,
                }
            )
            continue

        word_index = len(input_words)
        word_normalized = normalized_unit(
            word_text, f"{track_id} word {word.get('source_region_id')}"
        )
        input_word = {
            "unit_index": word_index,
            "text": word_text,
            "normalized_text": word_normalized,
        }
        input_words.append(input_word)
        ground_truth_words.append(
            {
                **input_word,
                "source_region_id": str(word.get("source_region_id", "")),
                "start_sec": word_start,
                "end_sec": word_end,
            }
        )

        syllables = word.get("syllables")
        if not isinstance(syllables, list) or not syllables:
            raise ValueError(f"{track_id}: lexical word {word_text!r} has no syllables")

        normalized_syllables: list[str] = []
        for index_in_word, syllable in enumerate(syllables):
            syllable_text = str(syllable.get("text", "")).strip()
            syllable_normalized = normalized_unit(
                syllable_text,
                f"{track_id} syllable {syllable.get('source_region_id')}",
            )
            syllable_start = float(syllable["start_sec"])
            syllable_end = float(syllable["end_sec"])
            if syllable_end <= syllable_start:
                raise ValueError(
                    f"{track_id}: invalid syllable interval for {syllable_text!r}"
                )

            syllable_index = len(input_syllables)
            input_syllable = {
                "unit_index": syllable_index,
                "parent_word_index": word_index,
                "index_in_word": index_in_word,
                "text": syllable_text,
                "normalized_text": syllable_normalized,
            }
            input_syllables.append(input_syllable)
            ground_truth_syllables.append(
                {
                    **input_syllable,
                    "source_region_id": str(syllable.get("source_region_id", "")),
                    "start_sec": syllable_start,
                    "end_sec": syllable_end,
                }
            )
            normalized_syllables.append(syllable_normalized)

        if "".join(normalized_syllables) != word_normalized:
            raise ValueError(
                f"{track_id}: syllables {normalized_syllables!r} do not "
                f"reconstruct word {word_normalized!r}"
            )

    lyrics_language = str(metadata.get("lyrics_language", "")).strip()
    reference_text = " ".join(word["normalized_text"] for word in input_words)
    input_track = {
        "track_id": track_id,
        "lyrics_language": lyrics_language,
        "reference_text_normalized": reference_text,
        "word_count": len(input_words),
        "syllable_count": len(input_syllables),
        "word_units": input_words,
        "syllable_units": input_syllables,
    }
    ground_truth_track = {
        "track_id": track_id,
        "lyrics_language": lyrics_language,
        "source_annotation_file": portable_path(path),
        "word_count": len(ground_truth_words),
        "syllable_count": len(ground_truth_syllables),
        "word_units": ground_truth_words,
        "syllable_units": ground_truth_syllables,
        "non_lexical_intervals": non_lexical_intervals,
    }
    return input_track, ground_truth_track


def build_alignment_data(
    annotation_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    annotation_paths = sorted(annotation_dir.glob("*_completed_annotation.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"No completed annotations found in {annotation_dir}")

    prepared = [prepare_track(path) for path in annotation_paths]
    input_tracks = sorted((item[0] for item in prepared), key=lambda x: x["track_id"])
    ground_truth_tracks = sorted(
        (item[1] for item in prepared), key=lambda x: x["track_id"]
    )
    track_ids = [track["track_id"] for track in input_tracks]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("Duplicate track IDs found")

    policy = {
        "input_audio": "complete track; non-lexical regions retained",
        "input_text": "ordered normalized lexical lyrics",
        "annotated_timestamps_available_to_model": False,
        "non_lexical_intervals_available_to_model": False,
        "syllabification_available_for_output_mapping": True,
    }
    inputs = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "task": "reference_text_lyrics_alignment",
        "policy": policy,
        "track_count": len(input_tracks),
        "tracks": input_tracks,
    }
    ground_truth = {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "task": "reference_text_lyrics_alignment_evaluation",
        "source_annotation_directory": portable_path(annotation_dir),
        "policy": policy,
        "track_count": len(ground_truth_tracks),
        "tracks": ground_truth_tracks,
    }
    return inputs, ground_truth


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare timing-free inputs and ground truth for lyrics alignment."
    )
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--input-output", type=Path, default=DEFAULT_INPUT_OUTPUT)
    parser.add_argument(
        "--ground-truth-output", type=Path, default=DEFAULT_GROUND_TRUTH_OUTPUT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_dir = args.annotation_dir.resolve()
    inputs, ground_truth = build_alignment_data(annotation_dir)

    totals = {
        "tracks": int(inputs["track_count"]),
        "words": sum(track["word_count"] for track in inputs["tracks"]),
        "syllables": sum(track["syllable_count"] for track in inputs["tracks"]),
        "non_lexical_intervals": sum(
            len(track["non_lexical_intervals"])
            for track in ground_truth["tracks"]
        ),
    }
    if annotation_dir == DEFAULT_ANNOTATION_DIR.resolve():
        mismatches = {
            name: (EXPECTED_RELEASE_TOTALS[name], value)
            for name, value in totals.items()
            if value != EXPECTED_RELEASE_TOTALS[name]
        }
        if mismatches:
            raise ValueError(
                "Released annotation totals do not match the expected IKD release: "
                f"{mismatches}"
            )

    write_json(args.input_output.resolve(), inputs)
    write_json(args.ground_truth_output.resolve(), ground_truth)

    print(f"Prepared {totals['tracks']} alignment tracks.")
    print(f"Lexical words: {totals['words']}")
    print(f"Syllables: {totals['syllables']}")
    print(
        "Non-lexical intervals retained only in ground truth: "
        f"{totals['non_lexical_intervals']}"
    )
    print(f"Timing-free input: {args.input_output.resolve()}")
    print(f"Evaluation ground truth: {args.ground_truth_output.resolve()}")


if __name__ == "__main__":
    main()
