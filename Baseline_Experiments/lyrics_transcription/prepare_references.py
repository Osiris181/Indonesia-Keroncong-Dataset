#!/usr/bin/env python3
"""Build lexical lyrics-transcription references from released IKD annotations."""

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
DEFAULT_OUTPUT = TASK_ROOT / "data" / "transcription_references.json"

NON_LEXICAL_MARKER = "[non-lexical]"
SCHEMA_VERSION = "IKD_lyrics_transcription_references_v1"
NORMALIZATION_VERSION = "ikd_lexical_text_v1"
APOSTROPHE_CHARACTERS = {"'", "’", "‘", "ʼ", "＇"}


def normalize_lexical_text(text: str) -> str:
    """Apply the fixed lexical-text normalization used for every score."""

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


def portable_path(path: Path) -> str:
    """Return a release-relative path, never a machine-specific absolute path."""

    try:
        return path.resolve().relative_to(RELEASE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as annotation_file:
        annotation = json.load(annotation_file)
    if not isinstance(annotation, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return annotation


def prepare_track(path: Path) -> dict[str, Any]:
    annotation = load_annotation(path)
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
    lexical_words: list[dict[str, Any]] = []
    non_lexical_intervals: list[dict[str, Any]] = []

    for word in ordered_words:
        text = str(word.get("text", "")).strip()
        start_sec = float(word["start_sec"])
        end_sec = float(word["end_sec"])
        if not text:
            raise ValueError(f"{track_id}: word with empty text")
        if end_sec <= start_sec:
            raise ValueError(
                f"{track_id}: invalid interval {start_sec:.6f}--{end_sec:.6f}"
            )

        base_record = {
            "source_region_id": str(word.get("source_region_id", "")),
            "text": text,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
        if is_non_lexical(text):
            non_lexical_intervals.append(base_record)
            continue

        normalized_text = normalize_lexical_text(text)
        if not normalized_text:
            raise ValueError(f"{track_id}: lexical word normalizes to empty: {text!r}")
        lexical_words.append({**base_record, "normalized_text": normalized_text})

    raw_text = " ".join(word["text"] for word in lexical_words)
    normalized_text = normalize_lexical_text(raw_text)
    return {
        "track_id": track_id,
        "lyrics_language": str(metadata.get("lyrics_language", "")).strip(),
        "source_annotation_file": portable_path(path),
        "reference_text_raw": raw_text,
        "reference_text_normalized": normalized_text,
        "reference_word_count": len(lexical_words),
        "normalized_token_count": len(normalized_text.split()),
        "reference_words": lexical_words,
        "non_lexical_intervals": non_lexical_intervals,
    }


def build_references(annotation_dir: Path) -> dict[str, Any]:
    annotation_paths = sorted(annotation_dir.glob("*_completed_annotation.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"No completed annotations found in {annotation_dir}")

    tracks = sorted(
        (prepare_track(path) for path in annotation_paths),
        key=lambda track: track["track_id"],
    )
    track_ids = [str(track["track_id"]) for track in tracks]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("Duplicate track IDs found in completed annotations")

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "lexical_lyrics_transcription",
        "source_annotation_directory": portable_path(annotation_dir),
        "non_lexical_policy": {
            "marker": NON_LEXICAL_MARKER,
            "input_audio": "retained",
            "reference_text": "excluded",
            "metric_scoring": "excluded by evaluator using annotated intervals",
            "model_access_to_intervals": False,
        },
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "unicode_form": "NFC",
            "lowercase": True,
            "apostrophes": "remove without introducing a word boundary",
            "other_punctuation": "replace with a space",
            "whitespace": "collapse consecutive whitespace and strip edges",
            "diacritics": "preserved",
        },
        "track_count": len(tracks),
        "tracks": tracks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare normalized IKD lexical transcription references."
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing the released completed annotation JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    references = build_references(args.annotation_dir.resolve())
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(references, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    lexical_words = sum(
        int(track["reference_word_count"]) for track in references["tracks"]
    )
    non_lexical = sum(
        len(track["non_lexical_intervals"]) for track in references["tracks"]
    )
    print(f"Prepared {references['track_count']} transcription references.")
    print(f"Lexical words: {lexical_words}")
    print(f"Non-lexical intervals: {non_lexical}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
