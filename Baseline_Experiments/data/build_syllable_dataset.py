#!/usr/bin/env python3
"""Build the shared syllable-level modeling index from the IKD release."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
BASELINES_ROOT = SCRIPT_PATH.parents[1]
RELEASE_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DATASET_ROOT = RELEASE_ROOT / "IKD_Dataset"
DEFAULT_OUTPUT = SCRIPT_PATH.with_name("syllable_dataset.csv")

LABELS = ("luk", "cengkok", "gregel", "embat", "nggandhul")
RUNS = (1, 2, 3, 4, 5)
SPLITS = ("train", "validation", "test")
EXPECTED = {
    "tracks": 30,
    "syllables": 6_772,
    "with_technique": 2_676,
    "without_technique": 4_096,
    "multi_label": 736,
    "supports": {
        "luk": 1_658,
        "cengkok": 374,
        "gregel": 613,
        "embat": 563,
        "nggandhul": 351,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--track-index", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def release_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(RELEASE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_annotation_path(value: str, dataset_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    release_candidate = RELEASE_ROOT / path
    if release_candidate.is_file():
        return release_candidate
    return dataset_root / "Song_Annotations" / path.name


def load_split_roles(path: Path, track_ids: set[str]) -> dict[tuple[int, str], str]:
    roles: dict[tuple[int, str], str] = {}
    for row in read_csv(path):
        run = int(row["outer_fold"])
        track_id = row["track_id"].strip()
        split = row["split"].strip()
        if run not in RUNS or split not in SPLITS:
            raise ValueError(f"Invalid split assignment: {row}")
        key = (run, track_id)
        if key in roles:
            raise ValueError(f"Duplicate split assignment for run={run}, {track_id}")
        roles[key] = split
    expected = {(run, track_id) for run in RUNS for track_id in track_ids}
    if set(roles) != expected:
        missing = sorted(expected - set(roles))[:10]
        extra = sorted(set(roles) - expected)[:10]
        raise ValueError(f"Split coverage mismatch; missing={missing}, extra={extra}")
    for track_id in track_ids:
        observed = Counter(roles[(run, track_id)] for run in RUNS)
        if observed != Counter({"train": 3, "validation": 1, "test": 1}):
            raise ValueError(f"Invalid five-run rotation for {track_id}: {observed}")
    return roles


def build_rows(
    track_index: Path,
    split_file: Path,
    dataset_root: Path,
) -> list[dict[str, Any]]:
    tracks = read_csv(track_index)
    track_ids = {row["track_id"].strip() for row in tracks}
    if len(tracks) != EXPECTED["tracks"] or len(track_ids) != EXPECTED["tracks"]:
        raise ValueError("The released track index must contain 30 unique tracks")
    roles = load_split_roles(split_file, track_ids)
    output: list[dict[str, Any]] = []
    seen_syllable_ids: set[str] = set()

    for track in tracks:
        track_id = track["track_id"].strip()
        annotation_path = resolve_annotation_path(
            track["annotation_file"], dataset_root
        )
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Missing annotation for {track_id}: {annotation_path}"
            )
        with annotation_path.open(encoding="utf-8") as handle:
            annotation = json.load(handle)
        if annotation.get("track_id") != track_id:
            raise ValueError(f"Track ID mismatch in {annotation_path}")
        metadata = annotation["metadata"]
        if metadata.get("performer_group_id") != track["performer_group_id"]:
            raise ValueError(f"Performer-group mismatch for {track_id}")

        for word_index, word in enumerate(annotation.get("words", [])):
            word_id = str(word["source_region_id"])
            for syllable_index, syllable in enumerate(word.get("syllables", [])):
                syllable_id = str(syllable["source_region_id"])
                if syllable_id in seen_syllable_ids:
                    raise ValueError(f"Duplicate syllable ID: {syllable_id}")
                seen_syllable_ids.add(syllable_id)
                raw_labels = list(syllable.get("melisma_labels", []))
                positives = [label for label in LABELS if label in raw_labels]
                no_melisma = "no_melisma" in raw_labels
                if no_melisma == bool(positives):
                    raise ValueError(
                        f"{syllable_id} must contain either no_melisma or "
                        "one or more technique labels"
                    )
                start = float(syllable["start_sec"])
                end = float(syllable["end_sec"])
                if not 0.0 <= start < end:
                    raise ValueError(f"Invalid syllable interval: {syllable_id}")
                row: dict[str, Any] = {
                    "track_id": track_id,
                    "title": metadata.get("title", ""),
                    "artist": metadata.get("artist", ""),
                    "performer_group_id": metadata.get("performer_group_id", ""),
                    "genre": metadata.get("genre", ""),
                    "lyrics_language": metadata.get("lyrics_language", ""),
                    "annotation_file": release_relative(annotation_path),
                    "vocal_audio_relative_path": f"{track_id}/vocals.wav",
                    "word_index": word_index,
                    "word_id": word_id,
                    "word_text": word.get("text", ""),
                    "syllable_index_in_word": syllable_index,
                    "syllable_id": syllable_id,
                    "syllable_text": syllable.get("text", ""),
                    "start_sec": round(start, 9),
                    "end_sec": round(end, 9),
                    "duration_sec": round(end - start, 9),
                    "melisma_labels": "|".join(positives or ["no_melisma"]),
                    "no_melisma": int(no_melisma),
                    "is_ornamented": int(bool(positives)),
                    "label_count": len(positives),
                }
                row.update({label: int(label in positives) for label in LABELS})
                row.update(
                    {
                        f"fold_{run}_split": roles[(run, track_id)]
                        for run in RUNS
                    }
                )
                output.append(row)
    return output


def validate_counts(rows: list[dict[str, Any]]) -> None:
    with_technique = sum(int(row["is_ornamented"]) for row in rows)
    observed = {
        "syllables": len(rows),
        "with_technique": with_technique,
        "without_technique": len(rows) - with_technique,
        "multi_label": sum(int(row["label_count"]) > 1 for row in rows),
        "supports": {
            label: sum(int(row[label]) for row in rows) for label in LABELS
        },
    }
    expected = {key: value for key, value in EXPECTED.items() if key != "tracks"}
    if observed != expected:
        raise ValueError(f"Dataset statistics differ from the release: {observed}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    track_index = (
        args.track_index.resolve()
        if args.track_index
        else dataset_root / "Indonesian_Keroncong_Dataset.csv"
    )
    split_file = (
        args.split_file.resolve()
        if args.split_file
        else dataset_root / "Splits" / "track_fold_assignments.csv"
    )
    rows = build_rows(track_index, split_file, dataset_root)
    validate_counts(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} syllables to {args.output}")


if __name__ == "__main__":
    main()

