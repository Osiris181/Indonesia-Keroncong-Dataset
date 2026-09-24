#!/usr/bin/env python3
"""Evaluate masked and unmasked IKD lexical lyrics transcriptions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from prepare_references import normalize_lexical_text


SCRIPT_PATH = Path(__file__).resolve()
TASK_ROOT = SCRIPT_PATH.parent
DEFAULT_REFERENCE = TASK_ROOT / "data" / "transcription_references.json"
DEFAULT_CONFIG = TASK_ROOT / "config.json"
DEFAULT_INPUT_ROOT = TASK_ROOT / "outputs" / "predictions"
DEFAULT_OUTPUT_ROOT = TASK_ROOT / "outputs" / "evaluation"
POLICIES = ("masked", "unmasked")
EVALUATION_SCHEMA_VERSION = "IKD_lyrics_transcription_evaluation_v1"


@dataclass(frozen=True)
class EditCounts:
    reference_units: int
    hypothesis_units: int
    correct: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        if self.reference_units == 0:
            return 0.0 if self.hypothesis_units == 0 else float("inf")
        return self.errors / self.reference_units

    @property
    def substitution_rate(self) -> float:
        return self.substitutions / self.reference_units

    @property
    def deletion_rate(self) -> float:
        return self.deletions / self.reference_units

    @property
    def insertion_rate(self) -> float:
        return self.insertions / self.reference_units


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(TASK_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def config_fingerprint(config: dict[str, Any]) -> str:
    serialized = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def levenshtein_counts(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> EditCounts:
    """Return a deterministic minimum-edit decomposition.

    Exact matches are preferred. Other equal-cost paths use substitution,
    deletion, then insertion priority.
    """

    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distances = [[0] * columns for _ in range(rows)]
    operations = [[""] * columns for _ in range(rows)]

    for row in range(1, rows):
        distances[row][0] = row
        operations[row][0] = "deletion"
    for column in range(1, columns):
        distances[0][column] = column
        operations[0][column] = "insertion"

    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                distances[row][column] = distances[row - 1][column - 1]
                operations[row][column] = "correct"
                continue
            candidates = (
                (distances[row - 1][column - 1] + 1, 0, "substitution"),
                (distances[row - 1][column] + 1, 1, "deletion"),
                (distances[row][column - 1] + 1, 2, "insertion"),
            )
            distance, _, operation = min(candidates)
            distances[row][column] = distance
            operations[row][column] = operation

    correct = substitutions = deletions = insertions = 0
    row = len(reference)
    column = len(hypothesis)
    while row > 0 or column > 0:
        operation = operations[row][column]
        if operation == "correct":
            correct += 1
            row -= 1
            column -= 1
        elif operation == "substitution":
            substitutions += 1
            row -= 1
            column -= 1
        elif operation == "deletion":
            deletions += 1
            row -= 1
        elif operation == "insertion":
            insertions += 1
            column -= 1
        else:
            raise RuntimeError(
                f"Invalid edit backtrace at reference={row}, hypothesis={column}"
            )

    return EditCounts(
        reference_units=len(reference),
        hypothesis_units=len(hypothesis),
        correct=correct,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
    )


def normalized_characters(text: str) -> list[str]:
    return [character for character in text if not character.isspace()]


def midpoint_is_masked(
    start_sec: float, end_sec: float, intervals: Sequence[dict[str, Any]]
) -> bool:
    midpoint = (start_sec + end_sec) / 2.0
    return any(
        float(interval["start_sec"]) <= midpoint < float(interval["end_sec"])
        for interval in intervals
    )


def extract_hypotheses(
    prediction: dict[str, Any], intervals: Sequence[dict[str, Any]]
) -> tuple[str, str, int, int]:
    raw_segments = prediction.get("asr_segments")
    alignment = prediction.get("hypothesis_word_alignment")
    if not isinstance(raw_segments, list) or not isinstance(alignment, dict):
        raise ValueError("Prediction is missing ASR segments or word alignment")
    aligned_segments = alignment.get("segments")
    if not isinstance(aligned_segments, list):
        raise ValueError("Prediction is missing aligned hypothesis segments")

    unmasked_text = normalize_lexical_text(
        " ".join(str(segment.get("text", "")) for segment in raw_segments)
    )
    aligned_words: list[dict[str, Any]] = []
    for segment in aligned_segments:
        words = segment.get("words")
        if not isinstance(words, list):
            raise ValueError("Aligned hypothesis segment is missing its words list")
        aligned_words.extend(words)

    aligned_unmasked_text = normalize_lexical_text(
        " ".join(str(word.get("word", "")) for word in aligned_words)
    )
    untimestamped_word_count = sum(
        word.get("start") is None or word.get("end") is None
        for word in aligned_words
    )
    if aligned_unmasked_text != unmasked_text:
        if not intervals:
            reconciliation = levenshtein_counts(
                aligned_unmasked_text.split(), unmasked_text.split()
            )
            untimestamped_word_count += (
                reconciliation.substitutions + reconciliation.insertions
            )
            return (
                unmasked_text,
                unmasked_text,
                0,
                untimestamped_word_count,
            )
        raise ValueError(
            "Aligned hypothesis words do not reconstruct the raw ASR hypothesis "
            "for a track requiring non-lexical masking"
        )

    if not intervals:
        return unmasked_text, unmasked_text, 0, untimestamped_word_count

    masked_words: list[str] = []
    masked_word_count = 0
    for word in aligned_words:
        text = str(word.get("word", "")).strip()
        start = word.get("start")
        end = word.get("end")
        if start is None or end is None:
            masked_words.append(text)
            continue
        if midpoint_is_masked(float(start), float(end), intervals):
            masked_word_count += 1
            continue
        masked_words.append(text)

    return (
        normalize_lexical_text(" ".join(masked_words)),
        unmasked_text,
        masked_word_count,
        untimestamped_word_count,
    )


def discover_predictions(
    input_root: Path,
    expected_sources: Sequence[str],
    expected_track_ids: set[str],
    expected_experiment: str,
    expected_fingerprint: str,
    allow_incomplete: bool,
) -> dict[tuple[str, str], Path]:
    predictions: dict[tuple[str, str], Path] = {}
    for audio_source in expected_sources:
        source_directory = input_root / audio_source
        for path in sorted(source_directory.glob("*.json")):
            prediction = load_json(path)
            track_id = str(prediction.get("track_id", ""))
            file_source = str(prediction.get("audio_source", ""))
            if track_id not in expected_track_ids:
                raise ValueError(f"{path}: unexpected track ID {track_id!r}")
            if file_source != audio_source:
                raise ValueError(f"{path}: audio source does not match its directory")
            if prediction.get("experiment_name") != expected_experiment:
                raise ValueError(f"{path}: experiment name does not match config")
            if prediction.get("config_sha256") != expected_fingerprint:
                raise ValueError(f"{path}: configuration fingerprint mismatch")
            key = (audio_source, track_id)
            if key in predictions:
                raise ValueError(f"Duplicate prediction for {key}")
            predictions[key] = path

    expected_keys = {
        (audio_source, track_id)
        for audio_source in expected_sources
        for track_id in expected_track_ids
    }
    missing = sorted(expected_keys - set(predictions))
    if missing and not allow_incomplete:
        formatted = [f"{source}/{track_id}" for source, track_id in missing]
        raise FileNotFoundError(
            f"Missing {len(missing)} transcription outputs: {formatted}"
        )
    if not predictions:
        raise FileNotFoundError(f"No transcription outputs found under {input_root}")
    return predictions


def metric_record(
    track_id: str,
    language: str,
    audio_source: str,
    policy: str,
    reference_text: str,
    hypothesis_text: str,
    masked_word_count: int,
    untimestamped_word_count: int,
) -> dict[str, Any]:
    word_counts = levenshtein_counts(reference_text.split(), hypothesis_text.split())
    character_counts = levenshtein_counts(
        normalized_characters(reference_text),
        normalized_characters(hypothesis_text),
    )
    return {
        "track_id": track_id,
        "lyrics_language": language,
        "audio_source": audio_source,
        "policy": policy,
        "reference_text_normalized": reference_text,
        "hypothesis_text_normalized": hypothesis_text,
        "masked_hypothesis_word_count": masked_word_count,
        "untimestamped_hypothesis_word_count": untimestamped_word_count,
        "word": {
            **asdict(word_counts),
            "errors": word_counts.errors,
            "wer": word_counts.error_rate,
            "substitution_rate": word_counts.substitution_rate,
            "deletion_rate": word_counts.deletion_rate,
            "insertion_rate": word_counts.insertion_rate,
        },
        "character": {
            **asdict(character_counts),
            "errors": character_counts.errors,
            "cer": character_counts.error_rate,
            "substitution_rate": character_counts.substitution_rate,
            "deletion_rate": character_counts.deletion_rate,
            "insertion_rate": character_counts.insertion_rate,
        },
    }


def sum_edit_counts(records: Sequence[dict[str, Any]], unit: str) -> EditCounts:
    return EditCounts(
        reference_units=sum(int(record[unit]["reference_units"]) for record in records),
        hypothesis_units=sum(
            int(record[unit]["hypothesis_units"]) for record in records
        ),
        correct=sum(int(record[unit]["correct"]) for record in records),
        substitutions=sum(int(record[unit]["substitutions"]) for record in records),
        deletions=sum(int(record[unit]["deletions"]) for record in records),
        insertions=sum(int(record[unit]["insertions"]) for record in records),
    )


def summarize_overall(
    records: Sequence[dict[str, Any]], audio_source: str, policy: str
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if record["audio_source"] == audio_source and record["policy"] == policy
    ]
    if not selected:
        raise ValueError(f"No records for {audio_source}/{policy}")
    word_counts = sum_edit_counts(selected, "word")
    character_counts = sum_edit_counts(selected, "character")
    return {
        "audio_source": audio_source,
        "policy": policy,
        "track_count": len(selected),
        "word": {
            **asdict(word_counts),
            "errors": word_counts.errors,
            "wer": word_counts.error_rate,
            "substitution_rate": word_counts.substitution_rate,
            "deletion_rate": word_counts.deletion_rate,
            "insertion_rate": word_counts.insertion_rate,
        },
        "character": {
            **asdict(character_counts),
            "errors": character_counts.errors,
            "cer": character_counts.error_rate,
        },
        "masked_hypothesis_word_count": sum(
            int(record["masked_hypothesis_word_count"]) for record in selected
        ),
        "untimestamped_hypothesis_word_count": sum(
            int(record["untimestamped_hypothesis_word_count"])
            for record in selected
        ),
    }


def build_overall_summaries(
    records: Sequence[dict[str, Any]], audio_sources: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        summarize_overall(records, audio_source, policy)
        for audio_source in audio_sources
        for policy in POLICIES
        if any(
            record["audio_source"] == audio_source and record["policy"] == policy
            for record in records
        )
    ]


def write_per_track_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    columns = [
        "track_id",
        "lyrics_language",
        "audio_source",
        "policy",
        "reference_words",
        "hypothesis_words",
        "word_substitutions",
        "word_deletions",
        "word_insertions",
        "wer",
        "reference_characters",
        "hypothesis_characters",
        "character_substitutions",
        "character_deletions",
        "character_insertions",
        "cer",
        "masked_hypothesis_words",
        "untimestamped_hypothesis_words",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "track_id": record["track_id"],
                    "lyrics_language": record["lyrics_language"],
                    "audio_source": record["audio_source"],
                    "policy": record["policy"],
                    "reference_words": record["word"]["reference_units"],
                    "hypothesis_words": record["word"]["hypothesis_units"],
                    "word_substitutions": record["word"]["substitutions"],
                    "word_deletions": record["word"]["deletions"],
                    "word_insertions": record["word"]["insertions"],
                    "wer": record["word"]["wer"],
                    "reference_characters": record["character"]["reference_units"],
                    "hypothesis_characters": record["character"]["hypothesis_units"],
                    "character_substitutions": record["character"]["substitutions"],
                    "character_deletions": record["character"]["deletions"],
                    "character_insertions": record["character"]["insertions"],
                    "cer": record["character"]["cer"],
                    "masked_hypothesis_words": record[
                        "masked_hypothesis_word_count"
                    ],
                    "untimestamped_hypothesis_words": record[
                        "untimestamped_hypothesis_word_count"
                    ],
                }
            )


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    word = summary["word"]
    character = summary["character"]
    return {
        "audio_source": summary["audio_source"],
        "policy": summary["policy"],
        "track_count": summary["track_count"],
        "reference_words": word["reference_units"],
        "hypothesis_words": word["hypothesis_units"],
        "word_substitutions": word["substitutions"],
        "word_deletions": word["deletions"],
        "word_insertions": word["insertions"],
        "wer": word["wer"],
        "cer": character["cer"],
        "word_substitution_rate": word["substitution_rate"],
        "word_deletion_rate": word["deletion_rate"],
        "word_insertion_rate": word["insertion_rate"],
        "masked_hypothesis_words": summary["masked_hypothesis_word_count"],
        "untimestamped_hypothesis_words": summary[
            "untimestamped_hypothesis_word_count"
        ],
    }


def write_overall_csv(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    rows = [flatten_summary(summary) for summary in summaries]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_text(summaries: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'Audio source':<14} {'Policy':<10} {'Tracks':>6} "
        f"{'WER':>9} {'CER':>9} {'Sub rate':>9} {'Del rate':>9} {'Ins rate':>9}"
    )
    lines = ["IKD lyrics transcription baseline", "", header, "-" * len(header)]
    for summary in summaries:
        word = summary["word"]
        lines.append(
            f"{summary['audio_source']:<14} {summary['policy']:<10} "
            f"{summary['track_count']:>6d} {word['wer']:>9.4f} "
            f"{summary['character']['cer']:>9.4f} "
            f"{word['substitution_rate']:>9.4f} "
            f"{word['deletion_rate']:>9.4f} {word['insertion_rate']:>9.4f}"
        )
    lines.extend(
        [
            "",
            "Primary aggregation: corpus-level micro error rates.",
            "CER unit: normalized non-whitespace Unicode characters.",
            "Masked policy: remove timestamped hypothesis words whose midpoint falls",
            "inside an annotated [non-lexical] interval.",
            "Unmasked policy: retain the complete full-audio hypothesis.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate IKD lyrics-transcription predictions."
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Evaluate available predictions instead of requiring all 60.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = args.reference.resolve()
    config_path = args.config.resolve()
    input_root = args.input_root.resolve()
    references = load_json(reference_path)
    config = load_json(config_path)
    expected_fingerprint = config_fingerprint(config)
    tracks = references.get("tracks")
    if not isinstance(tracks, list):
        raise TypeError("Reference file is missing its tracks list")
    reference_by_track = {str(track["track_id"]): track for track in tracks}
    if len(reference_by_track) != len(tracks):
        raise ValueError("Reference file contains duplicate track IDs")

    audio_sources = list(config["audio_sources"])
    predictions = discover_predictions(
        input_root,
        audio_sources,
        set(reference_by_track),
        str(config["experiment_name"]),
        expected_fingerprint,
        args.allow_incomplete,
    )

    records: list[dict[str, Any]] = []
    for (audio_source, track_id), prediction_path in sorted(predictions.items()):
        reference = reference_by_track[track_id]
        prediction = load_json(prediction_path)
        try:
            masked_text, unmasked_text, masked_count, untimestamped_count = (
                extract_hypotheses(prediction, reference["non_lexical_intervals"])
            )
        except ValueError as error:
            raise ValueError(f"{audio_source}/{track_id}: {error}") from error
        for policy, hypothesis_text in (
            ("masked", masked_text),
            ("unmasked", unmasked_text),
        ):
            records.append(
                metric_record(
                    track_id=track_id,
                    language=str(reference["lyrics_language"]),
                    audio_source=audio_source,
                    policy=policy,
                    reference_text=str(reference["reference_text_normalized"]),
                    hypothesis_text=hypothesis_text,
                    masked_word_count=masked_count if policy == "masked" else 0,
                    untimestamped_word_count=untimestamped_count,
                )
            )

    summaries = build_overall_summaries(records, audio_sources)
    output_root = args.output_root.resolve()
    evaluation = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "reference_file": portable_path(reference_path),
        "prediction_root": portable_path(input_root),
        "config_file": portable_path(config_path),
        "config_sha256": expected_fingerprint,
        "complete_dataset_required": not args.allow_incomplete,
        "evaluated_prediction_count": len(predictions),
        "metric_definitions": {
            "wer": "(word substitutions + deletions + insertions) / reference words",
            "cer": "(character substitutions + deletions + insertions) / reference non-whitespace characters",
            "edit_tie_priority": ["substitution", "deletion", "insertion"],
            "primary_aggregation": "corpus-level micro",
            "masked_policy": "exclude a timestamped hypothesis word when its midpoint is inside a non-lexical interval",
            "untimestamped_hypothesis_policy": "retain and score",
        },
        "overall": summaries,
        "per_track": records,
    }
    write_json(output_root / "metrics.json", evaluation)
    write_per_track_csv(output_root / "per_track_metrics.csv", records)
    write_overall_csv(output_root / "overall_metrics.csv", summaries)
    rendered_summary = summary_text(summaries)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.txt").open("w", encoding="utf-8") as output_file:
        output_file.write(rendered_summary)

    print(rendered_summary, end="")
    print(f"Detailed results: {output_root / 'metrics.json'}")


if __name__ == "__main__":
    main()
