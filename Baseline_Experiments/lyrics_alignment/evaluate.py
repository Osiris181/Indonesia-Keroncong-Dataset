#!/usr/bin/env python3
"""Evaluate word- and syllable-level IKD lyrics alignment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
TASK_ROOT = SCRIPT_PATH.parent
RELEASE_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = TASK_ROOT / "config.json"
DEFAULT_INPUT = TASK_ROOT / "data" / "alignment_inputs.json"
DEFAULT_GROUND_TRUTH = TASK_ROOT / "data" / "alignment_ground_truth.json"
DEFAULT_PREDICTION_ROOT = TASK_ROOT / "outputs" / "predictions"
DEFAULT_OUTPUT_DIR = TASK_ROOT / "outputs" / "evaluation"
METRIC_SCHEMA_VERSION = "IKD_reference_text_alignment_metrics_v1"
TOLERANCES_SECONDS = (0.05, 0.10, 0.20, 0.30)
LEVEL_SORT_ORDER = {"word": 0, "syllable": 1}


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


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


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
    configured = list(config["audio_sources"])
    if not requested_sources:
        return configured
    unknown = sorted(set(requested_sources) - set(configured))
    if unknown:
        raise ValueError(f"Unknown audio sources: {unknown}")
    return requested_sources


def interval_overlap(
    reference_start: float,
    reference_end: float,
    predicted_start: float,
    predicted_end: float,
) -> float:
    return max(
        0.0, min(reference_end, predicted_end) - max(reference_start, predicted_start)
    )


def interval_iou(
    reference_start: float,
    reference_end: float,
    predicted_start: float,
    predicted_end: float,
) -> float:
    intersection = interval_overlap(
        reference_start, reference_end, predicted_start, predicted_end
    )
    union = max(reference_end, predicted_end) - min(reference_start, predicted_start)
    return intersection / union if union > 0.0 else 0.0


def evaluate_units(
    track: dict[str, Any],
    prediction: dict[str, Any],
    audio_source: str,
    level: str,
) -> list[dict[str, Any]]:
    reference_units = track[f"{level}_units"]
    predicted_units = prediction[f"predicted_{level}_units"]
    predictions_by_index = {
        int(unit["unit_index"]): unit for unit in predicted_units
    }
    if len(predictions_by_index) != len(predicted_units):
        raise ValueError(
            f"{track['track_id']} {audio_source} {level}: duplicate predicted indices"
        )
    reference_indices = {int(unit["unit_index"]) for unit in reference_units}
    unexpected_indices = sorted(set(predictions_by_index) - reference_indices)
    if unexpected_indices:
        raise ValueError(
            f"{track['track_id']} {audio_source} {level}: unexpected predicted "
            f"unit indices {unexpected_indices}"
        )

    rows: list[dict[str, Any]] = []
    for reference in reference_units:
        unit_index = int(reference["unit_index"])
        if unit_index not in predictions_by_index:
            raise ValueError(
                f"{track['track_id']} {audio_source} {level}: missing unit "
                f"{unit_index}"
            )
        predicted = predictions_by_index[unit_index]
        if predicted.get("normalized_text") != reference.get("normalized_text"):
            raise ValueError(
                f"{track['track_id']} {audio_source} {level} unit {unit_index}: "
                "predicted and reference text differ"
            )

        predicted_start = predicted.get("start_sec")
        predicted_end = predicted.get("end_sec")
        aligned = (
            finite_number(predicted_start)
            and finite_number(predicted_end)
            and float(predicted_end) >= float(predicted_start)
        )
        row: dict[str, Any] = {
            "track_id": track["track_id"],
            "lyrics_language": track["lyrics_language"],
            "audio_source": audio_source,
            "level": level,
            "unit_index": unit_index,
            "source_region_id": reference["source_region_id"],
            "text": reference["text"],
            "reference_start_sec": float(reference["start_sec"]),
            "reference_end_sec": float(reference["end_sec"]),
            "predicted_start_sec": float(predicted_start) if aligned else None,
            "predicted_end_sec": float(predicted_end) if aligned else None,
            "aligned": aligned,
            "score": predicted.get("score"),
            "character_count": predicted.get("character_count"),
            "aligned_character_count": predicted.get("aligned_character_count"),
        }
        if aligned:
            onset_signed = float(predicted_start) - float(reference["start_sec"])
            offset_signed = float(predicted_end) - float(reference["end_sec"])
            row.update(
                {
                    "onset_signed_error_sec": onset_signed,
                    "onset_absolute_error_sec": abs(onset_signed),
                    "offset_signed_error_sec": offset_signed,
                    "offset_absolute_error_sec": abs(offset_signed),
                    "segment_iou": interval_iou(
                        float(reference["start_sec"]),
                        float(reference["end_sec"]),
                        float(predicted_start),
                        float(predicted_end),
                    ),
                }
            )
        else:
            row.update(
                {
                    "onset_signed_error_sec": None,
                    "onset_absolute_error_sec": None,
                    "offset_signed_error_sec": None,
                    "offset_absolute_error_sec": None,
                    "segment_iou": None,
                }
            )
        rows.append(row)
    return rows


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def correct_segment_durations(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Return correctly overlapped unit/gap duration and evaluated span."""

    rows_by_track: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_track.setdefault(str(row["track_id"]), []).append(row)

    correct_duration = 0.0
    evaluated_span = 0.0
    for track_rows in rows_by_track.values():
        ordered = sorted(track_rows, key=lambda row: int(row["unit_index"]))
        if not ordered:
            continue
        evaluated_span += max(
            0.0,
            float(ordered[-1]["reference_end_sec"])
            - float(ordered[0]["reference_start_sec"]),
        )
        for index, row in enumerate(ordered):
            if row["aligned"]:
                correct_duration += interval_overlap(
                    float(row["reference_start_sec"]),
                    float(row["reference_end_sec"]),
                    float(row["predicted_start_sec"]),
                    float(row["predicted_end_sec"]),
                )
            if index >= len(ordered) - 1:
                continue
            next_row = ordered[index + 1]
            if row["aligned"] and next_row["aligned"]:
                correct_duration += interval_overlap(
                    float(row["reference_end_sec"]),
                    float(next_row["reference_start_sec"]),
                    float(row["predicted_end_sec"]),
                    float(next_row["predicted_start_sec"]),
                )
    return correct_duration, evaluated_span


def summarize_rows(
    rows: list[dict[str, Any]],
    group_type: str,
    group_value: str,
    audio_source: str,
    level: str,
) -> dict[str, Any]:
    total = len(rows)
    aligned_rows = [row for row in rows if row["aligned"]]
    aligned = len(aligned_rows)
    onset_absolute = [row["onset_absolute_error_sec"] for row in aligned_rows]
    offset_absolute = [row["offset_absolute_error_sec"] for row in aligned_rows]
    onset_signed = [row["onset_signed_error_sec"] for row in aligned_rows]
    offset_signed = [row["offset_signed_error_sec"] for row in aligned_rows]
    ious = [row["segment_iou"] for row in aligned_rows]
    correct_duration, evaluated_span = correct_segment_durations(rows)

    summary: dict[str, Any] = {
        "group_type": group_type,
        "group_value": group_value,
        "audio_source": audio_source,
        "level": level,
        "support": total,
        "aligned_units": aligned,
        "coverage": aligned / total if total else None,
        "correct_segment_duration_sec": correct_duration,
        "evaluated_span_sec": evaluated_span,
        "pcs": correct_duration / evaluated_span if evaluated_span > 0.0 else None,
        "onset_mae_sec": mean_or_none(onset_absolute),
        "onset_median_ae_sec": median_or_none(onset_absolute),
        "offset_mae_sec": mean_or_none(offset_absolute),
        "offset_median_ae_sec": median_or_none(offset_absolute),
        "onset_mean_signed_error_sec": mean_or_none(onset_signed),
        "offset_mean_signed_error_sec": mean_or_none(offset_signed),
        "mean_segment_iou": mean_or_none(ious),
        "median_segment_iou": median_or_none(ious),
    }
    for tolerance_sec in TOLERANCES_SECONDS:
        tolerance_name = str(int(round(tolerance_sec * 1000)))
        onset_correct = sum(
            row["onset_absolute_error_sec"] <= tolerance_sec for row in aligned_rows
        )
        offset_correct = sum(
            row["offset_absolute_error_sec"] <= tolerance_sec for row in aligned_rows
        )
        both_correct = sum(
            row["onset_absolute_error_sec"] <= tolerance_sec
            and row["offset_absolute_error_sec"] <= tolerance_sec
            for row in aligned_rows
        )
        summary[f"onset_within_{tolerance_name}ms"] = (
            onset_correct / total if total else None
        )
        summary[f"offset_within_{tolerance_name}ms"] = (
            offset_correct / total if total else None
        )
        summary[f"both_within_{tolerance_name}ms"] = (
            both_correct / total if total else None
        )
    summary["pco_at_0_3"] = summary["onset_within_300ms"]
    return summary


def build_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    conditions = sorted(
        {(row["audio_source"], row["level"]) for row in rows},
        key=lambda condition: (
            condition[0],
            LEVEL_SORT_ORDER[condition[1]],
        ),
    )
    languages = sorted({row["lyrics_language"] for row in rows})
    track_ids = sorted({row["track_id"] for row in rows})
    for audio_source, level in conditions:
        condition_rows = [
            row
            for row in rows
            if row["audio_source"] == audio_source and row["level"] == level
        ]
        summaries.append(
            summarize_rows(condition_rows, "overall", "all", audio_source, level)
        )
        for language in languages:
            language_rows = [
                row
                for row in condition_rows
                if row["lyrics_language"] == language
            ]
            if language_rows:
                summaries.append(
                    summarize_rows(
                        language_rows,
                        "lyrics_language",
                        language,
                        audio_source,
                        level,
                    )
                )
        for track_id in track_ids:
            track_rows = [
                row for row in condition_rows if row["track_id"] == track_id
            ]
            if track_rows:
                summaries.append(
                    summarize_rows(
                        track_rows, "track_id", track_id, audio_source, level
                    )
                )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary_text(path: Path, summaries: list[dict[str, Any]]) -> None:
    overall = [summary for summary in summaries if summary["group_type"] == "overall"]
    lines = [
        "IKD reference-text lyrics-alignment baseline",
        "",
        "PCO uses all reference units as its denominator; unaligned units fail.",
        "",
        "source | level | onset MAE (s) | onset MedAE (s) | PCO@0.3 | PCS | support",
    ]
    for summary in overall:
        lines.append(
            " | ".join(
                [
                    summary["audio_source"],
                    summary["level"],
                    format_value(summary["onset_mae_sec"]),
                    format_value(summary["onset_median_ae_sec"]),
                    format_value(summary["pco_at_0_3"]),
                    format_value(summary["pcs"]),
                    str(summary["support"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reference-text IKD lyrics-alignment predictions."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Alignment input used to generate predictions.",
    )
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument(
        "--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--track-id", action="append", dest="track_ids")
    parser.add_argument("--audio-source", action="append", dest="audio_sources")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    input_path = args.input.resolve()
    ground_truth_path = args.ground_truth.resolve()
    prediction_root = args.prediction_root.resolve()
    config = load_json(config_path)
    config_sha256 = config_fingerprint(config)
    input_sha256 = file_sha256(input_path)
    ground_truth = load_json(ground_truth_path)
    tracks = select_tracks(ground_truth["tracks"], args.track_ids)
    audio_sources = select_audio_sources(config, args.audio_sources)

    rows: list[dict[str, Any]] = []
    for track in tracks:
        for audio_source in audio_sources:
            prediction_path = (
                prediction_root
                / audio_source
                / f"{track['track_id']}_{config['experiment_name']}.json"
            )
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction = load_json(prediction_path)
            if prediction.get("experiment_name") != config["experiment_name"]:
                raise ValueError(f"{prediction_path}: experiment mismatch")
            if prediction.get("config_sha256") != config_sha256:
                raise ValueError(f"{prediction_path}: configuration hash mismatch")
            if prediction.get("alignment_input_sha256") != input_sha256:
                raise ValueError(f"{prediction_path}: alignment-input hash mismatch")
            if prediction.get("track_id") != track["track_id"]:
                raise ValueError(f"{prediction_path}: track ID mismatch")
            if prediction.get("audio_source") != audio_source:
                raise ValueError(f"{prediction_path}: audio-source mismatch")
            rows.extend(evaluate_units(track, prediction, audio_source, "word"))
            rows.extend(evaluate_units(track, prediction, audio_source, "syllable"))

    summaries = build_summaries(rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "config_file": portable_path(config_path),
        "config_sha256": config_sha256,
        "alignment_input_file": portable_path(input_path),
        "alignment_input_sha256": input_sha256,
        "ground_truth_file": portable_path(ground_truth_path),
        "prediction_root": portable_path(prediction_root),
        "tolerances_seconds": list(TOLERANCES_SECONDS),
        "pco_denominator": "all reference units; unaligned units are failures",
        "metric_definitions": {
            "onset_absolute_error": (
                "absolute displacement between predicted and reference onset"
            ),
            "pco_at_0_3": (
                "fraction of all reference units with onset error at most 0.3 seconds"
            ),
            "pcs": (
                "correctly overlapped unit and inter-unit-gap duration divided by "
                "the summed evaluated reference spans"
            ),
        },
        "track_count": len(tracks),
        "audio_sources": audio_sources,
        "summary": summaries,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    write_csv(output_dir / "per_unit_metrics.csv", rows)
    write_csv(output_dir / "summary_metrics.csv", summaries)
    write_summary_text(output_dir / "summary.txt", summaries)

    print((output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"Detailed results: {output_dir}")


if __name__ == "__main__":
    main()
