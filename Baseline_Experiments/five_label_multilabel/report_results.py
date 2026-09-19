#!/usr/bin/env python3
"""Aggregate the two selected five-label experiments in paper-table order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_ROOT = Path(__file__).resolve().parent
BASELINES_ROOT = TASK_ROOT.parent
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from common.syllable_index import LABEL_COLUMNS
from five_label_multilabel.train import calculate_metrics


DEFAULT_RUNS_ROOT = TASK_ROOT / "outputs" / "training"
DEFAULT_OUTPUT = TASK_ROOT / "outputs" / "evaluation"
EXPECTED_RUNS = (1, 2, 3, 4, 5)
EXPECTED_SEEDS = (64, 128, 256)
EXPERIMENTS = (
    ("temporal_five_label_musicfm", "musicfm", "MusicFM"),
    ("temporal_five_label_musicfm_pitch", "musicfm_pitch", "MusicFM + pitch"),
)
OVERALL_FIELDS = (
    "exact_match_accuracy",
    "hamming_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_average_precision",
    "macro_roc_auc",
)
LABEL_FIELDS = (
    "precision",
    "recall",
    "f1",
    "average_precision",
    "roc_auc",
    "threshold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def prediction_paths(root: Path, experiment: str) -> list[Path]:
    return sorted((root / experiment).glob("run_*/seed_*/test_predictions.csv"))


def verify_model_set(
    paths: list[Path], experiment: str, search_root: Path, allow_incomplete: bool
) -> None:
    if not paths:
        raise FileNotFoundError(
            f"No predictions found for {experiment} under {search_root}"
        )
    observed: set[tuple[int, int]] = set()
    for path in paths:
        run = int(path.parents[1].name.removeprefix("run_"))
        seed = int(path.parent.name.removeprefix("seed_"))
        if (run, seed) in observed:
            raise ValueError(f"Duplicate model for {experiment}, run={run}, seed={seed}")
        observed.add((run, seed))
    expected = {(run, seed) for run in EXPECTED_RUNS for seed in EXPECTED_SEEDS}
    unexpected = observed - expected
    if unexpected:
        raise ValueError(f"Unexpected run and seed pairs: {sorted(unexpected)}")
    missing = expected - observed
    if missing and not allow_incomplete:
        raise ValueError(f"{experiment} is missing models: {sorted(missing)}")


def load_model_result(path: Path) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    required = {
        "syllable_id",
        *[f"target_{label}" for label in LABEL_COLUMNS],
        *[f"probability_{label}" for label in LABEL_COLUMNS],
        *[f"prediction_{label}" for label in LABEL_COLUMNS],
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    summary_path = path.with_name("run_summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    thresholds = np.asarray(
        [summary["validation_thresholds"][label] for label in LABEL_COLUMNS],
        dtype=np.float64,
    )
    targets = frame[
        [f"target_{label}" for label in LABEL_COLUMNS]
    ].to_numpy(dtype=np.int64)
    probabilities = frame[
        [f"probability_{label}" for label in LABEL_COLUMNS]
    ].to_numpy(dtype=np.float64)
    predictions = frame[
        [f"prediction_{label}" for label in LABEL_COLUMNS]
    ].to_numpy(dtype=np.int64)
    return (
        calculate_metrics(targets, probabilities, thresholds, predictions),
        frame,
        targets,
    )


def mean_and_sample_deviation(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    deviation = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), deviation


def formatted(values: list[float]) -> str:
    mean, deviation = mean_and_sample_deviation(values)
    return f"{mean:.6f} +/- {deviation:.6f}"


def pairwise_prediction_associations(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row-normalized and raw pairwise prediction associations.

    Diagonal cells count true-positive predictions. An off-diagonal cell
    counts a prediction of the column label when that label is absent from a
    syllable containing the row label. Correctly predicted co-labels are not
    counted as errors.
    """
    targets = frame[
        [f"target_{label}" for label in LABEL_COLUMNS]
    ].to_numpy(dtype=np.int64)
    predictions = frame[
        [f"prediction_{label}" for label in LABEL_COLUMNS]
    ].to_numpy(dtype=np.int64)
    matrix = np.zeros((len(LABEL_COLUMNS), len(LABEL_COLUMNS)), dtype=np.int64)

    for target, prediction in zip(targets, predictions, strict=True):
        ground_truth_labels = np.flatnonzero(target == 1)
        false_positive_labels = np.flatnonzero((target == 0) & (prediction == 1))
        for ground_truth_label in ground_truth_labels:
            if prediction[ground_truth_label] == 1:
                matrix[ground_truth_label, ground_truth_label] += 1
            if false_positive_labels.size:
                matrix[ground_truth_label, false_positive_labels] += 1

    support = targets.sum(axis=0, dtype=np.int64)[:, np.newaxis]
    normalized = np.divide(
        matrix,
        support,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=support != 0,
    ) * 100.0
    return normalized, matrix


def write_pairwise_matrix(matrix: np.ndarray, path: Path) -> None:
    frame = pd.DataFrame(matrix, columns=LABEL_COLUMNS, index=LABEL_COLUMNS)
    frame.to_csv(path, index_label="ground_truth_label")


def main() -> None:
    args = parse_args()
    overall_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    text_sections: list[str] = []
    global_targets: dict[str, np.ndarray] = {}
    pairwise_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for experiment, setting, display_name in EXPERIMENTS:
        paths = prediction_paths(args.runs_root, experiment)
        verify_model_set(
            paths, experiment, args.runs_root.resolve() / experiment, args.allow_incomplete
        )
        metrics_by_model: list[dict[str, Any]] = []
        normalized_pairwise_by_model: list[np.ndarray] = []
        raw_pairwise_by_model: list[np.ndarray] = []
        unique_targets: dict[str, np.ndarray] = {}
        completed_runs: set[int] = set()
        for path in paths:
            metrics, frame, targets = load_model_result(path)
            metrics_by_model.append(metrics)
            normalized_pairwise, raw_pairwise = pairwise_prediction_associations(frame)
            normalized_pairwise_by_model.append(normalized_pairwise)
            raw_pairwise_by_model.append(raw_pairwise)
            completed_runs.add(int(path.parents[1].name.removeprefix("run_")))
            for syllable_id, target in zip(frame["syllable_id"].astype(str), targets):
                previous = unique_targets.get(syllable_id)
                if previous is not None and not np.array_equal(previous, target):
                    raise ValueError(f"Conflicting targets for {syllable_id}")
                unique_targets[syllable_id] = target.copy()
                global_previous = global_targets.get(syllable_id)
                if global_previous is not None and not np.array_equal(global_previous, target):
                    raise ValueError(f"Targets differ between experiments for {syllable_id}")
                global_targets[syllable_id] = target.copy()
        unique_target_matrix = np.stack(list(unique_targets.values()))
        pairwise_results[setting] = (
            np.stack(normalized_pairwise_by_model).mean(axis=0),
            np.stack(raw_pairwise_by_model).sum(axis=0),
        )

        overall_row: dict[str, Any] = {
            "experiment": experiment,
            "setting": setting,
            "display_name": display_name,
            "models": len(metrics_by_model),
            "support_unique_syllables": len(unique_targets),
        }
        for field in OVERALL_FIELDS:
            values = [float(metrics[field]) for metrics in metrics_by_model]
            mean, deviation = mean_and_sample_deviation(values)
            overall_row[f"{field}_mean"] = mean
            overall_row[f"{field}_sample_standard_deviation"] = deviation
        overall_rows.append(overall_row)

        for label_index, label in enumerate(LABEL_COLUMNS):
            label_row: dict[str, Any] = {
                "experiment": experiment,
                "setting": setting,
                "display_name": display_name,
                "label": label,
                "support_unique_syllables": int(unique_target_matrix[:, label_index].sum()),
                "models": len(metrics_by_model),
            }
            for field in LABEL_FIELDS:
                values = [
                    float(metrics["per_label"][label][field])
                    for metrics in metrics_by_model
                ]
                mean, deviation = mean_and_sample_deviation(values)
                label_row[f"{field}_mean"] = mean
                label_row[f"{field}_sample_standard_deviation"] = deviation
            per_label_rows.append(label_row)

        lines = [
            f"Experiment: {experiment}",
            f"Display name: {display_name}",
            f"Completed models: {len(metrics_by_model)}/15",
            "Completed folds: " + ", ".join(str(run) for run in sorted(completed_runs)),
            "Aggregation: mean +/- sample standard deviation across completed models",
            "Support: unique test syllables across completed folds",
            "",
            "scope\texact-acc\thamming-acc\tprecision\trecall\tF1-Score\tmAP\tROC-AUC\tthreshold\tsupport",
        ]
        values = [
            formatted([float(metrics[field]) for metrics in metrics_by_model])
            for field in OVERALL_FIELDS
        ]
        lines.append("\t".join(["Overall", *values, "--", str(len(unique_targets))]))
        for label_index, label in enumerate(LABEL_COLUMNS):
            label_values = [
                formatted(
                    [float(metrics["per_label"][label][field]) for metrics in metrics_by_model]
                )
                for field in LABEL_FIELDS
            ]
            support = int(unique_target_matrix[:, label_index].sum())
            lines.append("\t".join([label, "--", "--", *label_values, str(support)]))
        text_sections.append("\n".join(lines))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overall_rows).to_csv(output_dir / "overall_metrics.csv", index=False)
    pd.DataFrame(per_label_rows).to_csv(output_dir / "per_label_metrics.csv", index=False)
    for setting, (normalized_matrix, raw_matrix) in pairwise_results.items():
        write_pairwise_matrix(
            normalized_matrix,
            output_dir / f"pairwise_{setting}_row_normalized_percent.csv",
        )
        write_pairwise_matrix(
            raw_matrix,
            output_dir / f"pairwise_{setting}_pooled_raw_counts.csv",
        )
    (output_dir / "experiment_summary.txt").write_text(
        "\n\n".join(text_sections) + "\n", encoding="utf-8"
    )

    overall = {str(row["setting"]): row for row in overall_rows}
    per_label = {
        (str(row["setting"]), str(row["label"])): row for row in per_label_rows
    }
    print("IKD five-label multi-label vocal-technique classification")
    print()
    print("MusicFM setting | P | R | F1 | mAP")
    for setting, display_name in (
        ("musicfm", "MusicFM"),
        ("musicfm_pitch", "MusicFM + pitch"),
    ):
        row = overall[setting]
        print(
            "{} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(
                display_name,
                float(row["macro_precision_mean"]),
                float(row["macro_recall_mean"]),
                float(row["macro_f1_mean"]),
                float(row["macro_average_precision_mean"]),
            )
        )
    print()
    print(
        "Technique | Support | MusicFM P | MusicFM R | MusicFM F1 | MusicFM AP | "
        "MusicFM + pitch P | MusicFM + pitch R | MusicFM + pitch F1 | "
        "MusicFM + pitch AP"
    )
    for label in LABEL_COLUMNS:
        base = per_label[("musicfm", label)]
        pitch = per_label[("musicfm_pitch", label)]
        print(
            "{} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | "
            "{:.4f} | {:.4f} | {:.4f} | {:.4f}".format(
                label,
                int(base["support_unique_syllables"]),
                float(base["precision_mean"]),
                float(base["recall_mean"]),
                float(base["f1_mean"]),
                float(base["average_precision_mean"]),
                float(pitch["precision_mean"]),
                float(pitch["recall_mean"]),
                float(pitch["f1_mean"]),
                float(pitch["average_precision_mean"]),
            )
        )
    print()
    print(f"Detailed results: {output_dir}")


if __name__ == "__main__":
    main()
