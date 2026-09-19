#!/usr/bin/env python3
"""Train the cached-frame MusicFM and MusicFM-plus-pitch baselines."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


SCRIPT_PATH = Path(__file__).resolve()
TASK_ROOT = SCRIPT_PATH.parent
BASELINES_ROOT = TASK_ROOT.parent
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from common.reproducibility import (
    clone_state_dict_to_cpu,
    configuration_sha256,
    load_toml,
    resolve_device,
    resolve_requested_values,
    set_deterministic_seed,
    threshold_grid,
    write_json,
)
from common.syllable_index import LABEL_COLUMNS, TRACEABILITY_COLUMNS, load_syllable_index
from five_label_multilabel.data import (
    FeatureSelection,
    FoldDataLoaders,
    TemporalTrackDataset,
    TemporalTrackStore,
    load_track_store,
    make_fold_loaders,
)
from five_label_multilabel.model import (
    TemporalConvConfig,
    TemporalTechniqueModel,
    load_temporal_config,
    normalized_syllable_bce_with_logits,
)


DEFAULT_CONFIG = TASK_ROOT / "config.toml"
DEFAULT_DATASET = BASELINES_ROOT / "data" / "syllable_dataset.csv"
DEFAULT_MUSICFM = TASK_ROOT / "outputs" / "features" / "musicfm"
DEFAULT_PITCH = TASK_ROOT / "outputs" / "features" / "pitch_25hz"
DEFAULT_OUTPUT = TASK_ROOT / "outputs" / "training"
LABEL_NAMES = tuple(LABEL_COLUMNS)


@dataclass(frozen=True)
class PredictionOutput:
    targets: np.ndarray
    probabilities: np.ndarray
    row_indices: np.ndarray
    syllable_ids: tuple[str, ...]
    track_ids: tuple[str, ...]
    weighted_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--musicfm-dir", type=Path, default=DEFAULT_MUSICFM)
    parser.add_argument("--pitch-dir", type=Path, default=DEFAULT_PITCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--experiment",
        action="append",
        help="Experiment name to run; repeat to select both configurations.",
    )
    selection.add_argument("--all-enabled", action="store_true")
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--run", dest="runs", nargs="+", type=int)
    parser.add_argument("--seed", dest="seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def locked(values: Mapping[str, Any], expected: Mapping[str, Any], scope: str) -> None:
    for name, wanted in expected.items():
        observed = values.get(name)
        if isinstance(wanted, float):
            equal = isinstance(observed, (int, float)) and np.isclose(observed, wanted)
        else:
            equal = observed == wanted
        if not equal:
            raise ValueError(
                f"Locked {scope} setting {name} must be {wanted!r}, got {observed!r}"
            )


def validate_configuration(configuration: dict[str, Any]) -> TemporalConvConfig:
    if configuration.get("schema_version") != "IKD_five_label_framewise_musicfm_pitch_v1":
        raise ValueError("Unsupported five-label configuration")
    locked(
        configuration["task"],
        {
            "name": "five_label_multilabel",
            "label_order": list(LABEL_NAMES),
            "target_formulation": "five_label",
        },
        "task",
    )
    locked(
        configuration["cross_validation"],
        {
            "runs": [1, 2, 3, 4, 5],
            "seeds": [64, 128, 256],
            "split_column_template": "fold_{run}_split",
            "group_column": "performer_group_id",
        },
        "cross-validation",
    )
    locked(
        configuration["features"]["musicfm"],
        {
            "encoder": "MusicFM25Hz",
            "checkpoint": "MSD",
            "source": "htdemucs_vocal_stem",
            "sample_rate_hz": 24000,
            "frame_rate_hz": 25,
            "token_step_samples": 960,
            "hidden_layer_index": 7,
            "embedding_dimension": 1024,
            "chunk_core_seconds": 30,
            "chunk_context_seconds": 5,
            "boundary_padding": False,
            "normalization": "per_frame_layer_norm",
            "layer_norm_epsilon": 1e-5,
            "random_seed": 20260816,
        },
        "MusicFM feature",
    )
    locked(
        configuration["features"]["pitch"],
        {
            "extractor": "torchcrepe",
            "model_capacity": "full",
            "source": "htdemucs_vocal_stem",
            "sample_rate_hz": 16000,
            "source_frame_rate_hz": 100,
            "hop_length_samples": 160,
            "minimum_frequency_hz": 50.0,
            "maximum_frequency_hz": 1000.0,
            "batch_size_frames": 2048,
            "decoder": "viterbi_bin_center",
            "frequency_dither": False,
            "random_seed": 20260815,
            "target_frame_rate_hz": 25,
            "alignment": "select_every_fourth_frame",
            "interpolation": False,
            "periodicity_threshold": 0.50,
            "representation": "track_relative_semitone_periodicity_reliability",
            "dimension": 3,
        },
        "pitch feature",
    )
    temporal = load_temporal_config(configuration)
    locked(
        configuration["model"]["temporal"],
        {
            "type": "residual_dilated_conv1d",
            "projection_channels": 256,
            "temporal_channels": 256,
            "kernel_size": 5,
            "dilations": [1, 2, 4, 8],
            "residual_blocks": 4,
            "activation": "relu",
            "dropout": 0.30,
            "dropout_type": "channel",
            "normalization": "per_frame_layer_norm",
            "causal": False,
            "padding": "same",
            "syllable_pooling": "mean_logits",
            "output_dimension": 5,
        },
        "temporal model",
    )
    locked(
        configuration["training"],
        {
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_unit": "complete_track",
            "batch_size_tracks": 1,
            "shuffle_training_tracks": True,
            "gradient_accumulation_steps": 1,
            "maximum_epochs": 100,
            "gradient_clip_norm": 5.0,
            "scheduler": "none",
            "num_workers": 0,
            "pin_memory": False,
            "mixed_precision": False,
            "deterministic_algorithms": True,
            "loss_normalization": (
                "sum_valid_syllable_losses_divided_by_batch_tracks_times_"
                "training_mean_syllables_per_track"
            ),
        },
        "training",
    )
    expected_experiments = {
        "temporal_five_label_musicfm": (False, 1024, "MusicFM"),
        "temporal_five_label_musicfm_pitch": (True, 1027, "MusicFM + pitch"),
    }
    observed = {
        str(item["name"]): (
            bool(item["include_pitch"]),
            int(item["input_dimension"]),
            str(item["display_name"]),
        )
        for item in configuration["experiments"]
    }
    if observed != expected_experiments:
        raise ValueError("Configure exactly the two selected framewise experiments")
    locked(
        configuration["loss"],
        {
            "name": "bce_with_logits",
            "positive_weight_formula": "negative_count / positive_count",
            "positive_weight_source": "training_partition_only",
            "positive_weight_cap_enabled": False,
        },
        "loss",
    )
    locked(
        configuration["early_stopping"],
        {
            "metric": "macro_average_precision",
            "mode": "max",
            "patience": 12,
            "minimum_delta": 1e-4,
            "restore_best_checkpoint": True,
        },
        "early stopping",
    )
    locked(
        configuration["threshold_selection"],
        {
            "scope": "one_per_label",
            "source": "validation_partition_only",
            "metric": "f1",
            "minimum": 0.05,
            "maximum": 0.95,
            "step": 0.01,
            "tie_break": "closest_to_0.5_then_lower",
        },
        "threshold selection",
    )
    return temporal


def choose_experiments(
    configuration: dict[str, Any], requested: Sequence[str] | None, all_enabled: bool
) -> list[dict[str, Any]]:
    experiments = configuration["experiments"]
    by_name = {str(item["name"]): item for item in experiments}
    if all_enabled:
        return [item for item in experiments if item["enabled"]]
    if not requested:
        raise ValueError("Select --experiment NAME or --all-enabled")
    if len(requested) != len(set(requested)):
        raise ValueError("An experiment was requested more than once")
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    return [by_name[name] for name in requested]


def target_matrix(dataset: TemporalTrackDataset) -> np.ndarray:
    blocks = [
        dataset.store.tracks[int(index)].targets for index in dataset.track_indices
    ]
    if not blocks:
        raise ValueError("A partition contains no tracks")
    return np.concatenate(blocks, axis=0)


def positive_weights(train_loader: DataLoader) -> torch.Tensor:
    targets = target_matrix(train_loader.dataset).astype(np.float64)
    positives = targets.sum(axis=0)
    negatives = targets.shape[0] - positives
    if np.any(positives <= 0):
        raise ValueError("A training partition contains a label with no positives")
    return torch.from_numpy((negatives / positives).astype(np.float32))


def mean_training_syllables(train_loader: DataLoader) -> float:
    dataset = train_loader.dataset
    counts = [
        dataset.store.tracks[int(index)].syllable_count
        for index in dataset.track_indices
    ]
    return float(np.mean(counts))


def move_inputs(
    batch: dict[str, Any], device: torch.device, non_blocking: bool
) -> dict[str, torch.Tensor]:
    names = (
        "features",
        "frame_mask",
        "syllable_frame_starts",
        "syllable_frame_ends",
        "syllable_mask",
    )
    return {
        name: batch[name].to(device, non_blocking=non_blocking) for name in names
    }


def train_one_epoch(
    model: TemporalTechniqueModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    mean_syllables: float,
    device: torch.device,
    gradient_clip_norm: float,
    non_blocking: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_tracks = 0
    for batch in loader:
        inputs = move_inputs(batch, device, non_blocking)
        targets = batch["targets"].to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        loss = normalized_syllable_bce_with_logits(
            output.syllable_logits,
            targets,
            inputs["syllable_mask"],
            weights,
            mean_syllables,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_tracks = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_tracks
        total_tracks += batch_tracks
    if total_tracks == 0:
        raise ValueError("Training loader produced no tracks")
    return total_loss / total_tracks


def predict(
    model: TemporalTechniqueModel,
    loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
    non_blocking: bool,
) -> PredictionOutput:
    model.eval()
    target_blocks: list[np.ndarray] = []
    probability_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    syllable_ids: list[str] = []
    track_ids: list[str] = []
    total_loss = 0.0
    total_syllables = 0
    with torch.no_grad():
        for batch in loader:
            inputs = move_inputs(batch, device, non_blocking)
            targets = batch["targets"].to(device, non_blocking=non_blocking)
            output = model(**inputs)
            mask = inputs["syllable_mask"]
            elementwise = F.binary_cross_entropy_with_logits(
                output.syllable_logits,
                targets,
                pos_weight=weights,
                reduction="none",
            )
            per_syllable = elementwise.mean(dim=2)
            total_loss += float(per_syllable[mask].sum().cpu())
            total_syllables += int(mask.sum())
            target_blocks.append(targets[mask].cpu().numpy())
            probability_blocks.append(
                torch.sigmoid(output.syllable_logits[mask]).cpu().numpy()
            )
            row_blocks.append(batch["row_indices"][batch["syllable_mask"]].numpy())
            for index, count in enumerate(batch["syllable_lengths"].tolist()):
                syllable_ids.extend(str(value) for value in batch["syllable_ids"][index][:count])
                track_ids.extend([str(batch["track_ids"][index])] * count)
    if total_syllables == 0:
        raise ValueError("Evaluation loader produced no syllables")
    return PredictionOutput(
        targets=np.concatenate(target_blocks),
        probabilities=np.concatenate(probability_blocks),
        row_indices=np.concatenate(row_blocks),
        syllable_ids=tuple(syllable_ids),
        track_ids=tuple(track_ids),
        weighted_loss=total_loss / total_syllables,
    )


def macro_average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float:
    values = [
        average_precision_score(targets[:, index], probabilities[:, index])
        for index in range(len(LABEL_NAMES))
    ]
    return float(np.mean(values))


def select_validation_thresholds(
    targets: np.ndarray, probabilities: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    thresholds = np.empty(targets.shape[1], dtype=np.float64)
    for label_index in range(targets.shape[1]):
        scores = np.asarray(
            [
                f1_score(
                    targets[:, label_index].astype(np.int64),
                    probabilities[:, label_index] >= candidate,
                    zero_division=0,
                )
                for candidate in grid
            ]
        )
        tied = grid[np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12)]
        thresholds[label_index] = min(
            tied.tolist(), key=lambda value: (abs(value - 0.5), value)
        )
    return thresholds


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    binary_predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    if binary_predictions is None:
        binary_predictions = probabilities >= thresholds.reshape(1, -1)
    binary_predictions = np.asarray(binary_predictions, dtype=bool)
    integer_targets = targets.astype(np.int64)
    per_label: dict[str, dict[str, Any]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    average_precisions: list[float] = []
    aucs: list[float] = []
    for index, label in enumerate(LABEL_NAMES):
        label_targets = integer_targets[:, index]
        label_predictions = binary_predictions[:, index].astype(np.int64)
        precision, recall, label_f1, _ = precision_recall_fscore_support(
            label_targets,
            label_predictions,
            average="binary",
            zero_division=0,
        )
        average_precision = float(
            average_precision_score(label_targets, probabilities[:, index])
        )
        auc = float(roc_auc_score(label_targets, probabilities[:, index]))
        precisions.append(float(precision))
        recalls.append(float(recall))
        average_precisions.append(average_precision)
        aucs.append(auc)
        per_label[label] = {
            "threshold": float(thresholds[index]),
            "support": int(label_targets.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(label_f1),
            "average_precision": average_precision,
            "roc_auc": auc,
        }
    return {
        "sample_count": int(integer_targets.shape[0]),
        "exact_match_accuracy": float(
            np.all(integer_targets == binary_predictions, axis=1).mean()
        ),
        "hamming_accuracy": float(
            np.equal(integer_targets, binary_predictions).mean()
        ),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(
            f1_score(integer_targets, binary_predictions, average="macro", zero_division=0)
        ),
        "macro_average_precision": float(np.mean(average_precisions)),
        "macro_roc_auc": float(np.mean(aucs)),
        "micro_f1": float(
            f1_score(integer_targets, binary_predictions, average="micro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(integer_targets, binary_predictions, average="weighted", zero_division=0)
        ),
        "per_label": per_label,
    }


def partition_description(loader: DataLoader) -> dict[str, Any]:
    dataset = loader.dataset
    row_indices = np.concatenate(
        [
            dataset.store.tracks[int(index)].row_indices
            for index in dataset.track_indices
        ]
    )
    metadata = dataset.store.metadata.iloc[row_indices]
    return {
        "syllables": int(len(metadata)),
        "tracks": sorted(metadata["track_id"].astype(str).unique().tolist()),
        "performer_groups": sorted(
            metadata["performer_group_id"].astype(str).unique().tolist()
        ),
        "mean_syllables_per_track": float(len(metadata) / len(dataset.track_indices)),
    }


def prediction_frame(
    output: PredictionOutput,
    store: TemporalTrackStore,
    experiment_name: str,
    run: int,
    seed: int,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    metadata = store.metadata.iloc[output.row_indices]
    if output.syllable_ids != tuple(metadata["syllable_id"].astype(str)):
        raise ValueError("Predicted syllable IDs differ from metadata")
    if output.track_ids != tuple(metadata["track_id"].astype(str)):
        raise ValueError("Predicted track IDs differ from metadata")
    frame = metadata.loc[:, TRACEABILITY_COLUMNS].reset_index(drop=True).copy()
    frame.insert(0, "test_base_fold", run)
    frame.insert(0, "seed", seed)
    frame.insert(0, "run", run)
    frame.insert(0, "experiment", experiment_name)
    predictions = output.probabilities >= thresholds.reshape(1, -1)
    for index, label in enumerate(LABEL_NAMES):
        frame[f"target_{label}"] = output.targets[:, index].astype(np.int64)
        frame[f"probability_{label}"] = output.probabilities[:, index]
        frame[f"prediction_{label}"] = predictions[:, index].astype(np.int64)
    return frame


def save_outputs(
    directory: Path,
    configuration_path: Path,
    configuration: dict[str, Any],
    experiment: dict[str, Any],
    run: int,
    seed: int,
    device: torch.device,
    loaders: FoldDataLoaders,
    model: TemporalTechniqueModel,
    weights: torch.Tensor,
    mean_syllables: float,
    thresholds: np.ndarray,
    validation_output: PredictionOutput,
    test_output: PredictionOutput,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    history: list[dict[str, Any]],
    best_epoch: int,
    best_validation_score: float,
    stopped_early: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(configuration_path, directory / "configuration_snapshot.toml")
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
    prediction_frame(
        test_output,
        loaders.test.dataset.store,
        str(experiment["name"]),
        run,
        seed,
        thresholds,
    ).to_csv(directory / "test_predictions.csv", index=False)
    write_json(directory / "feature_names.json", list(loaders.feature_names))
    if configuration["output"]["save_best_checkpoint"]:
        torch.save(
            {
                "schema_version": "IKD_cached_framewise_checkpoint_v1",
                "experiment": experiment["name"],
                "run": run,
                "seed": seed,
                "input_dimension": loaders.input_dimension,
                "label_order": list(LABEL_NAMES),
                "model_configuration": asdict(model.configuration),
                "model_state_dict": clone_state_dict_to_cpu(model),
            },
            directory / "best_model.pt",
        )
    summary = {
        "schema_version": "IKD_cached_framewise_run_summary_v1",
        "configuration_file": configuration_path.name,
        "configuration_sha256": configuration_sha256(configuration_path),
        "experiment": copy.deepcopy(experiment),
        "run": run,
        "test_base_fold": run,
        "seed": seed,
        "device": str(device),
        "library_versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_dimension": loaders.input_dimension,
        "label_order": list(LABEL_NAMES),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "receptive_field_frames": model.configuration.receptive_field_frames,
        "partitions": {
            name: partition_description(loader)
            for name, loader in (
                ("train", loaders.train),
                ("validation", loaders.validation),
                ("test", loaders.test),
            )
        },
        "positive_weights": {
            label: float(value) for label, value in zip(LABEL_NAMES, weights.tolist())
        },
        "training_mean_syllables_per_track": mean_syllables,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_validation_macro_average_precision": best_validation_score,
        "validation_weighted_loss_at_best_checkpoint": validation_output.weighted_loss,
        "test_weighted_loss_at_best_checkpoint": test_output.weighted_loss,
        "validation_thresholds": {
            label: float(value) for label, value in zip(LABEL_NAMES, thresholds)
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_configuration": asdict(model.configuration),
        "training_configuration": copy.deepcopy(configuration["training"]),
        "loss_configuration": copy.deepcopy(configuration["loss"]),
        "early_stopping_configuration": copy.deepcopy(configuration["early_stopping"]),
        "threshold_configuration": copy.deepcopy(configuration["threshold_selection"]),
    }
    write_json(directory / "run_summary.json", summary)


def train_single_run(
    configuration: dict[str, Any],
    configuration_path: Path,
    experiment: dict[str, Any],
    temporal_configuration: TemporalConvConfig,
    store: TemporalTrackStore,
    run: int,
    seed: int,
    device: torch.device,
    directory: Path,
) -> None:
    training = configuration["training"]
    early_stopping = configuration["early_stopping"]
    set_deterministic_seed(seed, bool(training["deterministic_algorithms"]))
    loaders = make_fold_loaders(
        store,
        run,
        seed,
        batch_size=int(training["batch_size_tracks"]),
        num_workers=int(training["num_workers"]),
        pin_memory=bool(training["pin_memory"]),
    )
    model = TemporalTechniqueModel(loaders.input_dimension, temporal_configuration).to(device)
    weights_cpu = positive_weights(loaders.train)
    weights = weights_cpu.to(device)
    mean_syllables = mean_training_syllables(loaders.train)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    non_blocking = bool(training["pin_memory"] and device.type == "cuda")

    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    without_improvement = 0
    history: list[dict[str, Any]] = []
    stopped_early = False
    progress = tqdm(
        range(1, int(training["maximum_epochs"]) + 1),
        desc=f"{experiment["name"]} run={run} seed={seed}",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in progress:
        training_loss = train_one_epoch(
            model,
            loaders.train,
            optimizer,
            weights,
            mean_syllables,
            device,
            float(training["gradient_clip_norm"]),
            non_blocking,
        )
        validation_output = predict(
            model, loaders.validation, weights, device, non_blocking
        )
        validation_score = macro_average_precision(
            validation_output.targets, validation_output.probabilities
        )
        improved = validation_score > best_score + float(early_stopping["minimum_delta"])
        if improved:
            best_score = validation_score
            best_epoch = epoch
            best_state = clone_state_dict_to_cpu(model)
            without_improvement = 0
        else:
            without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "training_weighted_loss": training_loss,
                "validation_weighted_loss": validation_output.weighted_loss,
                "validation_macro_average_precision": validation_score,
                "improved": improved,
            }
        )
        progress.set_postfix(
            train_loss=f"{training_loss:.4f}",
            validation_loss=f"{validation_output.weighted_loss:.4f}",
            validation_mAP=f"{validation_score:.4f}",
            best_epoch=best_epoch,
            patience=f"{without_improvement}/{early_stopping["patience"]}",
        )
        if without_improvement >= int(early_stopping["patience"]):
            stopped_early = True
            break
    progress.close()
    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    validation_output = predict(model, loaders.validation, weights, device, non_blocking)
    threshold_settings = configuration["threshold_selection"]
    grid = threshold_grid(
        float(threshold_settings["minimum"]),
        float(threshold_settings["maximum"]),
        float(threshold_settings["step"]),
    )
    thresholds = select_validation_thresholds(
        validation_output.targets, validation_output.probabilities, grid
    )
    validation_metrics = calculate_metrics(
        validation_output.targets, validation_output.probabilities, thresholds
    )
    test_output = predict(model, loaders.test, weights, device, non_blocking)
    test_metrics = calculate_metrics(
        test_output.targets, test_output.probabilities, thresholds
    )
    save_outputs(
        directory,
        configuration_path,
        configuration,
        experiment,
        run,
        seed,
        device,
        loaders,
        model,
        weights_cpu,
        mean_syllables,
        thresholds,
        validation_output,
        test_output,
        validation_metrics,
        test_metrics,
        history,
        best_epoch,
        best_score,
        stopped_early,
    )
    tqdm.write(
        f"Completed {experiment["name"]} run={run} seed={seed}: "
        f"F1={test_metrics["macro_f1"]:.6f}, "
        f"mAP={test_metrics["macro_average_precision"]:.6f}"
    )


def required_outputs(directory: Path, configuration: dict[str, Any]) -> list[Path]:
    names = [
        "configuration_snapshot.toml",
        "training_history.csv",
        "test_predictions.csv",
        "feature_names.json",
        "run_summary.json",
    ]
    if configuration["output"]["save_best_checkpoint"]:
        names.append("best_model.pt")
    return [directory / name for name in names]


def dry_run(
    configuration: dict[str, Any],
    temporal: TemporalConvConfig,
    experiment: dict[str, Any],
    store: TemporalTrackStore,
    runs: Sequence[int],
    seeds: Sequence[int],
) -> None:
    for run in runs:
        loaders = make_fold_loaders(store, run, seeds[0])
        model = TemporalTechniqueModel(loaders.input_dimension, temporal)
        weights = positive_weights(loaders.train)
        counts = "/".join(
            str(partition_description(loader)["syllables"])
            for loader in (loaders.train, loaders.validation, loaders.test)
        )
        tracks = "/".join(
            str(len(loader.dataset))
            for loader in (loaders.train, loaders.validation, loaders.test)
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        weight_text = ", ".join(
            f"{label}={value:.4f}" for label, value in zip(LABEL_NAMES, weights.tolist())
        )
        print(
            f"DRY RUN {experiment["name"]} run={run}: seeds={list(seeds)}, "
            f"input={loaders.input_dimension}, parameters={parameter_count}, "
            f"train/validation/test_tracks={tracks}, "
            f"train/validation/test_syllables={counts}, pos_weight=[{weight_text}]"
        )


def main() -> None:
    args = parse_args()
    configuration_path = args.config.resolve()
    configuration = load_toml(configuration_path)
    temporal = validate_configuration(configuration)
    if args.list_experiments:
        for experiment in configuration["experiments"]:
            state = "enabled" if experiment["enabled"] else "disabled"
            print(
                f"{experiment["name"]}: {state}, "
                f"input={experiment["input_dimension"]}, "
                f"pitch={experiment["include_pitch"]}"
            )
        return
    experiments = choose_experiments(
        configuration, args.experiment, args.all_enabled
    )
    runs = resolve_requested_values(
        args.runs, configuration["cross_validation"]["runs"], "run"
    )
    seeds = resolve_requested_values(
        args.seeds, configuration["cross_validation"]["seeds"], "seed"
    )
    metadata = load_syllable_index(args.dataset.resolve())
    musicfm_dir = args.musicfm_dir.resolve()
    pitch_dir = args.pitch_dir.resolve()
    output_root = args.output_dir.resolve()
    device = resolve_device(args.device)
    print(
        f"Selected {len(experiments)} experiment(s), {len(runs)} fold(s), and "
        f"{len(seeds)} seed(s): {len(experiments) * len(runs) * len(seeds)} "
        f"model run(s). Device={device}."
    )

    stores: dict[bool, TemporalTrackStore] = {}
    completed = 0
    skipped = 0
    for experiment in experiments:
        include_pitch = bool(experiment["include_pitch"])
        if include_pitch not in stores:
            stores[include_pitch] = load_track_store(
                metadata,
                musicfm_dir,
                pitch_dir,
                FeatureSelection(include_pitch=include_pitch),
            )
        store = stores[include_pitch]
        if store.input_dimension != int(experiment["input_dimension"]):
            raise ValueError("Constructed input dimension differs from configuration")
        if args.dry_run:
            dry_run(configuration, temporal, experiment, store, runs, seeds)
            continue
        for run in runs:
            for seed in seeds:
                directory = (
                    output_root
                    / str(experiment["name"])
                    / f"run_{run}"
                    / f"seed_{seed}"
                )
                paths = required_outputs(directory, configuration)
                existing = [path for path in paths if path.exists()]
                missing = [path for path in paths if not path.exists()]
                if existing and not args.overwrite:
                    if not missing:
                        skipped += 1
                        print(
                            f"Skipped completed {experiment["name"]} "
                            f"run={run} seed={seed}: {directory}"
                        )
                        continue
                    raise FileExistsError(
                        f"Partial output exists in {directory}; "
                        f"existing={[path.name for path in existing]}, "
                        f"missing={[path.name for path in missing]}"
                    )
                train_single_run(
                    configuration,
                    configuration_path,
                    experiment,
                    temporal,
                    store,
                    run,
                    seed,
                    device,
                    directory,
                )
                completed += 1
    print(f"Training request finished: completed={completed}, skipped={skipped}.")


if __name__ == "__main__":
    main()
