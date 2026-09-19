"""Residual Conv1D classifier with differentiable frame-to-syllable pooling."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from five_label_multilabel.musicfm import EMBEDDING_DIMENSION


@dataclass(frozen=True)
class TemporalConvConfig:
    projection_channels: int
    temporal_channels: int
    kernel_size: int
    dilations: tuple[int, ...]
    residual_blocks: int
    activation: str
    dropout: float
    dropout_type: str
    normalization: str
    causal: bool
    padding: str
    syllable_pooling: str
    output_dimension: int

    def __post_init__(self) -> None:
        if self.projection_channels <= 0 or self.temporal_channels <= 0:
            raise ValueError("Projection and temporal channels must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("Dilations must be nonempty and positive")
        if self.residual_blocks != len(self.dilations):
            raise ValueError("residual_blocks must equal the number of dilations")
        if self.activation != "relu":
            raise ValueError("The published activation is ReLU")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")
        if self.dropout_type != "channel":
            raise ValueError("The published temporal dropout is channel-wise")
        if self.normalization != "per_frame_layer_norm":
            raise ValueError("The published normalization is per-frame LayerNorm")
        if self.causal or self.padding != "same":
            raise ValueError("The published model is non-causal with same padding")
        if self.syllable_pooling != "mean_logits":
            raise ValueError("The published syllable pooling averages logits")
        if self.output_dimension != 5:
            raise ValueError("The released task has exactly five output labels")

    @property
    def receptive_field_frames(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)


def load_temporal_config(configuration: Mapping[str, Any]) -> TemporalConvConfig:
    try:
        values = configuration["model"]["temporal"]
    except (KeyError, TypeError) as error:
        raise ValueError("Missing [model.temporal] configuration") from error
    return TemporalConvConfig(
        projection_channels=int(values["projection_channels"]),
        temporal_channels=int(values["temporal_channels"]),
        kernel_size=int(values["kernel_size"]),
        dilations=tuple(int(value) for value in values["dilations"]),
        residual_blocks=int(values["residual_blocks"]),
        activation=str(values["activation"]),
        dropout=float(values["dropout"]),
        dropout_type=str(values["dropout_type"]),
        normalization=str(values["normalization"]),
        causal=bool(values["causal"]),
        padding=str(values["padding"]),
        syllable_pooling=str(values["syllable_pooling"]),
        output_dimension=int(values["output_dimension"]),
    )


@dataclass(frozen=True)
class TemporalModelOutput:
    frame_logits: torch.Tensor
    syllable_logits: torch.Tensor


class PerFrameLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.temporal_convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.normalization = PerFrameLayerNorm(channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout1d(dropout)
        self.pointwise_convolution = nn.Conv1d(channels, channels, kernel_size=1)
        self.output_activation = nn.ReLU()

    def forward(
        self, values: torch.Tensor, frame_mask: torch.Tensor
    ) -> torch.Tensor:
        residual = values
        transformed = self.temporal_convolution(values)
        transformed = self.normalization(transformed)
        transformed = self.activation(transformed)
        transformed = self.dropout(transformed)
        transformed = self.pointwise_convolution(transformed)
        output = self.output_activation(residual + transformed)
        return output * frame_mask.unsqueeze(1).to(dtype=output.dtype)


class TemporalTechniqueModel(nn.Module):
    """Produce five framewise logits, then mean-pool each syllable interval."""

    def __init__(
        self, input_dimension: int, configuration: TemporalConvConfig
    ) -> None:
        super().__init__()
        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive")
        self.configuration = configuration
        self.input_dimension = input_dimension
        self.output_dimension = configuration.output_dimension
        self.input_projection = nn.Conv1d(
            self.input_dimension, configuration.projection_channels, kernel_size=1
        )
        if configuration.projection_channels == configuration.temporal_channels:
            self.temporal_projection: nn.Module = nn.Identity()
        else:
            self.temporal_projection = nn.Conv1d(
                configuration.projection_channels,
                configuration.temporal_channels,
                kernel_size=1,
            )
        self.input_normalization = PerFrameLayerNorm(
            configuration.temporal_channels
        )
        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout1d(configuration.dropout)
        self.temporal_blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    configuration.temporal_channels,
                    configuration.kernel_size,
                    dilation,
                    configuration.dropout,
                )
                for dilation in configuration.dilations
            ]
        )
        self.output_head = nn.Conv1d(
            configuration.temporal_channels,
            configuration.output_dimension,
            kernel_size=1,
        )

    def framewise_logits(
        self, features: torch.Tensor, frame_mask: torch.Tensor
    ) -> torch.Tensor:
        _validate_feature_batch(features, frame_mask, self.input_dimension)
        mask = frame_mask.unsqueeze(1).to(dtype=features.dtype)
        values = features * mask
        values = self.input_projection(values)
        values = self.temporal_projection(values)
        values = self.input_normalization(values)
        values = self.input_activation(values)
        values = self.input_dropout(values)
        values = values * mask
        for block in self.temporal_blocks:
            values = block(values, frame_mask)
        return self.output_head(values) * mask

    def forward(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
        syllable_frame_starts: torch.Tensor,
        syllable_frame_ends: torch.Tensor,
        syllable_mask: torch.Tensor,
    ) -> TemporalModelOutput:
        frame_logits = self.framewise_logits(features, frame_mask)
        syllable_logits = mean_pool_syllable_logits(
            frame_logits,
            frame_mask,
            syllable_frame_starts,
            syllable_frame_ends,
            syllable_mask,
        )
        return TemporalModelOutput(frame_logits, syllable_logits)


def mean_pool_syllable_logits(
    frame_logits: torch.Tensor,
    frame_mask: torch.Tensor,
    syllable_frame_starts: torch.Tensor,
    syllable_frame_ends: torch.Tensor,
    syllable_mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiably average frame logits over every valid ``[start, end)``."""

    if frame_logits.ndim != 3:
        raise ValueError("frame_logits must have shape [batch, label, frame]")
    batch_size, label_count, frame_count = frame_logits.shape
    if frame_mask.shape != (batch_size, frame_count) or frame_mask.dtype != torch.bool:
        raise ValueError("frame_mask shape or dtype is invalid")
    if syllable_frame_starts.shape != syllable_frame_ends.shape:
        raise ValueError("Syllable start and end shapes differ")
    if syllable_mask.shape != syllable_frame_starts.shape:
        raise ValueError("syllable_mask shape differs from syllable boundaries")
    if syllable_mask.dtype != torch.bool:
        raise ValueError("syllable_mask must be Boolean")
    if syllable_frame_starts.ndim != 2 or syllable_frame_starts.shape[0] != batch_size:
        raise ValueError("Syllable boundaries must have shape [batch, syllable]")
    if syllable_frame_starts.dtype != torch.long or syllable_frame_ends.dtype != torch.long:
        raise ValueError("Syllable boundaries must use torch.long")

    frame_lengths = frame_mask.sum(dim=1, dtype=torch.long)
    limits = frame_lengths.unsqueeze(1).expand_as(syllable_frame_ends)
    valid_starts = syllable_frame_starts[syllable_mask]
    valid_ends = syllable_frame_ends[syllable_mask]
    if valid_starts.numel() == 0:
        raise ValueError("A batch must contain at least one valid syllable")
    if torch.any(valid_starts < 0) or torch.any(valid_starts >= valid_ends):
        raise ValueError("A valid syllable has an invalid frame interval")
    if torch.any(valid_ends > limits[syllable_mask]):
        raise ValueError("A valid syllable interval exceeds its real track frames")

    safe_starts = syllable_frame_starts.masked_fill(~syllable_mask, 0)
    safe_ends = syllable_frame_ends.masked_fill(~syllable_mask, 0)
    prefix = F.pad(torch.cumsum(frame_logits, dim=2), (1, 0))
    gather_starts = safe_starts.unsqueeze(1).expand(-1, label_count, -1)
    gather_ends = safe_ends.unsqueeze(1).expand(-1, label_count, -1)
    summed = torch.gather(prefix, 2, gather_ends) - torch.gather(
        prefix, 2, gather_starts
    )
    lengths = (safe_ends - safe_starts).clamp_min(1).unsqueeze(1)
    pooled = (summed / lengths).transpose(1, 2)
    return pooled * syllable_mask.unsqueeze(2).to(dtype=pooled.dtype)


def _validate_feature_batch(
    features: torch.Tensor, frame_mask: torch.Tensor, input_dimension: int
) -> None:
    if features.ndim != 3 or features.shape[1] != input_dimension:
        raise ValueError(
            f"Expected features [batch, {input_dimension}, frame], got {features.shape}"
        )
    if frame_mask.shape != (features.shape[0], features.shape[2]):
        raise ValueError("frame_mask shape differs from features")
    if frame_mask.dtype != torch.bool:
        raise ValueError("frame_mask must be Boolean")
    if not torch.isfinite(features).all():
        raise ValueError("Features contain non-finite values")
    frame_lengths = frame_mask.sum(dim=1, dtype=torch.long)
    expected = torch.arange(frame_mask.shape[1], device=frame_mask.device).unsqueeze(0)
    expected = expected < frame_lengths.unsqueeze(1)
    if not torch.equal(frame_mask, expected):
        raise ValueError("frame_mask must be left-aligned and contiguous")




def normalized_syllable_bce_with_logits(
    syllable_logits: torch.Tensor,
    targets: torch.Tensor,
    syllable_mask: torch.Tensor,
    positive_weights: torch.Tensor,
    mean_training_syllables_per_track: float,
) -> torch.Tensor:
    """Return the original track-sampled estimate of mean syllable BCE."""

    if syllable_logits.shape != targets.shape or syllable_logits.ndim != 3:
        raise ValueError("Logits and targets must share [batch, syllable, label]")
    if syllable_mask.shape != syllable_logits.shape[:2]:
        raise ValueError("syllable_mask shape differs from logits")
    if positive_weights.shape != (syllable_logits.shape[2],):
        raise ValueError("positive_weights must contain one value per label")
    if not mean_training_syllables_per_track > 0.0:
        raise ValueError("mean_training_syllables_per_track must be positive")
    elementwise = F.binary_cross_entropy_with_logits(
        syllable_logits,
        targets,
        pos_weight=positive_weights,
        reduction="none",
    )
    per_syllable = elementwise.mean(dim=2)
    valid_sum = (per_syllable * syllable_mask.to(per_syllable.dtype)).sum()
    denominator = syllable_logits.shape[0] * mean_training_syllables_per_track
    return valid_sum / denominator
