"""Frozen MusicFM loading and framewise layer-7 extraction."""

from __future__ import annotations

import importlib
import importlib.machinery
import math
import sys
import types
from pathlib import Path
from unittest import mock

import torch
from torch import nn


SAMPLE_RATE_HZ = 24_000
MODEL_HOP_SAMPLES = 240
MODEL_SUBSAMPLING_FACTOR = 4
MODEL_TOKEN_STEP_SAMPLES = MODEL_HOP_SAMPLES * MODEL_SUBSAMPLING_FACTOR
MODEL_TOKEN_STEP_SEC = MODEL_TOKEN_STEP_SAMPLES / SAMPLE_RATE_HZ
FRAME_RATE_HZ = SAMPLE_RATE_HZ / MODEL_TOKEN_STEP_SAMPLES
EMBEDDING_DIMENSION = 1_024
OUTPUT_LAYER_INDEX = 7
CORE_DURATION_SEC = 30
CONTEXT_DURATION_SEC = 5
CORE_FRAMES = int(CORE_DURATION_SEC * FRAME_RATE_HZ)
CONTEXT_FRAMES = int(CONTEXT_DURATION_SEC * FRAME_RATE_HZ)
BUNDLED_CONFORMER_CONFIG = Path(__file__).with_name("musicfm_conformer_config.json")


def expected_token_count(input_samples: int) -> int:
    """Return the MusicFM token count for a waveform length."""

    if input_samples <= 0:
        raise ValueError("MusicFM input must contain at least one sample")
    mel_frames_after_drop = input_samples // MODEL_HOP_SAMPLES
    return math.ceil(mel_frames_after_drop / MODEL_SUBSAMPLING_FACTOR)


def chunk_plan(total_frames: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return core and context intervals on the global 25 Hz grid."""

    if total_frames <= 0:
        raise ValueError("A track must contain at least one MusicFM frame")
    chunks: list[tuple[int, int, int, int]] = []
    for core_start in range(0, total_frames, CORE_FRAMES):
        core_end = min(total_frames, core_start + CORE_FRAMES)
        input_start = max(0, core_start - CONTEXT_FRAMES)
        input_end = min(total_frames, core_end + CONTEXT_FRAMES)
        chunks.append((core_start, core_end, input_start, input_end))
    return tuple(chunks)


def validate_musicfm_root(root: Path) -> tuple[Path, Path]:
    """Validate the external MusicFM checkout and return data files."""

    root = root.resolve()
    code = root / "model" / "musicfm_25hz.py"
    checkpoint = root / "data" / "pretrained_msd.pt"
    statistics = root / "data" / "msd_stats.json"
    missing = [path for path in (code, checkpoint, statistics) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete MusicFM repository: " + ", ".join(map(str, missing))
        )
    return checkpoint, statistics


def _external_musicfm_class(root: Path) -> type[nn.Module]:
    """Import MusicFM from the supplied checkout without a name collision."""

    root = root.resolve()
    package = sys.modules.get("musicfm")
    if package is None:
        package = types.ModuleType("musicfm")
        package.__package__ = "musicfm"
        package.__path__ = [str(root)]  # type: ignore[attr-defined]
        specification = importlib.machinery.ModuleSpec(
            "musicfm", loader=None, is_package=True
        )
        specification.submodule_search_locations = [str(root)]
        package.__spec__ = specification
        sys.modules["musicfm"] = package
    else:
        search_locations = {
            Path(value).resolve() for value in getattr(package, "__path__", ())
        }
        if root not in search_locations:
            raise ImportError(
                "A different top-level musicfm module is already imported; "
                "start a fresh process and pass the intended --musicfm-root"
            )
    module = importlib.import_module("musicfm.model.musicfm_25hz")
    musicfm_class = getattr(module, "MusicFM25Hz", None)
    if not isinstance(musicfm_class, type):
        raise ImportError("MusicFM25Hz was not found in model/musicfm_25hz.py")
    return musicfm_class


def load_musicfm(root: Path, device: torch.device) -> nn.Module:
    """Load frozen MusicFM-MSD in deterministic evaluation mode."""

    checkpoint, statistics = validate_musicfm_root(root)
    from transformers import Wav2Vec2ConformerConfig

    musicfm_class = _external_musicfm_class(root)
    base_configuration = Wav2Vec2ConformerConfig.from_json_file(
        str(BUNDLED_CONFORMER_CONFIG)
    )
    with mock.patch.object(
        Wav2Vec2ConformerConfig,
        "from_pretrained",
        return_value=base_configuration,
    ):
        model = musicfm_class(
            is_flash=False,
            stat_path=str(statistics),
            model_path=str(checkpoint),
        )
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


@torch.inference_mode()
def extract_frozen_layer(
    model: nn.Module,
    waveform: torch.Tensor,
    layer_index: int = OUTPUT_LAYER_INDEX,
) -> torch.Tensor:
    """Return frozen hidden states with shape [1, frame, 1024]."""

    if model.training:
        raise ValueError("MusicFM must remain in evaluation mode")
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError("MusicFM waveform must have shape [1, samples]")
    features = model.preprocessing(waveform, features=["melspec_2048"])
    features = model.normalize(features)
    encoded = model.conv(features["melspec_2048"])
    output = model.conformer(encoded, output_hidden_states=True)
    hidden = output.hidden_states[layer_index]
    expected_shape = (
        1,
        expected_token_count(int(waveform.shape[1])),
        EMBEDDING_DIMENSION,
    )
    if tuple(hidden.shape) != expected_shape:
        raise RuntimeError(
            f"Expected MusicFM output {expected_shape}, got {tuple(hidden.shape)}"
        )
    if not torch.isfinite(hidden).all():
        raise RuntimeError("MusicFM produced non-finite representations")
    return hidden
