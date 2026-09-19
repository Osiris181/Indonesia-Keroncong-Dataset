"""Configuration, device, and deterministic-training helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tomllib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def configuration_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_deterministic_seed(seed: int, enabled: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(enabled)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = enabled


def resolve_requested_values(
    requested: Sequence[int] | None,
    configured: Sequence[int],
    name: str,
) -> list[int]:
    values = list(configured if requested is None else requested)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {name} values were requested")
    invalid = set(values) - set(configured)
    if invalid:
        raise ValueError(
            f"Requested {name} values {sorted(invalid)} are not configured"
        )
    return values


def threshold_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    count = int(round((maximum - minimum) / step)) + 1
    grid = minimum + np.arange(count, dtype=np.float64) * step
    if not np.isclose(grid[-1], maximum):
        raise ValueError("Threshold range is not evenly divisible by its step")
    return np.round(grid, decimals=12)


def clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

