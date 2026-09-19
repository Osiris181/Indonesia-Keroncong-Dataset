"""Load and validate the common IKD syllable-level modeling index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COLUMNS = ("luk", "cengkok", "gregel", "embat", "nggandhul")
RUNS = (1, 2, 3, 4, 5)
SPLITS = ("train", "validation", "test")
EXPECTED_SYLLABLES = 6_772
EXPECTED_WITH_TECHNIQUE = 2_676
EXPECTED_LABEL_SUPPORT = {
    "luk": 1_658,
    "cengkok": 374,
    "gregel": 613,
    "embat": 563,
    "nggandhul": 351,
}

TRACEABILITY_COLUMNS = (
    "syllable_id",
    "track_id",
    "performer_group_id",
    "word_id",
    "word_text",
    "syllable_index_in_word",
    "syllable_text",
    "start_sec",
    "end_sec",
)


def load_syllable_index(path: Path) -> pd.DataFrame:
    """Read the released index and enforce the published corpus invariants."""

    if not path.is_file():
        raise FileNotFoundError(f"Syllable index not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    fold_columns = [f"fold_{run}_split" for run in RUNS]
    required = set(TRACEABILITY_COLUMNS) | set(LABEL_COLUMNS) | set(fold_columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Syllable index is missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_SYLLABLES:
        raise ValueError(
            f"Expected {EXPECTED_SYLLABLES} syllables, found {len(frame)}"
        )
    if frame["syllable_id"].isna().any() or frame["syllable_id"].duplicated().any():
        raise ValueError("syllable_id must be present and unique")

    for label, expected_support in EXPECTED_LABEL_SUPPORT.items():
        values = pd.to_numeric(frame[label], errors="raise")
        if not values.isin((0, 1)).all():
            raise ValueError(f"{label} must contain only 0 and 1")
        frame[label] = values.astype(np.float32)
        observed_support = int(values.sum())
        if observed_support != expected_support:
            raise ValueError(
                f"Expected support {expected_support} for {label}, "
                f"found {observed_support}"
            )

    targets = frame.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.int64)
    with_technique = int((targets.sum(axis=1) > 0).sum())
    if with_technique != EXPECTED_WITH_TECHNIQUE:
        raise ValueError(
            f"Expected {EXPECTED_WITH_TECHNIQUE} technique-positive syllables, "
            f"found {with_technique}"
        )

    for run, column in zip(RUNS, fold_columns, strict=True):
        observed = set(frame[column].dropna().astype(str))
        if observed != set(SPLITS):
            raise ValueError(
                f"Run {run} must contain {list(SPLITS)}, found {sorted(observed)}"
            )
        if not (frame.groupby("track_id")[column].nunique() == 1).all():
            raise ValueError(f"Run {run} assigns a track to multiple partitions")
        if not (frame.groupby("performer_group_id")[column].nunique() == 1).all():
            raise ValueError(
                f"Run {run} assigns a performer group to multiple partitions"
            )

    return frame.reset_index(drop=True)


def partition_indices(frame: pd.DataFrame, run: int) -> dict[str, np.ndarray]:
    """Return row indices for one published train/validation/test rotation."""

    if run not in RUNS:
        raise ValueError(f"run must be one of {list(RUNS)}")
    values = frame[f"fold_{run}_split"].astype(str).to_numpy()
    output = {split: np.flatnonzero(values == split) for split in SPLITS}
    for split, indices in output.items():
        if indices.size == 0:
            raise ValueError(f"Run {run} has an empty {split} partition")
    return output

