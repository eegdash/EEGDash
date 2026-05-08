# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Run a splitter and capture train/test sample IDs as JSON-serialisable folds."""

from __future__ import annotations

import inspect
import json
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "apply_split_manifest",
    "make_split_manifest",
    "manifest_to_json",
]


def make_split_manifest(
    splitter: Any,
    y: np.ndarray,
    metadata: pd.DataFrame,
    target: Optional[str] = None,
    sample_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Run ``splitter`` and capture ``train``/``test`` sample IDs per fold.

    MOABB evaluation splitters expose ``split(y, metadata)``; sklearn-style
    splitters (e.g. :class:`moabb.evaluations.splitters.LearningCurveSplitter`)
    expose ``split(X, y, groups=...)``. This dispatches by signature.
    """
    if sample_ids is None:
        if "sample_id" in metadata.columns:
            sample_ids = metadata["sample_id"].tolist()
        else:
            sample_ids = [str(i) for i in range(len(metadata.index))]
    sample_ids = list(sample_ids)
    if len(sample_ids) != len(metadata.index):
        raise ValueError(
            "len(sample_ids) must match len(metadata) "
            f"(got {len(sample_ids)} vs {len(metadata.index)})."
        )

    if "groups" in inspect.signature(splitter.split).parameters:
        groups = (
            metadata["subject"].to_numpy() if "subject" in metadata.columns else None
        )
        fold_iter = splitter.split(
            np.zeros((len(metadata.index), 1)), np.asarray(y), groups=groups
        )
    else:
        fold_iter = splitter.split(np.asarray(y), metadata)

    folds = [
        {
            "train": [sample_ids[int(i)] for i in train_idx],
            "test": [sample_ids[int(i)] for i in test_idx],
        }
        for train_idx, test_idx in fold_iter
    ]
    cls = type(splitter)
    return {
        "splitter_class": f"{cls.__module__}.{cls.__qualname__}",
        "random_seed": getattr(splitter, "random_state", None),
        "n_folds": len(folds),
        "target": target,
        "folds": folds,
    }


def apply_split_manifest(
    metadata: pd.DataFrame,
    manifest: dict[str, Any],
    fold: int = 0,
    split: str = "train",
) -> np.ndarray:
    """Return a boolean mask aligned with ``metadata`` for ``fold``/``split``."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
    if not (0 <= fold < manifest.get("n_folds", 0)):
        raise IndexError(
            f"Fold {fold} out of range (n_folds={manifest.get('n_folds')})."
        )
    if "sample_id" not in metadata.columns:
        raise ValueError("metadata must have a 'sample_id' column to apply a manifest.")
    target_ids = set(manifest["folds"][fold][split])
    return metadata["sample_id"].isin(target_ids).to_numpy()


def manifest_to_json(manifest: dict[str, Any]) -> str:
    """Render a manifest as a stable JSON string (sorted keys)."""
    return json.dumps(manifest, sort_keys=True, default=str)
