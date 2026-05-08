# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Leakage-safe evaluation utilities for EEGDash datasets.

Opt-in subpackage; requires MOABB:

.. code-block:: bash

    pip install eegdash[moabb]

Public surface:

- :func:`get_splitter` — friendly factory over :mod:`moabb.evaluations.splitters`.
- :func:`make_split_manifest` / :func:`apply_split_manifest` /
  :func:`manifest_to_json` — JSON-serialisable folds.
- :func:`train_test_split`, :func:`k_fold` — HuggingFace-style entry
  points returning ``{"train": ..., "test": ...}`` and ``(train, test)``
  iterables.
- :func:`assert_no_leakage`, :class:`LeakageError` — disjointness check
  + JSON ``leakage_report`` line on stdout.
- :func:`describe_split` — one-screen audit of a manifest.
- :func:`majority_baseline`, :func:`median_baseline` — chance-level
  baselines for classification and regression.
"""

from __future__ import annotations

try:
    import moabb.evaluations.splitters  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised on bare installs
    raise ImportError(
        "eegdash.splits requires MOABB. Install with: pip install eegdash[moabb]"
    ) from exc

import inspect
import json
import sys
from collections import Counter
from collections.abc import Iterator
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._splitters import get_splitter

ArrayLike = Union[Sequence, np.ndarray]
ManifestLike = Union[dict, Sequence[tuple]]


# --------------------------------------------------------------------------- #
# Split manifest: serialisable folds                                          #
# --------------------------------------------------------------------------- #


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


_GROUP_TO_SPLITTER: dict[str, str] = {
    "subject": "cross_subject",
    "session": "cross_session",
    "dataset": "cross_dataset",
}


# --------------------------------------------------------------------------- #
# HuggingFace-style entry points                                              #
# --------------------------------------------------------------------------- #


def _resolve_metadata(
    dataset: Any, target: Optional[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return ``(y, metadata)`` from a braindecode dataset or a DataFrame."""
    if isinstance(dataset, pd.DataFrame):
        metadata = dataset
    elif hasattr(dataset, "get_metadata"):
        try:
            metadata = pd.DataFrame(dataset.get_metadata()).reset_index(drop=True)
        except (TypeError, AttributeError, ValueError):
            metadata = pd.DataFrame(dataset.description).reset_index(drop=True)
    elif hasattr(dataset, "description"):
        metadata = pd.DataFrame(dataset.description).reset_index(drop=True)
    else:
        raise TypeError(
            f"Cannot extract metadata from {type(dataset).__name__}; pass a "
            "braindecode dataset or a DataFrame."
        )
    if "sample_id" not in metadata.columns:
        metadata = metadata.copy()
        metadata["sample_id"] = [f"row-{i:06d}" for i in range(len(metadata.index))]
    if target is not None and target in metadata.columns:
        y = metadata[target].to_numpy()
    else:
        y = np.zeros(len(metadata.index), dtype=int)
    return y, metadata


def _select(dataset: Any, indices: np.ndarray) -> Any:
    """Return the subset of ``dataset`` at ``indices`` (HF, braindecode, or DataFrame)."""
    idx = [int(i) for i in indices]
    if hasattr(dataset, "select"):
        return dataset.select(idx)
    if hasattr(dataset, "split"):
        return dataset.split({"_subset": idx})["_subset"]
    if hasattr(dataset, "iloc"):
        return dataset.iloc[idx]
    return [dataset[i] for i in idx]


def _resolve_splitter(group: str, **kwargs: Any):
    if group not in _GROUP_TO_SPLITTER:
        raise ValueError(
            f"Unknown group '{group}'. Expected one of: "
            f"{sorted(_GROUP_TO_SPLITTER.keys())}."
        )
    return get_splitter(_GROUP_TO_SPLITTER[group], **kwargs)


def train_test_split(
    dataset: Any,
    *,
    test_size: float | None = None,
    group: str = "subject",
    target: str | None = "target",
    seed: int | None = 42,
    **splitter_kwargs: Any,
) -> dict[str, Any]:
    """Group-aware ``{"train": ..., "test": ...}`` split, HF-style.

    Mirrors :meth:`datasets.Dataset.train_test_split`. The chosen
    ``group`` (``"subject"`` / ``"session"`` / ``"dataset"``) stays
    disjoint between the two halves.
    """
    y, metadata = _resolve_metadata(dataset, target)
    if test_size is not None:
        splitter_kwargs.setdefault("test_size", float(test_size))
    splitter_kwargs.setdefault("random_state", seed)
    splitter = _resolve_splitter(group, **splitter_kwargs)
    for train_idx, test_idx in splitter.split(y, metadata):
        return {
            "train": _select(dataset, np.asarray(train_idx)),
            "test": _select(dataset, np.asarray(test_idx)),
        }
    raise RuntimeError("Splitter produced no folds.")


def k_fold(
    dataset: Any,
    *,
    n_folds: int = 5,
    group: str = "subject",
    target: str | None = "target",
    seed: int | None = 42,
    **splitter_kwargs: Any,
) -> Iterator[tuple[Any, Any]]:
    """Yield ``(train, test)`` pairs across ``n_folds`` group-aware folds."""
    y, metadata = _resolve_metadata(dataset, target)
    splitter_kwargs.setdefault("n_folds", n_folds)
    splitter_kwargs.setdefault("random_state", seed)
    splitter = _resolve_splitter(group, **splitter_kwargs)
    for train_idx, test_idx in splitter.split(y, metadata):
        yield (
            _select(dataset, np.asarray(train_idx)),
            _select(dataset, np.asarray(test_idx)),
        )


# --------------------------------------------------------------------------- #
# Leakage assertions                                                          #
# --------------------------------------------------------------------------- #


class LeakageError(ValueError):
    """Raised when a split manifest leaks groups across train/test."""


def _values_for_ids(metadata: pd.DataFrame, ids: Sequence, by: str) -> set:
    """Return the set of ``by`` values reachable from a list of ``sample_id``s or row indices."""
    if by not in metadata.columns:
        raise ValueError(
            f"Metadata has no column '{by}'. "
            f"Available columns: {list(metadata.columns)}"
        )
    if "sample_id" in metadata.columns:
        mask = metadata["sample_id"].isin(set(ids))
        if mask.any():
            return set(metadata.loc[mask, by].dropna().astype(str).unique().tolist())
    try:
        idx = [int(i) for i in ids]
    except (TypeError, ValueError):
        return set()
    if not idx:
        return set()
    return set(metadata.iloc[idx][by].dropna().astype(str).unique().tolist())


def assert_no_leakage(
    manifest_or_splits: ManifestLike,
    metadata: pd.DataFrame,
    by: str = "subject",
) -> int:
    """Assert no train/test overlap on the ``by`` column for any fold.

    Always emits a JSON line ``{"leakage_report": {"overlap": <int>,
    "by": "<by>"}}`` on stdout, even when the overlap is 0.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    if not isinstance(by, str) or not by:
        raise ValueError("`by` must be a non-empty string column name.")

    if isinstance(manifest_or_splits, dict) and "folds" in manifest_or_splits:
        pairs = [(f["train"], f["test"]) for f in manifest_or_splits["folds"]]
    else:
        pairs = list(manifest_or_splits)
        for pair in pairs:
            if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                raise TypeError("Each fold must be a (train_ids, test_ids) tuple/list.")

    fold_overlaps: list[dict] = []
    max_overlap = 0
    for fold_index, (train_ids, test_ids) in enumerate(pairs):
        overlap = _values_for_ids(metadata, train_ids, by) & _values_for_ids(
            metadata, test_ids, by
        )
        if overlap:
            fold_overlaps.append(
                {"fold": fold_index, "n": len(overlap), "values": sorted(overlap)[:10]}
            )
        max_overlap = max(max_overlap, len(overlap))

    sys.stdout.write(
        json.dumps({"leakage_report": {"overlap": int(max_overlap), "by": str(by)}})
        + "\n"
    )
    sys.stdout.flush()
    if max_overlap > 0:
        raise LeakageError(
            f"Detected train/test overlap on column '{by}' "
            f"in {len(fold_overlaps)} fold(s); max_overlap={max_overlap}. "
            f"Details (first 10 values per fold): {fold_overlaps}"
        )
    return max_overlap


# --------------------------------------------------------------------------- #
# Chance-level baselines                                                      #
# --------------------------------------------------------------------------- #


def majority_baseline(
    y_train: ArrayLike, y_test: ArrayLike
) -> dict[str, Union[float, str]]:
    """Predict the most frequent training label for every test sample.

    Returns ``{"chance_level", "baseline_score", "metric": "accuracy"}``.
    """
    y_train_arr = np.atleast_1d(np.asarray(y_train))
    y_test_arr = np.atleast_1d(np.asarray(y_test))
    if y_train_arr.size == 0:
        raise ValueError("y_train is empty; cannot fit a majority baseline.")
    if y_test_arr.size == 0:
        return {
            "chance_level": float("nan"),
            "baseline_score": float("nan"),
            "metric": "accuracy",
        }
    train_mode, _ = Counter(y_train_arr.tolist()).most_common(1)[0]
    test_counts = Counter(y_test_arr.tolist())
    return {
        "chance_level": float(max(test_counts.values()) / y_test_arr.size),
        "baseline_score": float(np.mean(y_test_arr == train_mode)),
        "metric": "accuracy",
    }


def median_baseline(
    y_train: ArrayLike, y_test: ArrayLike
) -> dict[str, Union[float, str]]:
    """Predict the training median for every test sample.

    Returns ``{"chance_level": 0.0, "baseline_score", "metric": "r2"}``.
    """
    y_train_arr = np.atleast_1d(np.asarray(y_train)).astype(float)
    y_test_arr = np.atleast_1d(np.asarray(y_test)).astype(float)
    if y_train_arr.size == 0:
        raise ValueError("y_train is empty; cannot fit a median baseline.")
    if y_test_arr.size == 0:
        return {
            "chance_level": float("nan"),
            "baseline_score": float("nan"),
            "metric": "r2",
        }
    train_median = float(np.median(y_train_arr))
    ss_res = float(np.sum((y_test_arr - train_median) ** 2))
    ss_tot = float(np.sum((y_test_arr - float(np.mean(y_test_arr))) ** 2))
    baseline_score = 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "chance_level": 0.0,
        "baseline_score": float(baseline_score),
        "metric": "r2",
    }


# --------------------------------------------------------------------------- #
# Split summary                                                               #
# --------------------------------------------------------------------------- #


def _nunique_safe(df: pd.DataFrame, col: str) -> int:
    return int(df[col].nunique()) if col in df.columns else 0


def _fold_stats(
    fold: dict[str, list[str]], metadata: pd.DataFrame, target: Optional[str]
) -> dict[str, Any]:
    train = metadata[metadata["sample_id"].isin(set(fold["train"]))]
    test = metadata[metadata["sample_id"].isin(set(fold["test"]))]
    out: dict[str, Any] = {"n_train": len(train.index), "n_test": len(test.index)}
    for col in ("subject", "session", "dataset"):
        out[f"{col}s_train"] = _nunique_safe(train, col)
        out[f"{col}s_test"] = _nunique_safe(test, col)
    if target and target in metadata.columns:
        out["class_balance_train"] = dict(Counter(train[target].dropna().tolist()))
        out["class_balance_test"] = dict(Counter(test[target].dropna().tolist()))
    return out


def describe_split(
    manifest: dict[str, Any],
    metadata: pd.DataFrame,
    target: Optional[str] = None,
    print_report: bool = True,
) -> dict[str, Any]:
    """Return a structured summary of a manifest and optionally print it."""
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    if "sample_id" not in metadata.columns:
        raise ValueError("metadata must have a 'sample_id' column.")

    folds = manifest.get("folds", [])
    per_fold = [_fold_stats(f, metadata, target) for f in folds]
    coverage = {
        "n_samples": int(len(metadata.index)),
        **{
            f"n_{col}s": int(metadata[col].nunique())
            for col in ("subject", "session", "run", "dataset")
            if col in metadata.columns
        },
    }
    warnings: list[str] = []
    for i, stats in enumerate(per_fold):
        if stats["n_test"] == 0:
            warnings.append(f"Fold {i} has an empty test set.")
        if stats.get("subjects_test", 0) < 2:
            warnings.append(
                f"Fold {i} test set covers only "
                f"{stats.get('subjects_test', 0)} subject(s)."
            )
        classes = stats.get("class_balance_test") or {}
        if target and 0 < len(classes) < 2:
            warnings.append(
                f"Fold {i} test set has a single class "
                f"({list(classes.keys())[0]!r}) -- chance level will be 100%."
            )
    summary = {
        "n_folds": int(manifest.get("n_folds", len(folds))),
        "splitter_class": manifest.get("splitter_class"),
        "random_seed": manifest.get("random_seed"),
        "target": manifest.get("target", target),
        "coverage": coverage,
        "per_fold": per_fold,
        "warnings": warnings,
    }
    if print_report:
        print(
            f"Split summary -- folds={summary['n_folds']}, "
            f"splitter={summary['splitter_class']}, "
            f"seed={summary['random_seed']}, target={summary['target']}"
        )
        print(f"Coverage: {coverage}")
        for i, stats in enumerate(per_fold):
            line = (
                f"Fold {i}: train={stats['n_train']} "
                f"({stats.get('subjects_train', 0)} subj), "
                f"test={stats['n_test']} ({stats.get('subjects_test', 0)} subj)"
            )
            if "class_balance_test" in stats:
                line += f", classes_test={stats['class_balance_test']}"
            print(line)
        for w in warnings:
            print(f"Warning: {w}")
    return summary


__all__ = [
    "LeakageError",
    "apply_split_manifest",
    "assert_no_leakage",
    "describe_split",
    "get_splitter",
    "k_fold",
    "majority_baseline",
    "make_split_manifest",
    "manifest_to_json",
    "median_baseline",
    "train_test_split",
]
