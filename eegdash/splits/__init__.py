# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Leakage-safe evaluation utilities for EEGDash datasets.

Opt-in subpackage; requires MOABB:

.. code-block:: bash

    pip install eegdash[moabb]

Public surface (single-API: every entry point yields boolean mask pairs):

- :func:`get_splitter` — factory over :mod:`moabb.evaluations.splitters`.
- :func:`train_test_split`, :func:`k_fold` — return / yield
  ``(train_mask, test_mask)`` boolean arrays aligned with ``metadata`` rows.
  Pass ``splitter=`` to use any pre-built splitter (``learning_curve``,
  ``within_subject``, ...); otherwise builds a group-aware MOABB splitter
  from ``group``/``n_folds``/``test_size``.
- :func:`assert_no_leakage`, :class:`LeakageError` — disjointness check
  + JSON ``leakage_report`` line on stdout.
- :func:`describe_split` — one-screen audit of a folds list.
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
from collections.abc import Iterable, Iterator
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._splitters import get_splitter

ArrayLike = Union[Sequence, np.ndarray]
FoldsLike = Iterable[tuple[np.ndarray, np.ndarray]]

_GROUP_TO_SPLITTER: dict[str, str] = {
    "subject": "cross_subject",
    "session": "cross_session",
    "dataset": "cross_dataset",
}


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
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


def _resolve_splitter(group: str, **kwargs: Any):
    if group not in _GROUP_TO_SPLITTER:
        raise ValueError(
            f"Unknown group '{group}'. Expected one of: "
            f"{sorted(_GROUP_TO_SPLITTER.keys())}."
        )
    return get_splitter(_GROUP_TO_SPLITTER[group], **kwargs)


def _iter_fold_masks(
    splitter: Any, y: np.ndarray, metadata: pd.DataFrame
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Run ``splitter`` and yield ``(train_mask, test_mask)`` boolean arrays.

    MOABB evaluation splitters expose ``split(y, metadata)``; sklearn-style
    splitters (e.g. :class:`moabb.evaluations.splitters.LearningCurveSplitter`)
    expose ``split(X, y, groups=...)``. Dispatched by signature.
    """
    n = len(metadata.index)
    if "groups" in inspect.signature(splitter.split).parameters:
        groups = (
            metadata["subject"].to_numpy() if "subject" in metadata.columns else None
        )
        fold_iter = splitter.split(np.zeros((n, 1)), np.asarray(y), groups=groups)
    else:
        fold_iter = splitter.split(np.asarray(y), metadata)
    for train_idx, test_idx in fold_iter:
        train_mask = np.zeros(n, dtype=bool)
        train_mask[np.asarray(train_idx, dtype=int)] = True
        test_mask = np.zeros(n, dtype=bool)
        test_mask[np.asarray(test_idx, dtype=int)] = True
        yield train_mask, test_mask


# --------------------------------------------------------------------------- #
# Single-API entry points                                                     #
# --------------------------------------------------------------------------- #


def k_fold(
    dataset: Any,
    *,
    splitter: Any = None,
    n_folds: int = 5,
    group: str = "subject",
    target: Optional[str] = "target",
    seed: Optional[int] = 42,
    **splitter_kwargs: Any,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_mask, test_mask)`` boolean arrays per fold.

    With ``splitter=None``, builds a group-aware MOABB splitter from
    ``group`` / ``n_folds`` / ``seed``. Pass ``splitter=`` to use any
    pre-built splitter (``"learning_curve"``, ``"within_subject"``, ...).
    Masks align with ``metadata`` rows: slice arrays with ``X[train_mask]``
    or DataFrames with ``metadata.loc[train_mask]``.
    """
    y, metadata = _resolve_metadata(dataset, target)
    if splitter is None:
        splitter_kwargs.setdefault("n_folds", n_folds)
        splitter_kwargs.setdefault("random_state", seed)
        splitter = _resolve_splitter(group, **splitter_kwargs)
    yield from _iter_fold_masks(splitter, y, metadata)


def train_test_split(
    dataset: Any,
    *,
    splitter: Any = None,
    test_size: Optional[float] = None,
    group: str = "subject",
    target: Optional[str] = "target",
    seed: Optional[int] = 42,
    **splitter_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_mask, test_mask)`` for the first fold of ``k_fold``.

    Mirrors the single-fold case of :func:`k_fold`. The chosen ``group``
    (``"subject"`` / ``"session"`` / ``"dataset"``) stays disjoint between
    train and test.
    """
    if test_size is not None:
        splitter_kwargs.setdefault("test_size", float(test_size))
    for fold in k_fold(
        dataset,
        splitter=splitter,
        group=group,
        target=target,
        seed=seed,
        **splitter_kwargs,
    ):
        return fold
    raise RuntimeError("Splitter produced no folds.")


# --------------------------------------------------------------------------- #
# Leakage assertions                                                          #
# --------------------------------------------------------------------------- #


class LeakageError(ValueError):
    """Raised when a folds list leaks groups across train/test."""


def _values_for_mask(metadata: pd.DataFrame, mask: ArrayLike, by: str) -> set:
    if by not in metadata.columns:
        raise ValueError(
            f"Metadata has no column '{by}'. "
            f"Available columns: {list(metadata.columns)}"
        )
    arr = np.asarray(mask)
    if arr.size == 0:
        return set()
    rows = metadata.loc[arr] if arr.dtype == bool else metadata.iloc[arr.astype(int)]
    return set(rows[by].dropna().astype(str).unique().tolist())


def assert_no_leakage(
    folds: FoldsLike, metadata: pd.DataFrame, by: str = "subject"
) -> int:
    """Assert no train/test overlap on the ``by`` column for any fold.

    Always emits a JSON line ``{"leakage_report": {"overlap": <int>,
    "by": "<by>"}}`` on stdout, even when the overlap is 0. ``folds`` is
    any iterable of ``(train_mask, test_mask)`` tuples (e.g. the output of
    ``list(k_fold(...))``).
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    if not isinstance(by, str) or not by:
        raise ValueError("`by` must be a non-empty string column name.")

    fold_overlaps: list[dict] = []
    max_overlap = 0
    for fold_index, pair in enumerate(folds):
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise TypeError("Each fold must be a (train_mask, test_mask) tuple/list.")
        train_mask, test_mask = pair
        overlap = _values_for_mask(metadata, train_mask, by) & _values_for_mask(
            metadata, test_mask, by
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


def _slice_mask(metadata: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    return metadata.loc[mask] if mask.dtype == bool else metadata.iloc[mask.astype(int)]


def _fold_stats(
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    metadata: pd.DataFrame,
    target: Optional[str],
) -> dict[str, Any]:
    train = _slice_mask(metadata, train_mask)
    test = _slice_mask(metadata, test_mask)
    out: dict[str, Any] = {"n_train": len(train.index), "n_test": len(test.index)}
    for col in ("subject", "session", "dataset"):
        out[f"{col}s_train"] = _nunique_safe(train, col)
        out[f"{col}s_test"] = _nunique_safe(test, col)
    if target and target in metadata.columns:
        out["class_balance_train"] = dict(Counter(train[target].dropna().tolist()))
        out["class_balance_test"] = dict(Counter(test[target].dropna().tolist()))
    return out


def describe_split(
    folds: FoldsLike,
    metadata: pd.DataFrame,
    target: Optional[str] = None,
    print_report: bool = True,
) -> dict[str, Any]:
    """Audit ``folds`` (list of ``(train_mask, test_mask)`` tuples) against ``metadata``.

    Returns a dict with per-fold sizes, distinct subjects/sessions per side,
    optional class balance, and a list of structural warnings (empty test
    set, single-class test set, single-subject test set).
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    folds_list = [(np.asarray(tr), np.asarray(te)) for tr, te in folds]
    per_fold = [_fold_stats(tr, te, metadata, target) for tr, te in folds_list]
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
        "n_folds": len(folds_list),
        "target": target,
        "coverage": coverage,
        "per_fold": per_fold,
        "warnings": warnings,
    }
    if print_report:
        print(
            f"Split summary -- folds={summary['n_folds']}, target={summary['target']}"
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
    "assert_no_leakage",
    "describe_split",
    "get_splitter",
    "k_fold",
    "majority_baseline",
    "median_baseline",
    "train_test_split",
]
