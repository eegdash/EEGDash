# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Leakage-safe evaluation utilities for EEGDash datasets.

Opt-in subpackage; requires MOABB:

.. code-block:: bash

    pip install eegdash[moabb]

Public surface (one short file per real concern):

- :func:`get_splitter` (in ``_splitters``) — friendly factory over
  :mod:`moabb.evaluations.splitters`.
- :func:`make_split_manifest` / :func:`apply_split_manifest` /
  :func:`manifest_to_json` (in ``_manifest``) — JSON-serialisable folds.
- :func:`train_test_split`, :func:`k_fold` — HuggingFace-style entry
  points returning ``{"train": ..., "test": ...}`` and ``(train, test)``
  iterables.
- :func:`assert_no_leakage`, :class:`LeakageError` — disjointness check
  + the ``leakage_report`` JSON line consumed by runtime validators.
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

import json
import sys
from collections import Counter
from collections.abc import Iterator
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._manifest import (
    SCHEMA_VERSION,
    apply_split_manifest,
    make_split_manifest,
    manifest_to_json,
)
from ._splitters import get_splitter

ArrayLike = Union[Sequence, np.ndarray]
ManifestLike = Union[dict, Sequence[tuple]]

_GROUP_TO_SPLITTER: dict[str, str] = {
    "subject": "cross_subject",
    "session": "cross_session",
    "dataset": "cross_dataset",
}

# Test folds smaller than this trigger a describe_split warning.
_MIN_TEST_GROUP_WARNING: int = 2


# --------------------------------------------------------------------------- #
# HuggingFace-style entry points                                              #
# --------------------------------------------------------------------------- #


def _resolve_metadata(
    dataset: Any, target: Optional[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return ``(y, metadata)`` from a dataset or a DataFrame.

    Uses :meth:`braindecode.datasets.BaseConcatDataset.get_metadata` (one
    row per window) when available; otherwise treats ``dataset`` as the
    metadata frame itself.
    """
    if hasattr(dataset, "get_metadata"):
        metadata = pd.DataFrame(dataset.get_metadata()).reset_index(drop=True)
    elif isinstance(dataset, pd.DataFrame):
        metadata = dataset
    else:
        raise TypeError(
            f"Cannot extract metadata from {type(dataset).__name__}; pass a "
            "Braindecode dataset (with get_metadata) or a DataFrame."
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


def _first_fold(splitter, y: np.ndarray, metadata) -> tuple[np.ndarray, np.ndarray]:
    for train_idx, test_idx in splitter.split(y, metadata):
        return np.asarray(train_idx), np.asarray(test_idx)
    raise RuntimeError("Splitter produced no folds.")


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
    train_idx, test_idx = _first_fold(splitter, y, metadata)
    return {
        "train": _select(dataset, train_idx),
        "test": _select(dataset, test_idx),
    }


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


def _emit_report(overlap: int, by: str) -> None:
    payload = {"leakage_report": {"overlap": int(overlap), "by": str(by)}}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _iter_fold_pairs(
    manifest_or_splits: ManifestLike,
) -> Iterable[tuple[Sequence, Sequence]]:
    if isinstance(manifest_or_splits, dict) and "folds" in manifest_or_splits:
        for fold in manifest_or_splits["folds"]:
            yield fold["train"], fold["test"]
        return
    for pair in manifest_or_splits:  # type: ignore[arg-type]
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise TypeError("Each fold must be a (train_ids, test_ids) tuple/list.")
        yield pair[0], pair[1]


def _values_for_ids(metadata: pd.DataFrame, ids: Sequence, by: str) -> set:
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

    max_overlap = 0
    fold_overlaps: list[dict] = []
    for fold_index, (train_ids, test_ids) in enumerate(
        _iter_fold_pairs(manifest_or_splits)
    ):
        overlap = _values_for_ids(metadata, train_ids, by) & _values_for_ids(
            metadata, test_ids, by
        )
        n_overlap = len(overlap)
        if n_overlap:
            fold_overlaps.append(
                {"fold": fold_index, "n": n_overlap, "values": sorted(overlap)[:10]}
            )
        max_overlap = max(max_overlap, n_overlap)

    _emit_report(max_overlap, by)
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


def _to_numpy(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values)
    return array.reshape(1) if array.ndim == 0 else array


def majority_baseline(
    y_train: ArrayLike, y_test: ArrayLike
) -> dict[str, Union[float, str]]:
    """Predict the most frequent training label for every test sample.

    Returns ``{"chance_level", "baseline_score", "metric": "accuracy"}``.
    """
    y_train_arr = _to_numpy(y_train)
    y_test_arr = _to_numpy(y_test)
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
    y_train_arr = _to_numpy(y_train).astype(float)
    y_test_arr = _to_numpy(y_test).astype(float)
    if y_train_arr.size == 0:
        raise ValueError("y_train is empty; cannot fit a median baseline.")
    if y_test_arr.size == 0:
        return {
            "chance_level": float("nan"),
            "baseline_score": float("nan"),
            "metric": "r2",
        }
    train_median = float(np.median(y_train_arr))
    test_mean = float(np.mean(y_test_arr))
    ss_res = float(np.sum((y_test_arr - train_median) ** 2))
    ss_tot = float(np.sum((y_test_arr - test_mean) ** 2))
    baseline_score = 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "chance_level": 0.0,
        "baseline_score": float(baseline_score),
        "metric": "r2",
    }


# --------------------------------------------------------------------------- #
# Split summary                                                               #
# --------------------------------------------------------------------------- #


def _describe_fold(
    fold: dict[str, list[str]],
    metadata: pd.DataFrame,
    target: Optional[str],
) -> dict[str, Any]:
    if "sample_id" not in metadata.columns:
        raise ValueError(
            "metadata must have a 'sample_id' column to describe a manifest."
        )
    train_rows = metadata[metadata["sample_id"].isin(set(fold["train"]))]
    test_rows = metadata[metadata["sample_id"].isin(set(fold["test"]))]
    out: dict[str, Any] = {
        "n_train": len(train_rows.index),
        "n_test": len(test_rows.index),
        "subjects_train": int(train_rows.get("subject", pd.Series([])).nunique()),
        "subjects_test": int(test_rows.get("subject", pd.Series([])).nunique()),
        "sessions_train": int(train_rows.get("session", pd.Series([])).nunique()),
        "sessions_test": int(test_rows.get("session", pd.Series([])).nunique()),
        "datasets_train": int(train_rows.get("dataset", pd.Series([])).nunique()),
        "datasets_test": int(test_rows.get("dataset", pd.Series([])).nunique()),
    }
    if target is not None and target in metadata.columns:
        out["class_balance_train"] = dict(Counter(train_rows[target].dropna().tolist()))
        out["class_balance_test"] = dict(Counter(test_rows[target].dropna().tolist()))
    return out


def _print_report(summary: dict[str, Any]) -> None:
    cov = summary["coverage"]
    print(
        "Split summary -- "
        f"folds={summary['n_folds']}, "
        f"splitter={summary['splitter_class']}, "
        f"seed={summary['random_seed']}, "
        f"target={summary['target']}"
    )
    print(
        f"Coverage: n_samples={cov['n_samples']}, "
        f"subjects={cov['n_subjects']}, sessions={cov['n_sessions']}, "
        f"runs={cov['n_runs']}, datasets={cov['n_datasets']}"
    )
    for index, stats in enumerate(summary["per_fold"]):
        line = (
            f"Fold {index}: "
            f"train={stats['n_train']} ({stats['subjects_train']} subj), "
            f"test={stats['n_test']} ({stats['subjects_test']} subj)"
        )
        if "class_balance_test" in stats:
            line += f", classes_test={stats['class_balance_test']}"
        print(line)
    if summary["warnings"]:
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")


def describe_split(
    manifest: dict[str, Any],
    metadata: pd.DataFrame,
    target: Optional[str] = None,
    print_report: bool = True,
) -> dict[str, Any]:
    """Return a structured summary of a manifest and optionally print it."""
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    folds = manifest.get("folds", [])
    per_fold = [_describe_fold(f, metadata, target) for f in folds]
    coverage = {
        "n_samples": int(len(metadata.index)),
        "n_subjects": int(metadata.get("subject", pd.Series([])).nunique()),
        "n_sessions": int(metadata.get("session", pd.Series([])).nunique()),
        "n_runs": int(metadata.get("run", pd.Series([])).nunique()),
        "n_datasets": int(metadata.get("dataset", pd.Series([])).nunique()),
    }
    warnings: list[str] = []
    for index, stats in enumerate(per_fold):
        if stats["n_test"] == 0:
            warnings.append(f"Fold {index} has an empty test set.")
        if stats["subjects_test"] < _MIN_TEST_GROUP_WARNING:
            warnings.append(
                f"Fold {index} test set covers only "
                f"{stats['subjects_test']} subject(s)."
            )
        if target is not None and "class_balance_test" in stats:
            classes = stats["class_balance_test"]
            if classes and len(classes) < 2:
                warnings.append(
                    f"Fold {index} test set has a single class "
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
        _print_report(summary)
    return summary


__all__ = [
    "LeakageError",
    "SCHEMA_VERSION",
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
