# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Metadata adapters that bridge EEGDash/Braindecode datasets and MOABB-style
splitters.

The purpose of this file is *not* to introduce a new metadata format: it
normalizes the heterogeneous representations used across EEGDash
(``EEGDashDataset``), Braindecode windows datasets and ``FeaturesConcatDataset``
into a single tabular form expected by MOABB splitters and scikit-learn
``Group*`` splitters. The main entry points are :func:`to_split_metadata` and
:func:`to_moabb_split_inputs`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

# Required column names used across the splits package. ``subject``, ``session``,
# ``run`` and ``dataset`` are all MOABB conventions; ``sample_id`` is an
# EEGDash-specific stable identifier emitted per row. ``target`` is filled in
# only when a target column can be resolved.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "subject",
    "session",
    "run",
    "dataset",
    "sample_id",
)


DatasetLike = (
    Any  # ``BaseConcatDataset``/``EEGDashDataset``/``FeaturesConcatDataset``/DataFrame
)


def _coerce_str(value: Any, default: str) -> str:
    """Return a deterministic string for a metadata value, replacing missing values."""
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    return str(value)


def _description_to_row(description: Any, index: int) -> dict[str, Any]:
    """Pull the columns we need from a description record (dict or Series)."""
    if description is None:
        return {}
    if isinstance(description, pd.Series):
        record = description.to_dict()
    elif isinstance(description, dict):
        record = dict(description)
    else:  # pragma: no cover - defensive
        try:
            record = dict(description)
        except Exception:
            record = {}
    out: dict[str, Any] = {}
    for key in ("subject", "session", "run", "dataset"):
        out[key] = record.get(key)
    # Pass through any other potentially useful columns (``site``, ``task``,
    # ``family``, ``age``, ``gender``, ``group``, etc.) so describe_split can
    # report on them and assert_no_leakage can check them.
    for key, value in record.items():
        if key not in out:
            out[key] = value
    return out


def _iter_subdatasets(dataset: DatasetLike) -> list[Any]:
    """Return the list of contained datasets for a concatenated container."""
    sub = getattr(dataset, "datasets", None)
    if sub is None:
        return []
    return list(sub)


def _resolve_target_value(
    sub_dataset: Any,
    sample_index: int,
    target: Optional[str],
) -> Optional[Any]:
    """Resolve a target value for a single sample within ``sub_dataset``."""
    if target is None:
        return None

    # ``FeaturesDataset`` and braindecode windows datasets carry per-sample
    # metadata. If a ``target`` column exists there, prefer it over a constant
    # description value.
    metadata = getattr(sub_dataset, "metadata", None)
    if isinstance(metadata, pd.DataFrame):
        if target in metadata.columns:
            try:
                return metadata.iloc[sample_index][target]
            except (IndexError, KeyError):  # pragma: no cover - defensive
                return None
        if target == "target" and "target" in metadata.columns:
            try:
                return metadata.iloc[sample_index]["target"]
            except (IndexError, KeyError):  # pragma: no cover - defensive
                return None

    # Constant target stored on the description (e.g. clinical group label).
    description = getattr(sub_dataset, "description", None)
    if isinstance(description, pd.Series):
        if target in description.index:
            return description[target]
    elif isinstance(description, dict):
        if target in description:
            return description[target]

    # Fallback: per-sample ``y`` exposed by braindecode windows datasets.
    if target == "target":
        y = getattr(sub_dataset, "y", None)
        if y is not None:
            try:
                return y[sample_index]
            except (IndexError, TypeError):  # pragma: no cover - defensive
                return None
    return None


def _samples_in_subdataset(sub_dataset: Any) -> int:
    """Return the number of windows/samples in a single sub-dataset."""
    metadata = getattr(sub_dataset, "metadata", None)
    if isinstance(metadata, pd.DataFrame):
        return len(metadata.index)
    try:
        return len(sub_dataset)
    except TypeError:  # pragma: no cover - defensive
        return 0


def _from_concat_dataset(
    dataset: DatasetLike,
    target: Optional[str],
) -> pd.DataFrame:
    """Build a one-row-per-sample metadata frame from a concat-style dataset."""
    rows: list[dict[str, Any]] = []
    for ds_index, sub_dataset in enumerate(_iter_subdatasets(dataset)):
        description = getattr(sub_dataset, "description", None)
        base = _description_to_row(description, ds_index)
        n_samples = _samples_in_subdataset(sub_dataset)
        if n_samples <= 0:
            continue
        # Stable per-record key used to build sample_id.
        record_key = "{ds}__{subj}__{sess}__{run}__{idx}".format(
            ds=_coerce_str(base.get("dataset"), "ds"),
            subj=_coerce_str(base.get("subject"), "subj"),
            sess=_coerce_str(base.get("session"), "sess"),
            run=_coerce_str(base.get("run"), "run"),
            idx=ds_index,
        )
        for sample_index in range(n_samples):
            row = dict(base)
            row["dataset_index"] = ds_index
            row["sample_id"] = f"{record_key}__w{sample_index:06d}"
            row["window_index"] = sample_index
            if target is not None:
                row["target"] = _resolve_target_value(sub_dataset, sample_index, target)
            rows.append(row)
    return pd.DataFrame(rows)


def _coerce_metadata_frame(frame: pd.DataFrame, target: Optional[str]) -> pd.DataFrame:
    """Validate and fill in defaults for a user-provided metadata DataFrame."""
    out = frame.copy()
    for column in ("subject", "session", "run", "dataset"):
        if column not in out.columns:
            out[column] = None
    if "sample_id" not in out.columns:
        # Derive deterministic sample ids from row position + key columns.
        out = out.reset_index(drop=True)
        out["sample_id"] = [
            "{ds}__{subj}__{sess}__{run}__r{i:06d}".format(
                ds=_coerce_str(row.get("dataset"), "ds"),
                subj=_coerce_str(row.get("subject"), "subj"),
                sess=_coerce_str(row.get("session"), "sess"),
                run=_coerce_str(row.get("run"), "run"),
                i=i,
            )
            for i, row in out.iterrows()
        ]
    if target is not None and target not in out.columns:
        out[target] = None
    return out


def to_split_metadata(
    dataset: DatasetLike, target: Optional[str] = None
) -> pd.DataFrame:
    """Normalize a dataset into a one-row-per-sample metadata DataFrame.

    Parameters
    ----------
    dataset
        Any of: a Braindecode ``BaseConcatDataset`` (including
        ``EEGDashDataset``), a ``FeaturesConcatDataset``, a Braindecode windows
        dataset, or a pre-existing ``pandas.DataFrame``. When a DataFrame is
        passed in the columns are validated and missing identifier columns are
        filled with ``None``.
    target
        Optional column name to materialise as ``metadata['target']``. The
        target is looked up first in the per-sample ``metadata`` table, then in
        the description (constant target per record), then on ``ds.y``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with at least the columns ``subject``, ``session``, ``run``,
        ``dataset``, ``sample_id``. When ``target`` is provided, an additional
        ``target`` column is added (renamed from ``target`` if provided).

    """
    if isinstance(dataset, pd.DataFrame):
        return _coerce_metadata_frame(dataset, target)

    # ``FeaturesConcatDataset``/``BaseConcatDataset`` both expose a ``datasets``
    # attribute. A bare ``FeaturesDataset`` or ``BaseDataset`` is wrapped in a
    # one-element list to reuse the same code path.
    if hasattr(dataset, "datasets"):
        return _from_concat_dataset(dataset, target)

    # Fall back to wrapping a single dataset.
    return _from_concat_dataset(_SingleDatasetShim(dataset), target)


class _SingleDatasetShim:
    """Wrap a single dataset to expose a ``datasets`` attribute."""

    def __init__(self, dataset: Any):
        self.datasets = [dataset]


def to_moabb_split_inputs(
    dataset: DatasetLike,
    target: Optional[str] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return ``(y, metadata)`` aligned to the order MOABB splitters expect.

    MOABB splitters consume ``y`` (NumPy array of labels) plus a metadata
    DataFrame with ``subject``, ``session``, ``run`` columns. EEGDash adds
    ``dataset`` and ``sample_id`` columns; MOABB ignores extra columns.

    Parameters
    ----------
    dataset
        Same set of accepted inputs as :func:`to_split_metadata`.
    target
        Column to use as the per-sample target. If ``None`` we emit a zero
        vector so MOABB stratified splitters do not crash.

    Returns
    -------
    (numpy.ndarray, pandas.DataFrame)
        ``y`` of shape ``(n_samples,)`` and the metadata DataFrame.

    """
    metadata = to_split_metadata(dataset, target=target)
    if target is not None and target in metadata.columns:
        y = metadata[target].to_numpy()
    elif "target" in metadata.columns:
        y = metadata["target"].to_numpy()
    else:
        y = np.zeros(len(metadata.index), dtype=int)
    return y, metadata


__all__ = [
    "to_moabb_split_inputs",
    "to_split_metadata",
]
