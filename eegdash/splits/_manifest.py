# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Build and consume cross-validation split manifests.

A split manifest is a small, JSON-serializable dictionary that captures
exactly enough provenance to reproduce a split: which splitter class was used
with which kwargs, what random seed, what target definition, what library
versions, and the actual ``train``/``test`` index lists per fold.

The manifest is produced once (by :func:`make_split_manifest`), serialized to
disk for benchmark submissions, and then consumed at training time by
:func:`apply_split_manifest`.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib
import json
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION: str = "1.0.0"


def _hash_metadata(metadata: pd.DataFrame) -> str:
    """Return a stable sha256 of the metadata frame's string representation.

    We only hash the columns that are leakage-relevant; that keeps the manifest
    invariant when the user enriches the metadata with cosmetic columns later.
    """
    columns = [
        c
        for c in ("subject", "session", "run", "dataset", "sample_id")
        if c in metadata.columns
    ]
    if not columns:  # pragma: no cover - defensive
        return hashlib.sha256(b"empty").hexdigest()
    canonical = (
        metadata[columns]
        .astype(str)
        .reset_index(drop=True)
        .to_csv(index=False, lineterminator="\n")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _library_versions() -> dict[str, str]:
    """Snapshot of versions for tools that materially affect a split."""
    out: dict[str, str] = {}
    for module_name in ("eegdash", "moabb", "sklearn", "numpy", "pandas"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        if version is not None:
            out[module_name] = str(version)
    return out


def _splitter_class_name(splitter: Any) -> str:
    """Return ``module.ClassName`` for a splitter instance."""
    cls = type(splitter)
    return f"{cls.__module__}.{cls.__qualname__}"


def _splitter_kwargs(splitter: Any) -> dict[str, Any]:
    """Best-effort extraction of constructor kwargs for provenance."""
    if hasattr(splitter, "splitter_kwargs"):
        return dict(splitter.splitter_kwargs)
    keys = (
        "n_folds",
        "n_splits",
        "shuffle",
        "random_state",
        "test_size",
        "n_perms",
        "data_size",
        "group_column",
        "stratified",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if hasattr(splitter, key):
            value = getattr(splitter, key)
            if not callable(value):
                # JSON cannot serialize np.int64/np.bool_; coerce.
                if isinstance(value, np.generic):
                    value = value.item()
                out[key] = value
    return out


def _resolve_random_seed(splitter: Any) -> Optional[int]:
    seed = getattr(splitter, "random_state", None)
    if seed is None:
        return None
    if isinstance(seed, np.generic):
        return int(seed.item())
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    return None


def make_split_manifest(
    splitter: Any,
    y: np.ndarray,
    metadata: pd.DataFrame,
    sample_ids: Optional[Sequence[str]] = None,
    target: Optional[str] = None,
) -> dict[str, Any]:
    """Run the splitter and return a JSON-serializable manifest.

    Parameters
    ----------
    splitter
        Object returned by :func:`get_splitter` (or any object exposing a
        ``split(y, metadata)`` method).
    y
        Per-sample labels (or zeros when ``target`` is ``None``).
    metadata
        DataFrame produced by :func:`to_split_metadata`.
    sample_ids
        Optional explicit sample identifier list. When omitted the function
        looks for a ``sample_id`` column on ``metadata``; otherwise integer row
        indices are used.
    target
        Name of the target column (recorded for provenance only).

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

    fold_records: list[dict[str, list[str]]] = []
    for train_idx, test_idx in splitter.split(np.asarray(y), metadata):
        train_idx_arr = np.asarray(list(train_idx), dtype=int)
        test_idx_arr = np.asarray(list(test_idx), dtype=int)
        fold_records.append(
            {
                "train": [sample_ids[i] for i in train_idx_arr.tolist()],
                "test": [sample_ids[i] for i in test_idx_arr.tolist()],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "splitter_class": _splitter_class_name(splitter),
        "splitter_kwargs": _splitter_kwargs(splitter),
        "random_seed": _resolve_random_seed(splitter),
        "n_folds": len(fold_records),
        "target": target,
        "metadata_hash": _hash_metadata(metadata),
        "folds": fold_records,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "library_versions": _library_versions(),
    }


def apply_split_manifest(
    dataset: Any,
    manifest: dict[str, Any],
    fold: int = 0,
    split: str = "train",
) -> Any:
    """Materialize a fold/split as either a row mask or a sub-dataset.

    Parameters
    ----------
    dataset
        Either a metadata DataFrame (in which case a boolean mask is returned)
        or a Braindecode-style concat dataset whose ``description`` exposes the
        ``sample_id`` column. When ``dataset`` is a concat dataset that does
        *not* have a per-sample ``sample_id`` column, the function selects the
        sub-datasets that contain at least one matching sample and returns a
        new container of the same type.
    manifest
        Manifest as produced by :func:`make_split_manifest`.
    fold
        0-based fold index.
    split
        ``"train"`` or ``"test"``.

    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
    if not (0 <= fold < manifest.get("n_folds", 0)):
        raise IndexError(
            f"Fold {fold} out of range (n_folds={manifest.get('n_folds')})."
        )
    target_ids = set(manifest["folds"][fold][split])

    # 1. DataFrame: return boolean mask aligned to row order.
    if isinstance(dataset, pd.DataFrame):
        if "sample_id" not in dataset.columns:
            raise ValueError(
                "DataFrame must have a 'sample_id' column to apply manifest."
            )
        return dataset["sample_id"].isin(target_ids).to_numpy()

    # 2. Concat dataset: pick the sub-datasets whose sample_ids overlap.
    if hasattr(dataset, "datasets"):
        from ._metadata import to_split_metadata

        try:
            full_metadata = to_split_metadata(dataset, target=manifest.get("target"))
        except Exception:  # pragma: no cover - defensive
            full_metadata = None

        keep_indices: list[int] = []
        if full_metadata is not None and "sample_id" in full_metadata.columns:
            mask = full_metadata["sample_id"].isin(target_ids)
            for ds_index in full_metadata.loc[mask, "dataset_index"].unique():
                keep_indices.append(int(ds_index))
        if not keep_indices:
            raise ValueError(
                "No sub-dataset of the concat dataset matched the manifest's "
                "sample IDs. Did you regenerate the dataset without keeping "
                "the same identifiers?"
            )
        return _select_subdatasets(dataset, sorted(set(keep_indices)))

    raise TypeError(
        f"Unsupported dataset type: {type(dataset).__name__}. "
        "Pass a metadata DataFrame or a concat dataset."
    )


def _select_subdatasets(dataset: Any, indices: list[int]) -> Any:
    """Return a concat dataset of the same class containing only ``indices``."""
    cls = type(dataset)
    sub = [dataset.datasets[i] for i in indices]
    try:
        return cls(sub)
    except TypeError:
        # Some classes use ``list_of_ds=`` keyword; try that.
        try:
            return cls(list_of_ds=sub)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"Cannot rebuild a {cls.__name__} from a sub-list of datasets."
            ) from exc


def manifest_to_json(manifest: dict[str, Any]) -> str:
    """Render the manifest as a stable JSON string (sorted keys)."""
    return json.dumps(manifest, sort_keys=True, default=str)


__all__ = [
    "SCHEMA_VERSION",
    "apply_split_manifest",
    "make_split_manifest",
    "manifest_to_json",
]
