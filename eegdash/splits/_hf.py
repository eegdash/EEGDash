# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""HuggingFace-style split helpers on top of MOABB.

The two entry points mirror :class:`datasets.Dataset` semantics:

- :func:`train_test_split` returns a dict ``{"train": ..., "test": ...}``.
- :func:`k_fold` yields ``(train, test)`` pairs across N folds.

Both keep groups disjoint (subject / session / dataset) by deferring to
MOABB's evaluation splitters.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ._metadata import to_moabb_split_inputs
from ._splitters import get_splitter

_GROUP_TO_SPLITTER: dict[str, str] = {
    "subject": "cross_subject",
    "session": "cross_session",
    "dataset": "cross_dataset",
}


def _select(dataset: Any, indices: np.ndarray) -> Any:
    """Return a subset of ``dataset`` at the given integer ``indices``.

    Tries the HuggingFace and braindecode shapes in order: ``.select`` (HF
    Dataset), ``.split`` with a fold dict (braindecode ConcatDataset),
    ``.iloc`` (pandas DataFrame), then a plain list comprehension.
    """
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
    """Split ``dataset`` into ``{"train": ..., "test": ...}`` subsets.

    Mirrors :meth:`datasets.Dataset.train_test_split`. Groups named by
    ``group`` (``"subject"`` / ``"session"`` / ``"dataset"``) stay disjoint
    across the two subsets.

    Parameters
    ----------
    dataset
        EEGDash / braindecode / HuggingFace dataset.
    test_size
        Fraction of *groups* held out. Forwarded to MOABB as
        ``test_size`` when supported. Default delegates to MOABB.
    group
        Metadata column to keep disjoint.
    target
        Metadata column used to derive ``y``. Pass ``None`` to skip.
    seed
        Random seed.
    **splitter_kwargs
        Passed through to :func:`~eegdash.splits.get_splitter`.

    """
    y, metadata = to_moabb_split_inputs(dataset, target=target)
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
    """Yield ``(train, test)`` pairs across ``n_folds`` group-aware folds.

    Mirrors :class:`sklearn.model_selection.KFold` iteration with
    HuggingFace-style dataset views. Groups named by ``group`` stay
    disjoint between train and test in every fold.
    """
    y, metadata = to_moabb_split_inputs(dataset, target=target)
    splitter_kwargs.setdefault("n_folds", n_folds)
    splitter_kwargs.setdefault("random_state", seed)
    splitter = _resolve_splitter(group, **splitter_kwargs)
    for train_idx, test_idx in splitter.split(y, metadata):
        yield (
            _select(dataset, np.asarray(train_idx)),
            _select(dataset, np.asarray(test_idx)),
        )


__all__ = ["k_fold", "train_test_split"]
