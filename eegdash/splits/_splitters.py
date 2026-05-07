# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Friendly-named splitter factory mapping to MOABB (or sklearn fallback)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# Friendly name -> MOABB class name in ``moabb.evaluations.splitters``.
_MOABB_NAME_MAP: dict[str, str] = {
    "cross_subject": "CrossSubjectSplitter",
    "cross_session": "CrossSessionSplitter",
    "within_session": "WithinSessionSplitter",
    "within_subject": "WithinSubjectSplitter",
    "cross_dataset": "CrossDatasetSplitter",
    "learning_curve": "LearningCurveSplitter",
}

# Group column the sklearn fallback uses to enforce no-leakage semantics.
_FALLBACK_GROUP_COLUMN: dict[str, str] = {
    "cross_subject": "subject",
    "cross_session": "session",
    "within_session": "session",
    "within_subject": "subject",
    "cross_dataset": "dataset",
    "learning_curve": "subject",
}


def _moabb_available() -> bool:
    try:
        import moabb.evaluations.splitters  # noqa: F401
    except Exception:
        return False
    return True


def _build_moabb_splitter(name: str, **kwargs: Any):
    import moabb.evaluations.splitters as moabb_splitters

    cls = getattr(moabb_splitters, _MOABB_NAME_MAP[name])
    init_kwargs: dict[str, Any] = {}
    n_folds = kwargs.pop("n_folds", None)
    n_splits = kwargs.pop("n_splits", None)
    if name in ("within_session", "within_subject"):
        init_kwargs["n_folds"] = int(n_folds or n_splits or 5)
        init_kwargs["random_state"] = kwargs.pop("random_state", None)
        init_kwargs["shuffle"] = kwargs.pop("shuffle", True)
    elif name == "learning_curve":
        if "data_size" not in kwargs:
            raise ValueError("learning_curve requires a `data_size` mapping.")
        if "n_perms" not in kwargs:
            raise ValueError("learning_curve requires `n_perms`.")
        init_kwargs.update(kwargs)
        kwargs = {}
    else:
        init_kwargs["random_state"] = kwargs.pop("random_state", None)
    init_kwargs.update(kwargs)
    return cls(**init_kwargs)


@dataclass
class _SklearnGroupSplitter:
    """sklearn-only fallback that mirrors MOABB's ``split(y, metadata)`` API."""

    name: str
    n_splits: int = 5
    random_state: Optional[int] = None
    stratified: bool = False
    group_column: str = "subject"
    shuffle: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    splitter_class: str = field(default="_SklearnGroupSplitter", init=False, repr=False)

    @property
    def splitter_kwargs(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "stratified": self.stratified,
            "group_column": self.group_column,
            "shuffle": self.shuffle,
        }

    def split(
        self, y: np.ndarray, metadata: pd.DataFrame
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

        if self.group_column not in metadata.columns:
            raise ValueError(
                f"sklearn fallback splitter for '{self.name}' requires "
                f"a metadata column '{self.group_column}'."
            )
        groups = metadata[self.group_column].to_numpy()
        if y is None or len(y) != len(groups):
            y = np.zeros(len(groups), dtype=int)
        cv = (
            StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state if self.shuffle else None,
            )
            if self.stratified
            else GroupKFold(n_splits=self.n_splits)
        )
        yield from cv.split(np.zeros((len(groups), 1)), y, groups=groups)


@dataclass
class _SklearnLearningCurveSplitter:
    """Minimal sklearn fallback for MOABB's ``LearningCurveSplitter``."""

    data_size: dict[str, Any]
    n_perms: int = 1
    test_size: float = 0.2
    random_state: Optional[int] = None

    splitter_class: str = field(
        default="_SklearnLearningCurveSplitter", init=False, repr=False
    )

    @property
    def splitter_kwargs(self) -> dict[str, Any]:
        return {
            "data_size": dict(self.data_size),
            "n_perms": self.n_perms,
            "test_size": self.test_size,
            "random_state": self.random_state,
        }

    def split(
        self, y: np.ndarray, metadata: pd.DataFrame
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        from sklearn.model_selection import GroupShuffleSplit

        if "subject" not in metadata.columns:
            raise ValueError("learning_curve fallback requires a 'subject' column.")
        groups = metadata["subject"].to_numpy()
        rng = np.random.default_rng(self.random_state)
        sizes = self.data_size.get("value", [1.0])
        if isinstance(sizes, (int, float)):
            sizes = [sizes]
        n_perms = (
            max(self.n_perms)
            if isinstance(self.n_perms, (list, tuple))
            else int(self.n_perms)
        )
        for fraction in sizes:
            for _ in range(n_perms):
                seed = int(rng.integers(0, 2**31 - 1))
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=self.test_size, random_state=seed
                )
                train_idx, test_idx = next(
                    gss.split(np.zeros((len(groups), 1)), y, groups=groups)
                )
                train_subjects = np.unique(groups[train_idx])
                n_keep = max(1, int(np.ceil(len(train_subjects) * float(fraction))))
                kept = rng.choice(train_subjects, size=n_keep, replace=False)
                mask = np.isin(groups[train_idx], kept)
                yield train_idx[mask], test_idx


def get_splitter(name: str, engine: str = "moabb", **kwargs: Any):
    """Return a splitter instance for the friendly ``name``.

    ``name`` is one of ``cross_subject``, ``cross_session``, ``within_subject``,
    ``within_session``, ``cross_dataset``, ``learning_curve``. ``engine`` is
    ``"moabb"`` (default; falls back to sklearn when MOABB is missing) or
    ``"sklearn"``. Returns an object with a ``split(y, metadata)`` method.
    """
    if name not in _MOABB_NAME_MAP:
        raise ValueError(
            f"Unknown splitter '{name}'. Expected one of: "
            f"{sorted(_MOABB_NAME_MAP.keys())}."
        )
    if engine not in ("moabb", "sklearn"):
        raise ValueError(f"Unknown engine '{engine}'. Expected 'moabb' or 'sklearn'.")

    if engine == "moabb" and _moabb_available():
        try:
            return _build_moabb_splitter(name, **kwargs)
        except (ImportError, AttributeError):  # pragma: no cover - defensive
            pass  # fall through to sklearn

    if name == "learning_curve":
        return _SklearnLearningCurveSplitter(
            data_size=kwargs.get("data_size", {"policy": "ratio", "value": [0.5, 1.0]}),
            n_perms=kwargs.get("n_perms", 1),
            test_size=kwargs.get("test_size", 0.2),
            random_state=kwargs.get("random_state", None),
        )

    skip = {"n_splits", "n_folds", "random_state", "stratified", "shuffle"}
    return _SklearnGroupSplitter(
        name=name,
        n_splits=int(kwargs.get("n_splits", kwargs.get("n_folds", 5))),
        random_state=kwargs.get("random_state", None),
        stratified=bool(kwargs.get("stratified", False)),
        group_column=_FALLBACK_GROUP_COLUMN[name],
        shuffle=bool(kwargs.get("shuffle", False)),
        extra_kwargs={k: v for k, v in kwargs.items() if k not in skip},
    )


__all__ = ["get_splitter"]
