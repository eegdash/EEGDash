# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Friendly-named MOABB splitter factory."""

from __future__ import annotations

from typing import Any

import moabb.evaluations.splitters as moabb_splitters
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Friendly name -> MOABB class name in ``moabb.evaluations.splitters``.
_MOABB_NAME_MAP: dict[str, str] = {
    "cross_subject": "CrossSubjectSplitter",
    "cross_session": "CrossSessionSplitter",
    "within_session": "WithinSessionSplitter",
    "within_subject": "WithinSubjectSplitter",
    "cross_dataset": "CrossDatasetSplitter",
    "learning_curve": "LearningCurveSplitter",
}

# Splitters where MOABB defaults to ``LeaveOneGroupOut``. Swap to a
# parametrisable sklearn cv_class when the caller asks for a fold count or
# a test-size ratio.
_GROUP_BASED = {"cross_subject", "cross_session", "cross_dataset"}


def get_splitter(name: str, **kwargs: Any):
    """Return a MOABB splitter instance for the friendly ``name``.

    ``name`` is one of ``cross_subject``, ``cross_session``,
    ``within_subject``, ``within_session``, ``cross_dataset``,
    ``learning_curve``. ``kwargs`` flow through to the underlying MOABB
    class (``n_folds`` / ``random_state`` / ``shuffle`` / ``data_size`` /
    ``n_perms`` / ``test_size``). Returns an object with a
    ``split(y, metadata)`` method.

    For the group-based cross-validators (``cross_subject``,
    ``cross_session``, ``cross_dataset``) the MOABB default is
    ``LeaveOneGroupOut``; passing ``n_splits``/``n_folds`` swaps to
    ``GroupKFold`` and passing ``test_size`` swaps to ``GroupShuffleSplit``.
    """
    if name not in _MOABB_NAME_MAP:
        raise ValueError(
            f"Unknown splitter '{name}'. Expected one of: "
            f"{sorted(_MOABB_NAME_MAP.keys())}."
        )
    cls = getattr(moabb_splitters, _MOABB_NAME_MAP[name])
    init_kwargs: dict[str, Any] = {}
    n_folds = kwargs.pop("n_folds", None)
    n_splits = kwargs.pop("n_splits", None)
    test_size = kwargs.pop("test_size", None)
    if name in ("within_session", "within_subject"):
        init_kwargs["n_folds"] = int(n_folds or n_splits or 5)
        init_kwargs["random_state"] = kwargs.pop("random_state", None)
        init_kwargs["shuffle"] = kwargs.pop("shuffle", True)
    elif name == "learning_curve":
        if "data_size" not in kwargs:
            raise ValueError("learning_curve requires a `data_size` mapping.")
        if "n_perms" not in kwargs:
            raise ValueError("learning_curve requires `n_perms`.")
    else:
        # cross_subject / cross_session / cross_dataset.
        random_state = kwargs.pop("random_state", None)
        if test_size is not None:
            init_kwargs["cv_class"] = GroupShuffleSplit
            init_kwargs["n_splits"] = int(n_splits or n_folds or 1)
            init_kwargs["test_size"] = float(test_size)
            init_kwargs["random_state"] = random_state
        elif n_splits or n_folds:
            init_kwargs["cv_class"] = GroupKFold
            init_kwargs["n_splits"] = int(n_splits or n_folds)
        else:
            init_kwargs["random_state"] = random_state
    init_kwargs.update(kwargs)
    return cls(**init_kwargs)


__all__ = ["get_splitter"]
