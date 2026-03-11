# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""EEGDash: A comprehensive platform for EEG data management and analysis.

EEGDash provides a unified interface for accessing, querying, and analyzing large-scale
EEG datasets. It integrates with cloud storage and REST APIs to streamline EEG research
workflows.
"""

__version__ = "0.5.0"

# NOTE: Keep the top-level import lightweight to avoid importing heavy optional
# dependencies (e.g., braindecode/torch) when users only need small utilities.
# Public objects are exposed via lazy attribute access (PEP 562).

__all__ = [
    "EEGDash",
    "EEGDashDataset",
    "EEGChallengeDataset",
    "DataIntegrityError",
    "preprocessing",
    "schemas",
]


def __getattr__(name: str):
    if name == "EEGDash":
        from .api import EEGDash

        return EEGDash
    if name == "DataIntegrityError":
        from .dataset.exceptions import DataIntegrityError

        return DataIntegrityError
    if name in {"EEGDashDataset", "EEGChallengeDataset"}:
        from .dataset import EEGChallengeDataset, EEGDashDataset

        return EEGDashDataset if name == "EEGDashDataset" else EEGChallengeDataset
    if name == "preprocessing":
        from .hbn import preprocessing

        return preprocessing
    if name == "schemas":
        from . import schemas

        return schemas
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
