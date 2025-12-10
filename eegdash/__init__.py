# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""EEGDash: A comprehensive platform for EEG data management and analysis.

EEGDash provides a unified interface for accessing, querying, and analyzing large-scale
EEG datasets. It integrates with cloud storage and REST APIs to streamline EEG research
workflows.
"""

from .api import EEGDash
from .dataset import EEGChallengeDataset, EEGDashDataset
from .hbn import preprocessing

__all__ = [
    "EEGChallengeDataset",
    "EEGDash",
    "EEGDashDataset",
    "preprocessing",
]

__version__ = "0.5.0"
