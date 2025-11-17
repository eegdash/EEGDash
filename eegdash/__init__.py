# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""EEGDash: A comprehensive platform for EEG data management and analysis.

EEGDash provides a unified interface for accessing, querying, and analyzing large-scale
EEG datasets. It integrates with cloud storage, MongoDB databases, and machine learning
frameworks to streamline EEG research workflows.
"""

from .api import EEGDash
from .dataset import EEGChallengeDataset, EEGDashDataset
from .hbn import preprocessing
from .utils import _init_mongo_client

_init_mongo_client()

__all__ = ["EEGDash", "EEGDashDataset", "EEGChallengeDataset", "preprocessing"]

__version__ = "0.4.1"
