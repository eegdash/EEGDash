from .api import EEGDash, EEGDashDataset
from .dataset import EEGChallengeDataset
from .utils import __init__mongo_client

__init__mongo_client()

__all__ = ["EEGDash", "EEGDashDataset", "EEGChallengeDataset"]
__version__ = "0.2.0"
