from .api import EEGDash, EEGDashDataset
from .utils import __init__mongo_client
from .dataset import EEGChallengeDataset

__init__mongo_client()

__all__ = ["EEGDash", "EEGDashDataset", "EEGChallengeDataset"]
__version__ = "0.1.0"
