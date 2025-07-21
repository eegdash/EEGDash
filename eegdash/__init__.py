from .api import EEGDash, EEGDashDataset
from .utils import __init__mongo_client

__init__mongo_client()
__all__ = ["EEGDash", "EEGDashDataset"]
__version__ = "0.1.0"
