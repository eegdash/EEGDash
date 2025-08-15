from pathlib import Path

from .api import EEGDash, EEGDashDataset
from .dataset import EEGChallengeDataset
from .registry import register_openneuro_datasets
from .utils import __init__mongo_client

__init__mongo_client()


registered_classes = register_openneuro_datasets(
    summary_file=Path(__file__).parent / "dataset_summary.csv",
    base_class=EEGDashDataset,
    namespace=globals(),
)


__all__ = ["EEGDash", "EEGDashDataset", "EEGChallengeDataset"] + list(
    registered_classes.keys()
)

__version__ = "0.3.3"
