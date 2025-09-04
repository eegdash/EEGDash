import logging
from pathlib import Path

from mne.utils import warn

from .api import EEGDashDataset
from .const import RELEASE_TO_OPENNEURO_DATASET_MAP, SUBJECT_MINI_RELEASE_MAP
from .registry import register_openneuro_datasets

logger = logging.getLogger("eegdash")


class EEGChallengeDataset(EEGDashDataset):
    def __init__(
        self,
        release: str,
        cache_dir: str,
        mini: bool = True,
        query: dict | None = None,
        s3_bucket: str | None = "s3://nmdatasets/NeurIPS25",
        **kwargs,
    ):
        """Create a new EEGDashDataset from a given query or local BIDS dataset directory
        and dataset name. An EEGDashDataset is pooled collection of EEGDashBaseDataset
        instances (individual recordings) and is a subclass of braindecode's BaseConcatDataset.

        Parameters
        ----------
        release: str
            Release name. Can be one of ["R1", ..., "R11"]
        mini: bool, default True
            Whether to use the mini-release version of the dataset. It is recommended
            to use the mini version for faster training and evaluation.
        query : dict | None
            Optionally a dictionary that specifies a query to be executed,
            in addition to the dataset (automatically inferred from the release argument).
            See EEGDash.find() for details on the query format.
        cache_dir : str
            A directory where the dataset will be cached locally.
        s3_bucket : str | None
            An optional S3 bucket URI to use instead of the
            default OpenNeuro bucket for loading data files.
        kwargs : dict
            Additional keyword arguments to be passed to the EEGDashDataset
            constructor.

        """
        self.release = release
        self.mini = mini

        if release not in RELEASE_TO_OPENNEURO_DATASET_MAP:
            raise ValueError(
                f"Unknown release: {release}, expected one of {list(RELEASE_TO_OPENNEURO_DATASET_MAP.keys())}"
            )

        dataset_parameters = []
        if isinstance(release, str):
            dataset_parameters.append(RELEASE_TO_OPENNEURO_DATASET_MAP[release])
        else:
            raise ValueError(
                f"Unknown release type: {type(release)}, the expected type is str."
            )

        if query and "dataset" in query:
            raise ValueError(
                "Query using the parameters `dataset` with the class EEGChallengeDataset is not possible."
                "Please use the release argument instead, or the object EEGDashDataset instead."
            )

        if self.mini:
            # Disallow mixing subject selection with mini=True since mini already
            # applies a predefined subject subset.
            if (query and "subject" in query) or ("subject" in kwargs):
                raise ValueError(
                    "Query using the parameters `subject` with the class EEGChallengeDataset and `mini==True` is not possible."
                    "Please don't use the `subject` selection twice."
                    "Set `mini=False` to use the `subject` selection."
                )
            kwargs["subject"] = SUBJECT_MINI_RELEASE_MAP[release]
            s3_bucket = f"{s3_bucket}/{release}_mini_L100_bdf"
        else:
            s3_bucket = f"{s3_bucket}/{release}_L100_bdf"

        warn(
            "\n\n"
            "[EEGChallengeDataset] EEG 2025 Competition Data Notice:\n"
            "-------------------------------------------------------\n"
            "This object loads the HBN dataset that has been preprocessed for the EEG Challenge:\n"
            "  - Downsampled from 500Hz to 100Hz\n"
            "  - Bandpass filtered (0.5–50 Hz)\n"
            "\n"
            "For full preprocessing details, see:\n"
            "  https://github.com/eeg2025/downsample-datasets\n"
            "\n"
            "IMPORTANT: The data accessed via `EEGChallengeDataset` is NOT identical to what you get from `EEGDashDataset` directly.\n"
            "If you are participating in the competition, always use `EEGChallengeDataset` to ensure consistency with the challenge data.\n"
            "\n",
            UserWarning,
            module="eegdash",
        )

        super().__init__(
            dataset=RELEASE_TO_OPENNEURO_DATASET_MAP[release],
            query=query,
            cache_dir=cache_dir,
            s3_bucket=s3_bucket,
            **kwargs,
        )


registered_classes = register_openneuro_datasets(
    summary_file=Path(__file__).with_name("dataset_summary.csv"),
    base_class=EEGDashDataset,
    namespace=globals(),
)


__all__ = ["EEGChallengeDataset"] + list(registered_classes.keys())
