from pathlib import Path

from .api import EEGDashDataset
from .registry import register_openneuro_datasets

RELEASE_TO_OPENNEURO_DATASET_MAP = {
    "R11": "ds005516",
    "R10": "ds005515",
    "R9": "ds005514",
    "R8": "ds005512",
    "R7": "ds005511",
    "R6": "ds005510",
    "R4": "ds005508",
    "R5": "ds005509",
    "R3": "ds005507",
    "R2": "ds005506",
    "R1": "ds005505",
}


class EEGChallengeDataset(EEGDashDataset):
    def __init__(
        self,
        release: str | list[str],
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
        if release not in RELEASE_TO_OPENNEURO_DATASET_MAP:
            raise ValueError(f"Unknown release: {release}")

        dataset_parameters = []
        if isinstance(release, str):
            dataset_parameters.append(RELEASE_TO_OPENNEURO_DATASET_MAP[release])
        else:
            raise ValueError(
                f"Unknown release type: {type(release)}, the expected type is str."
            )

        if "dataset" in query:
            raise ValueError(
                "Query using the parameters `dataset` with the class EEGChallengeDataset is not possible"
                "Please use the release argument instead, or the object EEGDashDataset instead."
            )

        if mini:
            s3_bucket = f"{s3_bucket}/R{release}_mini_L100_bdf"
        else:
            s3_bucket = f"{s3_bucket}/R{release}_L100_bdf"

        super().__init__(
            dataset=dataset_parameters,
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
