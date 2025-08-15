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
        release: str,
        cache_dir: str,
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

        dataset = RELEASE_TO_OPENNEURO_DATASET_MAP[release]
        if query is None:
            query = {"dataset": dataset}
        elif "dataset" not in query:
            query["dataset"] = dataset
        elif query["dataset"] != dataset:
            raise ValueError(
                f"Query dataset {query['dataset']} does not match the release {release} "
                f"which corresponds to dataset {dataset}."
            )

        super().__init__(
            query=query,
            cache_dir=cache_dir,
            s3_bucket=f"{s3_bucket}/{release}_L100",
            **kwargs,
        )


# Now including all the openneuro datasets
# but first check if all the openneuro datasets are registered
# using the competition as parameters, but could any dataset....
if not all(name in globals() for name in RELEASE_TO_OPENNEURO_DATASET_MAP.values()):
    register_openneuro_datasets(
        summary_file=Path(__file__).with_name("dataset_summary.csv"),
        base_class=EEGDashDataset,
        namespace=globals(),
    )
