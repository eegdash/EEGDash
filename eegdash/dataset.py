from .api import EEGDashDataset

class EEGChallengeDataset(EEGDashDataset):
    def __init__(
        self,
        release: str = "R5",
        cache_dir: str = ".eegdash_cache",
        s3_bucket: str | None = "s3://nmdatasets/NeurIPS25/R5_L100",
        **kwargs,
    ):
        """Create a new EEGDashDataset from a given query or local BIDS dataset directory
        and dataset name. An EEGDashDataset is pooled collection of EEGDashBaseDataset
        instances (individual recordings) and is a subclass of braindecode's BaseConcatDataset.

        Parameters
        ----------
        query : dict | None
            Optionally a dictionary that specifies the query to be executed; see
            EEGDash.find() for details on the query format.
        data_dir : str | list[str] | None
            Optionally a string or a list of strings specifying one or more local
            BIDS dataset directories from which to load the EEG data files. Exactly one
            of query or data_dir must be provided.
        dataset : str | list[str] | None
            If data_dir is given, a name or list of names for for the dataset(s) to be loaded.
        description_fields : list[str]
            A list of fields to be extracted from the dataset records
            and included in the returned data description(s). Examples are typical
            subject metadata fields such as "subject", "session", "run", "task", etc.;
            see also data_config.description_fields for the default set of fields.
        cache_dir : str
            A directory where the dataset will be cached locally.
        s3_bucket : str | None
            An optional S3 bucket URI to use instead of the
            default OpenNeuro bucket for loading data files.
        kwargs : dict
            Additional keyword arguments to be passed to the EEGDashBaseDataset
            constructor.

        """
        dsnumber_release_map = {
            "R11": "ds005516",
            "R10": "ds005515",
            "R9": "ds005514",
            "R8": "ds005512",
            "R7": "ds005511",
            "R6": "ds005510",
            "R4": "ds005508",
            "R3": "ds005507",
            "R2": "ds005506",
            "R1": "ds005505",
        }
        super().__init__(
            query={"dataset": dsnumber_release_map[release]},
            cache_dir=cache_dir,
            s3_bucket=s3_bucket,
            **kwargs,
        )
