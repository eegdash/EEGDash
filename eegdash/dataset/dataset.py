from pathlib import Path
from typing import Any

from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from joblib import Parallel, delayed
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from braindecode.datasets import BaseConcatDataset

from .. import downloader
from ..bids_metadata import (
    build_query_from_kwargs,
    get_entities_from_record,
    merge_participants_fields,
    normalize_key,
)
from ..const import (
    ALLOWED_QUERY_FIELDS,
    RELEASE_TO_OPENNEURO_DATASET_MAP,
    SUBJECT_MINI_RELEASE_MAP,
)
from ..local_bids import discover_local_bids_records
from ..logging import logger
from ..paths import get_default_cache_dir
from ..schemas import validate_record
from .base import EEGDashRaw
from .bids_dataset import EEGBIDSDataset
from .registry import register_openneuro_datasets


class EEGDashDataset(BaseConcatDataset, metaclass=NumpyDocstringInheritanceInitMeta):
    """Create a new EEGDashDataset from a given query or local BIDS dataset directory
    and dataset name. An EEGDashDataset is pooled collection of EEGDashBaseDataset
    instances (individual recordings) and is a subclass of braindecode's BaseConcatDataset.

    Examples
    --------
    Basic usage with dataset and subject filtering:

    >>> from eegdash import EEGDashDataset
    >>> dataset = EEGDashDataset(
    ...     cache_dir="./data",
    ...     dataset="ds002718",
    ...     subject="012"
    ... )
    >>> print(f"Number of recordings: {len(dataset)}")

    Filter by multiple subjects and specific task:

    >>> subjects = ["012", "013", "014"]
    >>> dataset = EEGDashDataset(
    ...     cache_dir="./data",
    ...     dataset="ds002718",
    ...     subject=subjects,
    ...     task="RestingState"
    ... )

    Load and inspect EEG data from recordings:

    >>> if len(dataset) > 0:
    ...     recording = dataset[0]
    ...     print(f"Sampling rate: {raw.info['sfreq']} Hz")
    ...     print(f"Number of channels: {len(raw.ch_names)}")
    ...     print(f"Duration: {raw.times[-1]:.1f} seconds")

    Advanced filtering with raw MongoDB queries:

    >>> from eegdash import EEGDashDataset
    >>> query = {
    ...     "dataset": "ds002718",
    ...     "subject": {"$in": ["012", "013"]},
    ...     "task": "RestingState"
    ... }
    >>> dataset = EEGDashDataset(cache_dir="./data", query=query)

    Working with dataset collections and braindecode integration:

    >>> # EEGDashDataset is a braindecode BaseConcatDataset
    >>> for i, recording in enumerate(dataset):
    ...     if i >= 2:  # limit output
    ...         break
    ...     print(f"Recording {i}: {recording.description}")

    Parameters
    ----------
    cache_dir : str | Path
        Directory where data are cached locally.
    query : dict | None
        Raw MongoDB query to filter records. If provided, it is merged with
        keyword filtering arguments (see ``**kwargs``) using logical AND.
        You must provide at least a ``dataset`` (either in ``query`` or
        as a keyword argument). Only fields in ``ALLOWED_QUERY_FIELDS`` are
        considered for filtering.
    dataset : str
        Dataset identifier (e.g., ``"ds002718"``). Required if ``query`` does
        not already specify a dataset.
    task : str | list[str]
        Task name(s) to filter by (e.g., ``"RestingState"``).
    subject : str | list[str]
        Subject identifier(s) to filter by (e.g., ``"NDARCA153NKE"``).
    session : str | list[str]
        Session identifier(s) to filter by (e.g., ``"1"``).
    run : str | list[str]
        Run identifier(s) to filter by (e.g., ``"1"``).
    description_fields : list[str]
        Fields to extract from each record and include in dataset descriptions
        (e.g., "subject", "session", "run", "task").
    s3_bucket : str | None
        Optional S3 bucket URI (e.g., "s3://mybucket") to use instead of the
        default OpenNeuro bucket when downloading data files.
    records : list[dict] | None
        Pre-fetched metadata records. If provided, the dataset is constructed
        directly from these records and no MongoDB query is performed.
    download : bool, default True
        If False, load from local BIDS files only. Local data are expected
        under ``cache_dir / dataset``; no DB or S3 access is attempted.
    n_jobs : int
        Number of parallel jobs to use where applicable (-1 uses all cores).
    eeg_dash_instance : EEGDash | None
        Optional existing EEGDash client to reuse for DB queries. If None,
        a new client is created on demand, not used in the case of no download.
    database : str | None
        Database name to use (e.g., "eegdash", "eegdash_staging"). If None,
        uses the default database.
    auth_token : str | None
        Authentication token for accessing protected databases. Required for
        staging or admin operations.
    **kwargs : dict
        Additional keyword arguments serving two purposes:

        - Filtering: any keys present in ``ALLOWED_QUERY_FIELDS`` are treated as
          query filters (e.g., ``dataset``, ``subject``, ``task``, ...).
        - Dataset options: remaining keys are forwarded to
          ``EEGDashRaw``.

    """

    def __init__(
        self,
        cache_dir: str | Path,
        query: dict[str, Any] = None,
        description_fields: list[str] | None = None,
        s3_bucket: str | None = None,
        records: list[dict] | None = None,
        download: bool = True,
        n_jobs: int = -1,
        eeg_dash_instance: Any = None,
        database: str | None = None,
        auth_token: str | None = None,
        **kwargs,
    ):
        # Parameters that don't need validation
        _suppress_comp_warning: bool = kwargs.pop("_suppress_comp_warning", False)
        self._dedupe_records: bool = kwargs.pop("_dedupe_records", False)
        self.s3_bucket = s3_bucket
        self.database = database
        self.auth_token = auth_token
        self.records = records
        self.download = download
        self.n_jobs = n_jobs
        self.eeg_dash_instance = eeg_dash_instance

        if description_fields is None:
            description_fields = [
                "subject",
                "session",
                "run",
                "task",
                "age",
                "gender",
                "sex",
            ]

        self.cache_dir = cache_dir
        if self.cache_dir == "" or self.cache_dir is None:
            self.cache_dir = get_default_cache_dir()
            logger.warning(
                f"Cache directory is empty, using the eegdash default path: {self.cache_dir}"
            )

        self.cache_dir = Path(self.cache_dir)

        if not self.cache_dir.exists():
            logger.warning(
                f"Cache directory does not exist, creating it: {self.cache_dir}"
            )
            self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Extract query filters from kwargs (validates field names)
        query_kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED_QUERY_FIELDS}
        if query_kwargs:
            # Validate early: this raises ValueError for unknown fields or empty values
            build_query_from_kwargs(**query_kwargs)

        # Separate query kwargs from BaseDataset constructor kwargs
        self.query = query or {}
        self.query.update(query_kwargs)
        base_dataset_kwargs = {
            k: v for k, v in kwargs.items() if k not in ALLOWED_QUERY_FIELDS
        }

        if "dataset" not in self.query:
            # If explicit records are provided, infer dataset from records
            if isinstance(records, list) and records and isinstance(records[0], dict):
                inferred = records[0].get("dataset")
                if inferred:
                    self.query["dataset"] = inferred
                else:
                    raise ValueError("You must provide a 'dataset' argument")
            else:
                raise ValueError("You must provide a 'dataset' argument")

        # Decide on a dataset subfolder name for cache isolation. If using
        # challenge/preprocessed buckets (e.g., BDF, mini subsets), append
        # informative suffixes to avoid overlapping with the original dataset.
        dataset_folder = self.query["dataset"]

        self.data_dir = self.cache_dir / dataset_folder

        if (
            not _suppress_comp_warning
            and self.query["dataset"] in RELEASE_TO_OPENNEURO_DATASET_MAP.values()
        ):
            message_text = Text.from_markup(
                "[italic]This notice is only for users who are participating in the [link=https://eeg2025.github.io/]EEG 2025 Competition[/link].[/italic]\n\n"
                "[bold]EEG 2025 Competition Data Notice![/bold]\n"
                "You are loading one of the datasets that is used in competition, but via `EEGDashDataset`.\n\n"
                "[bold red]IMPORTANT[/bold red]: \n"
                "If you download data from `EEGDashDataset`, it is [u]NOT[/u] identical to the official \n"
                "competition data, which is accessed via `EEGChallengeDataset`. "
                "The competition data has been downsampled and filtered.\n\n"
                "[bold]If you are participating in the competition, \nyou must use the `EEGChallengeDataset` object to ensure consistency.[/bold] \n\n"
                "If you are not participating in the competition, you can ignore this message."
            )
            warning_panel = Panel(
                message_text,
                title="[yellow]EEG 2025 Competition Data Notice[/yellow]",
                subtitle="[cyan]Source: EEGDashDataset[/cyan]",
                border_style="yellow",
            )

            try:
                Console().print(warning_panel)
            except Exception:
                logger.warning(str(message_text))

        if records is not None:
            self.records = self._normalize_records(records)

            datasets = [
                EEGDashRaw(
                    record,
                    self.cache_dir,
                    **base_dataset_kwargs,
                )
                for record in self.records
            ]
        elif not download:  # only assume local data is complete if not downloading
            if not self.data_dir.exists():
                raise ValueError(
                    f"Offline mode is enabled, but local data_dir {self.data_dir} does not exist."
                )
            records = self._find_local_bids_records(self.data_dir, self.query)
            self.records = records
            # Try to enrich from local participants.tsv to restore requested fields
            try:
                bids_ds = EEGBIDSDataset(
                    data_dir=str(self.data_dir), dataset=self.query["dataset"]
                )  # type: ignore[index]
            except Exception:
                bids_ds = None

            datasets = []
            for record in records:
                # Start with entity values from filename (supports v1 and v2 formats)
                desc: dict[str, Any] = get_entities_from_record(record)

                if bids_ds is not None:
                    try:
                        rel_from_dataset = Path(record["bidspath"]).relative_to(
                            record["dataset"]
                        )  # type: ignore[index]
                        local_file = (self.data_dir / rel_from_dataset).as_posix()
                        part_row = bids_ds.subject_participant_tsv(local_file)
                        desc = merge_participants_fields(
                            description=desc,
                            participants_row=part_row
                            if isinstance(part_row, dict)
                            else None,
                            description_fields=description_fields,
                        )
                    except Exception:
                        pass

                datasets.append(
                    EEGDashRaw(
                        record=record,
                        cache_dir=self.cache_dir,
                        description=desc,
                        **base_dataset_kwargs,
                    )
                )

        elif self.query:
            if self.eeg_dash_instance is None:
                # to avoid circular import
                from ..api import EEGDash

                # Pass database and auth_token if specified
                eegdash_kwargs = {}
                if self.database:
                    eegdash_kwargs["database"] = self.database
                if self.auth_token:
                    eegdash_kwargs["auth_token"] = self.auth_token
                self.eeg_dash_instance = EEGDash(**eegdash_kwargs)
            datasets = self._find_datasets(
                query=build_query_from_kwargs(**self.query),
                description_fields=description_fields,
                base_dataset_kwargs=base_dataset_kwargs,
            )
            # We only need filesystem if we need to access S3
            self.filesystem = downloader.get_s3_filesystem()

            # Provide helpful error message when no datasets are found
            if len(datasets) == 0:
                query_str = build_query_from_kwargs(**self.query)
                raise ValueError(
                    f"No datasets found matching the query: {query_str}\n"
                    f"This could mean:\n"
                    f"  1. The dataset '{self.query.get('dataset', 'unknown')}' does not exist in the database\n"
                    f"  2. The specified filters (task, subject, etc.) are too restrictive\n"
                    f"  3. There is a connection issue with the MongoDB database\n"
                    f"The data exists at: {self.data_dir}"
                )

        # Attempt to fetch dataset-level metadata (for global files like participants.tsv)
        self.dataset_doc = None
        if self.download and self.eeg_dash_instance:
            try:
                self.dataset_doc = self.eeg_dash_instance.get_dataset(
                    self.query.get("dataset")
                )
            except Exception:
                pass

        super().__init__(datasets, lazy=True)

    def _normalize_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply dataset-level record normalization before building datasets."""
        if self.s3_bucket:
            for record in records:
                storage = record.setdefault("storage", {})
                storage["base"] = self.s3_bucket
                storage["backend"] = "s3"

        if self._dedupe_records:
            seen: set[str] = set()
            deduped: list[dict[str, Any]] = []
            # Reverse to keep the newest records (those inserted last)
            for record in reversed(records):
                key = (
                    record.get("bids_relpath")
                    or record.get("bidspath")
                    or record.get("data_name")
                )
                if key is None:
                    deduped.append(record)
                    continue
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(record)
            # Reverse back to maintain original relative order of successful records
            return list(reversed(deduped))

        return records

    def download_all(self, n_jobs: int | None = None) -> None:
        """Download missing remote files in parallel.

        Parameters
        ----------
        n_jobs : int | None
            Number of parallel workers to use. If None, defaults to ``self.n_jobs``.

        """
        if self.download is False:
            return

        if n_jobs is None:
            n_jobs = self.n_jobs

        targets: list[EEGDashRaw] = []
        for ds in self.datasets:
            if getattr(ds, "_raw_uri", None) is None:
                continue
            if not ds.filecache.exists() or any(
                not path.exists() for path in getattr(ds, "_dep_paths", [])
            ):
                targets.append(ds)

        if not targets:
            return

        if n_jobs == 1:
            for ds in targets:
                ds._download_required_files()
            return

        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(EEGDashRaw._download_required_files)(ds) for ds in targets
        )

        # Download global dataset files (participants.tsv, etc.)
        self._download_dataset_files()

    def _download_dataset_files(self) -> None:
        """Download global dataset files defined in dataset metadata."""
        if not self.dataset_doc or not self.download:
            return

        storage = self.dataset_doc.get("storage") or {}
        base = storage.get("base")
        backend = storage.get("backend")

        if not base or backend not in ("s3", "https"):
            return

        # Prepare list of files to download
        keys_to_download = set()
        if raw_key := storage.get("raw_key"):
            keys_to_download.add(raw_key)

        # Extract task filter if present
        task_filter = self.query.get("task")
        allowed_tasks = set()
        if isinstance(task_filter, str):
            allowed_tasks.add(task_filter)
        elif isinstance(task_filter, (list, tuple)):
            allowed_tasks.update(str(t) for t in task_filter)

        for key in storage.get("dep_keys", []):
            if allowed_tasks and key.startswith("task-") and "_" in key:
                # e.g. task-RestingState_events.json -> task_name="RestingState"
                task_name = key.split("_", 1)[0][5:]
                if task_name not in allowed_tasks:
                    continue
            keys_to_download.add(key)

        if not keys_to_download:
            return

        filesystem = downloader.get_s3_filesystem()

        # Download files that don't exist locally
        files_to_download = []
        for key in keys_to_download:
            dest = self.data_dir / key
            if not dest.exists():
                files_to_download.append((f"{base}/{key}", dest))

        if files_to_download:
            downloader.download_files(
                files_to_download,
                filesystem=filesystem,
                skip_existing=True,
            )

    def _find_local_bids_records(
        self, dataset_root: Path, filters: dict[str, Any]
    ) -> list[dict]:
        """Discover local BIDS EEG files and build v2 records.

        Enumerates EEG recordings under ``dataset_root`` using
        ``mne_bids.find_matching_paths`` and applies entity filters to produce
        v2 records suitable for :class:`EEGDashBaseDataset`. No network access is
        performed, and files are not read.

        Parameters
        ----------
        dataset_root : Path
            Local dataset directory (e.g., ``/path/to/cache/ds005509``).
        filters : dict
            Query filters. Must include ``'dataset'`` and may include BIDS
            entities like ``'subject'``, ``'session'``, etc.

        Returns
        -------
        list of dict
            A list of v2 records, one for each matched EEG file.

        Notes
        -----
        Matching is performed via :func:`mne_bids.find_matching_paths` using
        datatypes/suffixes derived from the ``'modality'`` filter (default:
        ``'eeg'``). For offline/local mode, storage backend is set to ``'local'``
        and the storage base points to the local dataset root.

        """
        return discover_local_bids_records(dataset_root, filters)

    def _find_key_in_nested_dict(self, data: Any, target_key: str) -> Any:
        """Recursively search for a key in nested dicts/lists.

        Performs a case-insensitive and underscore/hyphen-agnostic search.

        Parameters
        ----------
        data : Any
            The nested data structure (dicts, lists) to search.
        target_key : str
            The key to search for.

        Returns
        -------
        Any
            The value of the first matching key, or None if not found.

        """
        norm_target = normalize_key(target_key)
        if isinstance(data, dict):
            for k, v in data.items():
                if normalize_key(k) == norm_target:
                    return v
                res = self._find_key_in_nested_dict(v, target_key)
                if res is not None:
                    return res
        elif isinstance(data, list):
            for item in data:
                res = self._find_key_in_nested_dict(item, target_key)
                if res is not None:
                    return res
        return None

    def _find_datasets(
        self,
        query: dict[str, Any] | None,
        description_fields: list[str],
        base_dataset_kwargs: dict,
    ) -> list[EEGDashRaw]:
        """Find and construct datasets from a MongoDB query.

        Queries the database, then creates a list of
        :class:`EEGDashRaw` objects from the results. Records from the
        database must be v2 format with storage.base explicitly set.

        Parameters
        ----------
        query : dict, optional
            The MongoDB query to execute.
        description_fields : list of str
            Fields to extract from each record for the dataset description.
        base_dataset_kwargs : dict
            Additional keyword arguments to pass to the
            :class:`EEGDashRaw` constructor.

        Returns
        -------
        list of EEGDashRaw
            A list of dataset objects matching the query.

        Raises
        ------
        ValueError
            If records from the database are not v2 format.

        """
        datasets: list[EEGDashRaw] = []
        self.records = self._normalize_records(self.eeg_dash_instance.find(query))

        for record in self.records:
            # Validate v2 format
            errors = validate_record(record)
            if errors:
                raise ValueError(
                    f"Record from database must be v2 format: {errors}. "
                    f"Record data_name: {record.get('data_name', 'unknown')}"
                )

            description: dict[str, Any] = {}
            # Requested fields first (normalized matching)
            for field in description_fields:
                value = self._find_key_in_nested_dict(record, field)
                if value is not None:
                    description[field] = value
            # Merge all participants.tsv columns generically
            part = self._find_key_in_nested_dict(record, "participant_tsv")
            if isinstance(part, dict):
                description = merge_participants_fields(
                    description=description,
                    participants_row=part,
                    description_fields=description_fields,
                )
            datasets.append(
                EEGDashRaw(
                    record,
                    cache_dir=self.cache_dir,
                    description=description,
                    **base_dataset_kwargs,
                )
            )
        return datasets

    # just to fix the docstring inheritance until we solved it in braindecode.
    def save(self, path, overwrite=False):
        """Save the dataset to disk.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        overwrite : bool, default False
            If True, overwrite existing file.

        Returns
        -------
        None

        """
        return super().save(path, overwrite=overwrite)


class EEGChallengeDataset(EEGDashDataset):
    """A dataset helper for the EEG 2025 Challenge.

    This class simplifies access to the EEG 2025 Challenge datasets. It is a
    specialized version of :class:`~eegdash.api.EEGDashDataset` that is
    pre-configured for the challenge's data releases. It automatically maps a
    release name (e.g., "R1") to the corresponding OpenNeuro dataset and handles
    the selection of subject subsets (e.g., "mini" release).

    Parameters
    ----------
    release : str
        The name of the challenge release to load. Must be one of the keys in
        :const:`~eegdash.const.RELEASE_TO_OPENNEURO_DATASET_MAP`
        (e.g., "R1", "R2", ..., "R11").
    cache_dir : str
        The local directory where the dataset will be downloaded and cached.
    mini : bool, default True
        If True, the dataset is restricted to the official "mini" subset of
        subjects for the specified release. If False, all subjects for the
        release are included.
    query : dict, optional
        An additional MongoDB-style query to apply as a filter. This query is
        combined with the release and subject filters using a logical AND.
        The query must not contain the ``dataset`` key, as this is determined
        by the ``release`` parameter.
    s3_bucket : str, optional
        The base S3 bucket URI where the challenge data is stored. Defaults to
        the official challenge bucket.
    **kwargs
        Additional keyword arguments that are passed directly to the
        :class:`~eegdash.api.EEGDashDataset` constructor.

    Raises
    ------
    ValueError
        If the specified ``release`` is unknown, or if the ``query`` argument
        contains a ``dataset`` key. Also raised if ``mini`` is True and a
        requested subject is not part of the official mini-release subset.

    See Also
    --------
    EEGDashDataset : The base class for creating datasets from queries.

    """

    def __init__(
        self,
        release: str,
        cache_dir: str,
        mini: bool = True,
        query: dict | None = None,
        s3_bucket: str | None = None,
        **kwargs,
    ):
        self.release = release
        self.mini = mini

        if release not in RELEASE_TO_OPENNEURO_DATASET_MAP:
            raise ValueError(
                f"Unknown release: {release}, expected one of {list(RELEASE_TO_OPENNEURO_DATASET_MAP.keys())}"
            )

        if query and "dataset" in query:
            raise ValueError(
                "Query using the parameters `dataset` with the class EEGChallengeDataset is not possible."
                "Please use the release argument instead, or the object EEGDashDataset instead."
            )

        dataset_id = f"EEG2025r{release[1:]}"

        if self.mini:
            # When using the mini release, restrict subjects to the predefined subset.
            # If the user specifies subject(s), ensure they all belong to the mini subset;
            # otherwise, default to the full mini subject list for this release.

            allowed_subjects = set(SUBJECT_MINI_RELEASE_MAP[release])

            # Normalize potential 'subjects' -> 'subject' for convenience
            if "subjects" in kwargs and "subject" not in kwargs:
                kwargs["subject"] = kwargs.pop("subjects")

            # Collect user-requested subjects from kwargs/query. We canonicalize
            # kwargs via build_query_from_kwargs to leverage existing validation,
            # and support Mongo-style {"$in": [...]} shapes from a raw query.
            requested_subjects: list[str] = []

            # From kwargs
            if "subject" in kwargs and kwargs["subject"] is not None:
                # Use the shared query builder to normalize scalars/lists
                built = build_query_from_kwargs(subject=kwargs["subject"])
                s_val = built.get("subject")
                if isinstance(s_val, dict) and "$in" in s_val:
                    requested_subjects.extend(list(s_val["$in"]))
                elif s_val is not None:
                    requested_subjects.append(s_val)  # type: ignore[arg-type]

            # From query (top-level only)
            if query and isinstance(query, dict) and "subject" in query:
                qval = query["subject"]
                if isinstance(qval, dict) and "$in" in qval:
                    requested_subjects.extend(list(qval["$in"]))
                elif isinstance(qval, (list, tuple, set)):
                    requested_subjects.extend(list(qval))
                elif qval is not None:
                    requested_subjects.append(qval)

            # Validate if any subjects were explicitly requested
            if requested_subjects:
                invalid = sorted(
                    {s for s in requested_subjects if s not in allowed_subjects}
                )
                if invalid:
                    raise ValueError(
                        "Some requested subject(s) are not part of the mini release for "
                        f"{release}: {invalid}. Allowed subjects: {sorted(allowed_subjects)}"
                    )
                # Do not override user selection; keep their (validated) subjects as-is.
            else:
                # No subject specified by the user: default to the full mini subset
                kwargs["subject"] = sorted(allowed_subjects)

            # Construct dataset ID for mini
            dataset_id = f"{dataset_id}mini"

        if s3_bucket is None:
            if self.mini:
                s3_bucket = f"s3://nmdatasets/NeurIPS25/{release}_mini_L100_bdf"
            else:
                s3_bucket = f"s3://nmdatasets/NeurIPS25/{release}_L100_bdf"

        message_text = Text.from_markup(
            "This object loads the HBN dataset that has been preprocessed for the EEG Challenge:\n"
            "  * Downsampled from 500Hz to 100Hz\n"
            "  * Bandpass filtered (0.5-50 Hz)\n\n"
            "For full preprocessing applied for competition details, see:\n"
            "  [link=https://github.com/eeg2025/downsample-datasets]https://github.com/eeg2025/downsample-datasets[/link]\n\n"
            "The HBN dataset have some preprocessing applied by the HBN team:\n"
            "  * Re-reference (Cz Channel)\n\n"
            "[bold red]IMPORTANT[/bold red]: The data accessed via `EEGChallengeDataset` is [u]NOT[/u] identical to what you get from [link=https://github.com/eegdash/EEGDash/blob/develop/eegdash/api.py]EEGDashDataset[/link] directly.\n"
            "If you are participating in the competition, always use `EEGChallengeDataset` to ensure consistency with the challenge data."
        )

        warning_panel = Panel(
            message_text,
            title="[yellow]EEG 2025 Competition Data Notice[/yellow]",
            subtitle="[cyan]Source: EEGChallengeDataset[/cyan]",
            border_style="yellow",
        )

        # Render the panel directly to the console so it displays in IPython/terminals
        try:
            Console().print(warning_panel)
        except Exception:
            warning_message = str(message_text)
            logger.warning(warning_message)

        super().__init__(
            dataset=dataset_id,
            query=query,
            cache_dir=cache_dir,
            s3_bucket=s3_bucket,
            _suppress_comp_warning=True,
            _dedupe_records=True,
            **kwargs,
        )


registered_classes = register_openneuro_datasets(
    summary_file=Path(__file__).with_name("dataset_summary.csv"),
    base_class=EEGDashDataset,
    namespace=globals(),
    from_api=True,
)


__all__ = ["EEGDashDataset", "EEGChallengeDataset"] + list(registered_classes.keys())
