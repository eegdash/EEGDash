import copy
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from joblib import Parallel, delayed
from mne_bids.config import reader
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
from ._source_inference import correct_storage_inplace
from .base import EEGDashRaw
from .bids_dataset import EEGBIDSDataset
from .exceptions import PreviewError
from .registry import register_openneuro_datasets

# Valid extensions for EEG data files (from MNE-BIDS reader configuration)
_VALID_DATA_EXTENSIONS = frozenset(reader.keys())


@dataclass
class RecordingPreview:
    """Compact view of a single recording produced by :meth:`EEGDashDataset.preview`.

    Attributes
    ----------
    raw : mne.io.BaseRaw
        Loaded continuous data for the previewed recording.
    metadata : dict
        Description metadata for the recording (entities, demographics, etc.).
    snippet : numpy.ndarray
        The first 5 seconds of data with shape ``(n_channels, n_samples)``.
        If the recording is shorter than 5 seconds, the full recording is
        returned.
    annotations : list[dict]
        Annotations extracted from ``raw.annotations`` as a list of dicts
        with keys ``onset``, ``duration``, and ``description``.
    record : dict
        The underlying record metadata used to materialize ``raw``.
    index : int
        Index of the previewed recording within the parent dataset.

    """

    raw: Any
    metadata: dict[str, Any]
    snippet: Any
    annotations: list[dict[str, Any]]
    record: dict[str, Any] = field(default_factory=dict)
    index: int = 0

    def plot(self, **kwargs: Any) -> Any:
        """Plot the previewed recording via :meth:`mne.io.Raw.plot`.

        Parameters
        ----------
        **kwargs
            Passed through to :meth:`mne.io.Raw.plot`. ``show=False`` is
            recommended in non-interactive contexts to receive the
            ``matplotlib`` figure rather than blocking on a viewer.

        Returns
        -------
        matplotlib.figure.Figure
            The figure produced by ``raw.plot()``.

        """
        return self.raw.plot(**kwargs)


def _format_counter_compact(counter: Counter) -> str:
    """Render a ``Counter`` as ``key1xN, key2xM`` ordered by descending count."""
    if not counter:
        return "-"
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{key}x{count}" for key, count in items)


def _record_field_with_fallback(record: dict[str, Any], name: str) -> Any:
    """Read ``record[name]`` falling back to ``entities``/``entities_mne``."""
    if name in record and record.get(name) is not None:
        return record[name]
    for src_key in ("entities", "entities_mne"):
        src = record.get(src_key)
        if isinstance(src, dict) and src.get(name) is not None:
            return src[name]
    return None


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
    ...     raw = recording.load()
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
    ...     raw = recording.load()
    ...     print(f"  Channels: {len(raw.ch_names)}, Duration: {raw.times[-1]:.1f}s")

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
    on_error : str, default "raise"
        How to handle :class:`DataIntegrityError` when accessing ``.raw``
        on individual recordings:

        - ``"raise"`` (default): propagate the exception.
        - ``"warn"``: log the error as a warning and set ``.raw`` to ``None``.
        - ``"skip"``: silently set ``.raw`` to ``None``.

        Use :meth:`drop_bad` after iteration to remove skipped recordings.
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
        on_error: str = "raise",
        **kwargs,
    ):
        # Parameters that don't need validation
        _suppress_comp_warning: bool = kwargs.pop("_suppress_comp_warning", False)
        self._dedupe_records: bool = kwargs.pop("_dedupe_records", False)
        self._on_error = on_error
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
        base_dataset_kwargs["on_error"] = self._on_error

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

    def drop_bad(self) -> list[dict]:
        """Remove skipped datasets and return their records.

        Call after accessing ``.raw`` on all datasets (e.g. after iteration
        or preprocessing) to clean up the dataset list.

        Returns
        -------
        list of dict
            Records that were removed because loading failed.

        """
        bad = []
        valid_datasets = []
        valid_records = []
        for ds, record in zip(self.datasets, self.records):
            if getattr(ds, "_skipped", False):
                bad.append(record)
            else:
                valid_datasets.append(ds)
                valid_records.append(record)
        self.datasets = valid_datasets
        self.records = valid_records
        return bad

    def drop_short(self, min_samples: int) -> list[dict]:
        """Remove recordings shorter than *min_samples* and return their records.

        This is useful when downstream processing (e.g., fixed-length
        windowing) requires a minimum number of samples per recording.
        Recordings whose ``.raw`` is ``None`` (failed to load) are also
        dropped.

        Parameters
        ----------
        min_samples : int
            Minimum number of time-domain samples a recording must have
            to be kept.

        Returns
        -------
        list of dict
            Records that were removed.

        """
        dropped = []
        valid_datasets = []
        valid_records = []
        for ds, record in zip(self.datasets, self.records):
            raw = ds.raw
            if raw is None or raw.n_times < min_samples:
                dropped.append(record)
                ds._raw = None
            else:
                valid_datasets.append(ds)
                valid_records.append(record)
        self.datasets = valid_datasets
        self.records = valid_records
        return dropped

    def summary(self, verbose: bool = False) -> dict[str, Any]:
        """Summarize the dataset's records without touching the network.

        Aggregates fields available on ``self.records`` (and the dataset's
        description metadata when present) into a compact dictionary
        suitable for first-contact exploration. Optionally prints a
        formatted text report.

        Parameters
        ----------
        verbose : bool, default False
            If True, print a human-readable report in addition to
            returning the dictionary.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``n_records`` (int): Number of recordings.
            - ``n_subjects`` (int): Distinct subject identifiers.
            - ``n_tasks`` (int): Distinct task identifiers.
            - ``n_sessions`` (int): Distinct session identifiers.
            - ``n_runs`` (int): Distinct run identifiers.
            - ``modalities`` (set[str]): Modality codes (e.g. ``{"eeg"}``).
            - ``channel_counts`` (collections.Counter): Channel count
              histogram across recordings.
            - ``sampling_rates`` (collections.Counter): Sampling rate
              histogram (in Hz) across recordings.
            - ``total_duration_seconds`` (float): Sum of recording
              durations in seconds.
            - ``cache_path`` (str): The dataset's local cache directory.
            - ``estimated_size_bytes`` (int | None): Sum of known byte
              sizes across records. ``None`` when no record exposes a
              size.

        Notes
        -----
        Empty datasets are handled gracefully — counters are empty and
        scalar counts are zero.

        """
        records: list[dict[str, Any]] = list(getattr(self, "records", []) or [])

        n_records = len(records)
        modalities: set[str] = set()
        subjects: set[str] = set()
        tasks: set[str] = set()
        sessions: set[str] = set()
        runs: set[str] = set()
        channel_counts: Counter[int] = Counter()
        sampling_rates: Counter[float] = Counter()
        total_duration = 0.0
        total_size_bytes = 0
        size_seen = False

        for record in records:
            entities = record.get("entities") or record.get("entities_mne") or {}

            subject = record.get("subject") or entities.get("subject")
            if subject is not None:
                subjects.add(str(subject))
            task = record.get("task") or entities.get("task")
            if task is not None:
                tasks.add(str(task))
            session = record.get("session") or entities.get("session")
            if session is not None:
                sessions.add(str(session))
            run = record.get("run") or entities.get("run")
            if run is not None:
                runs.add(str(run))

            modality = record.get("modality")
            if modality is None:
                rec_mod = record.get("recording_modality")
                if isinstance(rec_mod, list):
                    modalities.update(str(m) for m in rec_mod if m)
                    modality = None
                else:
                    modality = rec_mod
            if isinstance(modality, str) and modality:
                modalities.add(modality)
            if not modalities:
                # Fall back to BIDS datatype when no explicit modality is set.
                datatype = record.get("datatype")
                if isinstance(datatype, str) and datatype:
                    modalities.add(datatype)

            nchans = record.get("nchans")
            if nchans is None:
                ch_names = record.get("ch_names")
                if isinstance(ch_names, list):
                    nchans = len(ch_names)
            if isinstance(nchans, (int, float)) and nchans > 0:
                channel_counts[int(nchans)] += 1

            sfreq = record.get("sampling_frequency") or record.get("sfreq")
            if isinstance(sfreq, (int, float)) and sfreq > 0:
                sampling_rates[float(sfreq)] += 1

            duration = record.get("duration")
            ntimes = record.get("ntimes")
            if duration is None and isinstance(sfreq, (int, float)) and sfreq:
                if isinstance(ntimes, (int, float)) and ntimes > 0:
                    duration = float(ntimes) / float(sfreq)
            if isinstance(duration, (int, float)) and duration > 0:
                total_duration += float(duration)

            size_val = (
                record.get("size_bytes")
                or record.get("file_size_bytes")
                or (record.get("storage") or {}).get("size_bytes")
            )
            if isinstance(size_val, (int, float)) and size_val > 0:
                total_size_bytes += int(size_val)
                size_seen = True

        cache_path = str(getattr(self, "cache_dir", "") or "")
        estimated_size: int | None = total_size_bytes if size_seen else None

        result: dict[str, Any] = {
            "n_records": n_records,
            "n_subjects": len(subjects),
            "n_tasks": len(tasks),
            "n_sessions": len(sessions),
            "n_runs": len(runs),
            "modalities": modalities,
            "channel_counts": channel_counts,
            "sampling_rates": sampling_rates,
            "total_duration_seconds": float(total_duration),
            "cache_path": cache_path,
            "estimated_size_bytes": estimated_size,
        }

        if verbose:
            self._print_summary(result)

        return result

    @staticmethod
    def _print_summary(report: dict[str, Any]) -> None:
        """Render a summary dict as a human-readable text block."""
        modalities = sorted(report["modalities"]) or ["-"]

        size = report["estimated_size_bytes"]
        size_str = f"{size:,} bytes" if isinstance(size, int) else "unknown"

        lines = [
            "EEGDashDataset summary",
            "----------------------",
            f"records:       {report['n_records']}",
            f"subjects:      {report['n_subjects']}",
            f"tasks:         {report['n_tasks']}",
            f"sessions:      {report['n_sessions']}",
            f"runs:          {report['n_runs']}",
            f"modalities:    {', '.join(modalities)}",
            f"channels:      {_format_counter_compact(report['channel_counts'])}",
            f"sampling rate: {_format_counter_compact(report['sampling_rates'])}",
            f"duration (s):  {report['total_duration_seconds']:.1f}",
            f"cache path:    {report['cache_path'] or '-'}",
            f"size on disk:  {size_str}",
        ]
        print("\n".join(lines))

    def preview(self, index: int = 0) -> RecordingPreview:
        """Load one recording and return a compact preview object.

        Useful as a first inspection step before building a full
        preprocessing pipeline. Only the recording at ``index`` is
        materialized; the remaining recordings stay lazy.

        Parameters
        ----------
        index : int, default 0
            Position of the recording within the dataset.

        Returns
        -------
        RecordingPreview
            Dataclass exposing ``raw``, ``metadata``, ``snippet``,
            ``annotations``, and a ``plot()`` helper.

        Raises
        ------
        PreviewError
            If the dataset is empty, the index is out of range, or the
            recording fails to load. The original exception is chained
            via ``__cause__``.

        """
        if not self.datasets:
            raise PreviewError(
                "Cannot preview: dataset is empty (no recordings).",
                index=index,
            )
        if index < 0:
            index += len(self.datasets)
        if index < 0 or index >= len(self.datasets):
            raise PreviewError(
                f"Preview index {index} is out of range "
                f"for a dataset with {len(self.datasets)} recordings.",
                index=index,
            )

        ds = self.datasets[index]
        record = (
            self.records[index]
            if getattr(self, "records", None) and index < len(self.records)
            else getattr(ds, "record", {}) or {}
        )

        try:
            raw = ds.raw
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise PreviewError(
                f"Failed to load recording at index {index}: {exc}",
                index=index,
                record=record if isinstance(record, dict) else {},
            ) from exc

        if raw is None:
            raise PreviewError(
                f"Failed to load recording at index {index}: "
                f"raw is None (record may be flagged as integrity-bad).",
                index=index,
                record=record if isinstance(record, dict) else {},
            )

        # Build snippet of the first 5 seconds (or whole recording).
        try:
            sfreq = float(raw.info["sfreq"])
        except Exception:
            sfreq = 0.0
        n_samples = getattr(raw, "n_times", None) or len(raw)
        if sfreq > 0:
            stop = min(int(round(5.0 * sfreq)), int(n_samples))
        else:
            stop = int(n_samples)
        stop = max(stop, 0)
        try:
            snippet, _ = raw[:, :stop]
        except Exception as exc:  # pragma: no cover - defensive
            raise PreviewError(
                f"Failed to slice recording at index {index}: {exc}",
                index=index,
                record=record if isinstance(record, dict) else {},
            ) from exc

        annotations: list[dict[str, Any]] = []
        annots = getattr(raw, "annotations", None)
        if annots is not None and len(annots) > 0:
            for onset, duration, description in zip(
                annots.onset, annots.duration, annots.description
            ):
                annotations.append(
                    {
                        "onset": float(onset),
                        "duration": float(duration),
                        "description": str(description),
                    }
                )

        # Build metadata dict from the dataset description (per-row).
        metadata: dict[str, Any] = {}
        ds_desc = getattr(ds, "description", None)
        if isinstance(ds_desc, dict):
            metadata.update(ds_desc)
        else:
            try:
                metadata.update(dict(ds_desc))  # pandas.Series support
            except Exception:
                pass
        # Surface a few canonical record fields if missing.
        for key in ("dataset", "subject", "task", "session", "run"):
            if key not in metadata and isinstance(record, dict) and key in record:
                metadata[key] = record[key]

        return RecordingPreview(
            raw=raw,
            metadata=metadata,
            snippet=snippet,
            annotations=annotations,
            record=record if isinstance(record, dict) else {},
            index=index,
        )

    def filter(self, **kwargs: Any) -> "EEGDashDataset":
        """Return a new dataset whose records match all keyword filters.

        Performs an in-memory filter over ``self.records`` without
        touching the network. The same kwargs accepted by the constructor
        are supported (``subject``, ``session``, ``task``, ``run``,
        ``dataset``, ``modality``, ...). A list value matches with OR
        semantics; multiple kwargs combine with AND semantics.

        Parameters
        ----------
        **kwargs
            Field/value pairs limited to
            :data:`eegdash.const.ALLOWED_QUERY_FIELDS`. Each value may be
            a scalar or a list/tuple/set of acceptable values.

        Returns
        -------
        EEGDashDataset
            A new instance whose ``.datasets`` and ``.records`` are the
            subset matching all filters. ``self`` is not mutated.

        Raises
        ------
        ValueError
            If a kwarg references a field outside
            :data:`eegdash.const.ALLOWED_QUERY_FIELDS`.

        """
        if not kwargs:
            return self._clone_with(
                datasets=list(self.datasets),
                records=list(getattr(self, "records", []) or []),
            )

        unknown = set(kwargs) - set(ALLOWED_QUERY_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown filter field(s): {sorted(unknown)}. "
                f"Allowed fields are: {', '.join(sorted(ALLOWED_QUERY_FIELDS))}"
            )

        # Normalize each kwarg into a set of acceptable string values.
        normalized: dict[str, set[str]] = {}
        for field_name, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                values = {str(v) for v in value if v is not None}
            else:
                values = {str(value)}
            if values:
                normalized[field_name] = values

        records = list(getattr(self, "records", []) or [])
        kept_pairs: list[tuple[Any, dict[str, Any]]] = []
        for ds, record in zip(self.datasets, records):
            keep = True
            for field_name, accepted in normalized.items():
                rv = _record_field_with_fallback(record, field_name)
                if rv is None or str(rv) not in accepted:
                    keep = False
                    break
            if keep:
                kept_pairs.append((ds, record))

        kept_datasets = [ds for ds, _ in kept_pairs]
        kept_records = [rec for _, rec in kept_pairs]

        return self._clone_with(datasets=kept_datasets, records=kept_records)

    def _clone_with(
        self,
        datasets: list[Any],
        records: list[dict[str, Any]],
    ) -> "EEGDashDataset":
        """Construct a new ``EEGDashDataset`` view sharing this instance's config.

        Skips re-running the constructor (no DB or filesystem access) by
        instantiating via ``__new__`` and copying over the relevant
        attributes. Used by :meth:`filter`.
        """
        clone = self.__class__.__new__(self.__class__)
        # Copy lightweight scalar / metadata attributes.
        for attr in (
            "_dedupe_records",
            "_on_error",
            "s3_bucket",
            "database",
            "auth_token",
            "download",
            "n_jobs",
            "eeg_dash_instance",
            "cache_dir",
            "data_dir",
            "dataset_doc",
        ):
            if hasattr(self, attr):
                setattr(clone, attr, getattr(self, attr))
        # Avoid sharing mutable state by deep-copying the query dict.
        clone.query = copy.deepcopy(getattr(self, "query", {}) or {})
        # Filesystem handle is reused if present.
        if hasattr(self, "filesystem"):
            clone.filesystem = self.filesystem
        clone.records = list(records)

        if datasets:
            # Initialize the BaseConcatDataset machinery without re-running
            # our own __init__ (which would re-query / re-discover).
            BaseConcatDataset.__init__(clone, list(datasets), lazy=True)
        else:
            # BaseConcatDataset rejects empty lists. Set the minimal state
            # needed for downstream helpers (summary, len, repr) to work.
            clone.datasets = []
            clone.cumulative_sizes_cache = []
            clone.target_transform = None

        return clone

    @property
    def cumulative_sizes(self) -> list[int]:
        """Recompute cumulative sizes from current dataset lengths.

        Overrides the cached version from BaseConcatDataset because individual
        dataset lengths can change after lazy raw loading (estimated ntimes
        from JSON metadata may differ from actual n_times in the raw file).
        """
        from torch.utils.data import ConcatDataset

        return ConcatDataset.cumsum(self.datasets)

    @cumulative_sizes.setter
    def cumulative_sizes(self, value):
        # Accept writes from ConcatDataset.__init__ but discard; we always recompute.
        pass

    def _ensure_cumulative_sizes(self) -> list[int]:
        """Always recompute cumulative sizes.

        Overrides BaseConcatDataset's cached version (used by __len__) to
        stay consistent with the dynamic cumulative_sizes property.
        """
        return self.cumulative_sizes

    def _normalize_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply dataset-level record normalization before building datasets.

        This method performs several normalizations:
        1. Filters out records with invalid extensions (e.g., .json, .tsv sidecar files)
        2. Self-heals records whose ``storage.base`` is misrouted for the
           dataset_id pattern (residual fallout from the pre-PR-#327 NEMAR
           ingestion bug — ``nm*`` records pointing at ``s3://openneuro.org``).
        3. Updates storage backend/base if s3_bucket is specified
        4. Deduplicates records if _dedupe_records is enabled
        """
        # Filter out records that are not valid EEG data files
        # (e.g., filter out .json, .tsv sidecar files that may be in the database)
        filtered_records = []
        for record in records:
            ext = record.get("extension", "")
            # Check if extension is valid for EEG data (case-insensitive)
            if ext and ext.lower() not in {e.lower() for e in _VALID_DATA_EXTENSIONS}:
                continue
            filtered_records.append(record)

        records = filtered_records

        # Self-heal misrouted storage.base before any user override applies.
        # The pre-PR-#327 NEMAR mislabel has since been patched in the DB, so
        # this is now a defensive net for stale caches / third-party records.
        seen_corrections: set[tuple[str, str, str]] = set()
        for record in records:
            corrected, old_base = correct_storage_inplace(record)
            if corrected:
                key = (record.get("dataset", ""), old_base, record["storage"]["base"])
                if key not in seen_corrections:
                    seen_corrections.add(key)
                    logger.info(
                        "Auto-corrected misrouted storage.base for dataset %s: "
                        "%s -> %s",
                        record.get("dataset"),
                        old_base,
                        record["storage"]["base"],
                    )

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
            self._download_dataset_files()
            return

        if n_jobs == 1:
            for ds in targets:
                ds._download_required_files()
        else:
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(EEGDashRaw._download_required_files)(ds) for ds in targets
            )

        # Download global dataset files (participants.tsv, etc.)
        self._download_dataset_files()

    def estimate_download_size(self) -> dict[str, Any]:
        """Estimate the on-disk size required to materialize this dataset.

        Inspects ``self.records`` for any per-recording byte counts
        (``size_bytes``, ``file_size_bytes``, or ``storage.size_bytes``)
        and sums them. When no record exposes a size, falls back to the
        dataset-level ``size_bytes`` field surfaced by
        :meth:`EEGDash.find_datasets` (and cached on
        ``self.dataset_doc``). The method never touches the network or
        the filesystem.

        Returns
        -------
        dict
            A dictionary with keys:

            - ``bytes`` (int | None): Total estimated bytes; ``None``
              when no information is available.
            - ``n_records`` (int): Number of recordings considered.
            - ``source`` (str): One of ``"records"`` (per-record sizes
              were summed), ``"summary"`` (dataset-level fallback was
              used), or ``"unknown"`` (no size hint anywhere).

        Notes
        -----
        Returns ``{"bytes": None, "n_records": 0, "source": "unknown"}``
        for an empty dataset. The ``"summary"`` fallback is best-effort:
        when only a subset of records is selected (e.g. one subject out
        of many) the dataset-level total will overestimate the actual
        download size.

        """
        records: list[dict[str, Any]] = list(getattr(self, "records", []) or [])
        n_records = len(records)

        total_bytes = 0
        size_seen = False
        for record in records:
            size_val = (
                record.get("size_bytes")
                or record.get("file_size_bytes")
                or (record.get("storage") or {}).get("size_bytes")
            )
            if isinstance(size_val, (int, float)) and size_val > 0:
                total_bytes += int(size_val)
                size_seen = True

        if size_seen:
            return {
                "bytes": int(total_bytes),
                "n_records": n_records,
                "source": "records",
            }

        # Fall back to the dataset-level summary cached on dataset_doc
        # (populated by EEGDash.get_dataset(...) during __init__).
        dataset_doc = getattr(self, "dataset_doc", None) or {}
        summary_size = dataset_doc.get("size_bytes")
        if isinstance(summary_size, (int, float)) and summary_size > 0:
            return {
                "bytes": int(summary_size),
                "n_records": n_records,
                "source": "summary",
            }

        return {"bytes": None, "n_records": n_records, "source": "unknown"}

    def ensure_downloaded(
        self,
        n_jobs: int | None = None,
        progress: bool = False,
    ) -> dict[str, Any]:
        """Download any missing files and return a small summary.

        Discoverable wrapper around :meth:`download_all` for tutorials
        and docs. Identifies which recordings still need fetching, then
        delegates the actual transfer to ``download_all``. Optionally
        renders a ``tqdm`` progress bar while iterating recordings; if
        ``tqdm`` is not installed, falls back silently to a no-op
        iterator.

        Parameters
        ----------
        n_jobs : int | None, optional
            Number of parallel workers forwarded to
            :meth:`download_all`. ``None`` (the default) defers to
            ``self.n_jobs``.
        progress : bool, default False
            If True, render a ``tqdm`` progress bar while inspecting
            recordings. Requires ``tqdm`` to be installed; otherwise
            the bar is silently skipped.

        Returns
        -------
        dict
            A dictionary with keys:

            - ``n_records`` (int): Total number of recordings.
            - ``n_downloaded`` (int): Recordings whose local cache was
              missing before this call (i.e., now scheduled for
              download).
            - ``n_skipped`` (int): Recordings already present locally
              and therefore skipped.
            - ``bytes_total`` (int | None): Estimated total bytes from
              :meth:`estimate_download_size`; ``None`` when no size
              hints are available.

        Raises
        ------
        NotImplementedError
            If the underlying class no longer exposes
            :meth:`download_all`.

        Notes
        -----
        For an empty dataset, the call is a no-op and returns a summary
        with all counts set to zero.

        """
        download_fn = getattr(self, "download_all", None)
        if not callable(download_fn):
            raise NotImplementedError(
                "ensure_downloaded() requires EEGDashDataset.download_all(); "
                "the current class does not provide it."
            )

        datasets = list(getattr(self, "datasets", []) or [])
        n_records = len(datasets)

        # Optional progress bar — tqdm is intentionally a soft dependency.
        iterator: Any = datasets
        if progress and datasets:
            try:
                from tqdm.auto import tqdm  # type: ignore[import-not-found]

                iterator = tqdm(
                    datasets,
                    total=n_records,
                    desc="ensure_downloaded",
                )
            except ImportError:
                iterator = datasets

        n_downloaded = 0
        n_skipped = 0
        for ds in iterator:
            filecache = getattr(ds, "filecache", None)
            dep_paths = getattr(ds, "_dep_paths", []) or []
            needs_download = False
            if filecache is None or not filecache.exists():
                needs_download = True
            elif any(not p.exists() for p in dep_paths):
                needs_download = True
            if needs_download:
                n_downloaded += 1
            else:
                n_skipped += 1

        size_estimate = self.estimate_download_size()

        if n_records:
            download_fn(n_jobs=n_jobs)

        return {
            "n_records": n_records,
            "n_downloaded": n_downloaded,
            "n_skipped": n_skipped,
            "bytes_total": size_estimate["bytes"],
        }

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
            for field_name in description_fields:
                value = self._find_key_in_nested_dict(record, field_name)
                if value is not None:
                    description[field_name] = value
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

    # ------------------------------------------------------------------ #
    # HuggingFace-style split helpers (require ``eegdash[moabb]``)        #
    # ------------------------------------------------------------------ #

    def train_test_split(
        self,
        *,
        test_size: float | None = None,
        group: str = "subject",
        target: str | None = "target",
        seed: int | None = 42,
        **splitter_kwargs: Any,
    ) -> dict[str, "EEGDashDataset"]:
        """Group-aware train/test split mirroring HuggingFace ``datasets``.

        Returns ``{"train": <subset>, "test": <subset>}`` where the chosen
        ``group`` (``"subject"``, ``"session"`` or ``"dataset"``) stays
        disjoint across the two halves. Requires
        ``pip install eegdash[moabb]``.
        """
        from ..splits import train_test_split as _split

        return _split(
            self,
            test_size=test_size,
            group=group,
            target=target,
            seed=seed,
            **splitter_kwargs,
        )

    def k_fold(
        self,
        *,
        n_folds: int = 5,
        group: str = "subject",
        target: str | None = "target",
        seed: int | None = 42,
        **splitter_kwargs: Any,
    ):
        """Iterate ``(train, test)`` pairs over group-aware folds.

        Mirrors :class:`sklearn.model_selection.KFold` iteration with the
        HuggingFace dataset-views shape. Requires
        ``pip install eegdash[moabb]``.
        """
        from ..splits import k_fold as _kf

        return _kf(
            self,
            n_folds=n_folds,
            group=group,
            target=target,
            seed=seed,
            **splitter_kwargs,
        )


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

        if not kwargs.get("download", True) and "modality" not in kwargs:
            kwargs["modality"] = "eeg"

        super().__init__(
            dataset=dataset_id,
            query=query,
            cache_dir=cache_dir,
            s3_bucket=s3_bucket,
            _suppress_comp_warning=True,
            _dedupe_records=True,
            **kwargs,
        )


_from_api = os.getenv("EEGDASH_DATASET_REGISTRY_FROM_API", "").lower() in {
    "1",
    "true",
    "yes",
}
registered_classes = register_openneuro_datasets(
    summary_file=Path(__file__).with_name("dataset_summary.csv"),
    base_class=EEGDashDataset,
    namespace=globals(),
    from_api=_from_api,
)


__all__ = ["EEGDashDataset", "EEGChallengeDataset"] + list(registered_classes.keys())
