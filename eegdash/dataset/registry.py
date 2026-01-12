from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _human_readable_size(num_bytes: int | float | None) -> str:
    """Convert bytes to human-readable string."""
    if num_bytes is None or num_bytes == 0:
        return "Unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def register_openneuro_datasets(
    summary_file: str | Path | None = None,
    *,
    base_class=None,
    namespace: Dict[str, Any] | None = None,
    add_to_all: bool = True,
    from_api: bool = False,
    api_url: str = "https://data.eegdash.org/api",
    database: str = "eegdash",
) -> Dict[str, type]:
    """Dynamically create and register dataset classes from a summary file or API.

    This function reads a CSV file or queries the API containing summaries of
    datasets and dynamically creates a Python class for each dataset. These
    classes inherit from a specified base class and are pre-configured with the
    dataset's ID.

    Parameters
    ----------
    summary_file : str or pathlib.Path
        The path to the CSV file containing the dataset summaries.
    base_class : type, optional
        The base class from which the new dataset classes will inherit. If not
        provided, :class:`eegdash.api.EEGDashDataset` is used.
    namespace : dict, optional
        The namespace (e.g., `globals()`) into which the newly created classes
        will be injected. Defaults to the local `globals()` of this module.
    add_to_all : bool, default True
        If True, the names of the newly created classes are added to the
        `__all__` list of the target namespace, making them importable with
        `from ... import *`.

    Returns
    -------
    dict[str, type]
        A dictionary mapping the names of the registered classes to the class
        types themselves.

    """
    if base_class is None:
        from ..api import EEGDashDataset as base_class  # lazy import

    namespace = namespace if namespace is not None else globals()
    module_name = namespace.get("__name__", __name__)
    registered: Dict[str, type] = {}

    df = pd.DataFrame()
    if from_api:
        try:
            df = _fetch_datasets_from_api(api_url, database)
        except Exception:
            # Fallback to CSV if API fails, or empty if no CSV provided
            pass

    if df.empty and summary_file:
        summary_path = Path(summary_file)
        if summary_path.exists():
            df = pd.read_csv(summary_path, comment="#", skip_blank_lines=True)

    for _, row_series in df.iterrows():
        # Use the explicit 'dataset' column, not the CSV index.
        dataset_id = str(row_series.get("dataset", "")).strip()
        if not dataset_id:
            continue

        class_name = dataset_id.upper()

        # avoid zero-arg super() here
        def make_init(_dataset: str):
            def __init__(
                self,
                cache_dir: str,
                query: dict | None = None,
                s3_bucket: str | None = None,
                **kwargs,
            ):
                q = {"dataset": _dataset}
                if query:
                    q.update(query)
                # call base_class.__init__ directly
                base_class.__init__(
                    self,
                    query=q,
                    cache_dir=cache_dir,
                    s3_bucket=s3_bucket,
                    **kwargs,
                )

            return __init__

        init = make_init(dataset_id)

        # Generate rich docstring with dataset metadata
        doc = _generate_rich_docstring(dataset_id, row_series, base_class)

        # init.__doc__ = doc

        cls = type(
            class_name,
            (base_class,),
            {
                "_dataset": dataset_id,
                "__init__": init,
                "__doc__": doc,
                "__module__": module_name,  #
            },
        )

        namespace[class_name] = cls
        registered[class_name] = cls

        if add_to_all:
            ns_all = namespace.setdefault("__all__", [])
            if isinstance(ns_all, list) and class_name not in ns_all:
                ns_all.append(class_name)

    return registered


def _generate_rich_docstring(
    dataset_id: str, row_series: pd.Series, base_class: type
) -> str:
    """Generate a comprehensive, well-formatted docstring for a dataset class.

    Parameters
    ----------
    dataset_id : str
        The identifier of the dataset (e.g., "ds002718").
    row_series : pandas.Series
        A pandas Series containing the metadata for the dataset, extracted
        from the summary CSV file.
    base_class : type
        The base class from which the new dataset class inherits. Used to
        generate the "See Also" section of the docstring.

    Returns
    -------
    str
        A formatted docstring.

    """

    def _clean_optional(value: object) -> str:
        text = str(value).strip() if value is not None else ""
        if not text or text.lower() in {"nan", "none", "null"}:
            return ""
        return text

    def _clean_or_unknown(value: object) -> str:
        cleaned = _clean_optional(value)
        return cleaned if cleaned else "Unknown"

    n_subjects = _clean_or_unknown(row_series.get("n_subjects"))
    n_records = _clean_or_unknown(row_series.get("n_records"))
    n_tasks = _clean_or_unknown(row_series.get("n_tasks"))

    modality = _clean_optional(row_series.get("record_modality"))
    if not modality:
        modality = _clean_optional(row_series.get("modality of exp"))
    exp_type = _clean_optional(row_series.get("type of exp"))
    subject_type = _clean_optional(row_series.get("Type Subject"))

    summary_bits: list[str] = []
    if modality:
        summary_bits.append(f"Modality: ``{modality}``")
    if exp_type:
        summary_bits.append(f"Experiment type: ``{exp_type}``")
    if subject_type:
        summary_bits.append(f"Subject type: ``{subject_type}``")

    summary_line = f"OpenNeuro dataset ``{dataset_id}``."
    if summary_bits:
        summary_line = f"{summary_line} {'; '.join(summary_bits)}."
    summary_lines = [summary_line]
    if any(value != "Unknown" for value in (n_subjects, n_records, n_tasks)):
        summary_lines.append(
            f"Subjects: {n_subjects}; recordings: {n_records}; tasks: {n_tasks}."
        )

    doi_raw = (
        row_series.get("dataset_doi")
        or row_series.get("DatasetDOI")
        or row_series.get("doi")
    )
    doi = _clean_optional(doi_raw)
    doi_clean = doi.replace("doi:", "").strip() if doi else ""

    openneuro_url = f"https://openneuro.org/datasets/{dataset_id}"
    nemar_url = f"https://nemar.org/dataexplorer/detail?dataset_id={dataset_id}"
    class_name = dataset_id.upper()

    references_lines = [
        f"OpenNeuro dataset: {openneuro_url}",
        f"NeMAR dataset: {nemar_url}",
    ]
    if doi_clean:
        references_lines.append(f"DOI: https://doi.org/{doi_clean}")

    docstring = f"""{chr(10).join(summary_lines)}

Parameters
----------
cache_dir : str | Path
    Directory where data are cached locally.
query : dict | None
    Additional MongoDB-style filters to AND with the dataset selection.
    Must not contain the key ``dataset``.
s3_bucket : str | None
    Base S3 bucket used to locate the data.
**kwargs : dict
    Additional keyword arguments forwarded to :class:`~{base_class.__module__}.{base_class.__name__}`.

Attributes
----------
data_dir : Path
    Local dataset cache directory (``cache_dir / dataset_id``).
query : dict
    Merged query with the dataset filter applied.
records : list[dict] | None
    Metadata records used to build the dataset, if pre-fetched.

Notes
-----
Each item is a recording; recording-level metadata are available via ``dataset.description``.
``query`` supports MongoDB-style filters on fields in ``ALLOWED_QUERY_FIELDS`` and is combined with the dataset filter.
Dataset-specific caveats are not provided in the summary metadata.

References
----------
{chr(10).join(references_lines)}

Examples
--------
>>> from eegdash.dataset import {class_name}
>>> dataset = {class_name}(cache_dir="./data")
>>> recording = dataset[0]
>>> raw = recording.load()
"""

    return docstring


# Datasets to explicitly ignore (synced with rules in 3_digest.py)
EXCLUDED_DATASETS = {
    "ABUDUKADI",
    "ABUDUKADI_2",
    "ABUDUKADI_3",
    "ABUDUKADI_4",
    "AILIJIANG",
    "AILIJIANG_3",
    "AILIJIANG_4",
    "AILIJIANG_5",
    "AILIJIANG_7",
    "AILIJIANG_8",
    "BAIHETI",
    "BAIHETI_2",
    "BAIHETI_3",
    "BIAN_3",
    "BIN_27",
    "BLIX",
    "BOJIN",
    "BOUSSAGOL",
    "AISHENG",
    "ACHOLA",
    "ANASHKIN",
    "ANJUM",
    "BARBIERI",
    "BIN_8",
    "BIN_9",
    "BING_4",
    "BING_8",
    "BOWEN_4",
    "AZIZAH",
    "BAO",
    "BAO-YOU",
    "BAO_2",
    "BENABBOU",
    "BING",
    "BOXIN",
    "test",
    "ds003380",
}


def fetch_datasets_from_api(
    api_url: str = "https://data.eegdash.org/api", database: str = "eegdash"
) -> pd.DataFrame:
    """Fetch dataset summaries from API and return as DataFrame matching CSV structure."""
    import os

    from ..paths import get_default_cache_dir

    cache_dir = get_default_cache_dir()
    cache_file = cache_dir / "dataset_summary.csv"

    # Try loading from cache first
    try:
        if cache_file.exists():
            return pd.read_csv(cache_file, comment="#", skip_blank_lines=True)
    except Exception:
        pass

    limit = int(os.environ.get("EEGDASH_DOC_LIMIT", 1000))

    # 1. Fetch Datasets Summary
    url = f"{api_url}/{database}/datasets/summary?limit={limit}"
    data: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        pass

    if not data or not data.get("success"):
        return pd.DataFrame()

    datasets = data.get("data", [])

    # 2. Fetch Global Stats (nchans, sfreq)
    # We collect IDs to optimize the query if needed, but the endpoint might just return all.
    # The stats endpoint is /datasets/stats/records
    stats_url = f"{api_url}/{database}/datasets/stats/records"
    global_stats: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(stats_url, timeout=60) as response:
            resp_json = json.loads(response.read().decode("utf-8"))
            global_stats = resp_json.get("data", {})
    except Exception:
        # Proceed without stats if this fails
        pass

    rows = []
    for ds in datasets:
        ds_id = ds.get("dataset_id", "").strip()
        # Filter test datasets and excluded ones
        if (
            ds_id.lower() in ("test", "test_dataset")
            or ds_id.upper() in EXCLUDED_DATASETS
        ):
            continue

        ds_stats = global_stats.get(ds_id, {})
        nchans_list = ds_stats.get("nchans_counts", [])
        sfreq_list = ds_stats.get("sfreq_counts", [])

        # Extract demographics
        demographics = ds.get("demographics", {}) or {}
        recording_modality = ds.get("recording_modality", []) or []
        if isinstance(recording_modality, str):
            recording_modality = [recording_modality]

        # Map API fields to expected CSV columns
        row = {
            "dataset": ds_id,
            "n_subjects": demographics.get("subjects_count", 0) or 0,
            "n_records": ds.get("total_files", 0) or 0,
            "n_tasks": len(ds.get("tasks", []) or []),
            "modality of exp": ", ".join(recording_modality),
            "type of exp": ds.get("study_design") or "Unknown",
            "Type Subject": ds.get("study_domain") or "Unknown",
            "duration_hours_total": 0.0,  # Not available in summary endpoint
            "size_bytes": ds.get("size_bytes") or 0,
            "size": ds.get("size_human") or _human_readable_size(ds.get("size_bytes")),
            "source": ds.get("source") or "unknown",
            # Extended fields for docs/summary tables
            "dataset_title": ds.get("name", ""),
            "record_modality": ", ".join(recording_modality),
            # We enforce JSON string for list/dict structures to survive CSV roundtrip reliably
            "nchans_set": json.dumps(nchans_list),
            "sampling_freqs": json.dumps(sfreq_list),
            "license": ds.get("license", ""),
            "doi": ds.get("dataset_doi", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save to cache if we got data
    if not df.empty:
        try:
            df.to_csv(cache_file, index=False)
        except Exception:
            pass

    return df


def _fetch_datasets_from_api(api_url: str, database: str) -> pd.DataFrame:
    """Fetch dataset summaries from API and return as DataFrame matching CSV structure."""
    import os

    limit = int(os.environ.get("EEGDASH_DOC_LIMIT", 1000))
    url = f"{api_url}/{database}/datasets/summary?limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return pd.DataFrame()

    if not data.get("success"):
        return pd.DataFrame()

    datasets = data.get("data", [])
    rows = []
    for ds in datasets:
        ds_id = ds.get("dataset_id", "").strip()
        # Filter test datasets and excluded ones
        if (
            ds_id.lower() in ("test", "test_dataset")
            or ds_id.upper() in EXCLUDED_DATASETS
        ):
            continue

        meta = ds.get("metadata", {})
        # Map API fields to expected CSV columns
        row = {
            "dataset": ds.get("dataset_id"),
            "n_subjects": meta.get("subject_count", 0),
            "n_records": ds.get("record_count", 0),
            "n_tasks": len(meta.get("tasks", [])),
            "modality of exp": ", ".join(meta.get("recording_modalities", []) or []),
            "type of exp": meta.get("type", "Unknown"),
            "Type Subject": meta.get("pathology", "Unknown"),
            "duration_hours_total": round(meta.get("duration_hours_total", 0) or 0, 2),
            "size": ds.get("size_human", "Unknown"),
            # internal/extra fields
            "source": ds.get("source", "unknown"),
        }
        rows.append(row)

    return pd.DataFrame(rows)
