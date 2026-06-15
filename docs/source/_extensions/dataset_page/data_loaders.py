"""Data-loading and value-cleaning helpers for the dataset_page package.

Centralises everything that turns a dataset id into the ``context`` mapping
that the section formatters consume:

* generic value cleaners (``_clean_value``, ``_normalize_list`` ...)
* the cached per-dataset detail fetch (``_load_dataset_details``)
* the dataset-summary loaders (``_load_dataset_rows`` ...)
* the context builder itself (``_build_dataset_context``)

The original monolith concatenated these with the section formatters; the
split here keeps them on the non-Sphinx side of the boundary so the formatters
can be reasoned about as pure functions over the context dict.
"""

from __future__ import annotations

import concurrent.futures
import csv
import importlib
import inspect
import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from sphinx.util import logging

import eegdash.dataset as dataset_module
from eegdash.dataset import EEGDashDataset
from eegdash.dataset.snapshot import DatasetSnapshot

from ._constants import (
    _PROBE_UA,
    _README_PAPER_DOI_RE,
    CLONE_ROOT,
    DEFAULT_METADATA_FIELDS,
    EEGDASH_API_BASE,
)

LOGGER = logging.getLogger(__name__)

_DATASET_DETAILS_CACHE: dict[str, dict[str, object]] = {}
_DATASET_DETAILS_CACHE_LOCK = threading.Lock()


def _get_json(
    url: str,
    *,
    timeout: float = 10.0,
    extra_headers: Mapping[str, str] | None = None,
) -> dict | None:
    """GET ``url`` and return parsed JSON, or ``None`` on a recoverable
    error (network / HTTP error / bad JSON).

    Centralises the boilerplate that was duplicated across every probe
    helper: UA spoofing, timeout, narrow exception handling, JSON
    decode. Unexpected errors propagate so real bugs aren't silently
    swallowed.
    """
    headers = {"Accept": "application/json", "User-Agent": _PROBE_UA}
    if extra_headers:
        headers.update(extra_headers)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _head_content_length(
    url: str,
    *,
    timeout: float = 8.0,
) -> int | None:
    """HEAD ``url`` and return the integer ``Content-Length`` (or None
    when the request fails / header is missing). Used by probes that
    only need to know whether a resource has a meaningful body.
    """
    try:
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": _PROBE_UA}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not (200 <= resp.status < 400):
                return None
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None


def _should_use_api_summary() -> bool:
    # Always try API first; set EEGDASH_NO_API=1 to disable.
    return not bool(os.environ.get("EEGDASH_NO_API"))


def _load_dataset_summary_from_api():
    """Return the dataset-summary DataFrame, or ``None`` when unavailable.

    Mirrors the legacy ``_load_dataset_summary_from_api`` in conf.py --
    moved here so the dataset-page builder owns its data source. The
    snapshot's own in-process cache is keyed by ``(api_base, database,
    limit)`` so a second call inside the same build reuses the result.
    """
    if not _should_use_api_summary():
        return None

    snapshot = DatasetSnapshot.build()
    if snapshot.api_errors:
        LOGGER.info(
            "[dataset-docs] DatasetSnapshot source=%s dataset_count=%d "
            "errors=%d (first: %s)",
            snapshot.source,
            snapshot.dataset_count,
            len(snapshot.api_errors),
            snapshot.api_errors[0],
        )
    df = snapshot.rows()
    if df.empty:
        return None
    return df


# ---------------------------------------------------------------------------
# Generic value-cleaning helpers
# ---------------------------------------------------------------------------


def _clean_value(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown"}:
        return default
    return text


def _format_stat_counts(value: object, default: str = "") -> str:
    """Format JSON arrays of {val, count} objects into human-readable strings.

    Handles formats like: [{"val": 64, "count": 30}, {"val": 32, "count": 24}]
    Returns strings like: "64 (30), 32 (24)" or just "64" for single values.
    """
    if value is None:
        return default

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return default

    if text.startswith("["):
        try:
            items = json.loads(text)
            if not items:
                return default

            valid_items = []
            for item in items:
                if isinstance(item, dict):
                    val = item.get("val")
                    count = item.get("count", 0)
                    if val is not None:
                        if count > 1:
                            valid_items.append(f"{val} ({count})")
                        else:
                            valid_items.append(str(val))

            if not valid_items:
                return default

            unique_vals = set(
                item.get("val") for item in items if isinstance(item, dict)
            )
            unique_vals.discard(None)
            if len(unique_vals) == 1:
                return str(unique_vals.pop())

            return ", ".join(valid_items)
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    return _clean_value(value, default)


def _collapse_whitespace(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _normalize_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [
            _collapse_whitespace(str(item).strip())
            for item in value
            if str(item).strip()
        ]
        return items
    text = _collapse_whitespace(str(value).strip())
    return [text] if text else []


def _value_or_unknown(value: str, field_type: str = "general") -> str:
    """Return ``value`` or a context-aware placeholder for missing data."""
    if value and value.strip() not in ("", "nan", "none", "null", "unknown", "—"):
        return value
    placeholders = {
        "n_channels": "Varies",
        "pathology": "Not specified",
        "duration": "Not calculated",
        "sampling_rate": "Varies",
        "subjects": "—",
        "recordings": "—",
        "tasks": "—",
        "license": "See source",
        "general": "—",
    }
    return placeholders.get(field_type, placeholders["general"])


def _normalize_doi(doi: str) -> str:
    if not doi:
        return ""
    return doi.replace("doi:", "").strip()


# ---------------------------------------------------------------------------
# Per-dataset detail fetch / cache
# ---------------------------------------------------------------------------


def _fetch_dataset_details_from_api(dataset_id: str) -> dict[str, object]:
    """Fetch detailed dataset information from the API.

    Uses the endpoint: /datasets/summary/{dataset_id}
    """
    if not _should_use_api_summary():
        return {}

    api_url = EEGDASH_API_BASE

    # API may be case-sensitive; try original then common variations.
    ids_to_try = [dataset_id]
    if dataset_id.startswith("ds"):
        ids_to_try.append(dataset_id.lower())
    elif dataset_id.lower().startswith("eeg2025"):
        ids_to_try.append(f"EEG2025r{dataset_id.lower().replace('eeg2025r', '')}")

    data = None
    for try_id in ids_to_try:
        url = f"{api_url}/datasets/summary/{try_id}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data.get("success"):
                    break
        except Exception as exc:
            LOGGER.debug("[dataset-docs] API fetch for %s failed: %s", try_id, exc)
            continue

    if not data or not data.get("success"):
        return {}

    ds = data.get("data", {})
    if not ds:
        return {}

    year = ""
    timestamps = ds.get("timestamps", {}) or {}
    created_at = timestamps.get("dataset_created_at", "")
    if created_at and len(created_at) >= 4:
        year = created_at[:4]

    title = _clean_value(ds.get("computed_title")) or _clean_value(ds.get("name"))
    if title and (
        title.lower().endswith((".tsv", ".json", ".csv", ".md"))
        or title.lower() == "readme"
    ):
        readme = _clean_value(ds.get("readme", ""))
        if readme.startswith("# "):
            title = readme.split("\n")[0][2:].strip()
            if title.lower().startswith("wrist:"):
                title = title.split(":", 1)[1].strip()

    details: dict[str, object] = {
        "title": title,
        "authors": _normalize_list(ds.get("authors")),
        "license": _clean_value(ds.get("license")),
        "doi": _clean_value(ds.get("dataset_doi")),
        "year": year,
        "readme": _clean_value(ds.get("readme")),
        "funding": _normalize_list(ds.get("funding")),
        "senior_author": _clean_value(ds.get("senior_author")),
        "contact_info": _normalize_list(ds.get("contact_info")),
        "n_subjects": ds.get("demographics", {}).get("subjects_count"),
        "total_files": ds.get("total_files"),
        "n_tasks": len(ds.get("tasks", []) or []),
        "tasks": ds.get("tasks") or [],
        "recording_modality": ds.get("recording_modality", []),
        "datatypes": ds.get("datatypes") or [],
        "size_bytes": ds.get("size_bytes"),
        "source": _clean_value(ds.get("source")),
        "demographics": ds.get("demographics"),
        "nchans_counts": ds.get("nchans_counts"),
        "sfreq_counts": ds.get("sfreq_counts"),
        "total_duration_s": ds.get("total_duration_s"),
        "bad_channels_info": ds.get("bad_channels_info"),
        # Editorial Brief — fields the rich field-card and provenance strip
        # need to render real values instead of TODO placeholders.
        "bids_version": _clean_value(ds.get("bids_version")),
        "tags": ds.get("tags") or {},
        "dataset_storage": ds.get("storage") or {},
        "associated_paper_doi": _clean_value(ds.get("associated_paper_doi")),
        "stats_computed_at": _clean_value(ds.get("stats_computed_at")),
        "digested_at": _clean_value((ds.get("timestamps") or {}).get("digested_at")),
        # Additional fields surfaced from the summary endpoint:
        "sessions": ds.get("sessions") or [],
        "dataset_created_at": _clean_value(
            (ds.get("timestamps") or {}).get("dataset_created_at")
        ),
        "dataset_modified_at": _clean_value(
            (ds.get("timestamps") or {}).get("dataset_modified_at")
        ),
        "data_processed": bool(ds.get("data_processed")),
        "contributing_labs": ds.get("contributing_labs") or [],
        "n_contributing_labs": ds.get("n_contributing_labs"),
        "experimental_modalities": ds.get("experimental_modalities") or [],
        "study_design": _clean_value(ds.get("study_design")),
        "study_domain": _clean_value(ds.get("study_domain")),
    }

    external_links = ds.get("external_links", {}) or {}
    details["source_url"] = _clean_value(external_links.get("source_url"))
    details["paper_url"] = _clean_value(external_links.get("paper_url"))
    details["github_url"] = _clean_value(external_links.get("github_url"))
    details["osf_url"] = _clean_value(external_links.get("osf_url"))

    # Some NEMAR-ingested datasets never populate ``external_links.paper_url``
    # even though the README ships a ``[![Paper DOI](...)](https://doi.org/…)``
    # badge. The lede block already turns that badge into a visible link, so
    # mirror it into the rail's quick-actions row when no structured field
    # exists yet.
    if not details["paper_url"]:
        readme = ds.get("readme") or ""
        if isinstance(readme, str) and "Paper" in readme:
            m = _README_PAPER_DOI_RE.search(readme)
            if m:
                details["paper_url"] = m.group(1)

    return details


# Suffixes the BIDS spec defines for the modality-specific sidecars the
# editorial field-card lists ("events · channels · electrodes · coordsystem").
# These match against the trailing path segment of each storage.dep_key.
_SIDECAR_SUFFIXES = {
    "events.tsv": "events",
    "events.json": "events.json",
    "channels.tsv": "channels",
    "electrodes.tsv": "electrodes",
    "coordsystem.json": "coordsystem",
    "eeg.json": "eeg.json",
    "meg.json": "meg.json",
    "ieeg.json": "ieeg.json",
    "physio.tsv": "physio",
    "stim.tsv": "stim",
}


# Spec-conventional order for sidecar labels we render in the field-card.
_SIDECAR_RENDER_ORDER = (
    "events",
    "events.json",
    "channels",
    "electrodes",
    "coordsystem",
    "eeg.json",
    "meg.json",
    "ieeg.json",
    "physio",
    "stim",
)


def _detect_sidecars_for_dataset(dataset_id: str) -> list[str]:
    """Return a sorted list of BIDS sidecar kinds present for ``dataset_id``.

    Probes ONE sample record via the eegdash records API and inspects
    ``storage.dep_keys`` for known BIDS sidecar suffixes. Returns an
    empty list if the probe fails (network error, dataset not yet
    ingested).
    """
    if not _should_use_api_summary():
        return []

    dataset_lower = dataset_id.lower()
    query = json.dumps(
        {"dataset": dataset_lower, "_has_missing_files": {"$ne": True}},
        separators=(",", ":"),
    )
    url = (
        f"{EEGDASH_API_BASE}/records"
        f"?{urllib.parse.urlencode({'limit': 1, 'filter': query})}"
    )
    body = _get_json(url)
    if not body or not body.get("success") or not body.get("data"):
        return []

    storage = body["data"][0].get("storage") or {}
    dep_keys = storage.get("dep_keys") or []
    if not isinstance(dep_keys, list):
        return []

    found: set[str] = set()
    for key in dep_keys:
        path = str(key)
        for suffix, label in _SIDECAR_SUFFIXES.items():
            if path.endswith(suffix):
                found.add(label)
                break
    return [k for k in _SIDECAR_RENDER_ORDER if k in found]


def _detect_huggingface_mirror(dataset_id: str) -> dict[str, object]:
    """Probe the EEGDash HuggingFace org for a mirror of ``dataset_id``.

    Returns a dict with ``available`` (bool), ``url`` (str, dataset-specific
    when available, org page otherwise), ``downloads`` (int when known),
    and ``last_modified`` (ISO-8601 str when known). The HF API returns
    200 with an ``{"error": "..."}`` body for missing datasets, so the
    discriminator is the presence of an ``id`` field in the JSON payload.
    """
    org_url = "https://huggingface.co/EEGDash"
    fallback: dict[str, object] = {
        "available": False,
        "url": org_url,
        "downloads": None,
        "last_modified": None,
    }
    if not _should_use_api_summary():
        return fallback

    dataset_lower = dataset_id.lower()
    body = _get_json(
        f"https://huggingface.co/api/datasets/EEGDash/{dataset_lower}",
        timeout=8.0,
    )
    if not isinstance(body, dict) or not body.get("id"):
        return fallback

    return {
        "available": True,
        "url": f"https://huggingface.co/datasets/EEGDash/{dataset_lower}",
        "downloads": body.get("downloads"),
        "last_modified": body.get("lastModified"),
    }


def _fetch_participants_from_records(dataset_id: str) -> list[dict[str, object]]:
    """Pull per-subject demographics for a dataset.

    Prefers the dedicated ``/api/eegdash/datasets/{dataset_id}/participants``
    endpoint, which deduplicates server-side via Mongo ``$group`` and
    returns one row per subject in a single request. Falls back to
    paginating ``/records`` for older server deployments that don't
    expose the participants endpoint yet.
    """
    if not _should_use_api_summary():
        return []

    dataset_lower = dataset_id.lower()

    # --- Primary path: dedicated participants endpoint ----------------
    body = _get_json(
        f"{EEGDASH_API_BASE}/datasets/{dataset_lower}/participants",
        timeout=12.0,
    )
    if body is not None and body.get("success") and isinstance(body.get("data"), list):
        participants: list[dict[str, object]] = []
        for entry in body["data"]:
            subject = str(entry.get("subject") or "").strip()
            if not subject:
                continue
            tsv = entry.get("participant_tsv") or {}
            if not isinstance(tsv, dict):
                tsv = {}
            participants.append({"subject": subject, **tsv})
        return participants

    # --- Fallback: paginate /records ----------------------------------
    query = json.dumps(
        {
            "dataset": dataset_lower,
            "suffix": {"$in": ["eeg", "ieeg", "emg", "meg"]},
            "_has_missing_files": {"$ne": True},
        },
        separators=(",", ":"),
    )
    seen: set[str] = set()
    participants = []
    skip = 0
    page_size = 1000
    max_skip = 20000  # safety bound — no real dataset has 20k recordings
    while skip < max_skip:
        url = (
            f"{EEGDASH_API_BASE}/records"
            f"?{urllib.parse.urlencode({'limit': page_size, 'skip': skip, 'filter': query})}"
        )
        page = _get_json(url, timeout=12.0)
        records = page.get("data") if isinstance(page, dict) else None
        if not records:
            break
        for record in records:
            subject = str(record.get("subject") or "").strip()
            if not subject or subject in seen:
                continue
            seen.add(subject)
            tsv = record.get("participant_tsv") or {}
            if not isinstance(tsv, dict):
                continue
            participants.append({"subject": subject, **tsv})
        if len(records) < page_size:
            break
        skip += page_size
    return participants


def _detect_hed_annotation(dataset_id: str) -> bool:
    """Return True when NEMAR has published a HED word-cloud for this dataset.

    Cheap signal: NEMAR only generates per-dataset HED word clouds when
    the events sidecar carries valid HED tags. A HEAD request to the
    SVG URL with a ``Content-Length > 1 KB`` check is enough — NEMAR's
    download endpoint returns 200 for missing files but with empty body.
    """
    if not _should_use_api_summary():
        return False

    url = (
        "https://nemar.org/dataexplorer/download"
        f"?filepath=/data/nemar/openneuro//processed/event_summaries/"
        f"{dataset_id.lower()}/word_cloud.svg&file_type=svg"
    )
    length = _head_content_length(url, timeout=8.0)
    return length is not None and length > 1024


def _load_dataset_details(dataset_id: str) -> dict[str, object]:
    """Aggregate per-dataset metadata from local files + API + probes.

    Thread-safe: the in-process cache is guarded by a lock so that
    duplicate workers don't waste 4 network probes on the same id.
    Network probes are issued concurrently because they hit different
    hosts (EEGDash API · NEMAR · HuggingFace) and don't depend on each
    other.
    """
    dataset_id = dataset_id.lower()

    with _DATASET_DETAILS_CACHE_LOCK:
        cached = _DATASET_DETAILS_CACHE.get(dataset_id)
        if cached is not None:
            return cached

    details: dict[str, object] = {}

    dataset_dir = CLONE_ROOT / dataset_id
    desc_path = dataset_dir / "dataset_description.json"
    if desc_path.exists():
        try:
            data = json.loads(desc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        details["title"] = _clean_value(data.get("Name"))
        details["authors"] = _normalize_list(data.get("Authors"))
        details["license"] = _clean_value(data.get("License"))
        details["doi"] = _clean_value(data.get("DatasetDOI"))
        details["how_to_acknowledge"] = _clean_value(data.get("HowToAcknowledge"))
        details["references"] = _normalize_list(data.get("ReferencesAndLinks"))
        details["funding"] = _normalize_list(data.get("Funding"))

    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        details.setdefault("doi", _clean_value(data.get("dataset_doi")))
        details["source_url"] = _clean_value(data.get("source_url"))

    # API call must run first; the parallel probes don't depend on it but
    # we want the api_details merged in before they overwrite.
    api_details = _fetch_dataset_details_from_api(dataset_id)
    for key, value in api_details.items():
        if value and not details.get(key):
            details[key] = value

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            "sidecars_detected": pool.submit(_detect_sidecars_for_dataset, dataset_id),
            "hed_annotated": pool.submit(_detect_hed_annotation, dataset_id),
            "huggingface": pool.submit(_detect_huggingface_mirror, dataset_id),
            "participants_rows": pool.submit(
                _fetch_participants_from_records, dataset_id
            ),
        }
        for key, future in futures.items():
            try:
                details[key] = future.result()
            except Exception as exc:  # noqa: BLE001 — probe results are best-effort
                LOGGER.warning(
                    "[dataset-docs] probe %s failed for %s: %s",
                    key,
                    dataset_id,
                    exc,
                )
                details[key] = (
                    False
                    if key == "hed_annotated"
                    else {}
                    if key == "huggingface"
                    else []
                )

    with _DATASET_DETAILS_CACHE_LOCK:
        existing = _DATASET_DETAILS_CACHE.setdefault(dataset_id, details)
    return existing


# ---------------------------------------------------------------------------
# Dataset-summary loaders (CSV/API row lookup; not the per-dataset detail
# fetch above). Both the directive's row-lookup and the listing builders
# share this entry point.
# ---------------------------------------------------------------------------


def _iter_dataset_classes() -> Sequence[str]:
    """Return the sorted dataset class names exported by ``eegdash.dataset``."""
    class_names: list[str] = []
    for name in getattr(dataset_module, "__all__", []):
        if name == "EEGChallengeDataset":
            continue
        obj = getattr(dataset_module, name, None)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, EEGDashDataset):
            continue
        if getattr(obj, "_dataset", None) is None:
            continue
        class_names.append(name)

    return tuple(sorted(class_names))


def _load_experiment_counts(dataset_names: Iterable[str]) -> list[tuple[str, int]]:
    """Return a sorted list of (experiment_type, count) pairs."""
    valid_names = {name.upper() for name in dataset_names}
    df = _load_dataset_summary_from_api()
    if df is not None and not df.empty:
        counter: Counter[str] = Counter()
        for _, row in df.iterrows():
            dataset_id = str(row.get("dataset", "")).strip().upper()
            if dataset_id not in valid_names:
                continue
            exp_type = str(row.get("type of exp") or "Unspecified").strip()
            counter[exp_type or "Unspecified"] += 1
        return sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    csv_path = Path(importlib.import_module("eegdash.dataset").__file__).with_name(
        "dataset_summary.csv"
    )
    if not csv_path.exists():
        return []

    counter: Counter[str] = Counter()

    with csv_path.open(encoding="utf-8") as handle:
        filtered = (
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        )
        reader = csv.DictReader(filtered)
        for row in reader:
            dataset_id = (row.get("dataset") or "").strip().upper()
            if dataset_id not in valid_names:
                continue
            exp_type = (row.get("type of exp") or "Unspecified").strip()
            counter[exp_type or "Unspecified"] += 1

    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _render_experiment_rows(pairs: Iterable[tuple[str, int]]) -> str:
    lines = []
    for exp_type, count in pairs:
        label = exp_type or "Unspecified"
        lines.append(f"   * - {label}\n     - {count}")
    if not lines:
        lines.append("   * - No experimental metadata available\n     - N/A")
    return "\n".join(lines)


def _render_toctree_entries(names: Sequence[str]) -> str:
    return "\n".join(f"   eegdash.dataset.{name}" for name in names)


def _load_dataset_rows(dataset_names: Sequence[str]) -> Mapping[str, Mapping[str, str]]:
    wanted = set(dataset_names)
    df = _load_dataset_summary_from_api()
    if df is not None and not df.empty:
        rows: dict[str, Mapping[str, str]] = {}
        for _, row in df.iterrows():
            dataset_id = str(row.get("dataset", "")).strip()
            if not dataset_id:
                continue
            class_name = dataset_id.upper()
            if class_name not in wanted:
                continue
            rows[class_name] = row.to_dict()
        if rows:
            return rows

    csv_path = Path(importlib.import_module("eegdash.dataset").__file__).with_name(
        "dataset_summary.csv"
    )
    if not csv_path.exists():
        return {}

    rows: dict[str, Mapping[str, str]] = {}

    with csv_path.open(encoding="utf-8") as handle:
        filtered = (
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        )
        reader = csv.DictReader(filtered)
        for row in reader:
            dataset_id = (row.get("dataset") or "").strip()
            if not dataset_id:
                continue
            class_name = dataset_id.upper()
            if class_name not in wanted:
                continue
            rows[class_name] = row

    return rows


# ---------------------------------------------------------------------------
# Dataset context: row + details -> dict consumed by the section formatters
# ---------------------------------------------------------------------------


def _build_dataset_context(
    class_name: str, row: Mapping[str, str] | None
) -> dict[str, object]:
    dataset_id = _clean_value(row.get("dataset") if row else "")
    dataset_id = dataset_id.lower() if dataset_id else class_name.lower()
    details = _load_dataset_details(dataset_id)

    modality = _clean_value((row or {}).get("record_modality"))
    if not modality:
        modality_raw = details.get("recording_modality", [])
        if isinstance(modality_raw, list):
            modality = ", ".join(str(m) for m in modality_raw)
        else:
            modality = _clean_value(modality_raw)
    if not modality:
        modality = _clean_value((row or {}).get("modality of exp"))

    source = _clean_value((row or {}).get("source"))
    if not source:
        source = "OpenNeuro"

    title = _collapse_whitespace(_clean_value(details.get("title")))
    if not title or title.lower().endswith((".tsv", ".json")):
        title = _collapse_whitespace(_clean_value((row or {}).get("dataset_title")))

    license_text = _clean_value(details.get("license"))
    if not license_text:
        license_text = _clean_value((row or {}).get("license"))

    doi = _clean_value(details.get("doi"))
    if not doi:
        doi = _clean_value((row or {}).get("doi"))

    year = _clean_value(details.get("year"))
    if not year or year == "—":
        refs = details.get("references", [])
        if not refs:
            readme = str(details.get("readme", ""))
            years = re.findall(r"\((\d{4})\)", readme)
            if not years:
                years = re.findall(r"\b(19|20)\d{2}\b", readme)
            if years:
                year = years[0]

    n_subjects = _clean_value((row or {}).get("n_subjects"))
    if not n_subjects or n_subjects == "0":
        n_subjects = _clean_value(details.get("n_subjects"))

    n_records = _clean_value((row or {}).get("n_records"))
    if not n_records or n_records == "0":
        n_records = _clean_value(details.get("total_files"))

    n_tasks = _clean_value((row or {}).get("n_tasks"))
    if not n_tasks or n_tasks == "0":
        n_tasks = _clean_value(details.get("n_tasks"))

    size = _clean_value((row or {}).get("size"))
    if not size or size == "Unknown":
        size_bytes = details.get("size_bytes")
        if size_bytes:
            try:
                s = float(size_bytes)
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if s < 1024.0:
                        size = (
                            f"{s:.2f} {unit}"
                            if unit not in ["B", "KB"]
                            else f"{int(s)} {unit}"
                        )
                        break
                    s /= 1024.0
            except (ValueError, TypeError):
                pass

    dataset_format = "—"
    if source.lower() in ["openneuro", "nemar"]:
        dataset_format = "BIDS"

    s3_item_count = _clean_value((row or {}).get("s3_item_count"))
    if not s3_item_count or s3_item_count == "0":
        s3_item_count = _clean_value(details.get("total_files"))

    # Canonical / author-year identifiers -- populated from CSV/API when
    # the name_suggester pipeline has run, else empty. Reuses the runtime
    # registry's parser so docs match the catalog's aliases.
    from eegdash.dataset.registry import _parse_canonical_names  # noqa: WPS433

    canonical_names = _parse_canonical_names((row or {}).get("canonical_name"))
    author_year_name = _clean_value((row or {}).get("author_year"))

    # NEMAR-native datasets have ids of the form ``nm000132``. The NEMAR
    # client uses these as primary keys; surface them through the
    # context so the section formatter does not need to re-derive on
    # every page. Datasets sourced elsewhere (OpenNeuro ``dsNNNNNN``)
    # leave ``nemar_id`` empty and the section is skipped.
    nemar_id = ""
    if dataset_id.startswith("nm"):
        nemar_id = dataset_id

    return {
        "class_name": class_name,
        "dataset_id": dataset_id,
        "dataset_upper": dataset_id.upper(),
        "nemar_id": nemar_id,
        "title": title,
        "year": year,
        "authors": details.get("authors", []),
        "license": license_text,
        "doi": doi,
        "canonical_names": canonical_names,
        "author_year_name": author_year_name,
        "source_url": _clean_value(details.get("source_url")),
        "paper_url": _clean_value(details.get("paper_url")),
        "github_url": _clean_value(details.get("github_url")),
        "osf_url": _clean_value(details.get("osf_url")),
        "references": details.get("references", []),
        "how_to_acknowledge": _clean_value(details.get("how_to_acknowledge")),
        "n_subjects": n_subjects,
        "n_records": n_records,
        "n_tasks": n_tasks,
        "n_channels": _format_stat_counts((row or {}).get("nchans_set")),
        "sampling_freqs": _format_stat_counts((row or {}).get("sampling_freqs")),
        "duration_hours_total": _clean_value((row or {}).get("duration_hours_total")),
        "size": size,
        "s3_item_count": s3_item_count,
        "modality": modality,
        "pathology": _clean_value((row or {}).get("Type Subject")),
        "tag_modality": _clean_value((row or {}).get("modality of exp")),
        "tag_type": _clean_value((row or {}).get("type of exp")),
        "source": source,
        "openneuro_url": f"https://openneuro.org/datasets/{dataset_id}",
        "nemar_url": f"https://nemar.org/dataexplorer/detail?dataset_id={dataset_id}",
        "metadata_fields": DEFAULT_METADATA_FIELDS,
        "format": dataset_format,
        "readme": _clean_value(details.get("readme")),
        "nemar_citation_count": _clean_value((row or {}).get("nemar_citation_count")),
        "demographics": details.get("demographics") or {},
        "participants_rows": details.get("participants_rows") or [],
        "nchans_counts": details.get("nchans_counts") or [],
        "sfreq_counts": details.get("sfreq_counts") or [],
        "total_duration_s": details.get("total_duration_s"),
        "bad_channels_info": details.get("bad_channels_info"),
        # Editorial Brief fields (surfaced from API; populated by
        # _fetch_dataset_details_from_api + the sidecar/HED probes).
        "bids_version": _clean_value(details.get("bids_version")),
        "tags": details.get("tags") or {},
        "datatypes": details.get("datatypes") or [],
        "tasks": details.get("tasks") or [],
        "funding": details.get("funding") or [],
        "senior_author": _clean_value(details.get("senior_author")),
        "contact_info": details.get("contact_info") or [],
        "sidecars_detected": details.get("sidecars_detected") or [],
        "hed_annotated": bool(details.get("hed_annotated")),
        "huggingface": details.get("huggingface")
        or {
            "available": False,
            "url": "https://huggingface.co/EEGDash",
            "downloads": None,
            "last_modified": None,
        },
        "associated_paper_doi": _clean_value(details.get("associated_paper_doi")),
        "digested_at": _clean_value(details.get("digested_at")),
        # Storage descriptor (backend, base S3 url, dep_keys).
        "dataset_storage": details.get("dataset_storage") or {},
        # Newly-surfaced fields for the editorial layout:
        "sessions": details.get("sessions") or [],
        "dataset_created_at": _clean_value(details.get("dataset_created_at")),
        "dataset_modified_at": _clean_value(details.get("dataset_modified_at")),
        "data_processed": bool(details.get("data_processed")),
        "contributing_labs": details.get("contributing_labs") or [],
        "n_contributing_labs": details.get("n_contributing_labs"),
        "experimental_modalities": details.get("experimental_modalities") or [],
        "study_design": _clean_value(details.get("study_design")),
        "study_domain": _clean_value(details.get("study_domain")),
    }


def _compute_quality_score(context: Mapping[str, object]) -> tuple[str, str, int]:
    """Compute a metadata-completeness score for the hero badge.

    Returns a (label, badge_color, percentage) triple. Categories:
    Complete (>=90), Good (>=70), Partial (>=50), Limited (<50).
    """
    fields_to_check = [
        ("title", context.get("title")),
        ("authors", context.get("authors")),
        ("license", context.get("license")),
        ("doi", context.get("doi")),
        ("n_subjects", context.get("n_subjects")),
        ("n_records", context.get("n_records")),
        ("n_channels", context.get("n_channels")),
        ("sampling_freqs", context.get("sampling_freqs")),
        ("modality", context.get("modality")),
        ("readme", context.get("readme")),
    ]

    filled = 0
    for _name, value in fields_to_check:
        if value:
            if isinstance(value, str):
                if value.strip() and value.strip() not in (
                    "—",
                    "Varies",
                    "Not specified",
                    "Not calculated",
                    "See source",
                ):
                    filled += 1
            elif isinstance(value, list) and len(value) > 0:
                filled += 1
            else:
                filled += 1

    percentage = int((filled / len(fields_to_check)) * 100)

    if percentage >= 90:
        return ("Complete", "success", percentage)
    if percentage >= 70:
        return ("Good", "primary", percentage)
    if percentage >= 50:
        return ("Partial", "warning", percentage)
    return ("Limited", "secondary", percentage)
