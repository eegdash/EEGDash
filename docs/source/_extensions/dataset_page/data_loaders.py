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

import csv
import importlib
import inspect
import json
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from sphinx.util import logging

import eegdash.dataset as dataset_module
from eegdash.dataset import EEGDashDataset
from eegdash.dataset.snapshot import DatasetSnapshot

from ._constants import CLONE_ROOT, DEFAULT_METADATA_FIELDS

LOGGER = logging.getLogger(__name__)

_DATASET_DETAILS_CACHE: dict[str, dict[str, object]] = {}


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

    api_url = "https://data.eegdash.org/api/eegdash"

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
        "n_subjects": ds.get("demographics", {}).get("subjects_count"),
        "total_files": ds.get("total_files"),
        "n_tasks": len(ds.get("tasks", []) or []),
        "recording_modality": ds.get("recording_modality", []),
        "size_bytes": ds.get("size_bytes"),
        "source": _clean_value(ds.get("source")),
        "demographics": ds.get("demographics"),
        "nchans_counts": ds.get("nchans_counts"),
        "sfreq_counts": ds.get("sfreq_counts"),
        "total_duration_s": ds.get("total_duration_s"),
        "bad_channels_info": ds.get("bad_channels_info"),
    }

    external_links = ds.get("external_links", {}) or {}
    details["source_url"] = _clean_value(external_links.get("source_url"))

    return details


def _load_dataset_details(dataset_id: str) -> dict[str, object]:
    dataset_id = dataset_id.lower()
    cached = _DATASET_DETAILS_CACHE.get(dataset_id)
    if cached is not None:
        return cached

    details: dict[str, object] = {}

    dataset_dir = CLONE_ROOT / dataset_id
    desc_path = dataset_dir / "dataset_description.json"
    if desc_path.exists():
        try:
            data = json.loads(desc_path.read_text(encoding="utf-8"))
        except Exception:
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
        except Exception:
            data = {}
        details.setdefault("doi", _clean_value(data.get("dataset_doi")))
        details["source_url"] = _clean_value(data.get("source_url"))

    if not details.get("title") or not details.get("authors"):
        api_details = _fetch_dataset_details_from_api(dataset_id)
        for key, value in api_details.items():
            if value and not details.get(key):
                details[key] = value

    _DATASET_DETAILS_CACHE[dataset_id] = details
    return details


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

    return {
        "class_name": class_name,
        "dataset_id": dataset_id,
        "dataset_upper": dataset_id.upper(),
        "title": title,
        "year": year,
        "authors": details.get("authors", []),
        "license": license_text,
        "doi": doi,
        "canonical_names": canonical_names,
        "author_year_name": author_year_name,
        "source_url": _clean_value(details.get("source_url")),
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
        "nchans_counts": details.get("nchans_counts") or [],
        "sfreq_counts": details.get("sfreq_counts") or [],
        "total_duration_s": details.get("total_duration_s"),
        "bad_channels_info": details.get("bad_channels_info"),
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
