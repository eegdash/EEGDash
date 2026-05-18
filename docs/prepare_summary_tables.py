"""Generate summary tables and charts for EEGDash documentation.

This script fetches data from the EEGDash API and generates:
- Interactive charts (bubble, sankey, treemap, growth, clinical, ridgeline)
- Summary statistics JSON
- HTML summary table with DataTables integration
"""

import bisect
import concurrent.futures
import json
import math
import os
import sys
import textwrap
from argparse import ArgumentParser
from collections import Counter
from functools import partial
from pathlib import Path
from shutil import copyfile as _copyfile
from typing import Callable

import pandas as pd
from plot_dataset import (
    generate_clinical_stacked_bar,
    generate_dataset_bubble,
    generate_dataset_growth,
    generate_dataset_sankey,
    generate_dataset_treemap,
    generate_moabb_bubble,
    generate_modality_ridgeline,
)
from plot_dataset.utils import get_dataset_url as _get_dataset_url
from plot_dataset.utils import human_readable_size
from table_tag_utils import _normalize_values, wrap_tags

# Ensure eegdash package is importable (this script lives in docs/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegdash.dataset.registry import fetch_chart_data_from_api

# Directories
DOCS_DIR = Path(__file__).resolve().parent
STATIC_DATASET_DIR = DOCS_DIR / "source" / "_static" / "dataset_generated"
BUILD_STATIC_DIR = DOCS_DIR / "_build" / "html" / "_static" / "dataset_generated"
PACKAGE_CSV = DOCS_DIR.parent / "eegdash" / "dataset" / "dataset_summary.csv"

# API Configuration
API_BASE_URL = "https://data.eegdash.org/api"
DEFAULT_DATABASE = "eegdash"

# Number of workers for parallel chart generation
MAX_CHART_WORKERS = 6

# Tokens to treat as unknown/empty
_UNKNOWN_TOKENS = {"unknown", "nothing", "nan", "none", "null", ""}

# Canonical mappings for normalizing tag values
DATASET_CANONICAL_MAP = {
    "pathology": {
        "healthy controls": "Healthy",
        "healthy": "Healthy",
        "control": "Healthy",
        "clinical": "Clinical",
        "patient": "Clinical",
    },
    "modality": {
        "auditory": "Auditory",
        "visual": "Visual",
        "somatosensory": "Somatosensory",
        "multisensory": "Multisensory",
    },
    "record_modality": {
        "eeg": "EEG",
        "emg": "EMG",
    },
    "type": {
        "perception": "Perception",
        "decision making": "Decision-making",
        "decision-making": "Decision-making",
        "rest": "Rest",
        "resting state": "Resting-state",
        "sleep": "Sleep",
    },
}


# =============================================================================
# Sparkbar helpers — module-level so pre-commit's no-nested-functions hook
# doesn't trip when they're called from the table-build pipeline.
# =============================================================================


def _int_label(v: float) -> str:
    """Format an integer count (thousand-separated)."""
    return f"{int(v):,}"


def _size_label(v: float) -> str:
    """Format a byte count in the largest human-readable unit."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024:
            return f"{int(v)} {unit}" if unit in ("B", "KB") else f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} PB"


def _percentile_of(v: float, positives: list[float], n_pos: int) -> int:
    """Rank ``v`` against a pre-sorted list of positive values as a 0–100 percentile."""
    if v <= 0 or n_pos == 0:
        return 0
    below = bisect.bisect_left(positives, v)
    return int(round(100 * below / n_pos))


def _sparkbar_cell(
    v: object,
    *,
    peak_log: float,
    positives: list[float],
    n_pos: int,
    label_fn: Callable[[float], str],
    noun: str,
) -> str:
    """Render one cell's ``<span class="sparkbar">`` markup for the summary table."""
    vi = 0 if v is None or pd.isna(v) else float(v)
    pct = max(0, min(100, round(100 * math.log10(vi + 1) / peak_log, 1)))
    label = label_fn(vi) if vi > 0 else "—"
    if vi > 0:
        pctile = _percentile_of(vi, positives, n_pos)
        if noun == "size":
            title = f"{label} — larger than {pctile}% of datasets"
        else:
            title = f"{label} {noun} — higher than {pctile}% of datasets"
    else:
        title = "Not reported"
    return (
        f'<span class="sparkbar" style="--pct:{pct}%;" '
        f'title="{title}" aria-label="{title}">'
        f'<span class="sparkbar-label">{label}</span></span>'
    )


def _log_sparkbar(
    values: pd.Series,
    *,
    label_fn: Callable[[float], str],
    noun: str,
) -> pd.Series:
    """Render a numeric Series as log-scaled sparkbar HTML cells.

    ``label_fn(vi)`` formats the displayed number; the ``--pct`` CSS
    variable drives the bar width; the hover ``title`` adds distribution
    context ("above 72% of datasets") so the length encoding is
    self-explanatory without a separate legend. The synthesized ``Total``
    row is excluded when computing the peak so a single outlier can't
    compress the rest of the distribution to near-zero.
    """
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    body = values.loc[[i for i in values.index if i != "Total"]]
    peak = body.max() if len(body) else 0
    peak_log = math.log10(peak + 1) if peak > 0 else 1
    positives = sorted(
        float(v) for v in body if v is not None and not pd.isna(v) and v > 0
    )
    n_pos = len(positives)
    return values.apply(
        partial(
            _sparkbar_cell,
            peak_log=peak_log,
            positives=positives,
            n_pos=n_pos,
            label_fn=label_fn,
            noun=noun,
        )
    )


# =============================================================================
# File utilities
# =============================================================================


def copyfile(src, dst):
    """Robust copyfile that ignores SameFileError."""
    try:
        _copyfile(src, dst)
    except Exception as exc:
        if "are the same file" not in str(exc):
            raise exc


def copy_to_static(src: Path, filename: str = None):
    """Copy file to both source and build static directories."""
    if filename is None:
        filename = Path(src).name
    copyfile(src, STATIC_DATASET_DIR / filename)
    BUILD_STATIC_DIR.mkdir(parents=True, exist_ok=True)
    copyfile(src, BUILD_STATIC_DIR / filename)


# =============================================================================
# Chart generation
# =============================================================================


def _generate_chart_task(
    name: str,
    generator: Callable,
    df: pd.DataFrame,
    output_path: Path,
    **kwargs,
) -> tuple[str, Path | None, str | None]:
    """Generate a single chart - worker function for parallel execution."""
    try:
        output = generator(df.copy(), output_path, **kwargs)
        return (name, output, None)
    except Exception as exc:
        return (name, None, str(exc))


def generate_charts_parallel(
    df_raw: pd.DataFrame,
    target_dir: Path,
    x_var: str = "subjects",
) -> list[tuple[str, Path | None, str | None]]:
    """Generate all charts in parallel using ThreadPoolExecutor."""
    # Prepare bubble chart DataFrame
    df_bubble = df_raw.copy()
    if "subjects" not in df_bubble.columns and "n_subjects" in df_bubble.columns:
        df_bubble["subjects"] = df_bubble["n_subjects"]
    if "records" not in df_bubble.columns and "n_records" in df_bubble.columns:
        df_bubble["records"] = df_bubble["n_records"]

    tasks = [
        (
            "Bubble",
            generate_dataset_bubble,
            df_bubble,
            target_dir / "dataset_bubble.html",
            {"x_var": x_var},
        ),
        (
            "MOABB Bubble",
            generate_moabb_bubble,
            df_raw,
            target_dir / "dataset_moabb_bubble.html",
            {"color_by": "modality"},
        ),
        (
            "Sankey",
            generate_dataset_sankey,
            df_raw,
            target_dir / "dataset_sankey.html",
            {},
        ),
        (
            "Treemap",
            generate_dataset_treemap,
            df_raw,
            target_dir / "dataset_treemap.html",
            {},
        ),
        (
            "Growth",
            generate_dataset_growth,
            df_raw,
            target_dir / "dataset_growth.html",
            {},
        ),
        (
            "Clinical",
            generate_clinical_stacked_bar,
            df_raw,
            target_dir / "dataset_clinical.html",
            {},
        ),
        (
            "Ridgeline",
            generate_modality_ridgeline,
            df_raw,
            target_dir / "dataset_ridgeline.html",
            {},
        ),
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CHART_WORKERS
    ) as executor:
        futures = {
            executor.submit(_generate_chart_task, name, gen, df, path, **kwargs): name
            for name, gen, df, path, kwargs in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            chart_name = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                results.append((chart_name, None, str(exc)))

    return results


def process_chart_results(results: list[tuple[str, Path | None, str | None]]) -> None:
    """Process and report chart generation results, copy successful outputs to static."""
    for name, output_path, error in results:
        if error:
            print(f"[{name}] Skipped: {error}")
        elif output_path:
            copy_to_static(output_path)
            print(f"Generated: {output_path.name}")


# =============================================================================
# Summary statistics
# =============================================================================


def save_summary_stats(df_raw: pd.DataFrame, aggregations: dict | None = None) -> None:
    """Calculate and save summary stats for the documentation cards.

    Prefers the pre-computed ``aggregations`` block returned by the
    ``/datasets/chart-data`` endpoint as the source of truth for the
    per-column counts (datasets, subjects, recording files, sources,
    modalities). The server computes these in a single MongoDB pass; the
    client recomputation that used to live here produced identical
    numbers (verified via parity check) but added a second source of
    truth. ``duration_hours`` is not exposed in the server ``totals``
    block today, so it is still summed client-side from the dataframe.

    When ``aggregations`` is missing keys (e.g. the chart-data endpoint
    fell back to ``/datasets/summary`` and returned ``{}``), the
    function falls back to the original client-side computation per
    field so the build never silently emits 0 for counts.
    """
    aggregations = aggregations or {}
    totals = aggregations.get("totals") or {}
    source_counts = aggregations.get("source_counts") or {}
    modality_counts = aggregations.get("modality_counts") or {}

    # datasets_total: prefer server. Client fallback is len(df_raw),
    # which the parity check confirmed equals totals.datasets.
    datasets_total = totals.get("datasets")
    if datasets_total is None:
        datasets_total = len(df_raw)

    # subjects_total: prefer server. Client fallback is sum of n_subjects.
    subjects_total = totals.get("subjects")
    if subjects_total is None:
        n_subj_col = "n_subjects" if "n_subjects" in df_raw.columns else "subjects"
        subjects_total = int(
            pd.to_numeric(df_raw.get(n_subj_col, 0), errors="coerce").sum()
        )

    # recording_total: prefer server. Server exposes this as totals.files
    # (one entry per recorded file); the client equivalent was a sum over
    # the n_records column, which the parity check confirmed matches.
    recording_total = totals.get("files")
    if recording_total is None:
        n_rec_col = "n_records" if "n_records" in df_raw.columns else "records"
        recording_total = int(
            pd.to_numeric(df_raw.get(n_rec_col, 0), errors="coerce").sum()
        )

    # duration_hours: not in server totals; keep client-side. May appear
    # as total_duration_s (seconds) or duration_hours_total (hours).
    if "duration_hours_total" in df_raw.columns:
        duration_hours = int(
            pd.to_numeric(df_raw["duration_hours_total"], errors="coerce").sum()
        )
    elif "total_duration_s" in df_raw.columns:
        duration_hours = int(
            pd.to_numeric(df_raw["total_duration_s"], errors="coerce").sum() / 3600
        )
    else:
        duration_hours = 0

    # modalities_total: prefer server (one canonical bucket per recording
    # modality, e.g. EEG/MEG/IEEG/FNIRS/EMG). The client fallback
    # tokenizes record_modality strings and lowercases — verified to
    # yield the same set size.
    if modality_counts:
        modalities_total = len(modality_counts)
    else:
        unique_modalities: set[str] = set()
        col = (
            "record_modality"
            if "record_modality" in df_raw.columns
            else "record modality"
        )
        if col in df_raw.columns:
            for m in df_raw[col].dropna():
                unique_modalities.update(_normalize_values(m))
        unique_modalities = {m.strip().lower() for m in unique_modalities if m.strip()}
        modalities_total = len(unique_modalities)

    # sources_total: prefer server source_counts dict size. Client
    # fallback is df['source'].nunique().
    if source_counts:
        sources_total = len(source_counts)
    elif "source" in df_raw.columns:
        sources_total = int(df_raw["source"].nunique())
    else:
        sources_total = 0

    summary_stats = {
        "datasets_total": int(datasets_total),
        "subjects_total": int(subjects_total),
        "recording_total": int(recording_total),
        "duration_hours": int(duration_hours),
        "modalities_total": int(modalities_total),
        "sources_total": int(sources_total),
    }

    stats_path = STATIC_DATASET_DIR / "summary_stats.json"
    with open(stats_path, "w") as f:
        json.dump(summary_stats, f)
    print(f"Generated summary stats: {stats_path}")

    if BUILD_STATIC_DIR.exists():
        with open(BUILD_STATIC_DIR / "summary_stats.json", "w") as f:
            json.dump(summary_stats, f)


def generate_search_index(df_raw: pd.DataFrame) -> None:
    """Generate search index JSON for client-side fuzzy search with Fuse.js."""
    search_index = []
    excluded = {"test", "ds003380"}

    for _, row in df_raw.iterrows():
        dataset_id = str(row.get("dataset", "")).strip()
        if not dataset_id or dataset_id.lower() in excluded:
            continue

        # Extract and normalize tag arrays
        pathology_col = "Type Subject" if "Type Subject" in row else "pathology"
        modality_col = "modality of exp" if "modality of exp" in row else "modality"
        type_col = "type of exp" if "type of exp" in row else "type"
        record_mod_col = (
            "record_modality" if "record_modality" in row else "record modality"
        )

        pathology_tags = list(_normalize_values(row.get(pathology_col, "")))
        modality_tags = list(_normalize_values(row.get(modality_col, "")))
        type_tags = list(_normalize_values(row.get(type_col, "")))
        record_modality = list(_normalize_values(row.get(record_mod_col, "")))

        # Get numeric values safely
        n_subjects = pd.to_numeric(row.get("n_subjects", 0), errors="coerce")
        n_records = pd.to_numeric(row.get("n_records", 0), errors="coerce")
        n_tasks = pd.to_numeric(row.get("n_tasks", 0), errors="coerce")

        entry = {
            "id": dataset_id,
            "title": str(row.get("dataset_title", "")).strip()[:200],
            "source": str(row.get("source", "")).strip(),
            "subjects": int(n_subjects) if pd.notna(n_subjects) else 0,
            "records": int(n_records) if pd.notna(n_records) else 0,
            "tasks": int(n_tasks) if pd.notna(n_tasks) else 0,
            "size": str(row.get("size", "")).strip(),
            "pathology": pathology_tags,
            "modality": modality_tags,
            "type": type_tags,
            "recordModality": record_modality,
            # URL for dataset detail page. Root-relative so the link
            # resolves correctly regardless of the document URL variant
            # (e.g. "/dataset_summary/" vs "/dataset_summary.html") —
            # without the leading slash, crawlers and share cards that
            # load the page under a trailing-slash URL would resolve it
            # to /dataset_summary/api/dataset/… (1,100+ Ahrefs 404s).
            "url": f"/api/dataset/eegdash.dataset.{dataset_id.upper()}.html",
        }
        search_index.append(entry)

    # Save to static directory
    index_path = STATIC_DATASET_DIR / "search_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(search_index, f, separators=(",", ":"))
    print(f"Generated search index: {index_path} ({len(search_index)} datasets)")

    # Copy to build directory
    if BUILD_STATIC_DIR.exists():
        with open(BUILD_STATIC_DIR / "search_index.json", "w", encoding="utf-8") as f:
            json.dump(search_index, f, separators=(",", ":"))


# =============================================================================
# Table preparation
# =============================================================================


def _normalise_tag(token: str, canonical: dict) -> str:
    """Normalize a tag token using a canonical mapping.

    Args:
        token: The tag token to normalize.
        canonical: A dict mapping lowercase strings to canonical forms.

    Returns:
        The normalized tag string, or None if the token is unknown.

    """
    text = " ".join(token.replace("_", " ").split())
    lowered = text.lower()
    if lowered in _UNKNOWN_TOKENS:
        return None
    if lowered in canonical:
        return canonical[lowered]
    return text


def _tag_normalizer(kind: str):
    """Create a normalizer function for a specific tag kind."""
    canonical = {k.lower(): v for k, v in DATASET_CANONICAL_MAP.get(kind, {}).items()}
    return partial(_normalise_tag, canonical=canonical)


def parse_freqs(value) -> str:
    """Parse frequencies/channels list and return mode with * if variable."""
    if not value:
        return ""

    if isinstance(value, str):
        value = value.strip()
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

    counts = Counter()

    # Handle API aggregation format (list of dicts)
    if (
        isinstance(value, list)
        and value
        and isinstance(value[0], dict)
        and "val" in value[0]
    ):
        for item in value:
            val = item.get("val")
            count = item.get("count", 1)
            if val is not None:
                try:
                    counts[int(float(val))] += count
                except (ValueError, TypeError):
                    pass
    else:
        # Handle simple list or string
        freqs = []
        if isinstance(value, str):
            value = value.strip("[]")
            if not value:
                return ""
            parts = [p.strip() for p in value.split(",") if p.strip()]
            try:
                freqs = [float(f) for f in parts]
            except ValueError:
                pass
        elif isinstance(value, (int, float)) and not pd.isna(value):
            freqs = [value]
        elif isinstance(value, list):
            try:
                freqs = [float(f) for f in value if f is not None]
            except (ValueError, TypeError):
                pass

        for f in freqs:
            try:
                counts[int(f)] += 1
            except ValueError:
                pass

    if not counts:
        return ""

    most_common_val, _ = counts.most_common(1)[0]
    return f"{most_common_val}" if len(counts) == 1 else f"{most_common_val}*"


def get_dataset_url(name: str) -> str | None:
    """Get URL for dataset documentation page."""
    return _get_dataset_url(name)


def wrap_dataset_name(name: str, modality: str = "", n_subjects: int = 0) -> str:
    """Wrap dataset name with link to documentation.

    If *modality* or *n_subjects* are available, a ``title`` attribute is
    added so hovering the link shows a tooltip like
    ``"ABSeqMEG — EEG dataset, 20 subjects"``.
    """
    name = name.strip()
    url = get_dataset_url(name)
    if not url:
        return name.upper()
    parts = []
    if modality:
        parts.append(f"{modality} dataset")
    if n_subjects and n_subjects > 0:
        parts.append(f"{n_subjects} subjects")
    title_attr = ""
    if parts:
        tooltip = f"{name.upper()} — {', '.join(parts)}"
        title_attr = f' title="{tooltip}"'
    return f'<a href="{url}"{title_attr}>{name.upper()}</a>'


def _strip_unknown(value: object) -> object:
    """Strip unknown/empty values, returning empty string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str) and value.strip().lower() in _UNKNOWN_TOKENS:
        return ""
    return value


def _format_canonical(raw: object, author_year_value: str) -> str:
    """Flatten a canonical_name cell into a comma-separated display string.

    Parses the catalog value with the registry's shared parser (so list,
    JSON-encoded list, and legacy comma-string inputs all work), and
    strips ``author_year_value`` so the same token doesn't render in
    both the "Author (year)" and "Canonical" columns.
    """
    from eegdash.dataset.registry import _parse_canonical_names

    names = _parse_canonical_names(raw)
    if author_year_value:
        names = [n for n in names if n != author_year_value]
    return ", ".join(names)


def prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for HTML table rendering."""
    # Filter excluded datasets
    excluded = {"test", "ds003380"}
    df = df[~df["dataset"].str.lower().isin(excluded)].copy()

    df["dataset"] = df.apply(
        lambda r: wrap_dataset_name(
            r["dataset"],
            modality=str(r.get("record_modality", "")).strip(),
            n_subjects=int(r["n_subjects"]) if pd.notna(r.get("n_subjects")) else 0,
        ),
        axis=1,
    )

    # Ensure required columns exist
    for col, default in [
        ("dataset_title", ""),
        ("source", ""),
        ("record_modality", df.get("record modality", "")),
        ("Type Subject", df.get("pathology", "")),
        ("modality of exp", df.get("modality", "")),
        ("type of exp", df.get("type", "")),
        ("license", ""),
        ("size", ""),
        ("size_bytes", 0),
        ("n_records", 0),
        ("n_subjects", 0),
        ("n_tasks", 0),
        ("n_sessions", 0),
        ("nchans_set", ""),
        ("sampling_freqs", ""),
        # Naming pipeline outputs (canonical aliases + author-year)
        ("canonical_name", ""),
        ("name_source", ""),
        ("author_year", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    # Flatten canonical_name (usually a JSON-encoded list in the CSV) into
    # a comma-separated string for table rendering. :func:`_format_canonical`
    # reuses the registry's parser so we treat the column exactly like
    # the runtime catalog does, and strips the author-year alias so the
    # same token doesn't appear in both the "Author (year)" and
    # "Canonical" columns.
    df["canonical_name"] = [
        _format_canonical(c, str(ay).strip() if not pd.isna(ay) else "")
        for c, ay in zip(df["canonical_name"], df["author_year"])
    ]
    df["author_year"] = df["author_year"].apply(
        lambda v: ""
        if v is None or (isinstance(v, float) and pd.isna(v))
        else str(v).strip()
    )

    df = df[
        [
            "dataset",
            "dataset_title",
            "author_year",
            "canonical_name",
            "source",
            "record_modality",
            "n_records",
            "n_subjects",
            "n_tasks",
            "n_sessions",
            "nchans_set",
            "sampling_freqs",
            "size",
            "size_bytes",
            "Type Subject",
            "modality of exp",
            "type of exp",
            "license",
        ]
    ]

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(_strip_unknown)

    df = df.rename(
        columns={
            "modality of exp": "modality",
            "type of exp": "type",
            "Type Subject": "pathology",
            "record_modality": "record modality",
        }
    )

    # Convert numeric columns
    df["n_subjects"] = df["n_subjects"].astype(int)
    df["n_tasks"] = df["n_tasks"].astype(int)
    df["n_sessions"] = df["n_sessions"].astype(int)
    df["n_records"] = df["n_records"].astype(int)
    df["sampling_freqs"] = df["sampling_freqs"].apply(parse_freqs)
    df["nchans_set"] = df["nchans_set"].apply(parse_freqs)

    # Apply tag wrapping
    for col, kind in [
        ("pathology", "pathology"),
        ("modality", "modality"),
        ("type", "type"),
        ("record modality", "record_modality"),
    ]:
        normalizer = _tag_normalizer(kind)
        df[col] = df[col].apply(
            lambda v: wrap_tags(
                v, kind=f"dataset-{kind.replace('_', '-')}", normalizer=normalizer
            )
        )

    # Add total row
    df.loc["Total"] = df.sum(numeric_only=True)
    df.loc["Total", "dataset"] = f"Total {len(df) - 1} datasets"
    for col in [
        "nchans_set",
        "sampling_freqs",
        "source",
        "pathology",
        "modality",
        "type",
        "record modality",
        "author_year",
        "canonical_name",
    ]:
        df.loc["Total", col] = ""
    df.loc["Total", "size"] = human_readable_size(df.loc["Total", "size_bytes"])
    # Keep ``size_bytes`` through the end of prepare_table — ``main_from_api``
    # builds a log-scale sparkbar from it and drops the raw column itself.
    df.index = df.index.astype(str)

    return df


# =============================================================================
# HTML Table template
# =============================================================================

DATA_TABLE_TEMPLATE = textwrap.dedent(r"""
<!-- jQuery + DataTables core -->
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<link rel="stylesheet" href="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.css"/>
<script src="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.js"></script>

<!-- Buttons + SearchPanes + ColVis -->
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.colVis.min.js"></script>
<link rel="stylesheet" href="https://cdn.datatables.net/select/1.7.0/css/select.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/searchpanes/2.3.1/css/searchPanes.dataTables.min.css">
<script src="https://cdn.datatables.net/select/1.7.0/js/dataTables.select.min.js"></script>
<script src="https://cdn.datatables.net/searchpanes/2.3.1/js/dataTables.searchPanes.min.js"></script>

<style>
    table.sd-table tfoot td {
        font-weight: 600;
        border-top: 2px solid rgba(0,0,0,0.2);
        background: #f9fafb;
        padding: 8px 10px !important;
        vertical-align: middle;
    }
    table.sd-table tbody td:nth-child(-n+6),
    table.sd-table tfoot td:nth-child(-n+6),
    table.sd-table thead th:nth-child(-n+6) { text-align: left; }
    table.sd-table tbody td:nth-child(n+7),
    table.sd-table tfoot td:nth-child(n+7),
    table.sd-table thead th:nth-child(n+7) { text-align: right; }

    /* FOUC guard: before DataTables wraps the raw <table>, the browser paints
       all 735 rows and 15 columns with no toolbar, no hidden-column handling,
       and no column-width reservation — a big unstyled grid that flashes for
       several hundred ms. Hide the raw table (no `.dataTable` class yet) and
       show a skeleton placeholder in its place until DataTables initialises. */
    #datasets-table:not(.dataTable) { display: none !important; }
    .dt-loading-skeleton {
        height: 420px;
        border-radius: 8px;
        background:
            linear-gradient(180deg,
                rgba(15,23,42,0.06) 0 46px,
                transparent        46px 100%),
            linear-gradient(90deg,
                rgba(15,23,42,0.03) 0%,
                rgba(15,23,42,0.07) 50%,
                rgba(15,23,42,0.03) 100%);
        background-size: 100% 100%, 200% 100%;
        animation: dt-skel 1.4s ease-in-out infinite;
    }
    .dt-loading-skeleton[hidden] { display: none !important; }
    @keyframes dt-skel {
        0%   { background-position: 0 0, 100% 0; }
        100% { background-position: 0 0, -100% 0; }
    }
</style>

<div class="dt-loading-skeleton" role="status" aria-live="polite" aria-label="Loading dataset table"></div>
<TABLE_HTML>

<!-- DataTables UI behaviour lives in a real, lintable JS file (see
     docs/source/_static/js/dataset_table.js). This page is read by the
     `.. dataset-figure:: table` directive and inlined into
     dataset_summary.html, where `_static/js/...` resolves correctly
     relative to the docs root. -->
<script src="_static/js/dataset_table.js" defer></script>
""")


# =============================================================================
# Main functions
# =============================================================================


def _load_local_dataset_summary() -> pd.DataFrame:
    """Load local dataset_summary.csv as fallback."""
    csv_path = PACKAGE_CSV
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path, index_col=False, header=0, skipinitialspace=True)
    except Exception:
        return pd.DataFrame()


def main_from_api(target_dir: str, database: str = DEFAULT_DATABASE, limit: int = 1000):
    """Generate summary tables and charts from API data."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    STATIC_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_STATIC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching chart data from API (database: {database})...")
    df_raw, aggregations = fetch_chart_data_from_api(
        API_BASE_URL, database, limit=limit
    )

    if aggregations:
        print(
            f"  Pre-computed aggregations: {len(aggregations.get('modality_counts', {}))} modalities, "
            f"{len(aggregations.get('source_counts', {}))} sources"
        )

    # Try to enrich with local CSV for better modality info
    fallback_df = _load_local_dataset_summary()
    if not fallback_df.empty and not df_raw.empty:
        print("Enriching API data with local dataset_summary.csv modality info...")
        if "dataset" not in df_raw.columns and "dataset_id" in df_raw.columns:
            df_raw["dataset"] = df_raw["dataset_id"]

        enrich_cols = ["dataset", "record_modality", "modality of exp"]
        enrich_df = fallback_df[
            [c for c in enrich_cols if c in fallback_df.columns]
        ].copy()

        if not enrich_df.empty and "dataset" in enrich_df.columns:
            df_raw = df_raw.merge(
                enrich_df, on="dataset", how="left", suffixes=("", "_csv")
            )
            if "record_modality_csv" in df_raw.columns:
                if "recording_modality" not in df_raw.columns:
                    df_raw["recording_modality"] = pd.Series(
                        index=df_raw.index, dtype="object"
                    )
                df_raw["recording_modality"] = df_raw[
                    "record_modality_csv"
                ].combine_first(df_raw["recording_modality"])
                df_raw["record_modality"] = df_raw["record_modality_csv"]

    if df_raw.empty:
        print("No datasets fetched from API!")
        return

    # Generate charts
    print("Generating charts in parallel...")
    chart_results = generate_charts_parallel(df_raw, target_dir, x_var="subjects")
    process_chart_results(chart_results)

    # Save summary stats — feed in the server-side aggregations block so
    # the per-column counts (datasets / subjects / recording / sources /
    # modalities) come from MongoDB's single-pass aggregation rather
    # than a second-source-of-truth recomputation over the dataframe.
    # Falls back to client-side computation when aggregations is empty
    # (chart-data 404 path).
    save_summary_stats(df_raw, aggregations=aggregations)

    # Generate search index for fuzzy autocomplete
    generate_search_index(df_raw)

    # Generate HTML table
    df = prepare_table(df_raw)
    df["n_subjects"] = df["n_subjects"].astype(int)
    df["n_tasks"] = df["n_tasks"].astype(int)
    df["n_sessions"] = df["n_sessions"].astype(int)
    df["n_records"] = df["n_records"].astype(int)
    int_cols = ["n_subjects", "n_tasks", "n_sessions", "n_records"]
    df[int_cols] = df[int_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")

    # Encode numeric scale as a CSS-variable-driven sparkbar on the cell.
    # Position is a more accurate perceptual channel than bare integers for
    # "which datasets are the big ones" — the primary scan task.
    # Log scale is used throughout because every numeric column (subjects,
    # records, bytes) is heavy-tailed — a linear bar compresses the body
    # of the distribution to <1% and only the single outlier registers.
    # Apply bars before rename so we can reference the original columns.
    df["n_subjects"] = _log_sparkbar(
        df["n_subjects"], label_fn=_int_label, noun="subjects"
    )
    df["n_records"] = _log_sparkbar(
        df["n_records"], label_fn=_int_label, noun="records"
    )
    if "size_bytes" in df.columns:
        df["size"] = _log_sparkbar(df["size_bytes"], label_fn=_size_label, noun="size")

    df = df.rename(
        columns={
            "dataset": "Dataset",
            "source": "Source",
            "nchans_set": "Channels",
            "sampling_freqs": "Sampling rate",
            "size": "Size",
            "n_records": "Records",
            "n_subjects": "Subjects",
            "n_tasks": "Tasks",
            "n_sessions": "Sessions",
            "pathology": "Pathology",
            "modality": "Modality",
            "type": "Type",
            "record modality": "Recording",
            "author_year": "Author (year)",
            "canonical_name": "Canonical",
        }
    )
    df = df[
        [
            "Dataset",
            "Author (year)",
            "Canonical",
            "Source",
            "Recording",
            "Pathology",
            "Modality",
            "Type",
            "Records",
            "Subjects",
            "Tasks",
            "Sessions",
            "Channels",
            "Sampling rate",
            "Size",
        ]
    ]

    html_table = df.to_html(
        classes=["sd-table", "sortable"],
        index=False,
        escape=False,
        table_id="datasets-table",
    )
    html_table = DATA_TABLE_TEMPLATE.replace("<TABLE_HTML>", html_table)
    table_path = target_dir / "dataset_summary_table.html"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(html_table)
    copy_to_static(table_path)
    print(f"Generated: {table_path.name}")

    # Generate KDE ridgeline
    try:
        kde_path = target_dir / "dataset_kde_modalities.html"
        kde_output = generate_modality_ridgeline(df_raw, kde_path)
        if kde_output:
            copy_to_static(kde_output)
            print(f"Generated: {kde_output.name}")
    except Exception as exc:
        print(f"[KDE] Skipped: {exc}")

    print(f"\nAll outputs saved to: {target_dir}")
    print(f"Static files copied to: {STATIC_DATASET_DIR}")
    print(f"Build files copied to: {BUILD_STATIC_DIR}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate EEGDash documentation charts and tables"
    )
    parser.add_argument(
        "--target",
        dest="target_dir",
        type=str,
        default="build",
        help="Output directory",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATABASE,
        help=f"Database (default: {DEFAULT_DATABASE})",
    )
    args = parser.parse_args()

    limit = int(os.environ.get("EEGDASH_DOC_LIMIT", 1000))
    main_from_api(args.target_dir, args.database, limit=limit)
    print(f"Output directory: {args.target_dir}")
