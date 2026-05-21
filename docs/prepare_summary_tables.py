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
    generate_api_study_explorer,
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
from eegdash.dataset.registry import fetch_chart_data_from_api, fetch_datasets_from_api

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
    database: str = DEFAULT_DATABASE,
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
            "API Volume",
            generate_api_study_explorer,
            df_raw,
            target_dir / "dataset_api_volume_scatter.html",
            {"api_url": API_BASE_URL, "database": database, "view": "volume"},
        ),
        (
            "API Coverage Matrix",
            generate_api_study_explorer,
            df_raw,
            target_dir / "dataset_api_coverage_matrix.html",
            {"api_url": API_BASE_URL, "database": database, "view": "matrix"},
        ),
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


def save_summary_stats(df_raw: pd.DataFrame) -> None:
    """Calculate and save summary stats for the documentation cards."""
    unique_modalities = set()
    col = (
        "record_modality" if "record_modality" in df_raw.columns else "record modality"
    )
    if col in df_raw.columns:
        for m in df_raw[col].dropna():
            unique_modalities.update(_normalize_values(m))
    unique_modalities = {m.strip().lower() for m in unique_modalities if m.strip()}

    n_subj_col = "n_subjects" if "n_subjects" in df_raw.columns else "subjects"
    subjects_total = int(
        pd.to_numeric(df_raw.get(n_subj_col, 0), errors="coerce").sum()
    )

    n_rec_col = "n_records" if "n_records" in df_raw.columns else "records"
    recording_total = int(
        pd.to_numeric(df_raw.get(n_rec_col, 0), errors="coerce").sum()
    )

    # Duration may appear as total_duration_s (seconds) or duration_hours_total (hours)
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

    summary_stats = {
        "datasets_total": len(df_raw),
        "subjects_total": subjects_total,
        "recording_total": recording_total,
        "duration_hours": duration_hours,
        "modalities_total": len(unique_modalities),
        "sources_total": df_raw["source"].nunique()
        if "source" in df_raw.columns
        else 0,
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

<script>
function tagsArrayFromHtml(html) {
    if (html == null) return [];
    if (typeof html === 'number') return [String(html)];
    if (typeof html === 'string' && html.indexOf('<') === -1) return [html.trim()];
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    const tags = Array.from(tmp.querySelectorAll('.tag')).map(el => (el.textContent || '').trim());
    const text = tmp.textContent.trim();
    return tags.length ? tags : (text ? [text] : []);
}

function parseSizeToBytes(text) {
    if (!text) return 0;
    const m = String(text).trim().match(/([\d,.]+)\s*(TB|GB|MB|KB|B)/i);
    if (!m) return 0;
    const value = parseFloat(m[1].replace(/,/g, ''));
    const unit = m[2].toUpperCase();
    const factor = { B:1, KB:1024, MB:1024**2, GB:1024**3, TB:1024**4 }[unit] || 1;
    return value * factor;
}

document.addEventListener('DOMContentLoaded', function () {
    const table = document.getElementById('datasets-table');
    if (!table || !window.jQuery || !window.jQuery.fn.DataTable) return;

    const $table = window.jQuery(table);
    if (window.jQuery.fn.DataTable.isDataTable(table)) return;

    const $tbody = $table.find('tbody');
    // Cell 0 now reads "Total 735 datasets" (rendered by prepare_table);
    // match on a leading "Total" token so the row moves into <tfoot> and
    // survives DataTables search/filter.
    const $total = $tbody.find('tr').filter(function(){
        return /^\s*Total\b/.test(window.jQuery(this).find('td').eq(0).text());
    });
    if ($total.length) {
        let $tfoot = $table.find('tfoot');
        if (!$tfoot.length) $tfoot = window.jQuery('<tfoot/>').appendTo($table);
        $total.appendTo($tfoot);
    }

    // Columns that expose a SearchPanes filter. Limit to low-cardinality
    // categorical fields (the four tag columns + Source); numeric columns
    // and free-text identity columns would produce a pick-list with
    // hundreds of options and swamp the filter panel.
    const FILTER_HEADERS = new Set(['source', 'recording', 'pathology', 'modality', 'type']);
    const FILTER_COLS = (function(){
        const cols = [];
        document.querySelectorAll('#datasets-table thead th').forEach((th, i) => {
            const t = (th.textContent || '').trim().toLowerCase();
            if (FILTER_HEADERS.has(t)) cols.push(i);
        });
        return cols;
    })();
    const TAG_COLS = (function(){
        const tagHeaders = new Set(['recording', 'pathology', 'modality', 'type']);
        const cols = [];
        $table.find('thead th').each(function(i){
            if (tagHeaders.has(window.jQuery(this).text().trim().toLowerCase())) cols.push(i);
        });
        return cols;
    })();
    const sizeIdx = (function(){
        let idx = -1;
        $table.find('thead th').each(function(i){
            const t = window.jQuery(this).text().trim().toLowerCase();
            if (t === 'size on disk' || t === 'size') idx = i;
        });
        return idx;
    })();

    // Power-user / rarely-distinguishing columns are hidden by default.
    // They stay available via the Columns button.
    const hiddenByDefaultIdxs = (function() {
        const targets = [];
        const wanted = new Set([
            'author (year)',
            'canonical',
            'source',          // only 2 values (openneuro / nemar) — low info
            'sessions',        // most datasets are 1 — low info density
        ]);
        const headers = table.querySelectorAll('thead th');
        for (let i = 0; i < headers.length; i++) {
            if (wanted.has((headers[i].textContent || '').trim().toLowerCase())) targets.push(i);
        }
        return targets;
    })();

    const dataTable = $table.DataTable({
        dom: 'Blfrtip',
        paging: false,
        searching: true,
        info: false,
        // Default sort = Dataset column (0) ascending; Canonical is hidden.
        order: [[0, 'asc']],
        language: {
            search: 'Filter dataset:',
            searchPanes: { collapse: { 0: 'Filters', _: 'Filters (%d)' } },
            emptyTable: '<span class="no-match">No datasets in this catalogue yet.</span>',
            zeroRecords: '<span class="no-match"><strong>No matches.</strong> Try clearing the search box or the Filters chip, or broaden your query.</span>'
        },
        buttons: [
            {
                extend: 'searchPanes',
                text: 'Filters',
                config: {
                    cascadePanes: true,
                    viewTotal: true,
                    layout: 'columns-4',
                    initCollapsed: false,
                    // Restrict the filter panel to the categorical columns.
                    // The `columns` array is resolved at runtime from the
                    // current table, below.
                    columns: FILTER_COLS,
                }
            },
            {
                extend: 'colvis',
                text: 'Columns',
                columns: ':not(:first-child)'  // don't let users hide the Dataset column
            }
        ],
        columnDefs: (function(){
            const findIdx = (label) => {
                let i = -1;
                $table.find('thead th').each(function (k) {
                    if ((this.textContent || '').trim().toLowerCase() === label) i = k;
                });
                return i;
            };
            // By default SearchPanes shows a pane for every column. That
            // would include Records (734 unique values), Canonical (~300),
            // and other high-cardinality columns — swamping the filter UI.
            // Only include the low-cardinality categorical columns that
            // are in FILTER_COLS; omit columnDefs for the rest and instead
            // turn SearchPanes off at the button level with a `columns`
            // whitelist. We still attach searchPanes opts on FILTER_COLS so
            // each pane can customise orthogonal / threshold.
            const defs = [
                { searchPanes: { show: true }, targets: FILTER_COLS },
            ];
            if (TAG_COLS.length) {
                defs.push({
                    targets: TAG_COLS,
                    searchPanes: { show: true, orthogonal: 'sp' },
                    render: function(data, type) { return type === 'sp' ? tagsArrayFromHtml(data) : data; }
                });
            }
            if (sizeIdx !== -1) {
                defs.push({
                    targets: sizeIdx,
                    render: function(data, type) {
                        return (type === 'sort' || type === 'type') ? parseSizeToBytes(data) : data;
                    }
                });
            }

            // Records / Subjects columns render as sparkbar HTML; the raw
            // `data` is that HTML, whose innerText is a comma-formatted
            // number. DataTables' default sort treats it as a string, so
            // "79" beats "40,360" in descending order. Parse it as an int
            // for sort/type, keep the HTML for display.
            ['records', 'subjects'].forEach(label => {
                const idx = findIdx(label);
                if (idx === -1) return;
                defs.push({
                    targets: idx,
                    render: function(data, type) {
                        if (type !== 'sort' && type !== 'type') return data;
                        const m = String(data).match(/>(\s*[\d,]+|—)\s*</);
                        if (!m) return 0;
                        const token = m[1].trim();
                        return token === '—' ? -1 : parseInt(token.replace(/,/g, ''), 10);
                    }
                });
            });

            // Channels + Sampling rate carry a trailing "*" for values that
            // are medians across recordings. DataTables sees the star as a
            // non-numeric suffix, falls back to string sort, and "999" ends
            // up above "8192". Strip the "*" and sort as int.
            ['channels', 'sampling rate'].forEach(label => {
                const idx = findIdx(label);
                if (idx === -1) return;
                defs.push({
                    targets: idx,
                    render: function(data, type) {
                        if (type !== 'sort' && type !== 'type') return data;
                        const s = String(data ?? '').trim();
                        if (!s || s === '—') return -1;
                        const n = parseInt(s.replace(/[*,\s]/g, ''), 10);
                        return Number.isNaN(n) ? -1 : n;
                    }
                });
            });
            if (hiddenByDefaultIdxs.length) {
                defs.push({ targets: hiddenByDefaultIdxs, visible: false });
            }
            // Reserve width for columns whose worst-case content is two
            // chips on one line (Type: "Clinical Intervention"; Recording:
            // "EEG + MEG"). Without a reservation these columns collapse to
            // ~80px and the multi-chip rows wrap to 2 lines.
            // Column-width reservations — pinned widths force DataTables to
            // give each column enough room for its worst-case content:
            //   - Dataset ID column fits the longest ID ("EEG2025R10MINI")
            //   - Sparkbar columns (records / subjects / size) need >= the
            //     ~150px sparkbar min-width plus padding
            //   - Tag columns need room for multi-chip values on one line
            //   - Scalar numeric columns (tasks / channels / sampling-rate)
            //     stay compact.
            const widthMap = {
                dataset: '8rem',
                // Hidden-by-default but reserve width anyway; without this,
                // revealing the column via ColVis collapses its cell to
                // ~80 px and long compound values (e.g. Canonical lists
                // like "BNCI2015_P300 BNCI2014_004 Schirrmeister2017 …")
                // wrap into 15-line cells that break the row rhythm.
                'author (year)': '9rem',
                canonical: '16rem',
                source: '7rem',
                recording: '9rem',
                pathology: '14rem',
                modality: '12rem',
                type: '12rem',
                records: '10rem',
                subjects: '10rem',
                tasks: '5rem',
                sessions: '6rem',
                channels: '6rem',
                'sampling rate': '6rem',
                size: '10rem',
            };
            Object.entries(widthMap).forEach(([label, w]) => {
                const idx = findIdx(label);
                if (idx !== -1) defs.push({ targets: idx, width: w });
            });
            return defs;
        })()
    });

    // Recompute column widths whenever a column is shown/hidden so newly
    // revealed columns don't overflow into their neighbours.
    dataTable.on('column-visibility.dt', function () {
        dataTable.columns.adjust();
    });

    // When the ColVis popup opens it is appended inside the .dt-buttons
    // container below the Columns chip. On short viewports the list
    // (14 items, ~580 px) overflows below the fold with most rows
    // unreachable. When that would happen, flip the popup so it opens
    // *upward* from the chip's top edge and cap its height to stay within
    // the viewport; also publish --dtc-top-offset so the CSS can size it.
    const positionCollection = function (node) {
        if (!(node instanceof HTMLElement)) return;
        if (!node.classList.contains('dt-button-collection')) return;
        // Reset any previous flip so re-measuring reflects natural position.
        node.style.removeProperty('transform');
        node.classList.remove('dtc-flipped');
        const rect = node.getBoundingClientRect();
        const vh = window.innerHeight;
        const safeBottom = 16; // 1rem gap before viewport edge
        if (rect.bottom > vh - safeBottom) {
            // Need to shift the menu up by (overflow amount + safeBottom)
            // so its bottom lands inside the viewport.
            const shift = Math.ceil(rect.bottom - vh + safeBottom);
            node.style.transform = `translateY(-${shift}px)`;
            node.classList.add('dtc-flipped');
        }
        // Publish the final top offset so CSS max-height can guarantee the
        // list fits even if content grew after the shift.
        const finalTop = node.getBoundingClientRect().top;
        node.style.setProperty('--dtc-top-offset', Math.max(8, finalTop) + 'px');
    };
    new MutationObserver(function (records) {
        records.forEach(function (rec) {
            rec.addedNodes.forEach(function (node) {
                if (node instanceof HTMLElement) {
                    positionCollection(node);
                    node.querySelectorAll?.('.dt-button-collection').forEach(positionCollection);
                }
            });
        });
    }).observe(document.body, { childList: true, subtree: true });

    // Search input: add a placeholder so first-time users know what the
    // field filters on, and an aria-label for screen readers.
    const $searchInput = $table.closest('.dataTables_wrapper').find('.dataTables_filter input');
    if ($searchInput.length) {
        $searchInput.attr('placeholder', 'e.g. DS000117, healthy, visual');
        $searchInput.attr('aria-label', 'Filter datasets by any field');
    }

    // Tag each column's td+th with a data-col-key attribute derived from
    // its header text. CSS can then hook specific columns (e.g. the Type
    // column with multi-tag cells) without fragile nth-child indices.
    //
    // Use the DataTables API (not `$table.find('thead th')`): by this
    // point DataTables has already removed hidden columns from the DOM,
    // so a raw DOM query would skip them. `dataTable.columns()` walks ALL
    // columns, visible or not, so when ColVis reveals one later its cells
    // already carry the expected attribute.
    dataTable.columns().every(function () {
        const headerEl = this.header();
        const key = (headerEl.textContent || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
        if (!key) return;
        headerEl.setAttribute('data-col-key', key);
        this.nodes().each(function (cell) {
            if (cell) cell.setAttribute('data-col-key', key);
        });
    });

    // Screen-reader hint on each Dataset ID link — without it, SRs just
    // announce the raw identifier ("DS000117") with no context.
    $table.find('tbody td:first-child a').each(function () {
        const id = (this.textContent || '').trim();
        if (id) this.setAttribute('aria-label', 'Open dataset ' + id + ' details');
    });

    // Click-to-filter on tag chips: clicking a tag filters the table to
    // rows whose same column contains that EXACT tag (not a substring).
    // Each tag in a multi-tag cell is wrapped in its own `<span>`, so the
    // ">TAG<" bracket pair uniquely identifies a complete tag text and
    // won't bleed "Clinical" into "Clinical Intervention" rows (which
    // contain `>Clinical</span> <span …>Intervention<` — "Clinical" is
    // followed by `<` either way, but also `>Intervention<` is there
    // separately, so we match on the bracketed text specifically).
    const escapeRe = s => String(s).replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
    $table.on('click', 'tbody .tag', function (e) {
        e.preventDefault();
        const cell = this.closest('td');
        if (!cell) return;
        const colKey = cell.dataset.colKey;
        if (!colKey) return;
        const col = dataTable.column('th[data-col-key="' + colKey + '"]');
        if (!col.length) return;
        const val = (this.textContent || '').trim();
        if (!val) return;
        // Match `>VALUE<` which only appears around the text of a
        // complete tag span. For cells containing a single plain text
        // token (no <span>) we also allow the value as the entire cell.
        const regex = '>' + escapeRe(val) + '<|^' + escapeRe(val) + '$';
        const current = col.search();
        if (current === regex) {
            col.search('').draw();
        } else {
            col.search(regex, true, false).draw();
        }
    });

    // Mount a "Clear filters" chip in the toolbar. Hidden at rest, shown
    // whenever any filter narrows the view (tag click, search box, column
    // search, or SearchPanes selection). One click wipes all three so the
    // user always has an obvious escape hatch. Added once; draw.dt below
    // only toggles its visibility.
    const clearChip = document.createElement('button');
    clearChip.type = 'button';
    clearChip.className = 'dt-button dt-clear-filters';
    clearChip.innerHTML = '\u2715&nbsp;Clear filters';
    clearChip.hidden = true;
    clearChip.setAttribute('aria-label', 'Clear all filters');
    clearChip.addEventListener('click', function () {
        // Reset column searches
        dataTable.columns().every(function () { this.search(''); });
        // Reset global search
        dataTable.search('');
        // Reset SearchPanes selections if the plugin is initialised
        if (dataTable.searchPanes && typeof dataTable.searchPanes.clearSelections === 'function') {
            try { dataTable.searchPanes.clearSelections(); } catch (e) {}
        }
        dataTable.draw();
    });
    const buttonsHost = $table.closest('.dataTables_wrapper').find('.dt-buttons')[0];
    if (buttonsHost) buttonsHost.appendChild(clearChip);

    // Capture the original Total row values on first draw so we can restore
    // them when all filters are cleared. The Tfoot is in the DOM before
    // DataTables takes over, so these reads give us the "unfiltered" state.
    const $tfootCells = $table.find('tfoot tr').children();
    const originalTotal = {};
    $tfootCells.each(function (i) { originalTotal[i] = this.innerText; });

    // Helper: sum of integers parsed from a column's rendered cells that
    // are currently in the filtered / searched subset. Note the options
    // go on the `column()` selector — passing them to .nodes() is a no-op
    // in DataTables 1.13 and quietly iterates every row.
    const sumColumn = (colKey, parser) => {
        let total = 0;
        const col = dataTable.column('th[data-col-key="' + colKey + '"]',
                                     { search: 'applied' });
        if (!col.length) return null;
        col.nodes().each(function (cell) {
            if (!cell) return;
            const v = parser(cell);
            if (Number.isFinite(v)) total += v;
        });
        return total;
    };
    const formatInt = n => n.toLocaleString('en-US');
    const formatBytes = n => {
        const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
        let v = n, u = 0;
        while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
        return u <= 1 ? Math.round(v) + ' ' + units[u] : v.toFixed(1) + ' ' + units[u];
    };
    const parseIntFromSparkbar = cell => {
        const lab = cell.querySelector('.sparkbar-label');
        if (!lab) return 0;
        const t = lab.textContent.trim();
        if (!t || t === '—') return 0;
        return parseInt(t.replace(/,/g, ''), 10);
    };
    const parseBytesFromSparkbar = cell => {
        const lab = cell.querySelector('.sparkbar-label');
        if (!lab) return 0;
        const t = lab.textContent.trim();
        if (!t || t === '—') return 0;
        const m = t.match(/([\d,.]+)\s*(TB|GB|MB|KB|B)/i);
        if (!m) return 0;
        const factor = { B:1, KB:1024, MB:1024**2, GB:1024**3, TB:1024**4, PB:1024**5 }[m[2].toUpperCase()] || 1;
        return parseFloat(m[1].replace(/,/g, '')) * factor;
    };
    const parseIntFromText = cell => {
        const t = (cell.textContent || '').trim().replace(/\*/g, '').replace(/,/g, '');
        return parseInt(t, 10);
    };

    // Resolve column key -> DOM tfoot cell on demand. The tfoot is reordered
    // the first time DataTables redraws (row sort) and its cells need to be
    // matched via the header order, not via "whatever tbody's first row
    // looks like at init".
    const getFootCell = (key) => {
        const visibleHeaders = [...$table.find('thead tr:last-child th')];
        const idx = visibleHeaders.findIndex(th => th.dataset.colKey === key);
        if (idx === -1) return null;
        return $table.find('tfoot tr').children()[idx] || null;
    };

    // Re-render Total row to reflect current (possibly filtered) subset.
    const recomputeTotalRow = () => {
        const visibleCount = dataTable.rows({ search: 'applied' }).count();
        const allCount = dataTable.rows().count();
        const isFiltered = visibleCount < allCount;
        const firstCell = $tfootCells[0];
        if (firstCell) {
            firstCell.innerText = isFiltered
                ? `Showing ${formatInt(visibleCount)} of ${formatInt(allCount)}`
                : originalTotal[0];
        }
        // Re-aggregate numeric columns from visible cells.
        const colMap = {
            records:  { parse: parseIntFromSparkbar,   fmt: formatInt },
            subjects: { parse: parseIntFromSparkbar,   fmt: formatInt },
            tasks:    { parse: parseIntFromText,       fmt: formatInt },
            // Channels / Sampling rate use medians per-dataset; summing across
            // datasets is meaningless, so leave them blank while filtered.
            size:     { parse: parseBytesFromSparkbar, fmt: formatBytes },
        };
        Object.entries(colMap).forEach(([key, spec]) => {
            const footCell = getFootCell(key);
            if (!footCell) return;
            const domIdx = [...footCell.parentElement.children].indexOf(footCell);
            if (!isFiltered) {
                footCell.innerText = originalTotal[domIdx];
                return;
            }
            const sum = sumColumn(key, spec.parse);
            footCell.innerText = sum ? spec.fmt(sum) : '';
        });
    };

    // After every draw, reflect active column filters on:
    //  (a) the column header (adds .dt-col-filtered so CSS can tint it)
    //  (b) body tags whose text matches the active filter (adds
    //     .tag-filter-active so they read as "currently applied").
    //  (c) the Clear-filters chip visibility + label ("Clear 2 filters").
    //  (d) the Total row numbers reflecting the currently visible subset.
    dataTable.on('draw.dt', function () {
        let activeColCount = 0;
        dataTable.columns().every(function () {
            const header = this.header();
            const colKey = header.dataset.colKey;
            const raw = this.search();
            const active = raw !== '';
            if (active) activeColCount++;
            header.classList.toggle('dt-col-filtered', active);
            if (!colKey) return;
            // The search value is now a tag-boundary regex of the form
            // ">Foo<|^Foo$"; recover the plain tag text "Foo" to match
            // against tags in the column.
            let filterText = raw;
            const m = raw.match(/^>(.+?)<\|\^\1\$$/);
            if (m) filterText = m[1].replace(/\\([-/\\^$*+?.()|\[\]{}])/g, '$1');
            $table.find('tbody td[data-col-key="' + colKey + '"] .tag').each(function () {
                this.classList.toggle('tag-filter-active',
                    active && (this.textContent || '').trim() === filterText);
            });
        });
        const globalSearchActive = dataTable.search() !== '';
        const visibleCount = dataTable.rows({ search: 'applied' }).count();
        const allCount = dataTable.rows().count();
        const subsetActive = visibleCount < allCount;
        // Treat global search / panes / subset narrowing as one additional
        // "implicit" filter for the count purposes if no column filter
        // explains the row reduction.
        let labelCount = activeColCount;
        if (!activeColCount && subsetActive) labelCount = 1;
        if (globalSearchActive && !activeColCount) labelCount = 1;
        clearChip.innerHTML = '\u2715&nbsp;Clear '
            + (labelCount > 1 ? labelCount + ' filters' : 'filter' + (labelCount === 1 ? '' : 's'));
        // Show chip if any filter signal is live.
        clearChip.hidden = !(activeColCount || globalSearchActive || subsetActive);
        recomputeTotalRow();
    });

    // Force a few columns to reserve space for their worst-case content
    // (compound tags like "Clinical Intervention" or "EEG + MEG"). The
    // columnDefs `width` option is ignored once DataTables has auto-sized,
    // so we set inline styles directly on the <th> and re-adjust.
    const $typeTh = $table.find('thead th[data-col-key="type"]');
    if ($typeTh.length) $typeTh.css('min-width', '13rem');
    const $recTh = $table.find('thead th[data-col-key="recording"]');
    if ($recTh.length) $recTh.css('min-width', '9rem');
    dataTable.columns.adjust();

    // Drive the right-edge "scroll to see more" fade via an `.is-overflowing`
    // class on BOTH the DataTables wrapper and the outer <figure>. The
    // ::after affordance is anchored on the figure (outside the scroll
    // container) so it stays pinned to the visible right edge when users
    // scroll the table horizontally.
    const wrapper = table.closest('.dataTables_wrapper') || document.getElementById(table.id + '_wrapper');
    const figure = wrapper && wrapper.closest('.eegdash-figure');
    const syncOverflow = function () {
        if (!wrapper) return;
        const sx = wrapper.scrollWidth > wrapper.clientWidth + 1;
        wrapper.classList.toggle('is-overflowing', sx);
        if (figure) figure.classList.toggle('is-overflowing', sx);
    };
    syncOverflow();
    window.addEventListener('resize', syncOverflow, { passive: true });
    dataTable.on('column-visibility.dt', function () { window.setTimeout(syncOverflow, 0); });

    // Header clicks follow DataTables' default: sort. A previous iteration
    // added a second click handler on the tag-column headers that opened
    // SearchPanes — it conflicted with the sort-on-click convention and its
    // tooltip ("Click to filter this column") lied about what clicking did.
    // Filtering is now exclusively reachable via the Filters chip above.

    // FOUC guard: tear down the loading skeleton once DataTables has wrapped
    // the table. The raw <table> reveal is driven by CSS (it gains the
    // `.dataTable` class at init), so all we need to do here is drop the
    // placeholder. Done last so any thrown error above leaves the skeleton
    // up — a visible placeholder beats a blank card.
    document
      .querySelectorAll('.dt-loading-skeleton')
      .forEach(function (el) { el.hidden = true; });
});
</script>
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


def _refresh_package_csv(database: str = DEFAULT_DATABASE) -> None:
    """Refresh the package-level dataset_summary.csv from the API.

    This ensures that ``register_openneuro_datasets()`` (called at import
    time by Sphinx) sees the same datasets that appear in the HTML summary
    tables, preventing broken links in the generated documentation.
    """
    print("Refreshing package dataset_summary.csv from API...")
    df_api = fetch_datasets_from_api(API_BASE_URL, database, force_refresh=True)
    if df_api.empty:
        print("  API returned no data; keeping existing CSV.")
        return

    # Preserve EEG2025 competition entries — the summary endpoint may omit them.
    try:
        df_existing = pd.read_csv(PACKAGE_CSV, comment="#", skip_blank_lines=True)
        api_datasets = set(df_api["dataset"])
        missing_eeg2025 = df_existing[
            df_existing["dataset"].str.startswith("EEG2025", na=False)
            & ~df_existing["dataset"].isin(api_datasets)
        ]
        if not missing_eeg2025.empty:
            df_api = pd.concat([df_api, missing_eeg2025], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as exc:
        print(f"  Could not read existing CSV for EEG2025 merge: {exc}")

    df_api.to_csv(PACKAGE_CSV, index=False)
    print(f"  Updated {PACKAGE_CSV.name}: {len(df_api)} datasets")


def main_from_api(target_dir: str, database: str = DEFAULT_DATABASE, limit: int = 1000):
    """Generate summary tables and charts from API data."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    STATIC_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # Refresh the package CSV so Sphinx class registration matches the API
    _refresh_package_csv(database)

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
    chart_results = generate_charts_parallel(
        df_raw, target_dir, x_var="subjects", database=database
    )
    process_chart_results(chart_results)

    # Save summary stats
    save_summary_stats(df_raw)

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
