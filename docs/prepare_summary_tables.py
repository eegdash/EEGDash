"""Generate summary tables and charts for EEGDash documentation.

This script fetches data from the EEGDash API and generates:
- Interactive charts (bubble, sankey, treemap, growth, clinical, ridgeline)
- Summary statistics JSON
- HTML summary table with DataTables integration
"""

import concurrent.futures
import json
import os
import textwrap
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile as _copyfile
from typing import Any, Callable

import pandas as pd
from plot_dataset import (
    generate_clinical_stacked_bar,
    generate_dataset_bubble,
    generate_dataset_growth,
    generate_dataset_sankey,
    generate_dataset_treemap,
    generate_modality_ridgeline,
)
from plot_dataset.utils import human_readable_size
from table_tag_utils import _normalize_values, wrap_tags

# Directories
DOCS_DIR = Path(__file__).resolve().parent
STATIC_DATASET_DIR = DOCS_DIR / "source" / "_static" / "dataset_generated"
BUILD_STATIC_DIR = DOCS_DIR / "_build" / "html" / "_static" / "dataset_generated"

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

    summary_stats = {
        "datasets_total": len(df_raw),
        "subjects_total": subjects_total,
        "recording_total": recording_total,
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


# =============================================================================
# Table preparation
# =============================================================================


def _tag_normalizer(kind: str):
    """Create a normalizer function for a specific tag kind."""
    canonical = {k.lower(): v for k, v in DATASET_CANONICAL_MAP.get(kind, {}).items()}

    def _normalise(token: str) -> str:
        text = " ".join(token.replace("_", " ").split())
        lowered = text.lower()
        if lowered in _UNKNOWN_TOKENS:
            return None
        if lowered in canonical:
            return canonical[lowered]
        return text

    return _normalise


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

    from collections import Counter

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
    from plot_dataset.utils import get_dataset_url

    return get_dataset_url(name)


def wrap_dataset_name(name: str) -> str:
    """Wrap dataset name with link to documentation."""
    name = name.strip()
    url = get_dataset_url(name)
    if not url:
        return name.upper()
    return f'<a href="{url}">{name.upper()}</a>'


def prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for HTML table rendering."""
    # Filter excluded datasets
    excluded = {"test", "ds003380"}
    df = df[~df["dataset"].str.lower().isin(excluded)].copy()

    df["dataset"] = df["dataset"].apply(wrap_dataset_name)

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
        ("nchans_set", ""),
        ("sampling_freqs", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    def _strip_unknown(value: object) -> object:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, str) and value.strip().lower() in _UNKNOWN_TOKENS:
            return ""
        return value

    df = df[
        [
            "dataset",
            "dataset_title",
            "source",
            "record_modality",
            "n_records",
            "n_subjects",
            "n_tasks",
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
    ]:
        df.loc["Total", col] = ""
    df.loc["Total", "size"] = human_readable_size(df.loc["Total", "size_bytes"])
    df = df.drop(columns=["size_bytes"])
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

<!-- Buttons + SearchPanes -->
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
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
</style>

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
    const $total = $tbody.find('tr').filter(function(){
        return window.jQuery(this).find('td').eq(0).text().trim() === 'Total';
    });
    if ($total.length) {
        let $tfoot = $table.find('tfoot');
        if (!$tfoot.length) $tfoot = window.jQuery('<tfoot/>').appendTo($table);
        $total.appendTo($tfoot);
    }

    const FILTER_COLS = [1, 2, 3, 4, 5, 6, 7, 8];
    const TAG_COLS = (function(){
        const tagHeaders = new Set(['record modality', 'pathology', 'modality', 'type']);
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

    const dataTable = $table.DataTable({
        dom: 'Blfrtip',
        paging: false,
        searching: true,
        info: false,
        language: {
            search: 'Filter dataset:',
            searchPanes: { collapse: { 0: 'Filters', _: 'Filters (%d)' } }
        },
        buttons: [{
            extend: 'searchPanes',
            text: 'Filters',
            config: { cascadePanes: true, viewTotal: true, layout: 'columns-4', initCollapsed: false }
        }],
        columnDefs: (function(){
            const defs = [{ searchPanes: { show: true }, targets: FILTER_COLS }];
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
            return defs;
        })()
    });

    $table.find('thead th').each(function (i) {
        if ([1, 2, 3, 4, 5].indexOf(i) === -1) return;
        window.jQuery(this).css('cursor','pointer').attr('title','Click to filter this column')
            .on('click', function () {
                dataTable.button('.buttons-searchPanes').trigger();
                window.setTimeout(function () {
                    const idx = [1,2,3,4,5].indexOf(i);
                    const $container = window.jQuery(dataTable.searchPanes.container());
                    const $pane = $container.find('.dtsp-pane').eq(idx);
                    const $title = $pane.find('.dtsp-title');
                    if ($title.length) $title.trigger('click');
                }, 0);
            });
    });
});
</script>
""")


# =============================================================================
# Main functions
# =============================================================================


def _load_local_dataset_summary() -> pd.DataFrame:
    """Load local dataset_summary.csv as fallback."""
    csv_path = (
        Path(__file__).resolve().parents[1]
        / "eegdash"
        / "dataset"
        / "dataset_summary.csv"
    )
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path, index_col=False, header=0, skipinitialspace=True)
    except Exception:
        return pd.DataFrame()


def main_from_api(target_dir: str, database: str = DEFAULT_DATABASE, limit: int = 1000):
    """Generate summary tables and charts from API data."""
    try:
        from eegdash.dataset.registry import fetch_chart_data_from_api
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).parents[1]))
        from eegdash.dataset.registry import fetch_chart_data_from_api

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

    # Save summary stats
    save_summary_stats(df_raw)

    # Generate HTML table
    df = prepare_table(df_raw)
    df["n_subjects"] = df["n_subjects"].astype(int)
    df["n_tasks"] = df["n_tasks"].astype(int)
    df["n_records"] = df["n_records"].astype(int)
    int_cols = ["n_subjects", "n_tasks", "n_records"]
    df[int_cols] = df[int_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")

    df = df.rename(
        columns={
            "dataset": "Dataset",
            "source": "Source",
            "nchans_set": "# of channels",
            "sampling_freqs": "sampling (Hz)",
            "size": "size",
            "n_records": "# of records",
            "n_subjects": "# of subjects",
            "n_tasks": "# of tasks",
            "pathology": "Pathology",
            "modality": "Modality",
            "type": "Type",
            "record modality": "Record modality",
        }
    )
    df = df[
        [
            "Dataset",
            "Source",
            "Record modality",
            "Pathology",
            "Modality",
            "Type",
            "# of records",
            "# of subjects",
            "# of tasks",
            "# of channels",
            "sampling (Hz)",
            "size",
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
