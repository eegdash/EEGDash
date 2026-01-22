import concurrent.futures
import glob
import json
import os
import textwrap
import urllib.request
from collections import Counter
from urllib.parse import quote
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copyfile as _copyfile
from tqdm import tqdm
from typing import Callable, Any


def copyfile(src, dst):
    """Robust copyfile that ignores SameFileError."""
    try:
        _copyfile(src, dst)
    except Exception as exc:
        if "are the same file" not in str(exc):
            raise exc


def copy_to_static(src: Path, filename: str = None):
    """Copy file to both source and build static directories.

    This ensures files are available both for:
    - Sphinx source builds (STATIC_DATASET_DIR)
    - Direct browsing of _build/html (BUILD_STATIC_DIR)
    """
    if filename is None:
        filename = Path(src).name

    # Copy to source static (for Sphinx builds)
    copyfile(src, STATIC_DATASET_DIR / filename)

    # Copy to build static (for direct browsing, bypasses Sphinx copy step)
    # Always create directory if it doesn't exist
    BUILD_STATIC_DIR.mkdir(parents=True, exist_ok=True)
    copyfile(src, BUILD_STATIC_DIR / filename)


import numpy as np
import pandas as pd
from plot_dataset import (
    generate_dataset_bubble,
    generate_dataset_sankey,
    generate_dataset_treemap,
    generate_modality_ridgeline,
    generate_dataset_growth,
    generate_clinical_stacked_bar,
)
from plot_dataset.utils import get_dataset_url, human_readable_size
from table_tag_utils import _normalize_values, wrap_tags

DOCS_DIR = Path(__file__).resolve().parent
STATIC_DATASET_DIR = DOCS_DIR / "source" / "_static" / "dataset_generated"
# Build output directory - files here are served directly without Sphinx copy step
BUILD_STATIC_DIR = DOCS_DIR / "_build" / "html" / "_static" / "dataset_generated"

# Number of workers for parallel chart generation
MAX_CHART_WORKERS = 6


def _generate_chart_task(
    name: str,
    generator: Callable,
    df: pd.DataFrame,
    output_path: Path,
    **kwargs,
) -> tuple[str, Path | None, str | None]:
    """Generate a single chart - worker function for parallel execution.

    Returns:
        tuple of (chart_name, output_path or None, error_message or None)
    """
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
    """Generate all charts in parallel using ThreadPoolExecutor.

    Parameters
    ----------
    df_raw : pd.DataFrame
        The raw dataset DataFrame
    target_dir : Path
        Output directory for charts
    x_var : str
        X-axis variable for bubble chart (default: "subjects")

    Returns
    -------
    list of tuples
        Each tuple contains (chart_name, output_path, error_message)
    """
    # Prepare bubble chart DataFrame
    df_bubble = df_raw.copy()
    if "subjects" not in df_bubble.columns and "n_subjects" in df_bubble.columns:
        df_bubble["subjects"] = df_bubble["n_subjects"]
    if "records" not in df_bubble.columns and "n_records" in df_bubble.columns:
        df_bubble["records"] = df_bubble["n_records"]

    # Define chart generation tasks
    # Each task is (name, generator_func, dataframe, output_path, kwargs)
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
        # Submit all tasks
        futures = {
            executor.submit(_generate_chart_task, name, gen, df, path, **kwargs): name
            for name, gen, df, path, kwargs in tasks
        }

        # Collect results as they complete
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
            print(f"[dataset {name}] Skipped due to error: {error}")
        elif output_path:
            copy_to_static(output_path)
            print(f"Generated: {output_path.name}")


# API Configuration
API_BASE_URL = "https://data.eegdash.org/api"
DEFAULT_DATABASE = "eegdash"


def wrap_dataset_name(name: str):
    # Remove any surrounding whitespace
    name = name.strip()
    # Link to the individual dataset API page
    # Updated structure: api/dataset/eegdash.dataset.<CLASS>.html
    url = get_dataset_url(name)
    if not url:
        return name.upper()
    return f'<a href="{url}">{name.upper()}</a>'

    return f'<a href="{url}">{name.upper()}</a>'


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

_UNKNOWN_TOKENS = {"unknown", "nothing", "nan", "none", "null"}

DATA_TABLE_TEMPLATE = textwrap.dedent(
    r"""
<!-- jQuery + DataTables core -->
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<link rel="stylesheet" href="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.css"/>
<script src="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.js"></script>

<!-- Buttons + SearchPanes (+ Select required by SearchPanes) -->
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<link rel="stylesheet" href="https://cdn.datatables.net/select/1.7.0/css/select.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/searchpanes/2.3.1/css/searchPanes.dataTables.min.css">
<script src="https://cdn.datatables.net/select/1.7.0/js/dataTables.select.min.js"></script>
<script src="https://cdn.datatables.net/searchpanes/2.3.1/js/dataTables.searchPanes.min.js"></script>

<style>
    /* Styling for the Total row (placed in tfoot) */
    table.sd-table tfoot td {
        font-weight: 600;
        border-top: 2px solid rgba(0,0,0,0.2);
        background: #f9fafb;
        /* Match body cell padding to keep perfect alignment */
        padding: 8px 10px !important;
        vertical-align: middle;
    }

    /* Left-align text columns (1..6) */
    table.sd-table tbody td:nth-child(-n+6),
    table.sd-table tfoot td:nth-child(-n+6),
    table.sd-table thead th:nth-child(-n+6) {
        text-align: left;
    }

    /* Right-align numeric-like columns (7..12) consistently for body & footer */
    table.sd-table tbody td:nth-child(n+7),
    table.sd-table tfoot td:nth-child(n+7),
    table.sd-table thead th:nth-child(n+7) {
        text-align: right;
    }
</style>

<TABLE_HTML>

<script>
// Helper: robustly extract values for SearchPanes when needed
function tagsArrayFromHtml(html) {
    if (html == null) return [];
    // If it's numeric or plain text, just return as a single value
    if (typeof html === 'number') return [String(html)];
    if (typeof html === 'string' && html.indexOf('<') === -1) return [html.trim()];
    // Else parse any .tag elements inside HTML
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    const tags = Array.from(tmp.querySelectorAll('.tag')).map(function(el){
        return (el.textContent || '').trim();
    });
    const text = tmp.textContent.trim();
    return tags.length ? tags : (text ? [text] : []);
}

// Helper: parse human-readable sizes like "4.31 GB" into bytes (number)
function parseSizeToBytes(text) {
    if (!text) return 0;
    const s = String(text).trim();
    const m = s.match(/([\d,.]+)\s*(TB|GB|MB|KB|B)/i);
    if (!m) return 0;
    const value = parseFloat(m[1].replace(/,/g, ''));
    const unit = m[2].toUpperCase();
    const factor = { B:1, KB:1024, MB:1024**2, GB:1024**3, TB:1024**4 }[unit] || 1;
    return value * factor;
}

document.addEventListener('DOMContentLoaded', function () {
    const table = document.getElementById('datasets-table');
    if (!table || !window.jQuery || !window.jQuery.fn || !window.jQuery.fn.DataTable) {
        return;
    }

    const $table = window.jQuery(table);
    if (window.jQuery.fn.DataTable.isDataTable(table)) {
        return;
    }

    // 1) Move the "Total" row into <tfoot> so sorting/filtering never moves it
    const $tbody = $table.find('tbody');
    const $total = $tbody.find('tr').filter(function(){
        return window.jQuery(this).find('td').eq(0).text().trim() === 'Total';
    });
    if ($total.length) {
        let $tfoot = $table.find('tfoot');
        if (!$tfoot.length) $tfoot = window.jQuery('<tfoot/>').appendTo($table);
        $total.appendTo($tfoot);
    }

    // 2) Initialize DataTable with SearchPanes button
    const FILTER_COLS = [1, 2, 3, 4, 5, 6, 7, 8];
    const TAG_COLS = (function(){
        const tagHeaders = new Set(['record modality', 'pathology', 'modality', 'type']);
        const cols = [];
        $table.find('thead th').each(function(i){
            const t = window.jQuery(this).text().trim().toLowerCase();
            if (tagHeaders.has(t)) cols.push(i);
        });
        return cols;
    })();
    // Detect the index of the size column by header text
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
            const defs = [
                { searchPanes: { show: true }, targets: FILTER_COLS }
            ];
            if (TAG_COLS.length) {
                defs.push({
                    targets: TAG_COLS,
                    searchPanes: { show: true, orthogonal: 'sp' },
                    render: function(data, type) {
                        if (type === 'sp') {
                            return tagsArrayFromHtml(data);
                        }
                        return data;
                    }
                });
            }
            if (sizeIdx !== -1) {
                defs.push({
                    targets: sizeIdx,
                    render: function(data, type) {
                        if (type === 'sort' || type === 'type') {
                            return parseSizeToBytes(data);
                        }
                        return data;
                    }
                });
            }
            return defs;
        })()
    });

    // 3) UX: click a header to open the relevant filter pane
    $table.find('thead th').each(function (i) {
        if ([1, 2, 3, 4, 5].indexOf(i) === -1) return; // Source, Record modality, Pathology, Modality, Type
        window.jQuery(this)
            .css('cursor','pointer')
            .attr('title','Click to filter this column')
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
"""
)


def _tag_normalizer(kind: str):
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


def prepare_table(df: pd.DataFrame):
    # drop test dataset and specific excluded datasets
    excluded_datasets = {"test", "ds003380"}
    df = df[~df["dataset"].str.lower().isin(excluded_datasets)].copy()

    df["dataset"] = df["dataset"].apply(wrap_dataset_name)
    # changing the column order
    if "dataset_title" not in df.columns:
        df["dataset_title"] = ""
    if "source" not in df.columns:
        df["source"] = ""
    if "record_modality" not in df.columns:
        df["record_modality"] = df.get("record modality", "")
    # Handle tags columns (new structure) or legacy column names
    if "Type Subject" not in df.columns:
        # Try to get from tags.pathology or default to empty
        if "pathology" in df.columns:
            df["Type Subject"] = df["pathology"]
        else:
            df["Type Subject"] = ""
    if "modality of exp" not in df.columns:
        # Try to get from tags.modality or default to empty
        if "modality" in df.columns and "Type Subject" in df.columns:
            # Only use if it's the experimental modality, not recording modality
            df["modality of exp"] = df["modality"]
        else:
            df["modality of exp"] = ""
    if "type of exp" not in df.columns:
        # Try to get from tags.type or default to empty
        if "type" in df.columns:
            df["type of exp"] = df["type"]
        else:
            df["type of exp"] = ""
    if "license" not in df.columns:
        df["license"] = ""
    if "size" not in df.columns:
        df["size"] = ""
    if "size_bytes" not in df.columns:
        df["size_bytes"] = 0
    if "n_records" not in df.columns:
        df["n_records"] = 0
    if "n_subjects" not in df.columns:
        df["n_subjects"] = 0
    if "n_tasks" not in df.columns:
        df["n_tasks"] = 0
    if "nchans_set" not in df.columns:
        df["nchans_set"] = ""
    if "sampling_freqs" not in df.columns:
        df["sampling_freqs"] = ""

    def _strip_unknown(value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        if isinstance(value, str) and value.strip().lower() in _UNKNOWN_TOKENS:
            return ""
        return value

    df = df[
        [
            "dataset",
            "dataset_title",  # Added
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
            "license",  # Added
        ]
    ]
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].apply(_strip_unknown)

    # renaming time for something small
    df = df.rename(
        columns={
            "modality of exp": "modality",
            "type of exp": "type",
            "Type Subject": "pathology",
            "record_modality": "record modality",
        }
    )
    # number of subject are always int
    df["n_subjects"] = df["n_subjects"].astype(int)
    # number of tasks are always int
    df["n_tasks"] = df["n_tasks"].astype(int)
    # number of records are always int
    df["n_records"] = df["n_records"].astype(int)

    # from the sample frequency list, I will apply str
    df["sampling_freqs"] = df["sampling_freqs"].apply(parse_freqs)
    # from the channels set, I will follow the same logic of freq
    df["nchans_set"] = df["nchans_set"].apply(parse_freqs)
    # Wrap categorical columns with styled tags for downstream rendering
    pathology_normalizer = _tag_normalizer("pathology")
    modality_normalizer = _tag_normalizer("modality")
    type_normalizer = _tag_normalizer("type")
    record_modality_normalizer = _tag_normalizer("record_modality")

    df["pathology"] = df["pathology"].apply(
        lambda value: wrap_tags(
            value,
            kind="dataset-pathology",
            normalizer=pathology_normalizer,
        )
    )
    df["modality"] = df["modality"].apply(
        lambda value: wrap_tags(
            value,
            kind="dataset-modality",
            normalizer=modality_normalizer,
        )
    )
    df["type"] = df["type"].apply(
        lambda value: wrap_tags(
            value,
            kind="dataset-type",
            normalizer=type_normalizer,
        )
    )
    df["record modality"] = df["record modality"].apply(
        lambda value: wrap_tags(
            value,
            kind="dataset-record-modality",
            normalizer=record_modality_normalizer,
        )
    )

    # Creating the total line
    df.loc["Total"] = df.sum(numeric_only=True)
    df.loc["Total", "dataset"] = f"Total {len(df) - 1} datasets"
    df.loc["Total", "nchans_set"] = ""
    df.loc["Total", "sampling_freqs"] = ""
    df.loc["Total", "source"] = ""
    df.loc["Total", "pathology"] = ""
    df.loc["Total", "modality"] = ""
    df.loc["Total", "type"] = ""
    df.loc["Total", "record modality"] = ""
    df.loc["Total", "size"] = human_readable_size(df.loc["Total", "size_bytes"])
    df = df.drop(columns=["size_bytes"])
    # arrounding the hours

    df.index = df.index.astype(str)

    return df


def main(source_dir: str, target_dir: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    STATIC_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_STATIC_DIR.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(Path(source_dir) / "dataset" / "*.csv"))
    for f in files:
        target_file = target_dir / Path(f).name
        print(f"Processing {f} -> {target_file}")
        df_raw = pd.read_csv(
            f, index_col=False, header=0, skipinitialspace=True
        )  # , sep=";")
        # Generate bubble chart from the raw data to have access to size_bytes
        bubble_path = target_dir / "dataset_bubble.html"
        bubble_output = generate_dataset_bubble(
            df_raw,
            bubble_path,
            x_var="subjects",
        )
        copy_to_static(bubble_output)
        print(f"Generated: {bubble_output.name}")

        # Save summary stats for documentation cards
        save_summary_stats(df_raw)

        # Generate Sankey diagram showing dataset flow across categories
        try:
            sankey_path = target_dir / "dataset_sankey.html"
            sankey_output = generate_dataset_sankey(df_raw, sankey_path)
            copy_to_static(sankey_output)
        except Exception as exc:
            print(f"[dataset Sankey] Skipped due to error: {exc}")

        try:
            treemap_path = target_dir / "dataset_treemap.html"
            treemap_output = generate_dataset_treemap(df_raw, treemap_path)
            copy_to_static(treemap_output)
        except Exception as exc:
            print(f"[dataset Treemap] Skipped due to error: {exc}")

        # Generate Dataset Growth Plot
        try:
            growth_path = target_dir / "dataset_growth.html"
            growth_output = generate_dataset_growth(df_raw, growth_path)
            copy_to_static(growth_output)
            print(f"Generated: {growth_output.name}")
        except Exception as exc:
            print(f"[dataset Growth] Skipped due to error: {exc}")

        # Generate Clinical Stacked Bar
        try:
            clinical_path = target_dir / "dataset_clinical.html"
            clinical_output = generate_clinical_stacked_bar(df_raw, clinical_path)
            copy_to_static(clinical_output)
            print(f"Generated: {clinical_output.name}")
        except Exception as exc:
            print(f"[dataset Clinical] Skipped due to error: {exc}")

        df = prepare_table(df_raw)
        # preserve int values
        df["n_subjects"] = df["n_subjects"].astype(int)
        df["n_tasks"] = df["n_tasks"].astype(int)
        df["n_records"] = df["n_records"].astype(int)
        int_cols = ["n_subjects", "n_tasks", "n_records"]

        # Coerce to numeric, allow NAs, and keep integer display
        df[int_cols] = (
            df[int_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")
        )
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
        # (If you add a 'Total' row after this, cast again or build it as Int64.)
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

        # Generate KDE ridgeline plot for modality participant distributions
        try:
            kde_path = target_dir / "dataset_kde_modalities.html"
            kde_output = generate_modality_ridgeline(df_raw, kde_path)
            if kde_output:
                copy_to_static(kde_output)
        except Exception as exc:
            print(f"[dataset KDE] Skipped due to error: {exc}")


def parse_freqs(value) -> str:
    """Parse frequencies/channels list and return mode with * if variable.

    Supports:
    - List of values: [64, 64, 64, 63]
    - List of dicts (from API aggregation): [{"val": 64, "count": 100}, {"val": 63, "count": 1}]
    - Single value: 64
    - String: "[64, 63]" or JSON string '[{"val": 64, "count": 100}]'
    """
    if not value:
        return ""

    # Try JSON parsing first if string
    if isinstance(value, str):
        value = value.strip()
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

    counts = Counter()

    # 1. Handle API aggregation format (list of dicts)
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
                    # Normalize to int for display (e.g. 64.0 -> 64)
                    val_int = int(float(val))
                    counts[val_int] += count
                except (ValueError, TypeError):
                    pass

    # 2. Handle simple list of values or string representation
    else:
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

        # Count frequencies
        for f in freqs:
            try:
                counts[int(f)] += 1
            except ValueError:
                pass

    if not counts:
        return ""

    most_common_val, _ = counts.most_common(1)[0]

    if len(counts) == 1:
        return f"{most_common_val}"
    else:
        return f"{most_common_val}*"


def save_summary_stats(df_raw: pd.DataFrame):
    """Calculate and save summary stats for the documentation cards."""
    unique_modalities = set()
    # Handle the fact that record_modality might be a join of modalities
    if "record_modality" in df_raw.columns:
        col = "record_modality"
    elif "record modality" in df_raw.columns:
        col = "record modality"
    else:
        col = None

    if col:
        for m in df_raw[col].dropna():
            unique_modalities.update(_normalize_values(m))
    unique_modalities = {m.strip().lower() for m in unique_modalities if m.strip()}

    # n_subjects might be in df_raw or subjects
    n_subj_col = "n_subjects" if "n_subjects" in df_raw.columns else "subjects"
    try:
        subjects_total = int(pd.to_numeric(df_raw[n_subj_col], errors="coerce").sum())
    except Exception:
        subjects_total = 0

    # n_records might be in df_raw or records
    n_rec_col = "n_records" if "n_records" in df_raw.columns else "records"
    try:
        recording_total = int(pd.to_numeric(df_raw[n_rec_col], errors="coerce").sum())
    except Exception:
        recording_total = 0

    summary_stats = {
        "datasets_total": len(df_raw),
        "subjects_total": subjects_total,
        "recording_total": recording_total,
        "modalities_total": len(unique_modalities),
        "sources_total": df_raw["source"].nunique()
        if "source" in df_raw.columns
        else 0,
    }

    # Write to source static directory
    stats_path = STATIC_DATASET_DIR / "summary_stats.json"
    with open(stats_path, "w") as f:
        json.dump(summary_stats, f)
    print(f"Generated summary stats: {stats_path}")

    # Also write to build static directory if it exists
    if BUILD_STATIC_DIR.exists():
        build_stats_path = BUILD_STATIC_DIR / "summary_stats.json"
        with open(build_stats_path, "w") as f:
            json.dump(summary_stats, f)


def _load_local_dataset_summary() -> pd.DataFrame:
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


def _needs_csv_fallback(df_raw: pd.DataFrame) -> bool:
    required_cols = ("n_subjects", "n_records", "n_tasks", "size_bytes")
    if any(col not in df_raw.columns for col in required_cols):
        return True
    numeric = (
        df_raw[list(required_cols)].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    totals = numeric.sum()
    return any(totals.get(col, 0) == 0 for col in required_cols)


def main_from_api(target_dir: str, database: str = DEFAULT_DATABASE, limit: int = 1000):
    """Generate summary tables from API data.

    Uses the optimized /datasets/chart-data endpoint which returns
    pre-aggregated data specifically for chart generation.
    """
    # Local import to avoid circular dependencies
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
    # Use optimized chart-data endpoint (single call, pre-aggregated)
    df_raw, aggregations = fetch_chart_data_from_api(
        API_BASE_URL, database, limit=limit
    )

    if aggregations:
        print(
            f"  Pre-computed aggregations: {len(aggregations.get('modality_counts', {}))} modalities, "
            f"{len(aggregations.get('source_counts', {}))} sources"
        )

    # Always try to load local summary to enrich API data (better modality tags)
    fallback_df = _load_local_dataset_summary()

    if df_raw.empty or _needs_csv_fallback(df_raw):
        if not fallback_df.empty:
            df_raw = fallback_df
            print("API summary incomplete; using local dataset_summary.csv.")
    elif not fallback_df.empty:
        # Merge modality info from CSV if API checks passed but CSV exists
        print("Enriching API data with local dataset_summary.csv modality info...")
        # CSV uses 'dataset' (id) and 'record_modality'
        # API uses 'dataset_id' or 'dataset'

        # Ensure ID column match
        if "dataset" not in df_raw.columns and "dataset_id" in df_raw.columns:
            df_raw["dataset"] = df_raw["dataset_id"]

        # Select relevant columns from CSV
        # We trust CSV 'record_modality' more than API 'experimental_modalities'
        enrich_cols = ["dataset", "record_modality", "modality of exp"]
        enrich_df = fallback_df[
            [c for c in enrich_cols if c in fallback_df.columns]
        ].copy()

        if not enrich_df.empty and "dataset" in enrich_df.columns:
            # Merge
            df_raw = df_raw.merge(
                enrich_df, on="dataset", how="left", suffixes=("", "_csv")
            )

            # Overwrite/Prioritize CSV modality
            if "record_modality_csv" in df_raw.columns:
                if "recording_modality" not in df_raw.columns:
                    df_raw["recording_modality"] = pd.Series(
                        index=df_raw.index, dtype="object"
                    )
                df_raw["recording_modality"] = df_raw[
                    "record_modality_csv"
                ].combine_first(df_raw["recording_modality"])
                # Also ensure our plotting scripts see it
                df_raw["record_modality"] = df_raw["record_modality_csv"]

    # Attempt to backfill timestamps from local consolidated JSON if missing
    if "dataset_created_at" not in df_raw.columns:
        json_backfill_path = (
            Path(__file__).resolve().parents[1]
            / "consolidated"
            / "openneuro_datasets.json"
        )
        if json_backfill_path.exists():
            try:
                print(f"Backfilling timestamps from {json_backfill_path}...")
                with open(json_backfill_path) as f:
                    local_data = json.load(f)

                # Create a mapping of dataset -> created_at
                # Handle both list of records or dict structure
                date_map = {}
                if isinstance(local_data, list):
                    for item in local_data:
                        ds = item.get("dataset_id") or item.get("dataset")
                        # Timestamps might be in 'timestamps' dict or top level
                        ts = item.get("dataset_created_at") or item.get(
                            "timestamps", {}
                        ).get("dataset_created_at")
                        if ds and ts:
                            date_map[ds] = ts

                if date_map:
                    # Map to df_raw
                    # Ensure join column is consistent (df_raw has 'dataset')
                    df_raw["dataset_created_at"] = df_raw["dataset"].map(date_map)
            except Exception as e:
                print(f"Failed to backfill timestamps: {e}")

    if df_raw.empty:
        print("No datasets fetched from API or local CSV!")
        return

    # Generate all charts in parallel
    print("Generating charts in parallel...")
    chart_results = generate_charts_parallel(df_raw, target_dir, x_var="subjects")
    process_chart_results(chart_results)

    # Save summary stats for documentation cards
    save_summary_stats(df_raw)

    # Prepare and generate HTML table
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

    # Generate KDE ridgeline plot
    try:
        kde_path = target_dir / "dataset_kde_modalities.html"
        kde_output = generate_modality_ridgeline(df_raw, kde_path)
        if kde_output:
            copy_to_static(kde_output)
            print(f"Generated: {kde_output.name}")
    except Exception as exc:
        print(f"[dataset KDE] Skipped due to error: {exc}")

    print(f"\nAll outputs saved to: {target_dir}")
    print(f"Static files copied to: {STATIC_DATASET_DIR}")
    print(f"Build files copied to: {BUILD_STATIC_DIR}")


def main_from_json(source_dir: str, target_dir: str):
    """Generate summary tables from local JSON digestion output."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    STATIC_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_STATIC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading datasets from {source_dir}...")
    dataset_files = list(source_dir.glob("*/ds*_dataset.json"))

    rows = []
    print(f"Found {len(dataset_files)} dataset files. loading...")

    for f in tqdm(dataset_files):
        try:
            with open(f) as fp:
                ds = json.load(fp)

            ds_id = ds.get("dataset_id", "").strip()
            if ds_id.upper() in EXCLUDED_DATASETS or ds_id.lower() in (
                "test",
                "test_dataset",
            ):
                continue

            # Handle modality list or single string
            rec_mod = ds.get("recording_modality")
            if isinstance(rec_mod, list):
                rec_mod = ", ".join(rec_mod)
            elif rec_mod is None:
                rec_mod = ""

            # Extract tags (new structure) or fallback to legacy fields
            tags = ds.get("tags", {}) or {}
            clinical = ds.get("clinical", {}) or {}
            paradigm = ds.get("paradigm", {}) or {}

            # Use tags.pathology if available, otherwise fallback
            pathology_list = tags.get("pathology", [])
            if pathology_list and isinstance(pathology_list, list):
                type_subject = ", ".join(pathology_list)
            elif clinical.get("is_clinical"):
                type_subject = clinical.get("purpose") or "Clinical"
            elif clinical.get("is_clinical") is False:
                type_subject = "Healthy"
            else:
                type_subject = ds.get("study_domain", "") or ""

            # Use tags.modality if available, otherwise fallback to recording_modality
            modality_list = tags.get("modality", [])
            if modality_list and isinstance(modality_list, list):
                modality_of_exp = ", ".join(modality_list)
            elif paradigm.get("modality"):
                modality_of_exp = paradigm.get("modality")
            else:
                # Fallback to recording_modality for bubble chart compatibility
                rec_mod = ds.get("recording_modality")
                if isinstance(rec_mod, list) and rec_mod:
                    modality_of_exp = rec_mod[0].upper() if rec_mod[0] else ""
                elif rec_mod:
                    modality_of_exp = str(rec_mod).upper()
                else:
                    modality_of_exp = ""

            # Use tags.type if available
            type_list = tags.get("type", [])
            if type_list and isinstance(type_list, list):
                type_of_exp = ", ".join(type_list)
            else:
                type_of_exp = (
                    paradigm.get("cognitive_domain") or ds.get("study_design", "") or ""
                )

            # Get timestamps for growth chart
            timestamps = ds.get("timestamps", {}) or {}
            dataset_created_at = timestamps.get("dataset_created_at") or ""

            row = {
                "dataset": ds_id,
                "dataset_title": ds.get("name", ""),  # New field
                "record_modality": rec_mod,
                "recording_modality": rec_mod,  # Alias for clinical/growth charts
                "n_records": ds.get("total_files", 0) or 0,
                # Demographics might be empty or null
                "n_subjects": ds.get("demographics", {}).get("subjects_count", 0) or 0,
                "n_tasks": len(ds.get("tasks", []) or []) or 0,
                "nchans_set": "",  # Not easily available in dataset.json
                "sampling_freqs": "",  # Not easily available
                "size": human_readable_size(ds.get("size_bytes") or 0),
                "size_bytes": ds.get("size_bytes") or 0,
                "duration_hours_total": 0,  # Not available in JSON, use 0
                # Mappings from tags
                "Type Subject": type_subject,
                "modality of exp": modality_of_exp,
                "type of exp": type_of_exp,
                # Extra
                "source": ds.get("source", ""),
                "license": ds.get("license", ""),
                # Timestamps for growth chart
                "dataset_created_at": dataset_created_at,
            }
            rows.append(row)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    df_raw = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df_raw)} datasets")

    if df_raw.empty:
        print("No valid datasets found!")
        return

    # Generate all charts in parallel
    print("Generating charts in parallel...")
    chart_results = generate_charts_parallel(df_raw, target_dir, x_var="records")
    process_chart_results(chart_results)

    # Summary Stats
    save_summary_stats(df_raw)

    # Table Generation
    df = prepare_table(df_raw)

    # Post-processing to match main_from_api logic
    df["n_subjects"] = df["n_subjects"].astype(int)
    df["n_tasks"] = df["n_tasks"].astype(int)
    df["n_records"] = df["n_records"].astype(int)
    int_cols = ["n_subjects", "n_tasks", "n_records"]
    df[int_cols] = df[int_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")

    # Rename and select columns
    # Added "Name" and "License" to selection
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "dataset_title": "Name",  # Ensure this exists (prepare_table keeps existing columns if not dropped)
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
            "license": "License",
        }
    )

    # Define column order
    cols_to_keep = [
        "Dataset",
        "Name",  # New
        "Source",
        "Record modality",
        "Pathology",
        "Modality",
        "Type",
        "# of records",
        "# of subjects",
        "# of tasks",
        # "# of channels", # Dropped as empty/null in local json
        # "sampling (Hz)", # Dropped
        "size",
        "License",  # New
    ]
    # Filter only existing columns
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep]

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

    # KDE
    try:
        kde_path = target_dir / "dataset_kde_modalities.html"
        kde_output = generate_modality_ridgeline(df_raw, kde_path)
        if kde_output:
            copy_to_static(kde_output)
            print(f"Generated: {kde_output.name}")
    except Exception as exc:
        print(f"[dataset KDE] Skipped due to error: {exc}")

    print(f"\nAll outputs saved to: {target_dir}")
    print(f"Static files copied to: {STATIC_DATASET_DIR}")
    print(f"Build files copied to: {BUILD_STATIC_DIR}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        dest="source_dir",
        type=str,
        default=None,
        help="Source directory with CSV files (legacy mode)",
    )
    parser.add_argument(
        "--json-dir",
        dest="json_dir",
        type=str,
        default=None,
        help="Source directory with JSON digestion output (digestion_full/)",
    )
    parser.add_argument(
        "--target",
        dest="target_dir",
        type=str,
        default="build",
        help="Target directory for output files",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATABASE,
        help=f"Database to fetch from API (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Use legacy CSV mode (requires --source)",
    )

    # Support legacy positional arguments
    parser.add_argument("legacy_source", nargs="?", type=str, default=None)
    parser.add_argument("legacy_target", nargs="?", type=str, default=None)

    args = parser.parse_args()

    # Handle legacy positional arguments
    if args.legacy_source and args.legacy_target:
        print("Using legacy CSV mode (positional arguments detected)")
        main(args.legacy_source, args.legacy_target)
    elif args.json_dir:
        print(f"Using JSON directory: {args.json_dir}")
        main_from_json(args.json_dir, args.target_dir)
    elif args.from_csv:
        if not args.source_dir:
            parser.error("--from-csv requires --source directory")
        main(args.source_dir, args.target_dir)
    else:
        # Default: fetch from API
        limit = int(os.environ.get("EEGDASH_DOC_LIMIT", 1000))
        main_from_api(args.target_dir, args.database, limit=limit)

    print(f"Output directory: {args.target_dir}")
