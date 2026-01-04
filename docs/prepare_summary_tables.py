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
from shutil import copyfile
from tqdm import tqdm

import numpy as np
import pandas as pd
from plot_dataset import (
    generate_dataset_bubble,
    generate_dataset_sankey,
    generate_dataset_treemap,
    generate_modality_ridgeline,
)
from plot_dataset.utils import get_dataset_url, human_readable_size
from table_tag_utils import _normalize_values, wrap_tags

DOCS_DIR = Path(__file__).resolve().parent
STATIC_DATASET_DIR = DOCS_DIR / "source" / "_static" / "dataset_generated"

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
    if "Type Subject" not in df.columns:
        df["Type Subject"] = ""
    if "modality of exp" not in df.columns:
        df["modality of exp"] = ""
    if "type of exp" not in df.columns:
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

    if "source" not in df.columns:
        df["source"] = ""

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
        copyfile(bubble_output, STATIC_DATASET_DIR / bubble_output.name)
        print(f"Generated: {bubble_output.name}")

        # Save summary stats for documentation cards
        save_summary_stats(df_raw)

        # Generate Sankey diagram showing dataset flow across categories
        try:
            sankey_path = target_dir / "dataset_sankey.html"
            sankey_output = generate_dataset_sankey(df_raw, sankey_path)
            copyfile(sankey_output, STATIC_DATASET_DIR / sankey_output.name)
        except Exception as exc:
            print(f"[dataset Sankey] Skipped due to error: {exc}")

        try:
            treemap_path = target_dir / "dataset_treemap.html"
            treemap_output = generate_dataset_treemap(df_raw, treemap_path)
            copyfile(treemap_output, STATIC_DATASET_DIR / treemap_output.name)
        except Exception as exc:
            print(f"[dataset Treemap] Skipped due to error: {exc}")

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
        copyfile(table_path, STATIC_DATASET_DIR / table_path.name)

        # Generate KDE ridgeline plot for modality participant distributions
        try:
            kde_path = target_dir / "dataset_kde_modalities.html"
            kde_output = generate_modality_ridgeline(df_raw, kde_path)
            if kde_output:
                copyfile(kde_output, STATIC_DATASET_DIR / kde_output.name)
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

    stats_path = STATIC_DATASET_DIR / "summary_stats.json"
    with open(stats_path, "w") as f:
        json.dump(summary_stats, f)
    print(f"Generated summary stats: {stats_path}")


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


def fetch_datasets_from_api(
    database: str = DEFAULT_DATABASE, limit: int = 1000
) -> pd.DataFrame:
    """Fetch all datasets from the EEGDash API and convert to DataFrame.

    Args:
        database: Database name (eegdash, eegdash_staging, eegdash_v1)
        limit: Maximum number of datasets per request (max 1000)

    Returns:
        DataFrame with columns matching the expected format for prepare_table()
    """
    all_datasets = []
    skip = 0
    total = None

    while True:
        url = f"{API_BASE_URL}/{database}/datasets/summary?limit={limit}&skip={skip}"
        print(f"Fetching: {url}")

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"Error fetching from API: {e}")
            raise

        if not data.get("success"):
            raise ValueError(f"API returned error: {data}")

        datasets = data.get("data", [])
        all_datasets.extend(datasets)

        if total is None:
            total = data.get("total", len(datasets))
            totals = data.get("totals", {})
            print(f"Total datasets in DB: {total}")
            print(
                f"Total subjects: {totals.get('subjects', '?')}, Total files: {totals.get('files', '?')}"
            )

        # Check if we got all datasets or reached the limit
        if len(datasets) < limit or len(all_datasets) >= limit:
            break

        skip += limit

    print(f"Fetched {len(all_datasets)} datasets total")

    # Fetch global stats once
    # We collect IDs to potentially optimize the query
    dataset_ids = {ds.get("dataset_id") for ds in all_datasets if ds.get("dataset_id")}
    global_stats = fetch_global_record_stats(database, dataset_ids)

    # Convert API response to DataFrame with expected columns
    rows = []
    for ds in all_datasets:
        ds_id = ds.get("dataset_id", "").strip()
        # Skip test datasets and excluded ones
        if (
            ds_id.lower() in ("test", "test_dataset")
            or ds_id.upper() in EXCLUDED_DATASETS
        ):
            continue

        # Get stats from global dict - now expecting counts dict or list of dicts
        ds_stats = global_stats.get(ds_id, {})
        nchans_list = ds_stats.get("nchans_counts", [])
        sfreq_list = ds_stats.get("sfreq_counts", [])

        row = {
            "dataset": ds.get("dataset_id", ds.get("name", "")),
            "record_modality": ds.get("recording_modality", ""),
            "n_records": ds.get("total_files", 0) or 0,
            "n_subjects": ds.get("demographics", {}).get("subjects_count", 0) or 0,
            "n_tasks": len(ds.get("tasks", [])) or 0,
            "nchans_set": nchans_list,
            "sampling_freqs": sfreq_list,
            "size": human_readable_size(ds.get("size_bytes") or 0),
            "size_bytes": ds.get("size_bytes") or 0,
            "dataset_title": ds.get("name", ""),
            # Map to expected categorical columns (these may need enrichment)
            "Type Subject": ds.get("study_domain", "") or "",
            "modality of exp": ", ".join(ds.get("modalities", []))
            if ds.get("modalities")
            else "",
            "type of exp": ds.get("study_design", "") or "",
            # Extra fields for reference
            "source": ds.get("source", ""),
            "license": ds.get("license", ""),
            "doi": ds.get("dataset_doi", ""),
            "duration_hours_total": 0.0,  # Fallback for treemap
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df)} datasets")
    return df


def fetch_global_record_stats(database: str, dataset_ids: set[str] = None):
    """Fetch nchans and sfreq aggregated stats for all datasets."""
    print("Fetching global record statistics (aggregated endpoint)...")
    url = f"{API_BASE_URL}/{database}/datasets/stats/records"

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            resp_json = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching global stats: {e}")
        return {}

    data = resp_json.get("data", {})
    print(f"Fetched stats for {len(data)} datasets")
    return data


def main_from_api(target_dir: str, database: str = DEFAULT_DATABASE, limit: int = 1000):
    """Generate summary tables from API data."""
    # Local import to avoid circular dependencies (depending on how this script is run)
    try:
        from eegdash.dataset.registry import fetch_datasets_from_api
    except ImportError:
        # Fallback if eegdash is not installed in editable mode but typically it is for docs
        import sys

        sys.path.insert(0, str(Path(__file__).parents[1]))
        from eegdash.dataset.registry import fetch_datasets_from_api

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    STATIC_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching data from API (database: {database})...")
    # Using registry's cached fetcher
    df_raw = fetch_datasets_from_api(API_BASE_URL, database)
    if df_raw.empty or _needs_csv_fallback(df_raw):
        fallback_df = _load_local_dataset_summary()
        if not fallback_df.empty:
            df_raw = fallback_df
            print("API summary incomplete; using local dataset_summary.csv.")

    if df_raw.empty:
        print("No datasets fetched from API or local CSV!")
        return

    # Generate bubble chart
    try:
        bubble_path = target_dir / "dataset_bubble.html"
        # Rename columns for bubble chart compatibility
        df_bubble = df_raw.rename(columns={"n_subjects": "subjects"})
        bubble_output = generate_dataset_bubble(
            df_bubble, bubble_path, x_var="subjects"
        )
        copyfile(bubble_output, STATIC_DATASET_DIR / bubble_output.name)
        print(f"Generated: {bubble_output.name}")
    except Exception as exc:
        print(f"[dataset Bubble] Skipped due to error: {exc}")

    # Save summary stats for documentation cards
    save_summary_stats(df_raw)

    # Generate Sankey diagram
    try:
        sankey_path = target_dir / "dataset_sankey.html"
        sankey_output = generate_dataset_sankey(df_raw, sankey_path)
        copyfile(sankey_output, STATIC_DATASET_DIR / sankey_output.name)
        print(f"Generated: {sankey_output.name}")
    except Exception as exc:
        print(f"[dataset Sankey] Skipped due to error: {exc}")

    # Generate Treemap
    try:
        treemap_path = target_dir / "dataset_treemap.html"
        treemap_output = generate_dataset_treemap(df_raw, treemap_path)
        copyfile(treemap_output, STATIC_DATASET_DIR / treemap_output.name)
        print(f"Generated: {treemap_output.name}")
    except Exception as exc:
        print(f"[dataset Treemap] Skipped due to error: {exc}")

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
    copyfile(table_path, STATIC_DATASET_DIR / table_path.name)
    print(f"Generated: {table_path.name}")

    # Generate KDE ridgeline plot
    try:
        kde_path = target_dir / "dataset_kde_modalities.html"
        kde_output = generate_modality_ridgeline(df_raw, kde_path)
        if kde_output:
            copyfile(kde_output, STATIC_DATASET_DIR / kde_output.name)
            print(f"Generated: {kde_output.name}")
    except Exception as exc:
        print(f"[dataset KDE] Skipped due to error: {exc}")

    print(f"\nAll outputs saved to: {target_dir}")
    print(f"Static files copied to: {STATIC_DATASET_DIR}")


def main_from_json(source_dir: str, target_dir: str):
    """Generate summary tables from local JSON digestion output."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    STATIC_DATASET_DIR.mkdir(parents=True, exist_ok=True)

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

            row = {
                "dataset": ds_id,
                "dataset_title": ds.get("name", ""),  # New field
                "record_modality": rec_mod,
                "n_records": ds.get("total_files", 0) or 0,
                # Demographics might be empty or null
                "n_subjects": ds.get("demographics", {}).get("subjects_count", 0) or 0,
                "n_tasks": len(ds.get("tasks", []) or []) or 0,
                "nchans_set": "",  # Not easily available in dataset.json
                "sampling_freqs": "",  # Not easily available
                "size": human_readable_size(ds.get("size_bytes") or 0),
                "size_bytes": ds.get("size_bytes") or 0,
                # Mappings
                "Type Subject": ds.get("study_domain", "") or "",
                "modality of exp": "",  # 'experimental_modalities' is null in my sample
                "type of exp": ds.get("study_design", "") or "",
                # Extra
                "source": ds.get("source", ""),
                "license": ds.get("license", ""),
            }
            rows.append(row)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    df_raw = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df_raw)} datasets")

    if df_raw.empty:
        print("No valid datasets found!")
        return

    # Generate visualizations (Bubble, Sankey, Treemap, KDE)
    # Reusing fetch_datasets logic...

    # Bubble
    try:
        bubble_path = target_dir / "dataset_bubble.html"
        df_bubble = df_raw.rename(columns={"n_subjects": "subjects"})
        bubble_output = generate_dataset_bubble(
            df_bubble, bubble_path, x_var="subjects"
        )
        copyfile(bubble_output, STATIC_DATASET_DIR / bubble_output.name)
        print(f"Generated: {bubble_output.name}")
    except Exception as exc:
        print(f"[dataset Bubble] Skipped due to error: {exc}")

    # Summary Stats
    save_summary_stats(df_raw)

    # Sankey
    try:
        sankey_path = target_dir / "dataset_sankey.html"
        sankey_output = generate_dataset_sankey(df_raw, sankey_path)
        copyfile(sankey_output, STATIC_DATASET_DIR / sankey_output.name)
        print(f"Generated: {sankey_output.name}")
    except Exception as exc:
        print(f"[dataset Sankey] Skipped due to error: {exc}")

    # Treemap
    try:
        treemap_path = target_dir / "dataset_treemap.html"
        treemap_output = generate_dataset_treemap(df_raw, treemap_path)
        copyfile(treemap_output, STATIC_DATASET_DIR / treemap_output.name)
        print(f"Generated: {treemap_output.name}")
    except Exception as exc:
        print(f"[dataset Treemap] Skipped due to error: {exc}")

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
    copyfile(table_path, STATIC_DATASET_DIR / table_path.name)
    print(f"Generated: {table_path.name}")

    # KDE
    try:
        kde_path = target_dir / "dataset_kde_modalities.html"
        kde_output = generate_modality_ridgeline(df_raw, kde_path)
        if kde_output:
            copyfile(kde_output, STATIC_DATASET_DIR / kde_output.name)
            print(f"Generated: {kde_output.name}")
    except Exception as exc:
        print(f"[dataset KDE] Skipped due to error: {exc}")

    print(f"\nAll outputs saved to: {target_dir}")


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
        "--from-api",
        action="store_true",
        default=True,
        help="Fetch data from API (default)",
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
