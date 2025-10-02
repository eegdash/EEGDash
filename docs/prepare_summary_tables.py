import glob
import sys
import textwrap
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from eegdash.logging import logger
from eegdash.plotting.dataset_summary_adapter import (
    BubbleSizeMetric,
    SummaryBubbleConfig,
    load_dataset_summary,
)
from eegdash.plotting.plotly_dataset_bubbles import generate_plotly_dataset_bubble

try:
    from plot_dataset import generate_dataset_sankey, generate_modality_ridgeline
    from plot_dataset.utils import get_dataset_url, human_readable_size
except ModuleNotFoundError:  # pragma: no cover - fallback for unit tests
    from docs.plot_dataset import (  # type: ignore
        generate_dataset_sankey,
        generate_modality_ridgeline,
    )
    from docs.plot_dataset.utils import (  # type: ignore
        get_dataset_url,
        human_readable_size,
    )

try:
    from table_tag_utils import wrap_tags
except ModuleNotFoundError:  # pragma: no cover - fallback for unit tests
    from docs.table_tag_utils import wrap_tags  # type: ignore

DOCS_DIR = Path(__file__).resolve().parent
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

STATIC_DATASET_DIR = DOCS_DIR / "source" / "_static" / "dataset_generated"


def wrap_dataset_name(name: str):
    # Remove any surrounding whitespace
    name = name.strip()
    # Link to the individual dataset API page
    # Updated structure: api/dataset/eegdash.dataset.<CLASS>.html
    url = get_dataset_url(name)
    if not url:
        return name.upper()
    return f'<a href="{url}">{name.upper()}</a>'


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
    "type": {
        "perception": "Perception",
        "decision making": "Decision-making",
        "decision-making": "Decision-making",
        "rest": "Rest",
        "resting state": "Resting-state",
        "sleep": "Sleep",
    },
}

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

    /* Right-align numeric-like columns (2..8) consistently for body & footer */
    table.sd-table tbody td:nth-child(n+2),
    table.sd-table tfoot td:nth-child(n+2) {
        text-align: right;
    }
    /* Keep first column (Dataset/Total) left-aligned */
    table.sd-table tbody td:first-child,
    table.sd-table tfoot td:first-child {
        text-align: left;
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
    return tags.length ? tags : [tmp.textContent.trim()];
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
    const FILTER_COLS = [1,2,3,4,5,6];
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
        if ([1,2,3,4].indexOf(i) === -1) return;
        window.jQuery(this)
            .css('cursor','pointer')
            .attr('title','Click to filter this column')
            .on('click', function () {
                dataTable.button('.buttons-searchPanes').trigger();
                window.setTimeout(function () {
                    const idx = [1,2,3,4].indexOf(i);
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
        if lowered in canonical:
            return canonical[lowered]
        return text

    return _normalise


def _copy_into_static(path: Path) -> Path:
    destination = STATIC_DATASET_DIR / path.name
    if path.resolve() == destination.resolve():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    copyfile(path, destination)
    return destination


def _build_bubble_config(
    *,
    bubble_limit: int | None,
    bubble_layout_seed: int | None,
    include_type_subject_badges: bool,
) -> SummaryBubbleConfig:
    config_kwargs: dict[str, object] = {
        "size_metric": BubbleSizeMetric.N_RECORDS,
        "legend_heading": "Record volume (log scale)",
    }
    if bubble_limit is not None:
        config_kwargs["limit"] = bubble_limit
    if bubble_layout_seed is not None:
        config_kwargs["layout_seed"] = bubble_layout_seed
    if include_type_subject_badges:
        config_kwargs["include_type_subject_badges"] = True
    return SummaryBubbleConfig(**config_kwargs)


def prepare_table(df: pd.DataFrame):
    # drop test dataset and create a copy to avoid SettingWithCopyWarning
    df = df[df["dataset"] != "test"].copy()

    df["dataset"] = df["dataset"].apply(wrap_dataset_name)
    # changing the column order
    df = df[
        [
            "dataset",
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
        ]
    ]

    # renaming time for something small
    df = df.rename(
        columns={
            "modality of exp": "modality",
            "type of exp": "type",
            "Type Subject": "pathology",
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

    # Creating the total line
    df.loc["Total"] = df.sum(numeric_only=True)
    df.loc["Total", "dataset"] = f"Total {len(df) - 1} datasets"
    df.loc["Total", "nchans_set"] = ""
    df.loc["Total", "sampling_freqs"] = ""
    df.loc["Total", "pathology"] = ""
    df.loc["Total", "modality"] = ""
    df.loc["Total", "type"] = ""
    df.loc["Total", "size"] = human_readable_size(df.loc["Total", "size_bytes"])
    df = df.drop(columns=["size_bytes"])
    # arrounding the hours

    df.index = df.index.astype(str)

    return df


def main(
    source_dir: str,
    target_dir: str,
    *,
    bubble_limit: int | None = None,
    bubble_layout_seed: int | None = 42,
    include_type_subject_badges: bool = False,
) -> None:
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    STATIC_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(Path(source_dir) / "dataset" / "*.csv")))
    bubble_config = _build_bubble_config(
        bubble_limit=bubble_limit,
        bubble_layout_seed=bubble_layout_seed,
        include_type_subject_badges=include_type_subject_badges,
    )

    for csv_path_str in files:
        csv_path = Path(csv_path_str)
        logger.info("Processing dataset summary CSV %s", csv_path)

        df_raw = pd.read_csv(csv_path, index_col=False, header=0, skipinitialspace=True)
        records = load_dataset_summary(csv_path)
        if not records:
            logger.warning(
                "No dataset summary records available after normalisation for %s; skipping Plotly bubble export.",
                csv_path,
            )
        else:
            bubble_path = target_dir_path / "dataset_bubble.html"
            figure = generate_plotly_dataset_bubble(
                records,
                bubble_config,
                outfile=bubble_path,
            )
            logger.info("Dataset bubble HTML written to %s", bubble_path)
            _copy_into_static(bubble_path)

        # Generate Sankey diagram showing dataset flow across categories
        try:
            sankey_path = target_dir_path / "dataset_sankey.html"
            sankey_output = generate_dataset_sankey(df_raw, sankey_path)
            if sankey_output:
                _copy_into_static(sankey_output)
        except Exception as exc:  # pragma: no cover - visual asset best effort
            logger.warning("[dataset Sankey] Skipped due to error: %s", exc)

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
                "nchans_set": "# of channels",
                "sampling_freqs": "sampling (Hz)",
                "size": "size",
                "n_records": "# of records",
                "n_subjects": "# of subjects",
                "n_tasks": "# of tasks",
                "pathology": "Pathology",
                "modality": "Modality",
                "type": "Type",
            }
        )
        df = df[
            [
                "Dataset",
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
        table_path = target_dir_path / "dataset_summary_table.html"
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(html_table)
        _copy_into_static(table_path)

        # Generate KDE ridgeline plot for modality participant distributions
        try:
            kde_path = target_dir_path / "dataset_kde_modalities.html"
            kde_output = generate_modality_ridgeline(df_raw, kde_path)
            if kde_output:
                _copy_into_static(kde_output)
        except Exception as exc:  # pragma: no cover - visual asset best effort
            logger.warning("[dataset KDE] Skipped due to error: %s", exc)


def parse_freqs(value) -> str:
    if isinstance(value, str):
        freq = [int(float(f)) for f in value.strip("[]").split(",")]
        if len(freq) == 1:
            return f"{int(freq[0])}"
        else:
            return f"{int(np.median(freq))}*"

    elif isinstance(value, (int, float)) and not pd.isna(value):
        return f"{int(value)}"
    return ""  # for other types like nan


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_dir", type=str)
    parser.add_argument("target_dir", type=str)
    parser.add_argument(
        "--bubble-limit",
        type=int,
        default=None,
        help="Optional maximum number of datasets to include in the bubble chart (default: adapter configuration).",
    )
    parser.add_argument(
        "--bubble-layout-seed",
        type=int,
        default=42,
        help="Deterministic layout seed forwarded to grouped bubbles (use a negative value to disable).",
    )
    parser.add_argument(
        "--bubble-include-type",
        action="store_true",
        help="Enable Type Subject badge legends in the grouped bubble output.",
    )
    args = parser.parse_args()
    layout_seed = args.bubble_layout_seed
    if layout_seed is not None and layout_seed < 0:
        layout_seed = None
    main(
        args.source_dir,
        args.target_dir,
        bubble_limit=args.bubble_limit,
        bubble_layout_seed=layout_seed,
        include_type_subject_badges=args.bubble_include_type,
    )
    print(args.target_dir)
