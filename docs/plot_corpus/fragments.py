"""Write every corpus fragment (``<key>.html``), ``numbers.inc``, ``manifest.json`` and the CSV downloads (the two
table views and the corpus table, ``eegdash-corpus-v0.csv``, copied byte for byte from the data folder).

The RST pages embed a fragment with ``.. corpus-figure:: <key>``; the
directive validates the key against ``manifest.json``.
"""

from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path

from . import charts, paradigms_table
from .data import (
    BLOCKED,
    CSV_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    EXCLUDED,
    GATED,
    RECORDING_COLUMNS,
    TABLE,
    numbers,
    of_status,
    read_buckets,
    read_electrodes,
    read_optional,
    read_rows,
    read_table,
    read_unplaced,
    reason,
    v0_rows,
)  # fmt: skip
from .electrodes import check_coverage, electrode_svg, electrodes_3d
from .svg import legend_html, order_toggle
from .table import DISPLAY_ALL, table_html

FRAGMENT_LIMIT = 200_000  # bytes per fragment
V0_NAME = "eegdash-corpus-v0"
BUCKETS = (
    "eegdash-corpus-v0-channels.csv"  # one row per channel set (bucket) of the v0 store
)
# Per-dataset recording lengths, recordings and hours per subject, tokens, and the subjects' ages and sex where
# known; optional, a chart without its table says so.
REC_V0 = "eegdash-corpus-v0-recordings.csv"
DEMOGRAPHICS = "eegdash-corpus-demographics.csv"
INPUTS = [
    "atlas.csv",
    BUCKETS,
    REC_V0,
    DEMOGRAPHICS,
    TABLE,
    "electrodes.json",
    "MANIFEST.json",
]


def figure(key: str, body: str, caption: str, legend: str = "") -> str:
    return (f'<figure class="corpus-figure" data-corpus-key="{key}">{legend}{body}'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')  # fmt: skip


def csv_text(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(
        buf, fieldnames=CSV_COLUMNS, lineterminator="\n", extrasaction="ignore"
    )
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


class _Fragments(dict):
    """key -> (body, caption, legend); ``add`` keeps the builder's lines short."""

    def add(self, key: str, pair: tuple[str, str], legend: str = "") -> None:
        self[key] = (pair[0], pair[1], legend)


def build_fragments(
    rows,
    buckets,
    electrodes,
    synced: str,
    recordings=None,
    demographics=None,
    unplaced=None,
) -> dict[str, tuple[str, str]]:
    """Key -> (fragment HTML, caption). Pure: no file is read or written. The counts, the licence chart and the
    full table are over every row (the paper's datasets and the release card's entries); the v0 page over v0's rows.
    ``unplaced`` are the channel names the position bank lacks (not drawn, counted in the electrode captions).
    """
    recordings, demographics = recordings or [], demographics or []
    check_coverage(electrodes, buckets, unplaced)
    v0 = v0_rows(rows)
    n_ds = len({b["dataset"] for b in buckets})
    legend_all = legend_html([r["status"] for r in rows])
    out = _Fragments()
    add = out.add
    # the universe: landing page and introduction
    add("kpis", charts.kpis_universe(rows))
    add("intro-kpis", charts.kpis_intro(rows))
    add("licences", charts.licences_chart(rows), legend_all)
    add(
        "table-all",
        (table_html(rows, "atlas.csv", columns=DISPLAY_ALL), table_all_caption(rows)),
    )
    # v0
    add("v0-kpis", charts.store_kpis(v0, buckets, electrodes))
    h, c = charts.hours_chart(v0)
    add("hours", (order_toggle() + h, c))
    h, c = charts.channels_chart(v0)
    add("channels", (order_toggle() + h, c))
    add("electrodes", electrode_svg(electrodes, n_ds, len(buckets), unplaced=unplaced))
    add(
        "electrodes-3d",
        electrodes_3d(electrodes, n_ds, len(buckets), unplaced=unplaced),
    )
    add("channel-bands", charts.channel_bands_buckets(buckets))
    add("subjects", charts.subjects_chart(v0))
    add("rates", charts.rates_chart(v0))
    add("paradigms", charts.paradigms_chart(v0))
    add(
        "paradigms-table", paradigms_table.paradigms_table_html(v0, buckets, recordings)
    )
    add("scatter", charts.scatter_chart(v0))
    add("lengths", charts.lengths_chart(v0, recordings))
    add("per-subject", charts.per_subject_chart(v0, recordings))
    add("people", charts.people_chart(v0, demographics))
    add("flow-licence", charts.flow_licence_chart(v0))
    add("tokens", charts.tokens_chart(v0, recordings))
    add("montages", charts.montage_chart(v0, buckets, electrodes))
    add(
        "table-v0",
        (
            table_html(v0, "atlas-v0.csv", status_filter=False),
            f"The {len(v0)} datasets of {V0_NAME}. Click a header to sort, hover a number for its source.",
        ),
    )
    return {k: (figure(k, body, cap, leg), cap) for k, (body, cap, leg) in out.items()}


def table_all_caption(rows: list[dict[str, str]]) -> str:
    """The full table's caption: how many datasets are not in the corpus, and where the reason is."""
    left = of_status(rows, BLOCKED, GATED, EXCLUDED)
    if not all(reason(r) for r in left):
        raise ValueError(
            f"no reason for {[r['dataset_id'] for r in left if not reason(r)]}"
        )
    out = len(left)
    return (f"All {len(rows)} datasets tracked (REVE's paper and its release card); the {out} not in the corpus "
            f"say why in the Reason column, and their notes give the detail. Filter by text or by status, click a "
            f"header to sort, hover a number for its source.")  # fmt: skip


def check_table(table: list[dict[str, str]], rows: list[dict[str, str]]) -> None:
    """The downloadable corpus table lists exactly v0's datasets: a reader who downloads it gets the corpus the
    pages describe.
    """
    have = {r["dataset_id"].strip() for r in table}
    want = {r["dataset_id"] for r in v0_rows(rows)}
    if have != want:
        raise ValueError(
            f"{TABLE} and atlas.csv's v0 rows differ: only in the table {sorted(have - want)}, "
            f"only in the atlas {sorted(want - have)}"
        )


def numbers_inc(nums: dict[str, str]) -> str:
    lines = [
        ".. Generated by docs/prepare_corpus.py from plot_corpus/data, do not hand-edit.",
        "",
    ]
    lines += [f".. |{k}| replace:: {v}" for k, v in nums.items()]
    return "\n".join(lines) + "\n"


def build(data: Path, out: Path) -> dict:
    """Read the data folder, write everything into ``out``; returns the manifest."""
    data, out = Path(data), Path(out)
    meta = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    synced = meta["synced"]
    rows = read_rows(data / "atlas.csv")
    buckets = read_buckets(data / BUCKETS)
    electrodes = read_electrodes(data / "electrodes.json")
    unplaced = read_unplaced(data / "electrodes.json")
    recordings = read_optional(data / REC_V0, RECORDING_COLUMNS)
    demographics = read_optional(data / DEMOGRAPHICS, DEMOGRAPHIC_COLUMNS)
    frags = build_fragments(
        rows, buckets, electrodes, synced, recordings, demographics, unplaced
    )
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob(
        "*.html"
    ):  # a fragment no page uses any more (a removed view) is not left behind
        if stale.stem not in frags:
            stale.unlink()
    for stale in out.glob("atlas-*.csv"):
        if stale.name != "atlas-v0.csv":
            stale.unlink()
    entries = {}
    for key, (text, cap) in frags.items():
        size = len(text.encode())
        if size > FRAGMENT_LIMIT:
            raise ValueError(
                f"fragment {key} is {size:,} bytes, over the {FRAGMENT_LIMIT:,} limit"
            )
        (out / f"{key}.html").write_text(text, encoding="utf-8")
        entries[key] = {"file": f"{key}.html", "caption": cap, "bytes": size}
    for name, sub in (("atlas.csv", rows), ("atlas-v0.csv", v0_rows(rows))):
        (out / name).write_text(csv_text(sub), encoding="utf-8")
    table = read_table(data / TABLE)
    check_table(table, rows)
    (out / TABLE).write_bytes(
        (data / TABLE).read_bytes()
    )  # the synced file itself: its hash is MANIFEST.json's
    nums = numbers(rows, buckets, electrodes, synced, recordings)
    (out / "numbers.inc").write_text(numbers_inc(nums), encoding="utf-8")
    manifest = {"about": "Generated by docs/prepare_corpus.py, do not hand-edit.", "synced": synced,
                "source_commit": meta.get("source_commit", ""), "fragments": entries, "numbers": nums,
                "rows": {"atlas": len(rows), "v0": len(v0_rows(rows)), "buckets": len(buckets),
                         "electrodes": len(electrodes), "unplaced": len(unplaced),
                         "recordings": len(recordings), "demographics": len(demographics),
                         "table": len(table)}}  # fmt: skip
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def is_stale(data: Path, out: Path) -> bool:
    """True when the outputs are missing or older than any data file or renderer module."""
    m = Path(out) / "manifest.json"
    if not m.is_file():
        return True
    newest = max([(Path(data) / n).stat().st_mtime for n in INPUTS if (Path(data) / n).is_file()]
                 + [p.stat().st_mtime for p in Path(__file__).parent.glob("*.py")])  # fmt: skip
    return newest > m.stat().st_mtime
