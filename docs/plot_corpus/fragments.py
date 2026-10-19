"""Write every corpus fragment (``<key>.html``), ``numbers.inc``, ``manifest.json`` and the CSV downloads.

The RST pages embed a fragment with ``.. corpus-figure:: <key>``; the
directive validates the key against ``manifest.json``.
"""

from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path

from . import charts
from .data import (
    BLOCKED,
    CONVERTED_V1,
    CSV_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    EXCLUDED,
    GATED,
    RECORDING_COLUMNS,
    TO_CONVERT,
    TO_INGEST,
    numbers,
    of_status,
    read_buckets,
    read_electrodes,
    read_optional,
    read_rows,
    read_unplaced,
    v0_rows,
    v1_added,
    v1_pending,
    v1_rows,
)  # fmt: skip
from .electrodes import check_coverage, electrode_svg, electrodes_3d
from .svg import legend_html, order_toggle
from .table import COMPACT, table_html

FRAGMENT_LIMIT = 200_000  # bytes per fragment
V0_NAME, V1_NAME = "eegdash-corpus-v0", "eegdash-corpus-v1"
ADDED = "eegdash-corpus-v1-channels.csv"  # channel sets the v1 store adds to v0 (its buckets v0 lacks); optional
# Per-dataset recording lengths, recordings and hours per subject, tokens (v1: the added datasets only), and the
# subjects' ages and sex where known; optional, a chart without its table says so.
REC_V0, REC_V1 = "eegdash-corpus-v0-recordings.csv", "eegdash-corpus-v1-recordings.csv"
DEMOGRAPHICS = "eegdash-corpus-demographics.csv"
INPUTS = [
    "atlas.csv",
    "eegdash-corpus-v0-channels.csv",
    ADDED,
    REC_V0,
    REC_V1,
    DEMOGRAPHICS,
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
    added=None,
    recordings=None,
    recordings_v1=None,
    demographics=None,
    unplaced=None,
) -> dict[str, tuple[str, str]]:
    """Key -> (fragment HTML, caption). Pure: no file is read or written. The counts, status charts, roadmap and
    full table are over every row (the paper's datasets and the release card's entries); the v0 and v1 pages over
    the rows of their statuses. ``unplaced`` are the channel names the position bank lacks (not drawn, counted in
    the electrode captions).
    """
    added, recordings, demographics = added or [], recordings or [], demographics or []
    rec_v1 = recordings + (
        recordings_v1 or []
    )  # the v1 file holds only the datasets v1 adds
    check_coverage(electrodes, buckets + added, unplaced)
    v0, v1 = v0_rows(rows), v1_rows(rows)
    n_ds = len({b["dataset"] for b in buckets})
    n_ds_v1 = n_ds + len({b["dataset"] for b in added})
    legend_all = legend_html([r["status"] for r in rows])
    legend_v1 = legend_html(
        [r["status"] for r in v1]
    )  # only the statuses v1 holds: no "to ingest" once nothing waits
    out = _Fragments()
    add = out.add
    add("kpis", charts.kpis_universe(rows))
    add("licences", charts.licences_chart(rows), legend_all)
    add("progress", charts.progress_html(rows))
    add("roadmap", charts.roadmap_html(rows, synced))
    add("flow", charts.flow_svg(rows))
    add("status", charts.status_svg(rows), legend_all)
    add("table-all", (table_html(rows, "atlas.csv"), table_all_caption(rows)))
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
    # v1
    new, pending = v1_added(rows), v1_pending(rows)
    add(
        "v1-adds-table",
        (
            table_html(new, "atlas-v1.csv", columns=COMPACT, controls=False),
            adds_caption(new, added),
        ),
    )
    add("v1-kpis", charts.store_kpis(v1, buckets + added, electrodes, added=True))
    add(
        "electrodes-v1",
        electrode_svg(
            electrodes,
            n_ds_v1,
            len(buckets) + len(added),
            added=True,
            unplaced=unplaced,
        ),
    )
    add(
        "electrodes-3d-v1",
        electrodes_3d(
            electrodes,
            n_ds_v1,
            len(buckets) + len(added),
            added=True,
            unplaced=unplaced,
        ),
    )
    h, c = charts.hours_chart(v1)
    add("hours-v1", (order_toggle() + h, c), legend_v1)
    h, c = charts.channels_chart(v1)
    add("channels-v1", (order_toggle() + h, c), legend_v1)
    # every v1 dataset in a store: measured per bucket as on v0; else by each dataset's mean channel count
    bands_v1 = (
        charts.channel_bands_rows(v1)
        if pending
        else charts.channel_bands_buckets(buckets, added)
    )
    add("channel-bands-v1", bands_v1, legend_v1)
    add("subjects-v1", charts.subjects_chart(v1), legend_v1)
    add("rates-v1", charts.rates_chart(v1), legend_v1)
    add("paradigms-v1", charts.paradigms_chart(v1), legend_v1)
    add("scatter-v1", charts.scatter_chart(v1), legend_v1)
    add("lengths-v1", charts.lengths_chart(v1, rec_v1), legend_v1)
    add("per-subject-v1", charts.per_subject_chart(v1, rec_v1), legend_v1)
    add("people-v1", charts.people_chart(v1, demographics), legend_v1)
    add("flow-licence-v1", charts.flow_licence_chart(v1), legend_v1)
    add("tokens-v1", charts.tokens_chart(v1, rec_v1, pending=pending))
    add("montages-v1", charts.montage_chart(v1, buckets + added, electrodes))
    add(
        "table-v1",
        (
            table_html(v1, "atlas-v1.csv"),
            f"The {len(v1)} datasets of {V1_NAME}. Click a header to sort, hover a number for its source.",
        ),
    )
    return {k: (figure(k, body, cap, leg), cap) for k, (body, cap, leg) in out.items()}


def table_all_caption(rows: list[dict[str, str]]) -> str:
    """The full table's caption: how many of the datasets tracked will not be converted (blocked, gated, excluded)."""
    out = len(of_status(rows, BLOCKED, GATED, EXCLUDED))
    pending = len(v1_pending(rows))
    wait = f", {pending} more in v1 but in no store yet" if pending else ""
    return (f"All {len(rows)} datasets tracked (REVE's paper and its release card); {out} of them will not be "
            f"converted{wait}. Filter by text or by status, click a header to sort, hover a number for its source.")  # fmt: skip


def adds_caption(new: list[dict[str, str]], added: list[dict[str, str]]) -> str:
    """The caption of the table of what v1 adds, each status named only when it holds a dataset."""
    n = {s: len(of_status(new, s)) for s in (CONVERTED_V1, TO_CONVERT, TO_INGEST)}
    said = [f"{n[s]} {words}" for s, words in ((CONVERTED_V1, "converted into the v1 store"),
                                              (TO_CONVERT, "ingested by EEGDash for the next conversion pass"),
                                              (TO_INGEST, "still to be ingested")) if n[s]]  # fmt: skip
    conv = {r["dataset_id"] for r in of_status(new, CONVERTED_V1)}
    sets = [
        b for b in added if b["dataset"] in conv
    ]  # summed over the store's buckets, not the rounded rows
    cap = f"The {len(new)} datasets v1 adds: {', '.join(said) or 'none'}"
    if sets:
        cap += f"; {sum(float(b['hours']) for b in sets):,.1f} h in {len(sets)} channel sets, measured in the v1 store"
    if n[TO_CONVERT] or n[TO_INGEST]:
        cap += "; the others' numbers come from EEGDash's catalogue"
    return cap + "."


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
    buckets = read_buckets(data / "eegdash-corpus-v0-channels.csv")
    electrodes = read_electrodes(data / "electrodes.json")
    unplaced = read_unplaced(data / "electrodes.json")
    added = read_buckets(data / ADDED) if (data / ADDED).is_file() else []
    recordings = read_optional(data / REC_V0, RECORDING_COLUMNS)
    recordings_v1 = read_optional(data / REC_V1, RECORDING_COLUMNS)
    demographics = read_optional(data / DEMOGRAPHICS, DEMOGRAPHIC_COLUMNS)
    frags = build_fragments(
        rows,
        buckets,
        electrodes,
        synced,
        added,
        recordings,
        recordings_v1,
        demographics,
        unplaced,
    )
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for key, (text, cap) in frags.items():
        size = len(text.encode())
        if size > FRAGMENT_LIMIT:
            raise ValueError(
                f"fragment {key} is {size:,} bytes, over the {FRAGMENT_LIMIT:,} limit"
            )
        (out / f"{key}.html").write_text(text, encoding="utf-8")
        entries[key] = {"file": f"{key}.html", "caption": cap, "bytes": size}
    for name, sub in (
        ("atlas.csv", rows),
        ("atlas-v0.csv", v0_rows(rows)),
        ("atlas-v1.csv", v1_rows(rows)),
    ):
        (out / name).write_text(csv_text(sub), encoding="utf-8")
    nums = numbers(rows, buckets, electrodes, synced, added)
    (out / "numbers.inc").write_text(numbers_inc(nums), encoding="utf-8")
    manifest = {"about": "Generated by docs/prepare_corpus.py, do not hand-edit.", "synced": synced,
                "source_commit": meta.get("source_commit", ""), "fragments": entries, "numbers": nums,
                "rows": {"atlas": len(rows), "v0": len(v0_rows(rows)), "v1": len(v1_rows(rows)),
                         "buckets": len(buckets), "added": len(added), "electrodes": len(electrodes),
                         "unplaced": len(unplaced),
                         "recordings": len(recordings), "recordings_v1": len(recordings_v1),
                         "demographics": len(demographics)}}  # fmt: skip
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
