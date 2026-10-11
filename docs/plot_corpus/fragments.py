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
    ATLAS_COLUMNS,
    CONVERTED,
    TO_CONVERT,
    TO_INGEST,
    numbers,
    of_status,
    read_buckets,
    read_electrodes,
    read_rows,
    v0_rows,
    v1_added,
    v1_rows,
)  # fmt: skip
from .electrodes import check_coverage, electrode_svg, electrodes_3d
from .svg import legend_html, order_toggle
from .table import COMPACT, table_html

FRAGMENT_LIMIT = 200_000  # bytes per fragment
V0_NAME, V1_NAME = "eegdash-corpus-v0", "eegdash-corpus-v1"
ADDED = "eegdash-channel-sets.csv"  # channel sets of the datasets v1 adds (EEGDash records); optional
INPUTS = [
    "atlas.csv",
    "reve-public-nc-channels.csv",
    ADDED,
    "electrodes.json",
    "MANIFEST.json",
]


def figure(key: str, body: str, caption: str, legend: str = "") -> str:
    return (f'<figure class="corpus-figure" data-corpus-key="{key}">{legend}{body}'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')  # fmt: skip


def csv_text(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(
        buf, fieldnames=ATLAS_COLUMNS, lineterminator="\n", extrasaction="ignore"
    )
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def add_fragment(
    out: dict[str, tuple[str, str, str]],
    key: str,
    pair: tuple[str, str],
    legend: str = "",
) -> None:
    out[key] = (pair[0], pair[1], legend)


def build_fragments(
    rows, buckets, electrodes, synced: str, added=None
) -> dict[str, tuple[str, str]]:
    """Key -> (fragment HTML, caption). Pure: no file is read or written."""
    added = added or []
    check_coverage(electrodes, buckets + added)
    v0, v1 = v0_rows(rows), v1_rows(rows)
    n_ds = len({b["dataset"] for b in buckets})
    n_ds_v1 = n_ds + len({b["dataset"] for b in added})
    legend_all = legend_html([r["status"] for r in rows])
    legend_v1 = legend_html([CONVERTED, TO_CONVERT, TO_INGEST])
    out: dict[str, tuple[str, str, str]] = {}

    add_fragment(out, "kpis", charts.kpis_universe(rows))
    add_fragment(out, "licences", charts.licences_chart(rows), legend_all)
    add_fragment(out, "progress", charts.progress_html(rows))
    add_fragment(out, "roadmap", charts.roadmap_html(rows, synced))
    add_fragment(out, "flow", charts.flow_svg(rows))
    add_fragment(out, "status", charts.status_svg(rows), legend_all)
    n_pending = sum(r["status"] != CONVERTED for r in rows)
    add_fragment(out, "table-all", (table_html(rows, "atlas.csv"),
                                    f"All {len(rows)} datasets tracked, {n_pending} of them not yet converted. "
                                    f"Click a header to sort, hover a number for its source."))  # fmt: skip
    # v0
    add_fragment(out, "v0-kpis", charts.kpis_v0(v0, buckets, electrodes))
    h, c = charts.hours_chart(v0)
    add_fragment(out, "hours", (order_toggle() + h, c))
    h, c = charts.channels_chart(v0)
    add_fragment(out, "channels", (order_toggle() + h, c))
    add_fragment(out, "electrodes", electrode_svg(electrodes, n_ds, len(buckets)))
    add_fragment(out, "electrodes-3d", electrodes_3d(electrodes, n_ds, len(buckets)))
    add_fragment(out, "channel-bands", charts.channel_bands_buckets(buckets))
    add_fragment(out, "subjects", charts.subjects_chart(v0))
    add_fragment(out, "rates", charts.rates_chart(v0))
    add_fragment(out, "paradigms", charts.paradigms_chart(v0))
    add_fragment(out, "scatter", charts.scatter_chart(v0))
    add_fragment(
        out,
        "table-v0",
        (
            table_html(v0, "atlas-v0.csv", status_filter=False),
            f"The {len(v0)} datasets of {V0_NAME}. Click a header to sort, hover a number for its source.",
        ),
    )
    # v1
    new = v1_added(rows)
    n_conv, n_ing = len(of_status(new, TO_CONVERT)), len(of_status(new, TO_INGEST))
    add_fragment(out, "v1-adds-table", (table_html(new, "atlas-v1.csv", columns=COMPACT, controls=False),
                                        f"The {len(new)} datasets v1 adds: {n_conv} ingested by EEGDash and not yet converted, "
                                        f"{n_ing} still to be ingested; numbers from EEGDash's records until converted."))  # fmt: skip
    add_fragment(out, "v1-kpis", charts.kpis_v1(v1))
    add_fragment(
        out,
        "electrodes-v1",
        electrode_svg(electrodes, n_ds_v1, len(buckets) + len(added), added=True),
    )
    add_fragment(
        out,
        "electrodes-3d-v1",
        electrodes_3d(electrodes, n_ds_v1, len(buckets) + len(added), added=True),
    )
    h, c = charts.hours_chart(v1)
    add_fragment(out, "hours-v1", (order_toggle() + h, c), legend_v1)
    h, c = charts.channels_chart(v1)
    add_fragment(out, "channels-v1", (order_toggle() + h, c), legend_v1)
    add_fragment(out, "channel-bands-v1", charts.channel_bands_rows(v1), legend_v1)
    add_fragment(out, "subjects-v1", charts.subjects_chart(v1), legend_v1)
    add_fragment(out, "rates-v1", charts.rates_chart(v1), legend_v1)
    add_fragment(out, "paradigms-v1", charts.paradigms_chart(v1), legend_v1)
    add_fragment(out, "scatter-v1", charts.scatter_chart(v1), legend_v1)
    add_fragment(
        out,
        "table-v1",
        (
            table_html(v1, "atlas-v1.csv", pending_option=False),
            f"The {len(v1)} datasets of {V1_NAME}. Click a header to sort, hover a number for its source.",
        ),
    )
    return {k: (figure(k, body, cap, leg), cap) for k, (body, cap, leg) in out.items()}


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
    buckets = read_buckets(data / "reve-public-nc-channels.csv")
    electrodes = read_electrodes(data / "electrodes.json")
    added = read_buckets(data / ADDED) if (data / ADDED).is_file() else []
    frags = build_fragments(rows, buckets, electrodes, synced, added)
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
                         "buckets": len(buckets), "added": len(added), "electrodes": len(electrodes)}}  # fmt: skip
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
