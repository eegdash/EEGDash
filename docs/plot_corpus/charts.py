"""The charts of the corpus pages. Each returns ``(html, caption)``; the caption is computed from the same rows.

Every chart takes the rows it shows (``rows`` already filtered to a version
or the universe), so the v0-only and v0+v1 variants share code.
"""

from __future__ import annotations

import html
import math
import re

from .data import (
    BLOCKED,
    CONVERTED,
    EXCLUDED,
    GATED,
    MONTAGE_MIN,
    SHORT,
    SLUGS,
    STATUSES,
    TENTEN,
    TO_CONVERT,
    TO_INGEST,
    WORDS,
    by_id,
    electrode_sets,
    fnum,
    hours_of,
    kpis,
    label_of,
    montage_names,
    name_of,
    progress,
    roadmap_items,
)  # fmt: skip
from .layout import sankey_flows, sankey_layout, squarify
from .svg import (
    LEFT,
    PITCH,
    axis_strip,
    dataset_bars,
    fill,
    grid_lines,
    hbars_svg,
    hscroll_frame,
    log_ticks,
    range_marks,
    range_rows,
    scroll_frame,
    svg_open,
    tick_text,
    value_axis,
)  # fmt: skip

e = html.escape
CHANNEL_BANDS = [(1, 16), (17, 32), (33, 64), (65, 128), (129, None)]
SUBJECT_BANDS = [(1, 10), (11, 30), (31, 100), (101, 300), (301, None)]


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.0f} %" if b else "0 %"


def median(vals: list[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2] if s else 0.0


# ------------------------------------------------------------------------------------------------- numbers, steps


def kpi_html(boxes: list[tuple[str, str, str | None]]) -> str:
    """KPI boxes: (number, label, status or None); a status box carries the status colour on its edge."""
    out = []
    for b, s, st in boxes:
        cls = f" corpus-bd-{SLUGS[st]}" if st else ""
        out.append(
            f'<div class="corpus-kpi{cls}"><b>{e(b)}</b><span>{e(s)}</span></div>'
        )
    return '<div class="corpus-kpis">' + "".join(out) + "</div>"


def kpis_universe(rows: list[dict[str, str]]) -> tuple[str, str]:
    """The corpus in numbers, over every dataset tracked (REVE's paper and its release card)."""
    k = kpis(rows)
    ing = f", {k['v1_to_ingest']} to ingest" if k["v1_to_ingest"] else ""
    boxes = [
        (f"{k['v0_datasets']}", "datasets converted (v0)", CONVERTED),
        (f"{k['v0_hours']:,.0f}", "hours converted (v0)", CONVERTED),
        (f"{k['v0_subjects']:,}", "subjects converted (v0)", CONVERTED),
        (
            f"{k['v1_datasets']}",
            f"datasets in v1 (+{k['v1_to_convert']} to convert{ing})",
            TO_CONVERT,
        ),
        (f"{k['v1_hours']:,.0f}", "hours in v1 (measured so far)", TO_CONVERT),
        (f"{k[BLOCKED]}", "blocked (ND licence)", BLOCKED),
        (f"{k[GATED]}", "gated (distribution forbidden)", GATED),
        (f"{k['datasets']}", "datasets tracked (paper and release card)", None),
    ]
    cap = (f"{k['v0_datasets']} of the {k['datasets']} datasets tracked (REVE's paper and its release card) are "
           f"converted (v0) and {k[TO_CONVERT]} wait for conversion (v1); {k[BLOCKED]} are blocked by a NoDerivatives "
           f"licence or none, {k[GATED]} are gated (their distribution is forbidden) and {k[EXCLUDED]} excluded.")  # fmt: skip
    return kpi_html(boxes), cap


def kpis_v0(
    rows: list[dict[str, str]], buckets: list[dict[str, str]], electrodes: list[dict]
) -> tuple[str, str]:
    k = kpis(rows)
    rates = {round(fnum(r["native_sfreq"])) for r in rows if fnum(r["native_sfreq"])}
    paradigms = {r["paradigm"] for r in rows if r["paradigm"]}
    boxes = [
        (f"{k['v0_datasets']}", "datasets", CONVERTED),
        (f"{k['v0_hours']:,.0f}", "hours", CONVERTED),
        (f"{k['v0_subjects']:,}", "subjects", None),
        (f"{k['v0_sessions']:,}", "sessions", None),
        (f"{len(buckets):,}", "channel sets (buckets)", None),
        (f"{len(electrodes)}", "electrode positions", None),
        (f"{len(rates)}", "native sampling rates", None),
        (f"{len(paradigms)}", "paradigms", None),
    ]
    cap = (f"Measured in the converted Zarr store: {k['v0_datasets']} datasets, {k['v0_hours']:,.0f} h from "
           f"{k['v0_subjects']:,} subjects, in {len(buckets):,} channel sets.")  # fmt: skip
    return kpi_html(boxes), cap


def kpis_v1(rows: list[dict[str, str]]) -> tuple[str, str]:
    k = kpis(rows)
    new = [r for r in rows if r["status"] == TO_CONVERT]
    ing = [r for r in rows if r["status"] == TO_INGEST]
    unmeasured = [r for r in new + ing if not fnum(r["hours"])]
    boxes = [
        (f"{k['v1_datasets']}", "datasets", None),
        (f"{k['v0_datasets']}", "converted (v0)", CONVERTED),
        (f"{len(new)}", "to convert", TO_CONVERT),
        *(
            [(f"{len(ing)}", "to ingest", TO_INGEST)] if ing else []
        ),  # only while a dataset waits for a deposit
        (f"{k['v1_hours']:,.0f}", "hours (measured so far)", None),
        (f"{k['v1_subjects']:,}", "subjects", None),
    ]
    cap = (
        f"v1 holds {k['v1_datasets']} datasets: v0's {k['v0_datasets']}"
        + (
            f", {len(new)} to convert and {len(ing)} to ingest"
            if ing
            else f" and {len(new)} to convert"
        )
        + "; the numbers of those come from EEGDash's records until converted"
        + (
            f" ({', '.join(name_of(r, 40) for r in unmeasured)} unmeasured)."
            if unmeasured
            else "."
        )
    )
    return kpi_html(boxes), cap


def progress_html(rows: list[dict[str, str]]) -> tuple[str, str]:
    steps = progress(rows)
    bar = "".join(f'<i class="corpus-bg-{SLUGS[st["status"]]}" style="flex:{st["count"]}" '
                  f'data-tip="{e(st["title"])}: {st["count"]} datasets"></i>' for st in steps if st["count"])  # fmt: skip
    lis = "".join(f'<li class="corpus-bd-{SLUGS[st["status"]]}" data-status="{e(st["status"])}" '
                  f'data-count="{st["count"]}"><b>{st["count"]}</b><span>{e(st["title"])}</span>'
                  f'<small>{e(st["sub"])}</small></li>' for st in steps)  # fmt: skip
    by = {st["status"]: st for st in steps}
    cap = (f"Of the {kpis(rows)['datasets']} datasets tracked, {by[CONVERTED]['count']} are converted, "
           f"{by[TO_CONVERT]['count']} ingested and waiting for conversion, {by[TO_INGEST]['count']} still to ingest; "
           f"{by[BLOCKED]['count']} are blocked by their licence and {by[GATED]['count']} gated (distribution "
           f"forbidden).")  # fmt: skip
    return (
        f'<div class="corpus-pbar">{bar}</div><ol class="corpus-psteps">{lis}</ol>',
        cap,
    )


def roadmap_html(rows: list[dict[str, str]], synced: str) -> tuple[str, str]:
    items = roadmap_items(rows)
    lis = "".join(f"<li>{e(t)}</li>" for t in items)
    return (f'<ol class="corpus-roadmap">{lis}</ol>',
            f"As of {synced}, the date of the data on this page; in order of what comes next.")  # fmt: skip


def flow_svg(rows: list[dict[str, str]]) -> tuple[str, str]:
    """REVE's paper -> release card -> v0 -> v1 (-> to be ingested, while any dataset is), over every row: the
    paper's datasets, the card's entries (the ones the paper does not name counted beside), the versions.
    """
    k = kpis(rows)
    n = lambda col: sum(r[col] == "yes" for r in rows)  # noqa: E731
    hrs = lambda col: hours_of([r for r in rows if r[col] == "yes"])  # noqa: E731
    card_only = k["card_only"]
    steps = [
        dict(key="paper", title="REVE's paper", count=n("in_reve_paper"), sub="datasets named", status=None),
        dict(key="card", title="Release card", count=n("in_release_card"),
             sub=f"entries ({card_only} not in the paper)" if card_only else "entries", status=None),
        dict(key="v0", title="eegdash-corpus-v0", count=n("in_v0"), sub=f"{hrs('in_v0'):,.0f} h", status=CONVERTED),
        dict(key="v1", title="eegdash-corpus-v1", count=n("in_v1"), sub=f"{hrs('in_v1'):,.0f} h", status=TO_CONVERT),
    ]  # fmt: skip
    if k[TO_INGEST]:
        steps.append(dict(key="ingest", title="To be ingested", count=k[TO_INGEST], sub=f"+ {k[BLOCKED]} blocked",
                          status=TO_INGEST))  # fmt: skip
    bw, gap, h = 164, 45, 96
    parts = [
        svg_open(
            len(steps) * bw + (len(steps) - 1) * gap,
            h,
            "flow from REVE's paper to the corpus versions",
        )
    ]
    for j, st in enumerate(steps):
        x = j * (bw + gap)
        if st["status"]:
            slug = SLUGS[st["status"]]
            box = (f'<rect x="{x + 1}" y="8" width="{bw - 2}" height="{h - 16}" rx="8" class="corpus-fill-{slug} '
                   f'corpus-stroke-{slug}" fill-opacity="0.14" stroke-width="1.5"/>')  # fmt: skip
        else:
            box = (f'<rect x="{x + 1}" y="8" width="{bw - 2}" height="{h - 16}" rx="8" fill="none" '
                   f'stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>')  # fmt: skip
        parts.append(f'<g data-step="{st["key"]}" data-count="{st["count"]}">{box}'
                     f'<text x="{x + bw / 2}" y="30" text-anchor="middle" font-weight="600">{e(st["title"])}</text>'
                     f'<text x="{x + bw / 2}" y="58" text-anchor="middle" font-size="24" font-weight="700">'
                     f'{st["count"]}</text><text x="{x + bw / 2}" y="78" text-anchor="middle" opacity="0.7">'
                     f'{e(st["sub"])}</text></g>')  # fmt: skip
        if j < len(steps) - 1:
            ax = x + bw + 6
            parts.append(f'<path d="M{ax} {h / 2}h{gap - 16}m-7 -6l7 6l-7 6" fill="none" stroke="currentColor" '
                         f'stroke-opacity="0.5" stroke-width="1.5"/>')  # fmt: skip
    parts.append("</svg>")
    by = {st["key"]: st for st in steps}
    v0h, v1h = hrs("in_v0"), hrs("in_v1")
    ni = k[TO_INGEST]
    wait = (
        f", and {ni} open dataset{' waits' if ni == 1 else 's wait'} to be ingested"
        if ni
        else ""
    )
    cap = (f"{by['v0']['count']} of the release card's {by['card']['count']} entries are converted ({v0h:,.0f} h); "
           f"v1 adds {by['v1']['count'] - by['v0']['count']} more of the paper's {by['paper']['count']} datasets "
           f"({v1h - v0h:,.0f} h){wait}.")  # fmt: skip
    return "".join(parts), cap


def status_svg(rows: list[dict[str, str]]) -> tuple[str, str]:
    """One stacked bar of every dataset tracked, per status."""
    k = kpis(rows)
    W, H = 1000, 46
    parts, x = [svg_open(W, H, "datasets by status")], 0.0
    for s in STATUSES:
        if not k[s]:
            continue
        w = k[s] / k["datasets"] * W
        label = (f'<text x="{x + w / 2:.1f}" y="{H / 2 + 5}" text-anchor="middle" class="corpus-on-mark" '
                 f'font-size="14" font-weight="700">{k[s]}</text>' if w > 22 else "")  # fmt: skip
        parts.append(f'<g data-status="{e(s)}" data-count="{k[s]}" data-tip="{e(s)}: {k[s]} datasets">'
                     f'<rect x="{x:.1f}" y="4" width="{w:.1f}" height="{H - 8}" class="{fill(s)}"/>{label}</g>')  # fmt: skip
        x += w
    parts.append("</svg>")
    out = k[BLOCKED] + k[GATED] + k[EXCLUDED]
    ing = f", {k[TO_INGEST]} wait to be ingested" if k[TO_INGEST] else ""
    cap = (f"{k[CONVERTED]} of the {k['datasets']} datasets tracked are converted, {k[TO_CONVERT]} wait for "
           f"conversion{ing}; {out} are blocked ({k[BLOCKED]}), gated ({k[GATED]}) or excluded ({k[EXCLUDED]}).")  # fmt: skip
    return "".join(parts), cap


# ------------------------------------------------------------------------------------------- per-dataset bar lists


def _bar_items(rows: list[dict[str, str]], col: str, text) -> list[dict]:
    items = []
    for r in rows:
        v = fnum(r[col])
        if not v:
            continue
        items.append({
            "label": label_of(r), "name": name_of(r, 40), "value": v, "status": r["status"], "text": text(v),
            "keys": {"hours": fnum(r["hours"]) or "", "channels": fnum(r["mean_channels"]) or "",
                     "name": name_of(r, 60).lower()},
            "tip": f"{name_of(r, 60)} ({r['dataset_id']}) · {fnum(r['hours']) or 0:,.1f} h · "
                   f"{fnum(r['mean_channels']) or 0:,.1f} channels · {SHORT[r['status']]}",
        })  # fmt: skip
    return items


def hours_chart(rows: list[dict[str, str]]) -> tuple[str, str]:
    """Every dataset with hours, most first, on a log axis, in a scrolling frame with a pinned axis."""
    items = sorted(
        _bar_items(rows, "hours", lambda v: f"{v:,.1f} h"), key=lambda it: -it["value"]
    )
    body, _ = dataset_bars(
        items, log=True, unit=" h", label="hours per dataset, log scale"
    )
    vals = [it["value"] for it in items]
    tot, k, cum = sum(vals), 0, 0.0
    while cum < tot / 2:
        cum += vals[k]
        k += 1
    names = ", ".join(it["name"] for it in items[:k]) if k <= 3 else ""
    lead = (
        "The largest dataset" if k == 1 else f"The {WORDS.get(k, k)} largest datasets"
    )
    cap = (f"{lead}{f' ({names})' if names else ''} hold{'s' if k == 1 else ''} {100 * cum / tot:.0f} % of the "
           f"{tot:,.0f} h shown; the median dataset has {median(vals):,.1f} h.")  # fmt: skip
    return body, cap


def channels_chart(rows: list[dict[str, str]]) -> tuple[str, str]:
    """Mean channel count per dataset, same layout and order toggle as the hours chart."""
    items = sorted(
        _bar_items(rows, "hours", lambda v: ""), key=lambda it: -it["value"]
    )  # hours order by default
    for it in items:
        it["value"] = it["keys"]["channels"] or 0
        it["text"] = f"{it['value']:,.0f} ch" if it["value"] else ""
    items = [it for it in items if it["value"]]
    body, _ = dataset_bars(
        items, log=False, unit="", label="mean channel count per dataset", step=32
    )
    vals = [it["value"] for it in items]
    conv = [it["value"] for it in items if it["status"] == CONVERTED]
    cap = (f"Median {median(vals):,.0f} channels; {pct(sum(v > 64 for v in conv), len(conv))} of the {len(conv)} "
           f"converted datasets have more than 64 channels and {sum(v <= 32 for v in conv)} have 32 or fewer.")  # fmt: skip
    return body, cap


# ---------------------------------------------------------------------------------------------- distributions


def band_label(lo: int, hi: int | None) -> str:
    return f"{lo}–{hi}" if hi else f">{lo - 1}"


def band_index(v: float, bands: list[tuple[int, int | None]]) -> int:
    """The band holding v (inclusive integer ranges; a mean of 19.7 channels rounds into one)."""
    v = round(v)
    return next(
        k for k, (lo, hi) in enumerate(bands) if v >= lo and (hi is None or v <= hi)
    )


def group_items(
    rows: list[dict[str, str]], key, measure: str, order=None
) -> list[dict]:
    """Rows grouped by key(row) (None = left out): segments are datasets or hours per status; the label says both."""
    groups: dict = {}
    for r in rows:
        k = key(r)
        if k is not None:
            groups.setdefault(k, []).append(r)
    keys = order if order is not None else sorted(groups, key=lambda k: (-sum(
        (fnum(r["hours"]) or 0) if measure == "hours" else 1 for r in groups[k]), str(k)))  # fmt: skip
    items = []
    for k in keys:
        g = groups.get(k, [])
        segs, tips = {}, {}
        for st in STATUSES:
            sub = [r for r in g if r["status"] == st]
            if sub:
                segs[st] = hours_of(sub) if measure == "hours" else len(sub)
                tips[st] = (
                    f"{len(sub)} dataset{'s' if len(sub) != 1 else ''}, {hours_of(sub):,.0f} h"
                )
        items.append({"key": k, "label": str(k), "segs": segs, "tips": tips, "n": len(g), "hours": hours_of(g),
                      "text": f"{len(g)} dataset{'s' if len(g) != 1 else ''} · {hours_of(g):,.0f} h"})  # fmt: skip
    return items


def channel_bands_buckets(buckets: list[dict[str, str]]) -> tuple[str, str]:
    """Hours of the v0 Zarr store by channel-count band, measured per bucket (a dataset with several montages counts in
    each band it uses).
    """
    tot = sum(float(b["hours"]) for b in buckets)
    items = []
    for k, (lo, hi) in enumerate(CHANNEL_BANDS):
        sub = [
            b for b in buckets if band_index(float(b["n_chans"]), CHANNEL_BANDS) == k
        ]
        hrs, nd = sum(float(b["hours"]) for b in sub), len({b["dataset"] for b in sub})
        items.append({"label": f"{band_label(lo, hi)} channels", "segs": {CONVERTED: hrs} if hrs else {},
                      "tips": {CONVERTED: f"{nd} datasets, {len(sub)} channel sets, {hrs:,.0f} h"},
                      "text": f"{nd} datasets · {hrs:,.0f} h ({pct(hrs, tot)})", "hours": hrs, "n": nd})  # fmt: skip
    top = max(items, key=lambda it: it["hours"])
    cap = (f"{pct(top['hours'], tot)} of the {tot:,.0f} converted hours come from "
           f"{top['label'].replace(' channels', '')}-channel recordings ({top['n']} datasets); a dataset recorded "
           f"with several montages counts in each band it uses.")  # fmt: skip
    return hbars_svg(items, "hours by channel-count band"), cap


def channel_bands_rows(rows: list[dict[str, str]]) -> tuple[str, str]:
    """Hours by the band of each dataset's mean channel count, stacked by status (v1: not every dataset has buckets)."""
    data = [r for r in rows if fnum(r["mean_channels"])]
    items = group_items(data, lambda r: band_index(float(r["mean_channels"]), CHANNEL_BANDS), "hours",
                        order=list(range(len(CHANNEL_BANDS))))  # fmt: skip
    for it in items:
        it["label"] = f"{band_label(*CHANNEL_BANDS[it['key']])} channels"
    tot = sum(it["hours"] for it in items)
    top = max(items, key=lambda it: it["hours"])
    cap = (f"{pct(top['hours'], tot)} of the {tot:,.0f} h are in datasets of {top['label']} ({top['n']} datasets), "
           f"by each dataset's mean channel count.")  # fmt: skip
    return hbars_svg(items, "hours by channel-count band"), cap


def subjects_chart(rows: list[dict[str, str]]) -> tuple[str, str]:
    data = [r for r in rows if fnum(r["n_subjects"])]
    items = group_items(data, lambda r: band_index(float(r["n_subjects"]), SUBJECT_BANDS), "datasets",
                        order=list(range(len(SUBJECT_BANDS))))  # fmt: skip
    for it in items:
        it["label"] = f"{band_label(*SUBJECT_BANDS[it['key']])} subjects"
    tot = sum(it["hours"] for it in items)
    mode, big = max(items, key=lambda it: it["n"]), items[-1]
    over = SUBJECT_BANDS[-1][0] - 1
    tail = (f"the {big['n']} with more than {over} subjects hold {pct(big['hours'], tot)} of the hours" if big["n"]
            else f"none has more than {over} subjects")  # fmt: skip
    cap = f"Most datasets ({mode['n']} of {len(data)}) have {mode['label']}; {tail}."
    return hbars_svg(items, "datasets by subject count"), cap


def rates_chart(rows: list[dict[str, str]]) -> tuple[str, str]:
    data = [r for r in rows if fnum(r["native_sfreq"])]
    rates = sorted({round(float(r["native_sfreq"])) for r in data})
    items = group_items(
        data, lambda r: round(float(r["native_sfreq"])), "datasets", order=rates
    )
    for it in items:
        it["label"] = f"{it['key']:,} Hz"
    conv = [r for r in data if r["status"] == CONVERTED]
    res = [r for r in conv if round(float(r["native_sfreq"])) != 200]
    native = sorted({round(float(r["native_sfreq"])) for r in res})
    span = f", from {native[0]:,} to {native[-1]:,} Hz" if len(native) > 1 else ""
    left = len(conv) - len(res)
    cap = (f"The converter resampled {len(res)} of the {len(conv)} converted datasets to 200 Hz from "
           f"{len(native)} native rate{'s' if len(native) != 1 else ''}{span}; "
           f"{left} {'was' if left == 1 else 'were'} recorded at 200 Hz.")  # fmt: skip
    return hbars_svg(items, "datasets by native sampling rate"), cap


def paradigms_chart(rows: list[dict[str, str]]) -> tuple[str, str]:
    items = group_items(rows, lambda r: r["paradigm"] or "unspecified", "hours")
    for it in items:
        it["label"] = label_of({"short_name": it["key"]}, 24)
    tot = sum(it["hours"] for it in items)
    top, most = items[0], max(items, key=lambda it: (it["n"], -items.index(it)))
    cap = (f"{top['key'].capitalize()} holds {pct(top['hours'], tot)} of the hours with {top['n']} "
           f"dataset{'s' if top['n'] != 1 else ''}; {most['key']} has the most datasets ({most['n']}).")  # fmt: skip
    return hbars_svg(items, "hours by paradigm"), cap


def licences_chart(rows: list[dict[str, str]]) -> tuple[str, str]:
    """Datasets by licence over the rows given, stacked by status (blocked licences show in the blocked colour)."""
    items = group_items(rows, lambda r: r["licence"] or "none stated", "datasets")
    top = items[0]
    v1 = [r for r in rows if r["status"] in (CONVERTED, TO_CONVERT)]
    cond = [r for r in v1 if re.search(r"SA|NC|GPL", r["licence"] or "")]
    blk = sum(r["status"] == BLOCKED for r in rows)
    cap = (f"{top['n']} of the {len(rows)} datasets are {top['key']}; of the {len(v1)} in v1, {len(cond)} carry "
           f"share-alike, non-commercial or copyleft terms, which a converted copy keeps, and {blk} "
           f"dataset{'s are' if blk != 1 else ' is'} blocked by NoDerivatives terms or no licence.")  # fmt: skip
    return hbars_svg(items, "datasets by licence"), cap


def scatter_chart(rows: list[dict[str, str]]) -> tuple[str, str]:
    """Subjects against hours, log-log, one point per dataset with both; the five largest by hours labelled, a group
    sharing its first word (the HBN releases) labelled once at its centre.
    """
    pts = [
        (fnum(r["n_subjects"]), fnum(r["hours"]), r)
        for r in rows
        if fnum(r["n_subjects"]) and fnum(r["hours"])
    ]
    xt = log_ticks(min(p[0] for p in pts), max(p[0] for p in pts))
    yt = log_ticks(min(p[1] for p in pts), max(p[1] for p in pts))
    left, top, w, h = 60, 14, 620, 330
    lx0, lx1, ly0, ly1 = (math.log10(v) for v in (xt[0], xt[-1], yt[0], yt[-1]))
    X = lambda v: left + (math.log10(v) - lx0) / (lx1 - lx0) * w  # noqa: E731
    Y = lambda v: top + h - (math.log10(v) - ly0) / (ly1 - ly0) * h  # noqa: E731
    parts = [
        svg_open(
            left + w + 30, top + h + 44, "subjects against hours, log-log", "760px"
        )
    ]
    for t in xt:
        parts.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{top + h}" stroke="currentColor" '
                     f'stroke-opacity="0.12"/><text x="{X(t):.1f}" y="{top + h + 16}" text-anchor="middle" '
                     f'opacity="0.7">{tick_text(t)}</text>')  # fmt: skip
    for t in yt:
        parts.append(f'<line x1="{left}" x2="{left + w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="currentColor" '
                     f'stroke-opacity="0.12"/><text x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" '
                     f'opacity="0.7">{tick_text(t, " h")}</text>')  # fmt: skip
    parts.append(
        f'<text x="{left + w / 2}" y="{top + h + 36}" text-anchor="middle">subjects</text>'
    )
    for s_, hr, r in pts:
        parts.append(f'<circle cx="{X(s_):.1f}" cy="{Y(hr):.1f}" r="4.5" class="corpus-pt {fill(r["status"])}" '
                     f'fill-opacity="0.85" data-tip="{e(name_of(r, 60))} ({r["dataset_id"]}) · {s_:,.0f} subjects · '
                     f'{hr:,.1f} h"/>')  # fmt: skip
    groups: dict[str, list] = {}
    for p in sorted(pts, key=lambda p: -p[1])[:5]:
        groups.setdefault(name_of(p[2]).split()[0], []).append(p)
    for word, g in groups.items():
        cx = sum(X(p[0]) for p in g) / len(g)
        cy = sum(Y(p[1]) for p in g) / len(g)
        text = label_of(g[0][2]) if len(g) == 1 else f"{word} ×{len(g)}"
        anchor, dx = ("end", -8) if cx > left + w * 0.8 else ("start", 8)
        parts.append(f'<text class="corpus-lab" x="{cx + dx:.1f}" y="{cy + 4:.1f}" text-anchor="{anchor}" '
                     f'font-weight="600">{e(text)}</text>')  # fmt: skip
    parts.append("</svg>")
    per = sorted((hr / s_, r["dataset_id"], r) for s_, hr, r in pts)
    cap = (f"Median {per[len(per) // 2][0]:.1f} h per subject over {len(pts)} datasets; {name_of(per[-1][2], 60)} "
           f"records the most, {per[-1][0]:.1f} h per subject.")  # fmt: skip
    return "".join(parts), cap


# ------------------------------------------------------------------------------------- recordings, people, budget

MINUTE_TICKS = [
    0.1,
    1,
    10,
    60,
    600,
    6000,
]  # 0.1 min .. 100 h: a log axis read in minutes and hours


def missing(what: str) -> tuple[str, str]:
    """The fragment of a chart whose table the data copy does not carry yet (the page still builds)."""
    return (f'<p class="corpus-note">No {e(what)} in this data copy yet: the table arrives with the next sync of the '
            f"corpus tables.</p>", f"No {what} in this data copy yet.")  # fmt: skip


def joined(
    rows: list[dict[str, str]], table: list[dict[str, str]]
) -> list[tuple[dict, dict]]:
    """(atlas row, derived row) for each row of the page that the derived table describes."""
    t = by_id(table)
    return [(r, t[r["dataset_id"]]) for r in rows if r["dataset_id"] in t]


def fmt_minutes(m: float) -> str:
    if m < 10:
        return f"{m:.1f} min"
    return f"{m:.0f} min" if m < 90 else f"{m / 60:,.1f} h"


def minute_tick(m: float) -> str:
    """A tick of the minute axis: ``0.1 min``, ``1 min``, ``10 min``, ``1 h``, ``10 h``, ``100 h``."""
    return f"{m:g} min" if m < 60 else f"{m / 60:g} h"


def fmt_tokens(t: float) -> str:
    return (
        f"{t / 1e9:,.2f} B"
        if t >= 1e9
        else f"{t / 1e6:,.0f} M"
        if t >= 1e6
        else f"{t:,.0f}"
    )


def fmt_num(v: float) -> str:
    return f"{v:,.0f}" if v >= 10 or v == int(v) else f"{v:.1f}"


def weighted_median(pairs: list[tuple[float, float]]) -> float:
    """The value at half the total weight of (value, weight) pairs."""
    pairs = sorted(pairs)
    half, cum = sum(w for _, w in pairs) / 2, 0.0
    for v, w in pairs:
        cum += w
        if cum >= half:
            return v
    return pairs[-1][0] if pairs else 0.0


def lengths_chart(
    rows: list[dict[str, str]], recordings: list[dict[str, str]]
) -> tuple[str, str]:
    """Per dataset the recording lengths as quantiles (p10..p90 bar, quartile box, median tick, longest as a mark),
    longest median first, on a log minute axis. A summary of the quantiles, not a density: the per-recording
    lengths are not shipped.
    """
    pairs = [(r, q) for r, q in joined(rows, recordings) if fnum(q["minutes_p50"])]
    if not pairs:
        return missing("recording lengths")
    pairs.sort(key=lambda p: (-fnum(p[1]["minutes_p50"]), p[0]["dataset_id"]))
    items, vals = [], []
    for r, q in pairs:
        m = {
            k: fnum(q[f"minutes_{k}"])
            for k in ("p10", "p25", "p50", "p75", "p90", "max")
        }
        vals += [v for v in m.values() if v]
        items.append({
            "label": label_of(r), "status": r["status"], "text": fmt_minutes(m["p50"]),
            "q": {"lo": m["p10"], "q1": m["p25"], "mid": m["p50"], "q3": m["p75"], "hi": m["p90"], "top": m["max"]},
            "tip": f"{name_of(r, 60)} ({r['dataset_id']}) · {fmt_num(fnum(q['n_recordings']) or 0)} recordings · "
                   f"median {fmt_minutes(m['p50'])}"
                   + (f", middle half {fmt_minutes(m['p25'])}–{fmt_minutes(m['p75'])}" if m["p25"] and m["p75"] else "")
                   + (f", longest {fmt_minutes(m['max'])}" if m["max"] else "") + f" · {SHORT[r['status']]}",
        })  # fmt: skip
    ticks, frac = value_axis(vals, log=True, candidates=MINUTE_TICKS)
    panel = {"x0": LEFT, "width": 300, "ticks": ticks, "frac": frac, "fmt": minute_tick}
    body, _ = range_rows(items, panel, "recording length per dataset, log scale")
    med = weighted_median(
        [(fnum(q["minutes_p50"]), fnum(q["n_recordings"]) or 1) for _, q in pairs]
    )
    n_rec = sum(fnum(q["n_recordings"]) or 0 for _, q in pairs)
    lo, hi = pairs[-1], pairs[0]
    longest = max(
        pairs, key=lambda p: fnum(p[1]["minutes_max"]) or fnum(p[1]["minutes_p50"])
    )
    top = fnum(longest[1]["minutes_max"]) or fnum(longest[1]["minutes_p50"])
    cap = (f"The median recording lasts {fmt_minutes(med)} over {n_rec:,.0f} recordings of {len(pairs)} datasets "
           f"(per-dataset medians weighted by recordings); medians run from {fmt_minutes(fnum(lo[1]['minutes_p50']))} "
           f"({name_of(lo[0], 40)}) to {fmt_minutes(fnum(hi[1]['minutes_p50']))} ({name_of(hi[0], 40)}), and "
           f"{name_of(longest[0], 40)} has the longest recordings, up to {fmt_minutes(top)}. Bars span the 10th to "
           f"90th percentile, boxes the middle half, the tick is the median and the dot the longest recording.")  # fmt: skip
    return body, cap


def lollipop(v_mid: float | None, v_top: float | None, X, cls: str) -> str:
    """A dot at the median subject and a line to the subject with the most."""
    if v_mid is None:
        return ""
    out = ""
    if v_top is not None and v_top > v_mid:
        out += (f'<rect x="{X(v_mid):.1f}" y="7" width="{X(v_top) - X(v_mid):.1f}" height="2" class="{cls}" '
                f'fill-opacity="0.55"/><rect x="{X(v_top) - 1:.1f}" y="3" width="2" height="10" fill="currentColor" '
                f'fill-opacity="0.6"/>')  # fmt: skip
    return out + f'<circle cx="{X(v_mid):.1f}" cy="8" r="4" class="{cls}"/>'


def per_subject_chart(
    rows: list[dict[str, str]], recordings: list[dict[str, str]]
) -> tuple[str, str]:
    """Per dataset, recordings per subject and hours per subject (median subject as a dot, the subject with the
    most at the end of the line), two log panels sharing the rows, most hours per subject first.
    """
    pairs = [
        (r, q) for r, q in joined(rows, recordings) if fnum(q["hours_per_subject_p50"])
    ]
    if not pairs:
        return missing("recordings per subject")
    pairs.sort(key=lambda p: (-fnum(p[1]["hours_per_subject_p50"]), p[0]["dataset_id"]))
    get = lambda q, k: fnum(q[k])  # noqa: E731
    rec_vals = [
        v
        for _, q in pairs
        for v in (get(q, "rec_per_subject_p50"), get(q, "rec_per_subject_max"))
        if v
    ]
    hrs_vals = [
        v
        for _, q in pairs
        for v in (get(q, "hours_per_subject_p50"), get(q, "hours_per_subject_max"))
        if v
    ]
    rt, rf = value_axis(rec_vals or [1, 10], log=True)
    ht, hf = value_axis(hrs_vals, log=True)
    pa = {
        "x0": LEFT,
        "width": 220,
        "ticks": rt,
        "frac": rf,
        "fmt": tick_text,
        "title": "recordings per subject",
    }
    pb = {"x0": LEFT + 260, "width": 220, "ticks": ht, "frac": hf, "fmt": lambda v: tick_text(v, " h"),
          "title": "hours per subject"}  # fmt: skip
    XA = lambda v: pa["x0"] + rf(v) * pa["width"]  # noqa: E731
    XB = lambda v: pb["x0"] + hf(v) * pb["width"]  # noqa: E731
    W = pb["x0"] + pb["width"] + 70
    h = PITCH * len(pairs) + 4
    body = [svg_open(W, h, "recordings and hours per subject, log scales", cls="corpus-bars"),
            grid_lines([pa, pb], h)]  # fmt: skip
    for j, (r, q) in enumerate(pairs):
        cls = fill(r["status"])
        rm, rx, hm, hx = (get(q, k) for k in ("rec_per_subject_p50", "rec_per_subject_max", "hours_per_subject_p50",
                                              "hours_per_subject_max"))  # fmt: skip
        tip = (f"{name_of(r, 60)} ({r['dataset_id']}) · "
               + (f"median {fmt_num(rm)} recordings per subject" + (f" (most {fmt_num(rx)})" if rx else "") + " · "
                  if rm else "")
               + f"median {hm:,.2f} h per subject" + (f" (most {hx:,.1f} h)" if hx else ""))  # fmt: skip
        ends = [XB(v) for v in (hm, hx) if v]
        body.append(f'<g class="corpus-row" transform="translate(0,{j * PITCH})" data-tip="{e(tip)}">'
                    f'<text x="{LEFT - 6}" y="12" text-anchor="end">{e(label_of(r))}</text>'
                    f"{lollipop(rm, rx, XA, cls)}{lollipop(hm, hx, XB, cls)}"
                    f'<text x="{max(ends) + 7:.1f}" y="12" opacity="0.75">{hm:,.1f} h</text></g>')  # fmt: skip
    body.append("</svg>")
    html_ = scroll_frame(
        axis_strip(W, [pa, pb], "recordings and hours per subject"), "".join(body)
    )
    med_h = median([get(q, "hours_per_subject_p50") for _, q in pairs])
    med_r = median(
        [
            get(q, "rec_per_subject_p50")
            for _, q in pairs
            if get(q, "rec_per_subject_p50")
        ]
        or [0]
    )
    top_h, low_h = pairs[0], pairs[-1]
    most = max(pairs, key=lambda p: get(p[1], "rec_per_subject_max") or 0)
    cap = (f"The median dataset records {fmt_num(med_r)} recording{'s' if med_r != 1 else ''} and {med_h:,.1f} h per "
           f"subject; {name_of(top_h[0], 40)} records the most per subject ({get(top_h[1], 'hours_per_subject_p50'):,.1f} h "
           f"median), {name_of(low_h[0], 40)} the least ({get(low_h[1], 'hours_per_subject_p50'):,.2f} h), and "
           f"{name_of(most[0], 40)} has the most recordings for one subject "
           f"({fmt_num(get(most[1], 'rec_per_subject_max') or 0)}). Dots mark the median subject, lines reach the "
           f"subject with the most.")  # fmt: skip
    return html_, cap


def age_of(d: dict[str, str]) -> dict | None:
    """The age quantiles of a demographics row, or None when no age is known."""
    q = {"lo": fnum(d["age_min"]), "q1": fnum(d["age_p25"]), "mid": fnum(d["age_p50"]), "q3": fnum(d["age_p75"]),
         "hi": fnum(d["age_max"])}  # fmt: skip
    if q["mid"] is None and (q["lo"] is None or q["hi"] is None):
        return None
    return q


def sex_of(d: dict[str, str]) -> tuple[int, int] | None:
    m, f = int(fnum(d["n_male"]) or 0), int(fnum(d["n_female"]) or 0)
    return (m, f) if m + f else None


def people_chart(
    rows: list[dict[str, str]], demographics: list[dict[str, str]]
) -> tuple[str, str]:
    """Ages per dataset (min..max bar, quartile box, median tick), youngest median last, and beside it the female
    and male shares with their counts; datasets with neither are left out and counted in the caption.
    """
    pairs = [(r, d, age_of(d), sex_of(d)) for r, d in joined(rows, demographics)]
    pairs = [p for p in pairs if p[2] or p[3]]
    if not pairs:
        return missing("ages or sex of the subjects")
    mid = lambda a: a["mid"] if a["mid"] is not None else (a["lo"] + a["hi"]) / 2  # noqa: E731
    pairs.sort(
        key=lambda p: (
            p[2] is None,
            -mid(p[2]) if p[2] else 0,
            name_of(p[0], 60).lower(),
        )
    )
    ages = [v for p in pairs if p[2] for v in p[2].values() if v is not None]
    at, af = value_axis(ages or [100], log=False, step=20)
    pa = {
        "x0": LEFT,
        "width": 260,
        "ticks": at,
        "frac": af,
        "fmt": tick_text,
        "title": "age (years)",
    }
    sx0, sw = LEFT + 290, 120
    W = sx0 + sw + 110
    XA = lambda v: LEFT + af(v) * 260  # noqa: E731
    axis = axis_strip(W, [pa], "ages and sex per dataset").replace(
        "</svg>",
        f'<rect x="{sx0}" y="3" width="9" height="9" class="corpus-q2"/><text x="{sx0 + 13}" y="11">female</text>'
        f'<rect x="{sx0 + 62}" y="3" width="9" height="9" class="corpus-q6"/><text x="{sx0 + 75}" y="11">male</text>'
        f'<text x="{sx0 + sw / 2}" y="29" text-anchor="middle" opacity="0.7">share of subjects</text></svg>',
    )
    h = PITCH * len(pairs) + 4
    body = [
        svg_open(W, h, "ages and sex per dataset", cls="corpus-bars"),
        grid_lines([pa], h),
    ]
    for j, (r, d, a, sx) in enumerate(pairs):
        tip = [f"{name_of(r, 60)} ({r['dataset_id']})"]
        if a:
            tip.append(f"ages {fmt_num(a['lo'] or a['mid'])}–{fmt_num(a['hi'] or a['mid'])}"
                       + (f", median {fmt_num(a['mid'])}" if a["mid"] is not None else ""))  # fmt: skip
        else:
            tip.append("ages unknown")
        sex_svg = ""
        if sx:
            m, f = sx
            fw = sw * f / (m + f)
            sex_svg = (f'<rect x="{sx0}" y="3" width="{fw:.1f}" height="10" class="corpus-q2"/>'
                       f'<rect x="{sx0 + fw:.1f}" y="3" width="{sw - fw:.1f}" height="10" class="corpus-q6"/>'
                       f'<text x="{sx0 + sw + 6}" y="12" opacity="0.75">{f:,} F · {m:,} M</text>')  # fmt: skip
            tip.append(f"{f:,} female, {m:,} male")
        else:
            tip.append("sex unknown")
        if d.get("tasks"):
            tip.append(d["tasks"])
        body.append(f'<g class="corpus-row" transform="translate(0,{j * PITCH})" data-tip="{e(" · ".join(tip))}">'
                    f'<text x="{LEFT - 6}" y="12" text-anchor="end">{e(label_of(r))}</text>'
                    f"{range_marks(a, XA, fill(r['status'])) if a else ''}{sex_svg}</g>")  # fmt: skip
    body.append("</svg>")
    html_ = scroll_frame(axis, "".join(body))
    aged = [p for p in pairs if p[2]]
    sexed = [p[3] for p in pairs if p[3]]
    cap = f"Ages known for {len(aged)} of {len(rows)} datasets, sex for {len(sexed)}"
    if len(aged) == 1:
        cap += f"; median age {fmt_num(mid(aged[0][2]))} ({name_of(aged[0][0], 40)})"
    elif aged:
        y, o = aged[-1], aged[0]
        cap += (f"; median ages run from {fmt_num(mid(y[2]))} ({name_of(y[0], 40)}) to {fmt_num(mid(o[2]))} "
                f"({name_of(o[0], 40)})")  # fmt: skip
    if sexed:
        m, f = sum(s[0] for s in sexed), sum(s[1] for s in sexed)
        cap += f"; {pct(f, m + f)} of the {m + f:,} subjects with a recorded sex are female"
    cap += ". Datasets with neither are not shown."
    return html_, cap


# The share of the tokens (%) at which each shade of the green ramp starts: corpus-q0 below the first edge, q7 from
# the last (half decades, so the long tail of small datasets spreads over the light shades).
SHARE_EDGES = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]


def share_shade(share_pct: float) -> int:
    """The ramp step (0..7) of a share of the tokens, in percent."""
    return sum(share_pct >= edge for edge in SHARE_EDGES)


def share_legend(width: float) -> str:
    """The ramp's eight swatches with their share ranges, under the treemap."""
    labels = ([f"< {SHARE_EDGES[0]:g} %"] + [f"{a:g}–{b:g} %" for a, b in zip(SHARE_EDGES, SHARE_EDGES[1:])]
              + [f"≥ {SHARE_EDGES[-1]:g} %"])  # fmt: skip
    sw = min(96.0, (width - 120) / len(labels))
    parts = [svg_open(120 + sw * len(labels), 34, "share of the tokens, colour legend", f"{120 + sw * len(labels):g}px"),
             '<text x="0" y="13" opacity="0.8">share of the tokens</text>']  # fmt: skip
    for k, lab in enumerate(labels):
        x = 120 + k * sw
        parts.append(f'<rect x="{x:.1f}" y="3" width="{sw - 3:.1f}" height="12" class="corpus-q{k}"/>'
                     f'<text x="{x + (sw - 3) / 2:.1f}" y="29" text-anchor="middle" opacity="0.75">{e(lab)}</text>')  # fmt: skip
    return "".join(parts) + "</svg>"


def tokens_chart(
    rows: list[dict[str, str]],
    recordings: list[dict[str, str]],
    pending: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    """Tokens per epoch per dataset as a squarified treemap grouped by paradigm (group outlined, named when it has
    room), each tile shaded by its share of the tokens on the green ramp (legend under the chart). Tokens exist only
    for converted data: ``pending`` (the datasets v1 adds) are named in the caption, not drawn.
    """
    pairs = [
        (r, fnum(q["tokens"])) for r, q in joined(rows, recordings) if fnum(q["tokens"])
    ]
    if not pairs:
        return missing("token counts")
    groups: dict[str, list] = {}
    for r, t in pairs:
        groups.setdefault(r["paradigm"] or "unspecified", []).append((r, t))
    order = sorted(groups, key=lambda g: (-sum(t for _, t in groups[g]), g))
    W, H = 960, 480
    tot = sum(t for _, t in pairs)
    parts = [
        svg_open(W, H, "tokens per dataset, grouped by paradigm"),
        '<g class="corpus-tm">',
    ]
    labels = []
    for g, (gx, gy, gw, gh) in zip(
        order, squarify([sum(t for _, t in groups[g]) for g in order], 0, 0, W, H)
    ):
        leaves = sorted(groups[g], key=lambda p: (-p[1], p[0]["dataset_id"]))
        gt = sum(t for _, t in leaves)
        for (r, t), (x, y, w, h) in zip(
            leaves, squarify([t for _, t in leaves], gx, gy, gw, gh)
        ):
            shade = share_shade(100 * t / tot)
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" class="corpus-q{shade}" '
                         f'data-tokens="{t:.0f}" data-tip="{e(name_of(r, 60))} ({r["dataset_id"]}) · {fmt_tokens(t)} '
                         f'tokens, {pct(t, tot)} · {e(g)} · {e(SHORT[r["status"]])}"/>')  # fmt: skip
            if w >= 64 and h >= 30:
                chars = int(w / 6.6)
                text = label_of(r, max(chars - len(r["dataset_id"]) - 3, 4))
                text = (
                    text if len(text) <= chars else name_of(r, chars)
                )  # no room for the id: the name alone
                ink = "corpus-ink-hi" if shade >= 4 else "corpus-ink-lo"
                labels.append(f'<text x="{x + w / 2:.1f}" y="{y + h / 2:.1f}" text-anchor="middle" class="{ink}">'
                              f'{e(text)}<tspan x="{x + w / 2:.1f}" dy="13" opacity="0.85">'
                              f"{pct(t, tot)}</tspan></text>")  # fmt: skip
        parts.append(f'<rect x="{gx:.2f}" y="{gy:.2f}" width="{gw:.2f}" height="{gh:.2f}" fill="none" '
                     f'stroke="currentColor" stroke-width="2" data-group="{e(g)}" data-tokens="{gt:.0f}"/>')  # fmt: skip
        if gw >= 110 and gh >= 44:
            labels.append(f'<text class="corpus-lab" x="{gx + 6:.1f}" y="{gy + 15:.1f}">'
                          f"{e(label_of({'short_name': g}, int(gw / 7)))}</text>")  # fmt: skip
    parts.append("</g>" + "".join(labels) + "</svg>")
    pairs.sort(key=lambda p: -p[1])
    top3 = pairs[:3]
    cap = (f"{fmt_tokens(tot)} tokens per epoch over {len(pairs)} datasets: {name_of(pairs[0][0], 40)} alone holds "
           f"{pct(pairs[0][1], tot)}, the three largest ({', '.join(name_of(r, 40) for r, _ in top3)}) "
           f"{pct(sum(t for _, t in top3), tot)}; areas are tokens, colour is each dataset's share of them (darker is "
           f"larger, legend below), outlines group the datasets by paradigm.")  # fmt: skip
    if pending:
        cap += (f" Tokens exist only for converted data: the {len(pending)} datasets v1 adds "
                f"({hours_of(pending):,.0f} h measured so far) enter this budget when they are converted.")  # fmt: skip
    return "".join(parts) + share_legend(W), cap


def sankey_svg(rows: list[dict[str, str]], measure: str) -> tuple[str, list, dict]:
    """Archive -> licence -> status, flows in hours or datasets; returns the SVG, the node values and the links."""
    keys = [
        lambda r: r["archive"] or "unspecified",
        lambda r: r["licence"] or "none stated",
        lambda r: r["status"],
    ]
    value = (lambda r: fnum(r["hours"]) or 0) if measure == "hours" else (lambda r: 1)
    nodes, links = sankey_flows(rows, keys, value)
    orders = [sorted(c, key=lambda n, c=c: (-c[n], n)) for c in nodes[:2]]
    orders.append([s for s in STATUSES if s in nodes[2]])
    count = [{} for _ in keys]
    for r in rows:
        if value(r):
            for c, k in enumerate(keys):
                count[c][k(r)] = count[c].get(k(r), 0) + 1
    top, nw, height = 10, 12, 340
    spans, bands = sankey_layout(nodes, orders, links, height)
    xs = [190, 470, 750]
    W = xs[2] + nw + 210
    H = (
        top + max(max(s[1] for s in sp.values()) for sp in spans) + 26
    )  # room for the column titles
    parts = [svg_open(W, H, f"archive to licence to status, by {measure}")]
    for (c, s, t), (ys0, ys1, yt0, yt1) in bands.items():
        x0, x1 = xs[c] + nw, xs[c + 1]
        xm = (x0 + x1) / 2
        look = (
            f'class="{fill(t)}" fill-opacity="0.45"'
            if c == 1
            else 'fill="currentColor" fill-opacity="0.16"'
        )
        name_t = SHORT[t] if c == 1 else t
        v = links[(c, s, t)]
        tip = f"{s} → {name_t}: " + (
            f"{v:,.0f} h" if measure == "hours" else plural_ds(v)
        )
        parts.append(f'<path d="M{x0} {top + ys0:.1f}C{xm} {top + ys0:.1f} {xm} {top + yt0:.1f} {x1} {top + yt0:.1f}'
                     f'L{x1} {top + yt1:.1f}C{xm} {top + yt1:.1f} {xm} {top + ys1:.1f} {x0} {top + ys1:.1f}Z" {look} '
                     f'data-from="{c}:{e(s)}" data-to="{c + 1}:{e(t)}" data-v="{v:g}" data-tip="{e(tip)}"/>')  # fmt: skip
    for c, sp in enumerate(spans):
        for n, (y0, y1) in sp.items():
            v, k = nodes[c][n], count[c][n]
            name = SHORT[n] if c == 2 else n
            look = (
                f'class="{fill(n)}"'
                if c == 2
                else 'fill="currentColor" fill-opacity="0.6"'
            )
            amount = f"{v:,.0f} h" if measure == "hours" else f"{k}"
            tip = f"{name}: {plural_ds(k)}" + (
                f", {v:,.0f} h" if measure == "hours" else ""
            )
            x, anchor = (xs[c] - 6, "end") if c == 0 else (xs[c] + nw + 6, "start")
            parts.append(f'<g data-node="{c}:{e(n)}" data-v="{v:g}" data-tip="{e(tip)}">'
                         f'<rect x="{xs[c]}" y="{top + y0:.1f}" width="{nw}" height="{y1 - y0:.1f}" {look}/>'
                         f'<text class="corpus-lab" x="{x}" y="{top + (y0 + y1) / 2 + 4:.1f}" text-anchor="{anchor}">'
                         f"{e(name)} · {amount}</text></g>")  # fmt: skip
    for c, title in enumerate(("archive", "licence", "status")):
        parts.append(
            f'<text x="{xs[c] + nw / 2}" y="{H - 1}" text-anchor="middle" opacity="0.6">{title}</text>'
        )
    parts.append("</svg>")
    return "".join(parts), nodes, links


def plural_ds(n: float) -> str:
    return f"{n:,.0f} dataset{'s' if n != 1 else ''}"


def flow_licence_chart(rows: list[dict[str, str]]) -> tuple[str, str]:
    """The Sankey in hours (default) and in datasets, switched by the ``flow`` toggle (the electrode map's
    ``data-corpus-show`` mechanism, its own group so the map is not switched with it).
    """
    hs, hn, _ = sankey_svg(rows, "hours")
    ds, _, _ = sankey_svg(rows, "datasets")
    toggle = ('<div class="corpus-toggle" role="group" aria-label="Flow width by" data-corpus-group="flow">'
              '<span class="corpus-toggle-label">Width by</span>'
              '<button type="button" data-value="hours" aria-pressed="true">Hours</button>'
              '<button type="button" data-value="datasets" aria-pressed="false">Datasets</button></div>')  # fmt: skip
    body = (f'{toggle}<div data-corpus-group="flow" data-corpus-state="hours"><div data-corpus-show="hours">{hs}</div>'
            f'<div data-corpus-show="datasets">{ds}</div></div>')  # fmt: skip
    tot = sum(hn[0].values())
    with_h = sum(1 for r in rows if fnum(r["hours"]))
    arch = max(hn[0], key=hn[0].get)
    lic = max(hn[1], key=hn[1].get)
    cap = (f"{tot:,.0f} h over {with_h} of the {len(rows)} datasets: {arch} hosts {pct(hn[0][arch], tot)} and "
           f"{lic} covers {pct(hn[1][lic], tot)}")  # fmt: skip
    if len(hn[2]) > 1:
        cap += f"; {pct(hn[2].get(CONVERTED, 0), tot)} of the hours are converted"
    if with_h < len(rows):
        n = len(rows) - with_h
        cap += f"; {n} dataset{'s' if n != 1 else ''} without hours show{'s' if n == 1 else ''} in the datasets view only"
    return body, cap + "."


def montage_bands(ds: list[str], n_of: dict[str, int]) -> list[tuple[str, list[str]]]:
    """The datasets in channel-count bands, most electrodes first: (label, datasets) per non-empty band; the top band
    reads up to its largest dataset (``129–265``).
    """
    out = []
    for k in reversed(range(len(CHANNEL_BANDS))):
        grp = [d for d in ds if band_index(n_of[d], CHANNEL_BANDS) == k]
        if grp:
            lo, hi = CHANNEL_BANDS[k]
            out.append((f"{lo}–{hi or max(n_of[d] for d in grp)}", grp))
    return out


def montage_chart(
    rows: list[dict[str, str]],
    buckets: list[dict[str, str]],
    electrodes: list[dict],
    min_datasets: int = MONTAGE_MIN,
) -> tuple[str, str]:
    """The 10-20 / 10-10 names that at least ``min_datasets`` datasets use (rows, front to back then left to right)
    x the datasets (columns, by their number n of distinct electrodes, most first, in channel-count bands with a
    bracket above each), a cell filled when one of the dataset's channel sets carries the electrode. A column's
    tooltip reads k of n: k of the dataset's n electrodes are among the rows (a net with its own naming, EGI's
    E1…E129, shows only the names that coincide). Each column is one path of merged runs, so the fragment stays
    small; the body scrolls sideways under the electrode names, pinned on the left.
    """
    have = electrode_sets(buckets, electrodes)
    names, uses = montage_names(have, electrodes, min_datasets)
    if not names:
        return missing("electrode names shared by enough datasets")
    meta = by_id(rows)
    ds = [d for d in have if d in meta] or list(have)
    n_of = {d: len(have[d]) for d in ds}
    on_of = {d: sum(n in have[d] for n in names) for d in ds}
    label = lambda d: label_of(meta[d], 20) if d in meta else d  # noqa: E731
    title = lambda d: f"{name_of(meta[d], 60)} ({d})" if d in meta else d  # noqa: E731
    ds.sort(key=lambda d: (-n_of[d], -on_of[d], label(d).lower(), d))
    bands = montage_bands(ds, n_of)
    cw, bar, pitch, gap, pad = (
        14,
        12,
        10,
        8,
        4,
    )  # column pitch and bar width, row pitch, gap between bands
    top, foot, pin_w = (
        42,
        150,
        40,
    )  # band labels and brackets above, dataset labels below, electrode names
    rows_h = pitch * len(names)
    H = top + rows_h + foot
    W = pad + cw * len(ds) + gap * (len(bands) - 1) + pad
    # region guides: a faint line where the name's region changes (Fp, AF, F, … front to back)
    region = [TENTEN.match(n).group(1).lower() for n in names]
    cuts = [j for j in range(1, len(names)) if region[j] != region[j - 1]]
    guide = lambda x0, x1: "".join(  # noqa: E731
        f'<line x1="{x0}" x2="{x1}" y1="{top + j * pitch - 0.5}" y2="{top + j * pitch - 0.5}" stroke="currentColor" '
        f'stroke-opacity="0.18"/>'
        for j in cuts
    )
    pin = [svg_open(pin_w, H, "electrode names", fixed=True), guide(0, pin_w)]
    for j, n in enumerate(names):
        pin.append(f'<text x="{pin_w - 5}" y="{top + j * pitch + 8}" text-anchor="end" font-size="8.5" '
                   f'data-tip="{e(n)} · used by {uses[n]} datasets">{e(n)}</text>')  # fmt: skip
    pin.append("</svg>")
    body = [svg_open(W, H, "electrodes per dataset", fixed=True), guide(0, W)]
    x, ends = (
        float(pad),
        [-1e9, -1e9],
    )  # the right end of the band labels on the lower and upper line
    for b, (span, grp) in enumerate(bands):
        x0, x1 = x, x + cw * len(grp) - (cw - bar)
        text = (
            f"{span} channels · {plural_ds(len(grp))}"
            if b == 0
            else f"{span} · {len(grp)}"
        )
        est = 5.6 * len(text)  # font-size 10
        start, anchor, ax = (
            (x0, "start", x0) if x0 + est <= W else (x1 - est, "end", x1)
        )
        line = 0 if start >= ends[0] + 6 else 1 if start >= ends[1] + 6 else 0
        ends[line] = start + est
        yb = top - 8
        body.append(f'<g data-band="{e(span)}" data-count="{len(grp)}">'
                    f'<path d="M{x0:g} {yb + 4}V{yb}H{x1:g}V{yb + 4}" fill="none" stroke="currentColor" '
                    f'stroke-opacity="0.55"/><text x="{ax:g}" y="{yb - 4 - 12 * line}" text-anchor="{anchor}" '
                    f'font-size="10" font-weight="600">{e(text)}</text></g>')  # fmt: skip
        for d in grp:
            on = [n in have[d] for n in names]
            runs, k = [], 0
            while k < len(on):
                if on[k]:
                    j = k
                    while j < len(on) and on[j]:
                        j += 1
                    runs.append(
                        f"M{x:g} {top + k * pitch}h{bar}v{(j - k) * pitch - 1}h{-bar}z"
                    )
                    k = j
                else:
                    k += 1
            tip = (f"{title(d)} · {on_of[d]} of its {n_of[d]} electrodes among the {len(names)} rows · "
                   f"{span} channels")  # fmt: skip
            lx = x + bar / 2 + 3
            body.append(f'<g class="corpus-col" data-tip="{e(tip)}" data-dataset="{e(d)}" data-on="{on_of[d]}" '
                        f'data-n="{n_of[d]}"><rect x="{x:g}" y="{top}" width="{bar}" height="{rows_h - 1}" '
                        f'fill="currentColor" fill-opacity="0.06"/>'
                        + (f'<path d="{"".join(runs)}" class="{fill(CONVERTED)}"/>' if runs else "")
                        + f'<text transform="translate({lx:g},{top + rows_h + 4}) rotate(-90)" text-anchor="end" '
                        f'font-size="8" data-tip="{e(tip)}">{e(label(d))}</text></g>')  # fmt: skip
            x += cw
        x += gap
    body.append("</svg>")
    counts = sorted(on_of.values())
    top_name = max(names, key=lambda n: (uses[n], -names.index(n)))
    own = sum(2 * on_of[d] < n_of[d] for d in ds)
    cap = (f"Columns are the {len(ds)} datasets, most electrodes first, in channel-count bands; rows are the "
           f"{len(names)} 10-20 / 10-10 names that at least {min_datasets} datasets use, front to back. {top_name} is "
           f"the most shared ({uses[top_name]} datasets); the median dataset carries {counts[len(counts) // 2]} of the "
           f"{len(names)}, and {own} dataset{'s use their' if own != 1 else ' uses its'} own naming (fewer than half of "
           f"{'their' if own != 1 else 'its'} electrodes among the rows), so {'their columns show' if own != 1 else 'its column shows'} "
           f"only the names that coincide. Hover a column for its k of n: k of the dataset's n distinct electrodes "
           f"are among the rows.")  # fmt: skip
    return hscroll_frame("".join(pin), "".join(body)), cap
