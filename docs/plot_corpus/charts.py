"""The charts of the corpus pages. Each returns ``(html, caption)``; the caption is computed from the same rows.

Every chart takes the rows it shows (``rows`` already filtered to a version
or the universe), so the v0-only and v0+v1 variants share code.
"""

from __future__ import annotations

import html
import math
import re

from .data import (
    BLOCKED, CONVERTED, EXCLUDED, GATED, SHORT, SLUGS, STATUSES, TO_CONVERT, TO_INGEST,
    fnum, hours_of, kpis, label_of, progress, roadmap_items,
)  # fmt: skip
from .svg import dataset_bars, fill, hbars_svg, log_ticks, svg_open, tick_text

e = html.escape
CHANNEL_BANDS = [(1, 16), (17, 32), (33, 64), (65, 128), (129, None)]
SUBJECT_BANDS = [(1, 10), (11, 30), (31, 100), (101, 300), (301, None)]
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


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
        out.append(f'<div class="corpus-kpi{cls}"><b>{e(b)}</b><span>{e(s)}</span></div>')
    return '<div class="corpus-kpis">' + "".join(out) + "</div>"


def kpis_universe(rows: list[dict[str, str]]) -> tuple[str, str]:
    k = kpis(rows)
    boxes = [
        (f"{k[CONVERTED]}", "datasets converted (v0)", CONVERTED),
        (f"{k['v0_hours']:,.0f}", "hours converted (v0)", CONVERTED),
        (f"{k['v0_subjects']:,}", "subjects converted (v0)", CONVERTED),
        (f"{k['v1_datasets']}", f"datasets in v1 (+{k[TO_CONVERT]} to convert, {k[TO_INGEST]} to ingest)", TO_CONVERT),
        (f"{k['v1_hours']:,.0f}", "hours in v1 (measured so far)", TO_CONVERT),
        (f"{k['pending']}", f"pending ({k[TO_INGEST]} to ingest, {k[BLOCKED]} blocked)", TO_INGEST),
        (f"{k['datasets']}", "datasets tracked in all", None),
    ]
    cap = (f"{k[CONVERTED]} of the {k['datasets']} datasets tracked are converted; {k[GATED]} are gated and "
           f"{k[EXCLUDED]} excluded.")  # fmt: skip
    return kpi_html(boxes), cap


def kpis_v0(rows: list[dict[str, str]], buckets: list[dict[str, str]], electrodes: list[dict]) -> tuple[str, str]:
    k = kpis(rows)
    rates = {round(fnum(r["native_sfreq"])) for r in rows if fnum(r["native_sfreq"])}
    paradigms = {r["paradigm"] for r in rows if r["paradigm"]}
    boxes = [
        (f"{k[CONVERTED]}", "datasets", CONVERTED),
        (f"{k['v0_hours']:,.0f}", "hours", CONVERTED),
        (f"{k['v0_subjects']:,}", "subjects", None),
        (f"{k['v0_sessions']:,}", "sessions", None),
        (f"{len(buckets):,}", "channel sets (buckets)", None),
        (f"{len(electrodes)}", "electrode positions", None),
        (f"{len(rates)}", "native sampling rates", None),
        (f"{len(paradigms)}", "paradigms", None),
    ]
    cap = (f"Measured in the converted Zarr store: {k[CONVERTED]} datasets, {k['v0_hours']:,.0f} h from "
           f"{k['v0_subjects']:,} subjects, in {len(buckets):,} channel sets.")  # fmt: skip
    return kpi_html(boxes), cap


def kpis_v1(rows: list[dict[str, str]]) -> tuple[str, str]:
    k = kpis(rows)
    new = [r for r in rows if r["status"] == TO_CONVERT]
    ing = [r for r in rows if r["status"] == TO_INGEST]
    unmeasured = [r for r in new + ing if not fnum(r["hours"])]
    boxes = [
        (f"{k['v1_datasets']}", "datasets", None),
        (f"{k[CONVERTED]}", "converted (v0)", CONVERTED),
        (f"{len(new)}", "to convert", TO_CONVERT),
        (f"{len(ing)}", "to ingest", TO_INGEST),
        (f"{k['v1_hours']:,.0f}", "hours (measured so far)", None),
        (f"{k['v1_subjects']:,}", "subjects", None),
    ]
    cap = (f"v1 holds {k['v1_datasets']} datasets: v0's {k[CONVERTED]}, {len(new)} to convert and {len(ing)} to ingest; "
           f"the numbers of those come from EEGDash's records until converted"
           + (f" ({', '.join(label_of(r, 40) for r in unmeasured)} unmeasured)." if unmeasured else "."))
    return kpi_html(boxes), cap


def progress_html(rows: list[dict[str, str]]) -> tuple[str, str]:
    steps = progress(rows)
    bar = "".join(f'<i class="corpus-bg-{SLUGS[st["status"]]}" style="flex:{st["count"]}" '
                  f'data-tip="{e(st["title"])}: {st["count"]} datasets"></i>' for st in steps if st["count"])  # fmt: skip
    lis = "".join(f'<li class="corpus-bd-{SLUGS[st["status"]]}" data-status="{e(st["status"])}" '
                  f'data-count="{st["count"]}"><b>{st["count"]}</b><span>{e(st["title"])}</span>'
                  f'<small>{e(st["sub"])}</small></li>' for st in steps)  # fmt: skip
    by = {st["status"]: st for st in steps}
    cap = (f"{by[CONVERTED]['count']} datasets converted, {by[TO_CONVERT]['count']} ingested and waiting for "
           f"conversion, {by[TO_INGEST]['count']} still to ingest; {by[BLOCKED]['count']} are blocked by their "
           f"licence and {by[GATED]['count']} are gated.")  # fmt: skip
    return f'<div class="corpus-pbar">{bar}</div><ol class="corpus-psteps">{lis}</ol>', cap


def roadmap_html(rows: list[dict[str, str]], synced: str) -> tuple[str, str]:
    items = roadmap_items(rows)
    lis = "".join(f"<li>{e(t)}</li>" for t in items)
    return (f'<ol class="corpus-roadmap">{lis}</ol>',
            f"As of {synced}, the date of the data on this page; in order of what comes next.")  # fmt: skip


def flow_svg(rows: list[dict[str, str]]) -> tuple[str, str]:
    """REVE's paper -> release card -> v0 -> v1 -> to be ingested, every count read from the rows."""
    k = kpis(rows)
    n = lambda col: sum(r[col] == "yes" for r in rows)  # noqa: E731
    hrs = lambda col: hours_of([r for r in rows if r[col] == "yes"])  # noqa: E731
    steps = [
        dict(key="paper", title="REVE's paper", count=n("in_reve_paper"), sub="datasets named", status=None),
        dict(key="card", title="Release card", count=n("in_release_card"), sub="entries", status=None),
        dict(key="v0", title="eegdash-corpus-v0", count=n("in_v0"), sub=f"{hrs('in_v0'):,.0f} h", status=CONVERTED),
        dict(key="v1", title="eegdash-corpus-v1", count=n("in_v1"), sub=f"{hrs('in_v1'):,.0f} h", status=TO_CONVERT),
        dict(key="ingest", title="To be ingested", count=k[TO_INGEST], sub=f"+ {k[BLOCKED]} blocked",
             status=TO_INGEST),
    ]  # fmt: skip
    bw, gap, h = 164, 45, 96
    parts = [svg_open(len(steps) * bw + (len(steps) - 1) * gap, h, "flow from REVE's paper to the corpus versions")]
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
    ni = by["ingest"]["count"]
    cap = (f"{by['v0']['count']} of the release card's {by['card']['count']} entries are converted ({v0h:,.0f} h); "
           f"v1 adds {by['v1']['count'] - by['v0']['count']} more ({v1h - v0h:,.0f} h), and "
           f"{ni} open dataset{' waits' if ni == 1 else 's wait'} to be ingested.")  # fmt: skip
    return "".join(parts), cap


def status_svg(rows: list[dict[str, str]]) -> tuple[str, str]:
    """One stacked bar of the datasets per status."""
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
    cap = (f"{k[CONVERTED]} of {k['datasets']} datasets ({pct(k[CONVERTED], k['datasets'])}) are ingested and "
           f"converted; {out} are blocked, gated or excluded.")  # fmt: skip
    return "".join(parts), cap


# ------------------------------------------------------------------------------------------- per-dataset bar lists


def _bar_items(rows: list[dict[str, str]], col: str, text) -> list[dict]:
    items = []
    for r in rows:
        v = fnum(r[col])
        if not v:
            continue
        items.append({
            "label": label_of(r, 22), "value": v, "status": r["status"], "text": text(v),
            "keys": {"hours": fnum(r["hours"]) or "", "channels": fnum(r["mean_channels"]) or "",
                     "name": label_of(r, 60).lower()},
            "tip": f"{label_of(r, 60)} ({r['dataset_id']}) · {fnum(r['hours']) or 0:,.1f} h · "
                   f"{fnum(r['mean_channels']) or 0:,.1f} channels · {SHORT[r['status']]}",
        })  # fmt: skip
    return items


def hours_chart(rows: list[dict[str, str]]) -> tuple[str, str]:
    """Every dataset with hours, most first, on a log axis, in a scrolling frame with a pinned axis."""
    items = sorted(_bar_items(rows, "hours", lambda v: f"{v:,.1f} h"), key=lambda it: -it["value"])
    body, _ = dataset_bars(items, log=True, unit=" h", label="hours per dataset, log scale")
    vals = [it["value"] for it in items]
    tot, k, cum = sum(vals), 0, 0.0
    while cum < tot / 2:
        cum += vals[k]
        k += 1
    names = ", ".join(it["label"] for it in items[:k]) if k <= 3 else ""
    lead = "The largest dataset" if k == 1 else f"The {WORDS.get(k, k)} largest datasets"
    cap = (f"{lead}{f' ({names})' if names else ''} hold{'s' if k == 1 else ''} {100 * cum / tot:.0f} % of the "
           f"{tot:,.0f} h shown; the median dataset has {median(vals):,.1f} h.")  # fmt: skip
    return body, cap


def channels_chart(rows: list[dict[str, str]]) -> tuple[str, str]:
    """Mean channel count per dataset, same layout and order toggle as the hours chart."""
    items = sorted(_bar_items(rows, "hours", lambda v: ""), key=lambda it: -it["value"])  # hours order by default
    for it in items:
        it["value"] = it["keys"]["channels"] or 0
        it["text"] = f"{it['value']:,.0f} ch" if it["value"] else ""
    items = [it for it in items if it["value"]]
    body, _ = dataset_bars(items, log=False, unit="", label="mean channel count per dataset", step=32)
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
    return next(k for k, (lo, hi) in enumerate(bands) if v >= lo and (hi is None or v <= hi))


def group_items(rows: list[dict[str, str]], key, measure: str, order=None) -> list[dict]:
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
                tips[st] = f"{len(sub)} dataset{'s' if len(sub) != 1 else ''}, {hours_of(sub):,.0f} h"
        items.append({"key": k, "label": str(k), "segs": segs, "tips": tips, "n": len(g), "hours": hours_of(g),
                      "text": f"{len(g)} dataset{'s' if len(g) != 1 else ''} · {hours_of(g):,.0f} h"})  # fmt: skip
    return items


def channel_bands_buckets(buckets: list[dict[str, str]]) -> tuple[str, str]:
    """Hours of the v0 Zarr store by channel-count band, measured per bucket (a dataset with several montages counts in
    each band it uses)."""
    tot = sum(float(b["hours"]) for b in buckets)
    items = []
    for k, (lo, hi) in enumerate(CHANNEL_BANDS):
        sub = [b for b in buckets if band_index(float(b["n_chans"]), CHANNEL_BANDS) == k]
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
    items = group_items(data, lambda r: round(float(r["native_sfreq"])), "datasets", order=rates)
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
    sharing its first word (the HBN releases) labelled once at its centre."""
    pts = [(fnum(r["n_subjects"]), fnum(r["hours"]), r) for r in rows if fnum(r["n_subjects"]) and fnum(r["hours"])]
    xt = log_ticks(min(p[0] for p in pts), max(p[0] for p in pts))
    yt = log_ticks(min(p[1] for p in pts), max(p[1] for p in pts))
    left, top, w, h = 60, 14, 620, 330
    lx0, lx1, ly0, ly1 = (math.log10(v) for v in (xt[0], xt[-1], yt[0], yt[-1]))
    X = lambda v: left + (math.log10(v) - lx0) / (lx1 - lx0) * w  # noqa: E731
    Y = lambda v: top + h - (math.log10(v) - ly0) / (ly1 - ly0) * h  # noqa: E731
    parts = [svg_open(left + w + 30, top + h + 44, "subjects against hours, log-log", "760px")]
    for t in xt:
        parts.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{top + h}" stroke="currentColor" '
                     f'stroke-opacity="0.12"/><text x="{X(t):.1f}" y="{top + h + 16}" text-anchor="middle" '
                     f'opacity="0.7">{tick_text(t)}</text>')  # fmt: skip
    for t in yt:
        parts.append(f'<line x1="{left}" x2="{left + w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="currentColor" '
                     f'stroke-opacity="0.12"/><text x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" '
                     f'opacity="0.7">{tick_text(t, " h")}</text>')  # fmt: skip
    parts.append(f'<text x="{left + w / 2}" y="{top + h + 36}" text-anchor="middle">subjects</text>')
    for s_, hr, r in pts:
        parts.append(f'<circle cx="{X(s_):.1f}" cy="{Y(hr):.1f}" r="4.5" class="corpus-pt {fill(r["status"])}" '
                     f'fill-opacity="0.85" data-tip="{e(label_of(r, 60))} · {s_:,.0f} subjects · {hr:,.1f} h"/>')  # fmt: skip
    groups: dict[str, list] = {}
    for p in sorted(pts, key=lambda p: -p[1])[:5]:
        groups.setdefault(label_of(p[2]).split()[0], []).append(p)
    for word, g in groups.items():
        cx = sum(X(p[0]) for p in g) / len(g)
        cy = sum(Y(p[1]) for p in g) / len(g)
        text = label_of(g[0][2]) if len(g) == 1 else f"{word} ×{len(g)}"
        anchor, dx = ("end", -8) if cx > left + w * 0.8 else ("start", 8)
        parts.append(f'<text class="corpus-lab" x="{cx + dx:.1f}" y="{cy + 4:.1f}" text-anchor="{anchor}" '
                     f'font-weight="600">{e(text)}</text>')  # fmt: skip
    parts.append("</svg>")
    per = sorted((hr / s_, r["dataset_id"], r) for s_, hr, r in pts)
    cap = (f"Median {per[len(per) // 2][0]:.1f} h per subject over {len(pts)} datasets; {label_of(per[-1][2], 60)} "
           f"records the most, {per[-1][0]:.1f} h per subject.")  # fmt: skip
    return "".join(parts), cap
