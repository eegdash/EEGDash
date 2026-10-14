"""Inline-SVG building blocks: open tag, log ticks, horizontal bar lists with a sticky axis, legends.

Text and axes use ``currentColor`` (the page's foreground); marks take their
colour from a class (``corpus-fill-<status slug>``) that reads a CSS variable,
so light and dark themes recolour them without regenerating.
"""

from __future__ import annotations

import html
import math

from .data import SHORT, SLUGS, STATUSES

e = html.escape


def svg_open(
    w: float,
    h: float,
    label: str,
    max_w: str = "100%",
    cls: str = "",
    fixed: bool = False,
) -> str:
    """The opening tag: scales to its container (``max_w`` caps it), or ``fixed`` at w x h pixels (a chart that
    scrolls sideways instead of shrinking).
    """
    c = f' class="{cls}"' if cls else ""
    size = (
        f'width="{w:g}" height="{h:g}"'
        if fixed
        else f'width="100%" style="max-width:{max_w}"'
    )
    return (f'<svg{c} viewBox="0 0 {w:g} {h:g}" {size} role="img" '
            f'aria-label="{e(label)}" font-size="11" fill="currentColor">')  # fmt: skip


def fill(status: str) -> str:
    return f"corpus-fill-{SLUGS[status]}"


def log_ticks(lo: float, hi: float) -> list[float]:
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    return [10.0**k for k in range(a, max(b, a + 1) + 1)]


def lin_ticks(hi: float, step: float) -> list[float]:
    top = step * max(1, math.ceil(hi / step))
    return [step * k for k in range(int(top / step) + 1)]


def tick_text(v: float, unit: str = "") -> str:
    return (f"{v:,.0f}" if v >= 1 else f"{v:g}") + unit


def scroll_frame(axis_svg: str, body_svg: str, max_px: int = 560) -> str:
    """A scrolling chart whose axis stays pinned: the axis SVG sits in a sticky strip above the body SVG (both share
    the viewBox width, so they scale together).
    """
    return (f'<div class="corpus-scroll" style="max-height:{max_px}px" tabindex="0">'
            f'<div class="corpus-scroll-axis">{axis_svg}</div>{body_svg}</div>')  # fmt: skip


def hscroll_frame(pin_svg: str, body_svg: str) -> str:
    """A chart wider than the page: the body scrolls sideways under its row labels, pinned on the left (both SVGs
    at fixed pixel sizes, the same height).
    """
    return (f'<div class="corpus-hscroll" tabindex="0"><div class="corpus-hscroll-pin">{pin_svg}</div>'
            f'<div class="corpus-hscroll-body">{body_svg}</div></div>')  # fmt: skip


def legend_html(statuses: list[str]) -> str:
    present = set(statuses)
    return (
        '<div class="corpus-legend">'
        + "".join(
            f'<span><i class="corpus-bg-{SLUGS[s]}"></i>{e(SHORT[s])}</span>'
            for s in STATUSES
            if s in present
        )
        + "</div>"
    )


# LEFT holds a dataset's axis label, "name (id)": up to 22 characters of name and the id in full.
LEFT, WIDTH, RIGHT, PITCH = 215, 300, 110, 16


def dataset_bars(
    items: list[dict], *, log: bool, unit: str, label: str, step: float = 32
) -> tuple[str, int]:
    """One bar per dataset, a sortable list: item = {id, label, value, status, keys {order: sort key}, tip}.

    Each row is a ``<g>`` carrying its sort keys, so corpus.js can reorder rows (``order`` toggle) by moving them;
    the axis is a separate SVG pinned above. Returns the HTML and the body height in pixels.
    """
    vals = [it["value"] for it in items]
    if log:
        ticks = log_ticks(min(vals), max(vals))
        lo, hi = math.log10(ticks[0]), math.log10(ticks[-1])
        X = lambda v: LEFT + (math.log10(v) - lo) / (hi - lo) * WIDTH  # noqa: E731
    else:
        ticks = lin_ticks(max(vals), step)
        X = lambda v: LEFT + v / ticks[-1] * WIDTH  # noqa: E731
    W = LEFT + WIDTH + RIGHT
    axis = [svg_open(W, 22, f"{label}, axis")]
    for t in ticks:
        axis.append(
            f'<text x="{X(t):.1f}" y="15" text-anchor="middle" opacity="0.7">{tick_text(t, unit)}</text>'
        )
    axis.append(
        f'<line x1="{LEFT}" x2="{LEFT + WIDTH}" y1="21" y2="21" stroke="currentColor" stroke-opacity="0.3"/>'
    )
    axis.append("</svg>")
    h = PITCH * len(items) + 4
    body = [svg_open(W, h, label, cls="corpus-bars")]
    for t in ticks:
        body.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="0" y2="{h}" stroke="currentColor" '
                    f'stroke-opacity="0.12"/>')  # fmt: skip
    for j, it in enumerate(items):
        keys = "".join(f' data-k-{k}="{e(str(v))}"' for k, v in it["keys"].items())
        x1 = X(it["value"])
        body.append(
            f'<g class="corpus-row" transform="translate(0,{j * PITCH})"{keys} data-tip="{e(it["tip"])}">'
            f'<text x="{LEFT - 6}" y="12" text-anchor="end">{e(it["label"])}</text>'
            f'<rect x="{LEFT}" y="2" width="{max(x1 - LEFT, 1):.1f}" height="{PITCH - 4}" rx="1.5" '
            f'class="{fill(it["status"])}"/>'
            f'<text x="{x1 + 5:.1f}" y="12" opacity="0.75">{e(it["text"])}</text></g>'
        )
    body.append("</svg>")
    frame = scroll_frame("".join(axis), "".join(body))
    return f'<div class="corpus-sortable" data-pitch="{PITCH}">{frame}</div>', h


def order_toggle(default: str = "hours") -> str:
    """Order of the per-dataset bar lists; every chart of the page follows the same toggle (``order`` group)."""
    opts = [("hours", "Hours"), ("channels", "Channels"), ("name", "Name")]
    btn = "".join(f'<button type="button" data-value="{v}" aria-pressed="{str(v == default).lower()}">{t}</button>'
                  for v, t in opts)  # fmt: skip
    return (f'<div class="corpus-toggle" role="group" aria-label="Order datasets by" data-corpus-group="order">'
            f'<span class="corpus-toggle-label">Order by</span>{btn}</div>')  # fmt: skip


def hbars_svg(items: list[dict], label: str) -> str:
    """Horizontal stacked bars: item = {label, segs {status: value}, tips {status: text}, text}; one linear scale."""
    left, width, right, pitch, top = 150, 250, 180, 26, 6
    peak = max([sum(it["segs"].values()) for it in items] + [1e-9])
    parts = [
        svg_open(
            left + width + right, top + pitch * max(len(items), 1) + 4, label, "640px"
        ).replace('font-size="11"', 'font-size="12.5"')
    ]
    for j, it in enumerate(items):
        y, x = top + j * pitch, float(left)
        parts.append(
            f'<text x="{left - 8}" y="{y + 15}" text-anchor="end">{e(it["label"])}</text>'
        )
        for st in STATUSES:
            v = it["segs"].get(st, 0)
            if not v:
                continue
            w = v / peak * width
            parts.append(f'<rect x="{x:.1f}" y="{y + 3}" width="{max(w - 1, 1.5):.1f}" height="{pitch - 8}" rx="2" '
                         f'class="{fill(st)}" data-tip="{e(it["label"])} · {e(SHORT[st])}: {e(it["tips"][st])}"/>')  # fmt: skip
            x += w
        parts.append(
            f'<text x="{x + 8:.1f}" y="{y + 15}" opacity="0.8">{e(it["text"])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def value_axis(
    vals: list[float],
    log: bool,
    step: float = 10.0,
    candidates: list[float] | None = None,
) -> tuple[list[float], object]:
    """Ticks spanning ``vals`` and the map value -> fraction of the axis. A log axis takes decades, or the
    ``candidates`` (e.g. minutes at 1 min, 10 min, 1 h, 10 h) that bracket the values; a linear axis starts at 0.
    """
    if not log:
        ticks = lin_ticks(max(vals), step)
        return ticks, lambda v: v / ticks[-1]
    lo, hi = min(vals), max(vals)
    if candidates:
        below = [c for c in candidates if c <= lo] or [candidates[0]]
        above = [c for c in candidates if c >= hi] or [candidates[-1]]
        ticks = [c for c in candidates if below[-1] <= c <= above[0]]
    else:
        ticks = log_ticks(lo, hi)
    a, b = math.log10(ticks[0]), math.log10(ticks[-1])
    return ticks, lambda v: (math.log10(max(v, ticks[0])) - a) / ((b - a) or 1)


def axis_strip(W: float, panels: list[dict], label: str, h: int = 22) -> str:
    """The pinned axis of a scrolling chart: per panel {x0, width, ticks, frac, fmt, title}; titles sit above the
    tick labels when any panel has one.
    """
    titled = any(p.get("title") for p in panels)
    h = h + 14 if titled else h
    out = [svg_open(W, h, f"{label}, axis")]
    for p in panels:
        X = lambda v, p=p: p["x0"] + p["frac"](v) * p["width"]  # noqa: E731
        if p.get("title"):
            out.append(f'<text x="{p["x0"] + p["width"] / 2:.1f}" y="11" text-anchor="middle" '
                       f'font-weight="600">{e(p["title"])}</text>')  # fmt: skip
        for t in p["ticks"]:
            out.append(
                f'<text x="{X(t):.1f}" y="{h - 7}" text-anchor="middle" opacity="0.7">{e(p["fmt"](t))}</text>'
            )
        out.append(f'<line x1="{p["x0"]}" x2="{p["x0"] + p["width"]}" y1="{h - 1}" y2="{h - 1}" '
                   f'stroke="currentColor" stroke-opacity="0.3"/>')  # fmt: skip
    out.append("</svg>")
    return "".join(out)


def grid_lines(panels: list[dict], h: float) -> str:
    """Faint vertical lines at every tick of every panel, the height of the body."""
    return "".join(
        f'<line x1="{p["x0"] + p["frac"](t) * p["width"]:.1f}" x2="{p["x0"] + p["frac"](t) * p["width"]:.1f}" '
        f'y1="0" y2="{h}" stroke="currentColor" stroke-opacity="0.12"/>'
        for p in panels
        for t in p["ticks"]
    )


def range_marks(q: dict, X, cls: str) -> str:
    """One quantile summary on a row of height PITCH: a thin bar lo..hi (p10..p90 or min..max), the quartile box,
    the median tick and the maximum as a small mark; any missing quantile is left out.
    """
    out = []
    lo, hi = q.get("lo"), q.get("hi")
    if lo is not None and hi is not None:
        out.append(f'<rect x="{X(lo):.1f}" y="7" width="{max(X(hi) - X(lo), 1):.1f}" height="2" class="{cls}" '
                   f'fill-opacity="0.55"/>')  # fmt: skip
    if q.get("q1") is not None and q.get("q3") is not None:
        out.append(f'<rect x="{X(q["q1"]):.1f}" y="3" width="{max(X(q["q3"]) - X(q["q1"]), 2):.1f}" height="10" '
                   f'rx="1.5" class="{cls}"/>')  # fmt: skip
    if q.get("mid") is not None:
        out.append(
            f'<rect x="{X(q["mid"]) - 1:.1f}" y="1" width="2" height="14" fill="currentColor"/>'
        )
    if q.get("top") is not None:
        out.append(
            f'<circle cx="{X(q["top"]):.1f}" cy="8" r="2" fill="currentColor" fill-opacity="0.6"/>'
        )
    return "".join(out)


def range_rows(
    items: list[dict], panel: dict, label: str, W: float | None = None
) -> tuple[str, int]:
    """One quantile summary per dataset in a scrolling frame with a pinned axis: item = {label, status, q {lo, q1,
    mid, q3, hi, top}, text, tip}; ``panel`` = {x0, width, ticks, frac, fmt}. Returns the HTML and body height.
    """
    W = W or LEFT + WIDTH + RIGHT
    X = lambda v: panel["x0"] + panel["frac"](v) * panel["width"]  # noqa: E731
    h = PITCH * len(items) + 4
    body = [svg_open(W, h, label, cls="corpus-bars"), grid_lines([panel], h)]
    for j, it in enumerate(items):
        ends = [X(v) for v in it["q"].values() if v is not None] or [panel["x0"]]
        body.append(
            f'<g class="corpus-row" transform="translate(0,{j * PITCH})" data-tip="{e(it["tip"])}">'
            f'<text x="{panel["x0"] - 6}" y="12" text-anchor="end">{e(it["label"])}</text>'
            f"{range_marks(it['q'], X, fill(it['status']))}"
            f'<text x="{max(ends) + 6:.1f}" y="12" opacity="0.75">{e(it["text"])}</text></g>'
        )
    body.append("</svg>")
    return scroll_frame(axis_strip(W, [panel], label), "".join(body)), h
