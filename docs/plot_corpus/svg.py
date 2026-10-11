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


def svg_open(w: float, h: float, label: str, max_w: str = "100%", cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return (f'<svg{c} viewBox="0 0 {w:g} {h:g}" width="100%" style="max-width:{max_w}" role="img" '
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
    the viewBox width, so they scale together)."""
    return (f'<div class="corpus-scroll" style="max-height:{max_px}px" tabindex="0">'
            f'<div class="corpus-scroll-axis">{axis_svg}</div>{body_svg}</div>')  # fmt: skip


def legend_html(statuses: list[str]) -> str:
    present = set(statuses)
    return '<div class="corpus-legend">' + "".join(
        f'<span><i class="corpus-bg-{SLUGS[s]}"></i>{e(SHORT[s])}</span>' for s in STATUSES if s in present
    ) + "</div>"


LEFT, WIDTH, RIGHT, PITCH = 150, 300, 110, 16


def dataset_bars(items: list[dict], *, log: bool, unit: str, label: str, step: float = 32) -> tuple[str, int]:
    """One bar per dataset, a sortable list: item = {id, label, value, status, keys {order: sort key}, tip}.

    Each row is a ``<g>`` carrying its sort keys, so corpus.js can reorder rows (``order`` toggle) by moving them;
    the axis is a separate SVG pinned above. Returns the HTML and the body height in pixels."""
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
        axis.append(f'<text x="{X(t):.1f}" y="15" text-anchor="middle" opacity="0.7">{tick_text(t, unit)}</text>')
    axis.append(f'<line x1="{LEFT}" x2="{LEFT + WIDTH}" y1="21" y2="21" stroke="currentColor" stroke-opacity="0.3"/>')
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
    parts = [svg_open(left + width + right, top + pitch * max(len(items), 1) + 4, label, "640px")
             .replace('font-size="11"', 'font-size="12.5"')]
    for j, it in enumerate(items):
        y, x = top + j * pitch, float(left)
        parts.append(f'<text x="{left - 8}" y="{y + 15}" text-anchor="end">{e(it["label"])}</text>')
        for st in STATUSES:
            v = it["segs"].get(st, 0)
            if not v:
                continue
            w = v / peak * width
            parts.append(f'<rect x="{x:.1f}" y="{y + 3}" width="{max(w - 1, 1.5):.1f}" height="{pitch - 8}" rx="2" '
                         f'class="{fill(st)}" data-tip="{e(it["label"])} · {e(SHORT[st])}: {e(it["tips"][st])}"/>')  # fmt: skip
            x += w
        parts.append(f'<text x="{x + 8:.1f}" y="{y + 15}" opacity="0.8">{e(it["text"])}</text>')
    parts.append("</svg>")
    return "".join(parts)
