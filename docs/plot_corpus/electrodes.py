"""Electrode positions of the converted Zarr store: aggregation, top-view projection, the 2-D map and the 3-D data.

``aggregate_positions`` and ``unplaced_names`` run at sync time (``sync_data.py``)
and write ``data/electrodes.json``; the renderer reads only that file. A channel
whose name the position bank lacks was placed by the converter from its
dataset's own coordinates (``electrodes.tsv``): it is listed as unplaced, not
drawn, and counted in the captions.
"""

from __future__ import annotations

import html
import json
import math

RAMP_STEPS = 8  # --corpus-q0 .. --corpus-q7, a green ramp defined in corpus.css
CLIP = 1.15  # electrodes below the equator are drawn outside the head circle, out to this radius
LANDMARKS = [("Fz",), ("Cz",), ("Pz",), ("Oz",), ("C3",), ("C4",), ("T7", "T3"), ("T8", "T4"), ("Fp1",), ("Fp2",),
             ("O1",), ("O2",)]  # fmt: skip
BUCKET_NOTE = "channel set (bucket): one exact set of channel names of one dataset in the Zarr store"
WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def aggregate_positions(
    buckets: list[dict[str, str]],
    bank: dict[str, list[float]],
    added: list[dict[str, str]] | None = None,
) -> list[dict]:
    """One entry per electrode *position* used by the buckets: hours, datasets and buckets summed over every bucket
    carrying it, most hours first. ``added`` are the channel sets of the v1 store that v0 does not hold: their sums go
    into ``added_hours`` / ``added_datasets`` / ``added_buckets``, so one file serves the v0 map and the v1 map. The
    bank holds case-fold duplicates sharing one coordinate (``CZ`` and ``Cz``), so channels are grouped by
    coordinates rounded to 1e-4 m and named by their spellings, most common first. A name the bank lacks is left
    out here (``unplaced_names`` lists it).
    """
    agg: dict[tuple, dict] = {}
    for prefix, rows in (("", buckets), ("added_", added or [])):
        for b in rows:
            hours = float(b["hours"])
            for name in b["channels"].split("|"):
                if name not in bank:
                    continue
                key = tuple(round(float(c), 4) for c in bank[name])
                e = agg.setdefault(key, {"hours": 0.0, "datasets": set(), "buckets": 0, "added_hours": 0.0,
                                         "added_datasets": set(), "added_buckets": 0, "spell": {}})  # fmt: skip
                e[prefix + "hours"] += hours
                e[prefix + "datasets"].add(b["dataset"])
                e[prefix + "buckets"] += 1
                e["spell"][name] = e["spell"].get(name, 0) + 1
    out = [
        {
            "names": sorted(e["spell"], key=lambda n: (-e["spell"][n], n)),
            "xyz": list(key),
            "hours": round(e["hours"], 1),
            "datasets": len(e["datasets"]),
            "buckets": e["buckets"],
            "added_hours": round(e["added_hours"], 1),
            "added_datasets": len(e["added_datasets"]),
            "added_buckets": e["added_buckets"],
        }
        for key, e in agg.items()
    ]
    return sorted(out, key=lambda e: (-e["hours"], -e["added_hours"], e["names"][0]))


def unplaced_names(
    buckets: list[dict[str, str]],
    bank: dict[str, list[float]],
    added: list[dict[str, str]] | None = None,
) -> list[dict]:
    """The channel names the bank lacks (the converter placed them from their dataset's own coordinates, which the
    store does not ship): per name its hours and datasets in v0 and in the channel sets v1 adds, most hours first.
    """
    agg: dict[str, dict] = {}
    for prefix, rows in (("", buckets), ("added_", added or [])):
        for b in rows:
            for name in b["channels"].split("|"):
                if name in bank:
                    continue
                u = agg.setdefault(name, {"hours": 0.0, "datasets": set(), "added_hours": 0.0,
                                          "added_datasets": set()})  # fmt: skip
                u[prefix + "hours"] += float(b["hours"])
                u[prefix + "datasets"].add(b["dataset"])
    out = [{"name": n, "hours": round(u["hours"], 1), "datasets": sorted(u["datasets"]),
            "added_hours": round(u["added_hours"], 1), "added_datasets": sorted(u["added_datasets"])}
           for n, u in agg.items()]  # fmt: skip
    return sorted(out, key=lambda u: (-u["hours"], -u["added_hours"], u["name"]))


def unplaced_note(unplaced: list[dict] | None, added: bool = False) -> str:
    """The caption's sentence on the electrodes placed from their dataset's own coordinates, counted from the data
    (empty when there are none in the view).
    """
    names, datasets = set(), set()
    for u in unplaced or []:
        ds = set(u["datasets"]) | (set(u.get("added_datasets", [])) if added else set())
        if ds:
            names.add(u["name"])
            datasets |= ds
    if not names:
        return ""
    n, k = len(names), len(datasets)
    return (f" {n} electrode{'s' if n != 1 else ''} of {WORDS.get(k, k)} dataset{'s' if k != 1 else ''} "
            f"{'were' if n != 1 else 'was'} placed from {'their' if k != 1 else 'its'} own coordinates and "
            f"{'are' if n != 1 else 'is'} not drawn.")  # fmt: skip


def view(e: dict, added: bool) -> tuple[float, int, int]:
    """(hours, datasets, buckets) of a position: v0 alone, or v0 plus the datasets v1 adds."""
    if not added:
        return e["hours"], e["datasets"], e["buckets"]
    return (e["hours"] + e.get("added_hours", 0), e["datasets"] + e.get("added_datasets", 0),
            e["buckets"] + e.get("added_buckets", 0))  # fmt: skip


def shown(electrodes: list[dict], added: bool) -> list[dict]:
    """The positions with data in the view (a position only the added datasets use is absent from the v0 map)."""
    return [e for e in electrodes if view(e, added)[0] > 0]


def check_coverage(
    electrodes: list[dict],
    buckets: list[dict[str, str]],
    unplaced: list[dict] | None = None,
) -> None:
    """Refuse buckets whose channels neither an electrode entry nor the unplaced list names (electrodes.json out of
    step with the buckets).
    """
    known = {n for e in electrodes for n in e["names"]} | {
        u["name"] for u in unplaced or []
    }
    for b in buckets:
        for name in b["channels"].split("|"):
            if name not in known:
                raise ValueError(
                    f"bucket {b['bucket']} ({b['dataset']}): channel {name!r} has no electrode position"
                )


def project(x: float, y: float, z: float, clip: float = CLIP) -> tuple[float, float]:
    """Azimuthal-equidistant top view from the head centre, in head radii: vertex 0, equator 1, nose up (+y), right ear
    right (+x); points further than ``clip`` (mastoids, face) are pulled in to ``clip``.
    """
    rho = min(math.atan2(math.hypot(x, y), z) / (math.pi / 2), clip)
    phi = math.atan2(y, x)
    return rho * math.cos(phi), rho * math.sin(phi)


def scales(electrodes: list[dict], added: bool = False) -> dict:
    """The two encodings (hours on a log scale from the decade below the least-used position, datasets linear)."""
    hours = [view(e, added)[0] for e in shown(electrodes, added)]
    hlo = math.floor(math.log10(min(hours)))
    hhi = math.log10(max(hours))
    dmax = max(view(e, added)[1] for e in electrodes)
    return {"hlo": hlo, "hhi": hhi, "dmax": dmax}


def t_hours(h: float, s: dict) -> float:
    return (
        (math.log10(h) - s["hlo"]) / (s["hhi"] - s["hlo"])
        if s["hhi"] > s["hlo"]
        else 1.0
    )


def t_datasets(d: int, s: dict) -> float:
    return (d - 1) / (s["dmax"] - 1) if s["dmax"] > 1 else 1.0


def step(t: float) -> int:
    return min(RAMP_STEPS - 1, max(0, int(t * RAMP_STEPS)))


def radius(t: float) -> float:
    return 2.5 + 5.5 * t


def tip(e: dict, n_datasets: int, n_buckets: int, added: bool = False) -> str:
    h, d, b = view(e, added)
    if not added:
        return f"{' / '.join(e['names'])} · {h:,.0f} h · in {d} of {n_datasets} datasets · in {b} of {n_buckets} channel sets"
    return (f"{' / '.join(e['names'])} · {h:,.0f} h ({e['hours']:,.0f} h in v0, {e['added_hours']:,.0f} h added) · "
            f"in {d} of {n_datasets} datasets ({e['datasets']} in v0, {e['added_datasets']} added) · "
            f"in {b} of {n_buckets} channel sets")  # fmt: skip


def tick_text(v: float, unit: str = "") -> str:
    return (f"{v:,.0f}" if v >= 1 else f"{v:g}") + unit


def toggle_html() -> str:
    """The hours / datasets toggle; every toggle of the ``encode`` group stays in step (2-D and 3-D)."""
    return (
        '<div class="corpus-toggle" role="group" aria-label="Colour and size by" data-corpus-group="encode">'
        '<button type="button" data-value="hours" aria-pressed="true">Hours</button>'
        '<button type="button" data-value="datasets" aria-pressed="false">Datasets</button></div>'
    )


def note_html(hint: str = "") -> str:
    """The note under a view, the same text and markup in the 2-D and the 3-D view."""
    return f'<p class="corpus-note">{hint}{html.escape(BUCKET_NOTE[0].upper() + BUCKET_NOTE[1:])}.</p>'


def legend_svg(
    electrodes: list[dict], n_datasets: int, W: float, y: float, added: bool = False
) -> str:
    """Both legends (hours, datasets); CSS shows the one of the current encoding."""
    s = scales(electrodes, added)
    hticks = [
        10.0**k for k in range(math.ceil(s["hlo"]), math.floor(s["hhi"]) + 1)
    ] or [10 ** s["hhi"]]
    dticks = sorted({1, s["dmax"], *(v for v in (10, 25) if v < s["dmax"] - 4)})
    parts = []
    for mode, title, ticks, tf, fmt in (
        ("hours", "hours of data at the position (log scale)", hticks, lambda v: t_hours(v, s),
         lambda v: tick_text(v, " h")),
        ("datasets", f"datasets (of {n_datasets}) that include the position", dticks, lambda v: t_datasets(v, s),
         lambda v: f"{v:,.0f}"),
    ):  # fmt: skip
        gap = W / (len(ticks) + 1)
        parts.append(f'<g class="corpus-lg" data-corpus-show="{mode}"><text x="{W / 2}" y="{y - 18}" '
                     f'text-anchor="middle" opacity="0.7">{title}</text>')  # fmt: skip
        for j, v in enumerate(ticks):
            t = tf(v)
            parts.append(f'<circle class="corpus-q{step(t)}" cx="{gap * (j + 1):.1f}" cy="{y}" r="{radius(t):.1f}"/>'
                         f'<text x="{gap * (j + 1):.1f}" y="{y + 22}" text-anchor="middle">{fmt(v)}</text>')  # fmt: skip
        parts.append("</g>")
    return "".join(parts)


def landmark_of(e: dict) -> str | None:
    """The landmark label an electrode entry carries (``T7`` for a ``T3`` spelling too), else None."""
    lower = {n.lower() for n in e["names"]}
    for group in LANDMARKS:
        hit = next((g for g in group if g.lower() in lower), None)
        if hit:
            return hit
    return None


def electrode_svg(
    electrodes: list[dict],
    n_datasets: int,
    n_buckets: int,
    *,
    added: bool = False,
    unplaced: list[dict] | None = None,
) -> tuple[str, str]:
    """The scalp map: one circle per position, ramp step and radius by hours (log) or datasets (linear), both
    precomputed so the toggle only swaps attributes; ``added`` draws v0 plus the datasets v1 adds. Returns the
    HTML and its caption.
    """
    s = scales(electrodes, added)
    R, cx, cy, W, H = 200, 250, 252, 500, 580
    e_ = html.escape
    parts = [
        toggle_html(),
        f'<svg class="corpus-emap" data-corpus-group="encode" data-corpus-state="hours" viewBox="0 0 {W} {H}" '
        f'width="100%" style="max-width:560px" role="img" aria-label="electrode positions of the converted Zarr store, '
        f'top view of the head" font-size="11" fill="currentColor">',
        f'<g fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5">'
        f'<circle cx="{cx}" cy="{cy}" r="{R}"/>'
        f'<path d="M{cx - 16} {cy - R + 2}L{cx} {cy - R - 18}L{cx + 16} {cy - R + 2}Z"/>'
        f'<ellipse cx="{cx - R - 7}" cy="{cy}" rx="8" ry="26"/><ellipse cx="{cx + R + 7}" cy="{cy}" rx="8" ry="26"/>'
        f'<path d="M{cx} {cy - R}V{cy + R}M{cx - R} {cy}H{cx + R}" stroke-opacity="0.12" stroke-dasharray="3 4"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{R / 2}" stroke-opacity="0.12" stroke-dasharray="3 4"/></g>',
    ]
    xy = {}
    for e in sorted(
        shown(electrodes, added), key=lambda e: -view(e, added)[0]
    ):  # big circles first, small on top
        u, v = project(*e["xyz"])
        x, y = cx + u * R, cy - v * R
        xy[e["names"][0]] = (x, y, e)
        h, d, _ = view(e, added)
        th, td = t_hours(h, s), t_datasets(d, s)
        parts.append(
            f'<circle class="corpus-el corpus-q{step(th)}" cx="{x:.1f}" cy="{y:.1f}" r="{radius(th):.1f}" '
            f'data-q-hours="{step(th)}" data-q-datasets="{step(td)}" data-r-hours="{radius(th):.1f}" '
            f'data-r-datasets="{radius(td):.1f}" data-tip="{e_(tip(e, n_datasets, n_buckets, added))}"/>'
        )
    for x, y, e in xy.values():  # the same landmarks are labelled in the 3-D view
        if hit := landmark_of(e):
            parts.append(f'<text class="corpus-lab" x="{x:.1f}" y="{y - radius(t_hours(view(e, added)[0], s)) - 3:.1f}" '
                         f'text-anchor="middle" font-size="10">{e_(hit)}</text>')  # fmt: skip
    parts.append(legend_svg(electrodes, n_datasets, W, H - 50, added))
    parts.append("</svg>")
    parts.append(note_html())
    return "".join(parts), caption(electrodes, added, unplaced)


def caption(
    electrodes: list[dict], added: bool = False, unplaced: list[dict] | None = None
) -> str:
    els = shown(electrodes, added)
    top = max(els, key=lambda e: view(e, added)[0])
    th, td, _ = view(top, added)
    single = sum(view(e, added)[1] == 1 for e in els)
    rare = (f"{single} position{'s appear' if single != 1 else ' appears'} in a single dataset" if single else
            f"none appears in a single dataset (the rarest in {min(view(e, added)[1] for e in els)})")  # fmt: skip
    new = len(els) - len(shown(electrodes, False)) if added else 0
    extra = f", {new} of them only in the datasets v1 adds" if new else ""
    return (f"{len(els)} electrode positions{extra}; {top['names'][0]} carries the most data ({th:,.0f} h "
            f"over {td} datasets); {rare}.{unplaced_note(unplaced, added)}")  # fmt: skip


def electrodes_3d(
    electrodes: list[dict],
    n_datasets: int,
    n_buckets: int,
    *,
    added: bool = False,
    unplaced: list[dict] | None = None,
) -> tuple[str, str]:
    """The 3-D view: a canvas drawn by corpus-3d.js from inline JSON (positions in head radii, both encodings,
    the landmark label where the 2-D map carries one), with the same legend and note as the 2-D view.
    """
    s = scales(electrodes, added)
    els = shown(electrodes, added)
    rmax = max(math.sqrt(sum(c * c for c in e["xyz"])) for e in els)
    pts = []
    for e in els:
        h, d, _ = view(e, added)
        th, td = t_hours(h, s), t_datasets(d, s)
        pts.append({
            "p": [round(c / rmax, 4) for c in e["xyz"]],
            "q": [step(th), step(td)],
            "r": [round(radius(th), 1), round(radius(td), 1)],
            "t": tip(e, n_datasets, n_buckets, added),
            **({"l": lab} if (lab := landmark_of(e)) else {}),
        })  # fmt: skip
    payload = json.dumps(pts, separators=(",", ":")).replace("</", "<\\/")
    body = (
        toggle_html()
        + f'<div class="corpus-3d" data-corpus-group="encode" data-corpus-state="hours">'
        f'<canvas class="corpus-3d-canvas" tabindex="0" role="img" aria-label="electrode positions of '
        f'the converted Zarr store in three dimensions; drag or use the arrow keys to rotate"></canvas>'
        f'<script type="application/json" class="corpus-3d-data">{payload}</script>'
        f'<svg class="corpus-3d-legend" viewBox="0 0 500 64" width="100%" font-size="11" fill="currentColor" '
        f'aria-hidden="true">{legend_svg(electrodes, n_datasets, 500, 34, added)}</svg></div>'
        + note_html("Drag, or focus the view and use the arrow keys, to rotate. ")
    )
    return body, caption(electrodes, added, unplaced)
