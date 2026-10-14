"""Pure geometry for the corpus charts: a squarified treemap and the nodes and bands of a three-column Sankey.

No SVG here; ``charts.py`` draws what these return, and the tests check the
invariants (rectangles tile their frame, every node's flows balance).
"""

from __future__ import annotations


def worst_ratio(row: list[float], side: float) -> float:
    """The worst aspect ratio of a strip of areas laid along a side of the given length."""
    s = sum(row)
    if not s or not side:
        return float("inf")
    return max(max(side * side * v / (s * s), (s * s) / (side * side * v)) for v in row if v > 0)


def squarify(
    values: list[float], x: float, y: float, w: float, h: float
) -> list[tuple[float, float, float, float]]:
    """Rectangles (x, y, w, h), one per value in the order given (sort descending first), tiling the frame exactly
    with areas proportional to the values (Bruls, Huizing and van Wijk's squarified layout). Zero values get an
    empty rectangle.
    """
    total = sum(values)
    out: list[tuple[float, float, float, float]] = [(x, y, 0.0, 0.0)] * len(values)
    if total <= 0 or w <= 0 or h <= 0:
        return out
    scale = w * h / total
    todo = [(i, v * scale) for i, v in enumerate(values) if v > 0]
    while todo:
        side = min(w, h)
        row = [todo[0]]
        k = 1
        while k < len(todo) and worst_ratio([a for _, a in row + [todo[k]]], side) <= worst_ratio(
            [a for _, a in row], side
        ):
            row.append(todo[k])
            k += 1
        todo = todo[k:]
        area = sum(a for _, a in row)
        if not todo:  # the last strip takes what is left, so the tiling closes exactly
            area = w * h
        if w >= h:  # a column on the left, its items stacked top to bottom
            cw = area / h
            yy = y
            for n, (i, a) in enumerate(row):
                ch = h - (yy - y) if n == len(row) - 1 else a / cw
                out[i] = (x, yy, cw, ch)
                yy += ch
            x, w = x + cw, w - cw
        else:  # a row on top, its items side by side
            rh = area / w
            xx = x
            for n, (i, a) in enumerate(row):
                rw = w - (xx - x) if n == len(row) - 1 else a / rh
                out[i] = (xx, y, rw, rh)
                xx += rw
            y, h = y + rh, h - rh
    return out


def sankey_flows(
    rows: list[dict[str, str]], keys: list, value
) -> tuple[list[dict[str, float]], dict[tuple, float]]:
    """Per column the node values ({name: value}) and the links ((column, source, target) -> value) of a Sankey
    whose columns are ``keys`` (functions of a row); ``value(row)`` is the flow a row carries (rows of value 0 are
    left out). Each row adds its value to one node per column and to one link per pair of columns, so the flows of
    every node balance by construction.
    """
    nodes: list[dict[str, float]] = [{} for _ in keys]
    links: dict[tuple, float] = {}
    for r in rows:
        v = value(r)
        if not v:
            continue
        names = [k(r) for k in keys]
        for c, n in enumerate(names):
            nodes[c][n] = nodes[c].get(n, 0) + v
        for c in range(len(names) - 1):
            key = (c, names[c], names[c + 1])
            links[key] = links.get(key, 0) + v
    return nodes, links


def sankey_layout(
    nodes: list[dict[str, float]],
    orders: list[list[str]],
    links: dict[tuple, float],
    height: float,
    gap: float = 12,
    min_h: float = 1.5,
) -> tuple[list[dict[str, tuple[float, float]]], dict[tuple, tuple[float, float, float, float]]]:
    """Node spans (y0, y1) per column and per link the band's (ys0, ys1, yt0, yt1) at its source and its target.
    One scale for every column (the column with the most nodes fills ``height``); a node is never thinner than
    ``min_h``; bands leave a node in the order of their targets and enter in the order of their sources.
    """
    total = max(sum(col.values()) for col in nodes) or 1
    scale = min((height - gap * (len(o) - 1)) / total for o in orders if o)
    spans: list[dict[str, tuple[float, float]]] = []
    for col, order in zip(nodes, orders):
        y, span = 0.0, {}
        for n in order:
            hh = max(col[n] * scale, min_h)
            span[n] = (y, y + hh)
            y += hh + gap
        spans.append(span)
    out_at = {(c, n): s[0] for c, sp in enumerate(spans) for n, s in sp.items()}
    in_at = dict(out_at)
    rank = [{n: j for j, n in enumerate(o)} for o in orders]
    bands = {}
    for key in sorted(links, key=lambda k: (k[0], rank[k[0]][k[1]], rank[k[0] + 1][k[2]])):
        c, s, t = key
        hs = links[key] / nodes[c][s] * (spans[c][s][1] - spans[c][s][0])
        ys = out_at[(c, s)]
        out_at[(c, s)] = ys + hs
        bands[key] = [ys, ys + hs, 0.0, 0.0]
    for key in sorted(links, key=lambda k: (k[0], rank[k[0] + 1][k[2]], rank[k[0]][k[1]])):
        c, s, t = key
        ht = links[key] / nodes[c + 1][t] * (spans[c + 1][t][1] - spans[c + 1][t][0])
        yt = in_at[(c + 1, t)]
        in_at[(c + 1, t)] = yt + ht
        bands[key][2:] = [yt, yt + ht]
    return spans, {k: tuple(v) for k, v in bands.items()}
