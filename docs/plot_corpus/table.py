"""The datasets table: sortable (numbers numerically, blanks last), filterable by text and status, header pinned,
CSV download. One renderer; the pages differ by row filter, preset status filter and columns."""

from __future__ import annotations

import html

from .data import CONVERTED, DIGITS, PROV_KEY, PROVENANCE, SHORT, SLUGS, STATUSES

e = html.escape
NOT_CONVERTED = "not-converted"
# (header, column or None for a composed cell, numeric)
DISPLAY = [("Dataset", "dataset_id", False), ("Name", "name", False), ("Status", "status", False),
           ("Source", "source", False), ("Archive", "archive", False), ("Paradigm", "paradigm", False),
           ("Subjects", "n_subjects", True), ("Sessions", "n_sessions", True), ("Hours", "hours", True),
           ("Channels", "mean_channels", True), ("Rate (Hz)", "native_sfreq", True), ("Size (GB)", "raw_GB", True),
           ("Licence", "licence", False), ("Redistributable", "redistributable", False), ("Listed in", None, False),
           ("Note", "note", False)]  # fmt: skip
COMPACT = [c for c in DISPLAY if c[1] in ("dataset_id", "name", "paradigm", "n_subjects", "n_sessions", "hours",
                                          "mean_channels", "licence", "note")]  # fmt: skip


def cell(r: dict[str, str], col: str | None, numeric: bool, prov: dict[str, str]) -> str:
    if col is None:
        lists = [k for k, c in (("paper", "in_reve_paper"), ("card", "in_release_card"), ("v0", "in_v0"),
                                ("v1", "in_v1")) if r[c] == "yes"]  # fmt: skip
        txt = " · ".join(lists)
        return f'<td data-v="{e(txt)}">{e(txt)}</td>'
    v = r[col]
    if col == "dataset_id":
        inner = f'<a href="{e(r["link"])}">{e(v)}</a>' if r["link"] else e(v)
        return f'<td class="corpus-id" data-v="{e(v)}">{inner}</td>'
    if col == "name":
        sub = f'<div class="corpus-sub">{e(r["short_name"])}</div>' if r["short_name"] else ""
        return f'<td data-v="{e(v)}">{e(v)}{sub}</td>'
    if numeric:
        try:
            sort = repr(float(v))
        except ValueError:
            sort = ""
        src = prov.get(PROV_KEY[col], "")
        tip = f' data-tip="source: {e(src)} ({e(PROVENANCE.get(src, src))})"' if src else ""
        shown = f"{float(v):,.{DIGITS[col]}f}" if sort else e(v)
        return f'<td class="corpus-n" data-v="{sort}"{tip}>{shown}</td>'
    if col == "status":
        return (f'<td data-v="{STATUSES.index(v)}"><span class="corpus-badge"><i class="corpus-bg-{SLUGS[v]}"></i>'
                f"{e(SHORT[v])}</span></td>")  # fmt: skip
    cls = ' class="corpus-note-cell"' if col == "note" else ""
    return f'<td{cls} data-v="{e(v)}">{e(v)}</td>'


def table_html(rows: list[dict[str, str]], csv_name: str, *, preset: str = "", status_filter: bool = True,
               columns: list[tuple] | None = None, controls: bool = True, pending_option: bool = True) -> str:
    """The table block; ``{static}`` in the CSV link is replaced by the directive with the page's path to _static.
    ``pending_option`` adds the "Not yet converted" shortcut to the status filter."""
    cols = columns or DISPLAY
    present = [s for s in STATUSES if any(r["status"] == s for r in rows)]
    head = "".join(f'<th scope="col" data-num="{int(n)}" tabindex="0">{e(h)}</th>' for h, _, n in cols)
    body = []
    for r in rows:
        prov = dict(p.split(":", 1) for p in r["provenance"].split("; ") if ":" in p)
        body.append(f'<tr data-status="{e(r["status"])}">' + "".join(cell(r, c, n, prov) for _, c, n in cols) + "</tr>")
    ctl = ""
    if controls:
        sel = ""
        if status_filter and len(present) > 1:
            counts = {s: sum(r["status"] == s for r in rows) for s in present}
            opts = [("", f"All statuses ({len(rows)})")]
            if pending_option and CONVERTED in present:
                opts.append((NOT_CONVERTED, f"Not yet converted ({len(rows) - counts[CONVERTED]})"))
            opts += [(s, f"{s} ({counts[s]})") for s in present]
            sel = (f'<select class="corpus-status" aria-label="Filter by status" data-converted="{e(CONVERTED)}">'
                   + "".join(f'<option value="{e(v)}"{" selected" if v == preset else ""}>{e(t)}</option>'
                             for v, t in opts) + "</select>")  # fmt: skip
        ctl = (f'<div class="corpus-controls"><input class="corpus-q" type="search" '
               f'placeholder="Filter: id, name, paradigm, licence…" aria-label="Filter the table">{sel}'
               f'<span class="corpus-count" aria-live="polite">{len(rows)} datasets</span>'
               f'<a class="corpus-csv" href="{{static}}corpus_generated/{e(csv_name)}" download>Download CSV</a></div>')  # fmt: skip
    return (f'<div class="corpus-table-block">{ctl}<div class="corpus-table-wrap" tabindex="0">'
            f'<table class="corpus-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
            f"</div></div>")  # fmt: skip
