"""Sphinx extension: emit a slim page index for the global search palette.

``_static/js/eegdash-search.js`` (the Ctrl+K palette) needs page titles +
section headings to search guides, tutorials and API pages. Sphinx's own
``searchindex.js`` has all of that but weighs ~2.3 MB because it is an
inverted index over every word of all ~900 generated dataset pages. This
extension writes ``_static/docs_index.json`` instead: one entry per
*non-dataset* page (datasets are covered by
``dataset_generated/search_index.json``), title + section titles only — a
few tens of KB.

Entry shape: ``{"u": docname, "t": title, "g": group, "s": [sections]}``
where group is ``api`` / ``examples`` / ``docs``.

Ported out of the legacy monolithic ``conf.py`` into a standalone
extension (one handler per extension, matching ``copy_dataset_summary``).
"""

from __future__ import annotations

import json
from pathlib import Path

from docutils import nodes
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)


def _write_docs_search_index(app, exception) -> None:
    if exception is not None or not getattr(app, "builder", None):
        return
    if app.builder.name != "html":
        return

    env = app.env
    skip_prefixes = (
        "api/dataset/eegdash.dataset.",
        "gen_modules/",
    )
    skip_exact = {"search", "genindex", "py-modindex"}

    entries = []
    for docname in sorted(env.titles):
        if docname in skip_exact or docname.startswith(skip_prefixes):
            continue
        # Gallery "execution times" stubs (top-level and per-gallery)
        # add noise, not value.
        if docname.endswith("sg_execution_times"):
            continue
        title = env.titles[docname].astext().strip()
        if not title or title == "<no title>":
            continue

        if docname.startswith("api/"):
            group = "api"
        elif docname.startswith("generated/"):
            # sphinx-gallery output lives under generated/auto_examples/;
            # a prefix check avoids misclassifying pages that merely
            # contain "auto_examples" somewhere in their name.
            group = "examples"
        else:
            group = "docs"

        sections = []
        toc = env.tocs.get(docname)
        if toc is not None:
            for ref in toc.findall(nodes.reference):
                text = ref.astext().strip()
                anchor = ref.get("anchorname", "")
                if text and text != title and anchor:
                    sections.append([text, anchor])
        entry = {"u": docname, "t": title, "g": group}
        if sections:
            entry["s"] = sections[:20]
        entries.append(entry)

    out_path = Path(app.outdir) / "_static" / "docs_index.json"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(entries, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info("Wrote docs search index: %s (%d pages)", out_path, len(entries))
    except OSError as exc:
        LOGGER.warning("Unable to write docs_index.json: %s", exc)


def setup(app) -> dict:
    app.connect("build-finished", _write_docs_search_index)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
