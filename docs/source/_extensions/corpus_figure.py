"""Sphinx directive that embeds one EEGDash Corpus fragment.

Usage::

    .. corpus-figure:: hours

The argument is a key of ``_static/corpus_generated/manifest.json``, written
by ``docs/prepare_corpus.py`` from ``docs/plot_corpus/data``. The fragment is
emitted as raw HTML inside ``<div class="corpus">`` (all corpus CSS is scoped
to that class); builders other than HTML get the figure's caption as a
paragraph instead. A ``builder-inited`` hook regenerates the fragments when
they are missing or older than the data or the renderer, so ``make
markdown`` and a bare ``sphinx-build`` work without the Makefile step.
An ``html-page-context`` hook attaches the corpus stylesheet and scripts to
the ``corpus/`` pages only (Sphinx reverts per-page assets after each page
and appends a checksum to their URLs, so a changed script is never served
from a stale cache). Modelled on ``dataset_figure.py``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Final

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.addnodes import only
from sphinx.util import logging

logger = logging.getLogger(__name__)

_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9\-]{0,63}$")
_OUT_REL: Final[str] = "_static/corpus_generated"


def _source_root() -> Path:
    """``docs/source``: this file lives in ``docs/source/_extensions``."""
    return Path(__file__).resolve().parent.parent


def _docs_root() -> Path:
    return _source_root().parent


def _manifest() -> dict:
    path = _source_root() / _OUT_REL / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class CorpusFigureDirective(Directive):
    """Embed the fragment ``<key>.html`` of the corpus pages."""

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self) -> list[nodes.Node]:
        key = self.arguments[0].strip()
        if not _KEY_RE.match(key):
            raise self.error(f"corpus-figure: invalid key {key!r}")
        fragments = _manifest().get("fragments", {})
        if key not in fragments:
            known = ", ".join(sorted(fragments)) or "none (run docs/prepare_corpus.py)"
            raise self.error(f"corpus-figure: unknown key {key!r}. Known keys: {known}.")
        path = _source_root() / _OUT_REL / fragments[key]["file"]
        env = self.state.document.settings.env
        env.note_dependency(str(path))
        env.note_dependency(str(_source_root() / _OUT_REL / "manifest.json"))
        static = "../" * env.docname.count("/") + "_static/"
        body = path.read_text(encoding="utf-8").replace("{static}", static)
        figure = only(expr="html")  # the markdown builder would pass raw HTML through
        figure += nodes.raw("", f'<div class="corpus">{body}</div>', format="html")
        fallback = only(expr="not html")
        fallback += nodes.paragraph(text=fragments[key]["caption"])
        return [figure, fallback]


def _regenerate(app) -> None:
    """Rebuild the fragments when stale; a failure stops the build, as a missing figure would."""
    docs = _docs_root()
    if str(docs) not in sys.path:
        sys.path.insert(0, str(docs))
    from plot_corpus.fragments import build, is_stale

    data, out = docs / "plot_corpus" / "data", _source_root() / _OUT_REL
    if is_stale(data, out):
        m = build(data, out)
        logger.info(f"corpus_figure: wrote {len(m['fragments'])} fragments (data as of {m['synced']})")


def _assets(app, pagename: str, templatename: str, context: dict, doctree) -> None:
    """The corpus CSS and JS on corpus pages only; the 3-D view's script on the two version pages."""
    if not pagename.startswith("corpus/"):
        return
    app.add_css_file("css/corpus.css")
    app.add_js_file("js/corpus.js", defer="defer")
    if pagename in ("corpus/v0", "corpus/v1"):
        app.add_js_file("js/corpus-3d.js", defer="defer")


def setup(app) -> dict:
    app.add_directive("corpus-figure", CorpusFigureDirective)
    app.connect("builder-inited", _regenerate)
    app.connect("html-page-context", _assets)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
