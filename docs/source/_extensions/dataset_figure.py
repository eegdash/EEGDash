"""Sphinx directive that renders one dataset-summary figure from the registry.

Usage::

    .. dataset-figure:: bubble

The single positional argument is a key in
``dataset_figures_registry.DATASET_FIGURES``. The directive emits the
``<header>``/``<figure>``/``<figcaption>`` scaffold that the eight prior
``docs/source/dataset_summary/<key>.rst`` files used to copy-paste, plus the
embedded chart HTML loaded from
``docs/source/_static/dataset_generated/<chart_filename>``.

Adding a ninth (or Nth) figure is a one-line addition to
``DATASET_FIGURES`` and a single ``.. dataset-figure:: <key>`` call in the
parent tab-set -- no new ``.rst`` file is required.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import dataset_figures_registry
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

# Lock the argument syntax to a conservative key shape. The registry only
# ever ships short slugs, so anything outside this character class indicates
# a typo or a template-injection attempt.
_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# Resolved relative to ``docs/source``; matches the existing
# ``:file: ../_static/dataset_generated/...`` paths the leaf RSTs used.
_CHART_ROOT_REL: Final[str] = "_static/dataset_generated"


def _docs_source_root() -> Path:
    """Locate ``docs/source`` from this extension's filesystem location.

    The extension lives at ``docs/source/_extensions/dataset_figure.py``, so
    its grandparent directory is ``docs/source``. Computed once per build
    rather than threading ``app.srcdir`` through every call -- the extension
    has no other state to share with the directive.
    """
    return Path(__file__).resolve().parent.parent


class DatasetFigureDirective(Directive):
    """Render one registered dataset-summary chart.

    Usage::

        .. dataset-figure:: bubble

    The directive emits the same DOM the pre-refactor leaf RSTs produced:

    * an optional ``<header class="eegdash-fig-title"><h3>...</h3></header>``
      (``title_style="header"``) or a ``.. rubric::`` (``"rubric"``),
    * an optional ``prologue`` block parsed as nested RST,
    * a ``<figure class="eegdash-figure">`` wrapping the chart HTML loaded
      verbatim from ``_static/dataset_generated/<chart_filename>``,
    * a ``<figcaption class="eegdash-caption">`` whose body is the
      registry's free-form HTML, and
    * an optional ``epilogue`` block parsed as nested RST.

    Adding a new figure: drop a new key into ``DATASET_FIGURES`` in
    ``dataset_figures_registry`` and call ``.. dataset-figure:: <key>``
    wherever the tab-set lives. No new ``.rst`` files needed.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self) -> list[nodes.Node]:
        key: str = self.arguments[0].strip()

        if not _KEY_RE.match(key):
            raise self.error(
                f"dataset-figure: invalid key {key!r}. "
                "Must match [A-Za-z0-9_-]{1,64}."
            )

        try:
            spec = dataset_figures_registry.get(key)
        except KeyError:
            known = ", ".join(dataset_figures_registry.keys())
            raise self.error(
                f"dataset-figure: unknown key {key!r}. Registered keys: {known}."
            )

        title: str = spec["title"]
        title_style: str = spec.get("title_style", "header")
        chart_filename: str = spec["chart_filename"]
        caption: str = spec["caption"]
        prologue: str = spec.get("prologue", "")
        epilogue: str = spec.get("epilogue", "")

        chart_path: Path = _docs_source_root() / _CHART_ROOT_REL / chart_filename
        if not chart_path.is_file():
            raise self.error(
                f"dataset-figure: chart file not found at {chart_path}. "
                f"Check the {key!r} entry in dataset_figures_registry."
            )

        # Note the chart as a build dependency so an incremental build picks
        # up regenerated chart HTML without forcing a clean rebuild. Mirrors
        # the ``env.note_dependency`` call in ``dataset_explorer.py``.
        env = self.state.document.settings.env
        env.note_dependency(str(chart_path))

        chart_html: str = chart_path.read_text(encoding="utf-8")

        # --- Title -----------------------------------------------------------
        title_nodes: list[nodes.Node] = []
        if title_style == "rubric":
            # ``.. rubric::`` -> ``<p class="rubric">`` in HTML, which is
            # what the kde/moabb leaf RSTs used.
            title_nodes.append(nodes.rubric(text=title))
        elif title_style == "header":
            header_html = (
                '<header class="eegdash-fig-title">'
                f"<h3>{_html_escape_text(title)}</h3>"
                "</header>"
            )
            title_nodes.append(nodes.raw("", header_html, format="html"))
        else:
            raise self.error(
                f"dataset-figure: unknown title_style {title_style!r} for "
                f"key {key!r}. Use 'header' or 'rubric'."
            )

        # --- Prologue / epilogue -------------------------------------------
        prologue_nodes: list[nodes.Node] = []
        if prologue:
            prologue_nodes = self._parse_rst_block(prologue)

        epilogue_nodes: list[nodes.Node] = []
        if epilogue:
            epilogue_nodes = self._parse_rst_block(epilogue)

        # --- Figure body ----------------------------------------------------
        # Three sibling raw-HTML nodes (open <figure>, chart payload, close
        # </figure> with caption) instead of one big concatenated string.
        # Keeping the chart HTML in its own node mirrors the pre-refactor
        # ``.. raw:: html :file:`` semantics: Sphinx treats the file contents
        # as an atomic chunk rather than re-encoding them.
        figure_open = nodes.raw(
            "",
            '<figure class="eegdash-figure" style="margin: 0 0 1.25rem 0;">',
            format="html",
        )
        chart_node = nodes.raw("", chart_html, format="html")
        figure_close = nodes.raw(
            "",
            (f'<figcaption class="eegdash-caption">{caption}</figcaption></figure>'),
            format="html",
        )

        return [
            *title_nodes,
            *prologue_nodes,
            figure_open,
            chart_node,
            figure_close,
            *epilogue_nodes,
        ]

    def _parse_rst_block(self, text: str) -> list[nodes.Node]:
        """Parse a multi-line RST string as nested directive content."""
        container = nodes.Element()
        lines = StringList(
            text.splitlines(),
            source=self.state.document.current_source or "<dataset-figure>",
        )
        self.state.nested_parse(lines, self.content_offset, container)
        return list(container.children)


def _html_escape_text(text: str) -> str:
    """Escape only the characters that would break an ``<h3>`` body.

    Cheap inline implementation -- we deliberately keep registry strings
    plain text, so the full ``html.escape`` apparatus would be overkill.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def setup(app) -> dict:
    app.add_directive("dataset-figure", DatasetFigureDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
