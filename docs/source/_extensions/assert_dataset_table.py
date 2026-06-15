"""Sphinx extension: assert the dataset-summary table inlines its JS stack.

The dataset_summary page's interactivity (sorting/filtering via DataTables)
depends on ``prepare_summary_tables.py`` inlining the DataTables/jQuery
``<script>`` tags directly into
``_static/dataset_generated/dataset_summary_table.html``. The global
``html_js_files`` in conf.py deliberately does **not** load that stack, so
if the generator stops inlining we'd lose the UI silently while the build
still passes.

This extension wires a ``builder-inited`` hook that fails fast when the
inline markers are missing.
"""

from __future__ import annotations

from pathlib import Path

from sphinx.util import logging

LOGGER = logging.getLogger(__name__)


def _assert_dataset_table_inlines_datatables(app) -> None:
    """Fail fast if the generated dataset-summary table lost its inline JS.

    ``html_css_files`` and ``html_js_files`` deliberately no longer load
    the DataTables/jQuery stack globally; the contract is that
    ``prepare_summary_tables.py`` inlines those CDN ``<script>`` tags
    directly into
    ``_static/dataset_generated/dataset_summary_table.html``. If the
    generator changes and drops the inlining, the dataset_summary page
    silently loses its interactivity — the HTML renders as a plain
    ``<table>`` with no sorting or filtering, and the build still passes.

    This hook verifies the marker scripts are present. Raise (not warn)
    so the failure is loud: a silent loss of the flagship page's UI is
    much worse than a build error.
    """
    table_path = (
        Path(app.srcdir)
        / "_static"
        / "dataset_generated"
        / "dataset_summary_table.html"
    )
    if not table_path.is_file():
        # File missing entirely — `prepare_summary_tables.py` hasn't run
        # yet (common on partial local rebuilds). Soft-warn rather than
        # block the build; `make html`/`html-noplot` already run the
        # generator before sphinx-build.
        LOGGER.warning(
            "Expected %s to exist; dataset_summary interactivity may be "
            "disabled until prepare_summary_tables.py runs.",
            table_path,
        )
        return
    content = table_path.read_text(encoding="utf-8", errors="replace")
    required_markers = ("datatables.min.js", "jquery-3.7.1.min.js")
    missing = [m for m in required_markers if m not in content]
    if missing:
        raise RuntimeError(
            f"{table_path.name} is missing expected inline CDN scripts: "
            f"{missing}. The global `html_js_files` in conf.py was slimmed "
            "on the assumption that prepare_summary_tables.py inlines the "
            "DataTables stack. Either restore the inlining in the "
            "generator, or re-add the scripts to `html_js_files`."
        )


def setup(app) -> dict:
    app.connect("builder-inited", _assert_dataset_table_inlines_datatables)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
