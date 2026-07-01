"""Sphinx extension: copy ``dataset_summary.csv`` into the build's ``_static``.

The dataset_summary page (and a few JS counters) loads the catalog CSV
from ``_static/dataset_summary.csv``. The CSV lives in the installed
``eegdash`` package, not the docs tree, so we copy it into the build
output at ``build-finished``.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from sphinx.util import logging

LOGGER = logging.getLogger(__name__)


def _copy_dataset_summary(app, exception) -> None:
    if exception is not None or not getattr(app, "builder", None):
        return

    csv_path = Path(importlib.import_module("eegdash.dataset").__file__).with_name(
        "dataset_summary.csv"
    )
    if not csv_path.exists():
        LOGGER.warning("dataset_summary.csv not found; skipping counter data copy.")
        return

    static_dir = Path(app.outdir) / "_static"
    try:
        static_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, static_dir / "dataset_summary.csv")
    except OSError as exc:
        LOGGER.warning("Unable to copy dataset_summary.csv to _static: %s", exc)


def setup(app) -> dict:
    app.connect("build-finished", _copy_dataset_summary)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
