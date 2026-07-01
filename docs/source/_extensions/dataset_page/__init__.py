"""Sphinx extension: render one dataset catalog page from a directive.

Usage::

    .. dataset-page:: DS002718

Replaces the legacy ``DATASET_PAGE_TEMPLATE`` f-string + ``_generate_dataset_docs``
pipeline (write 700+ ephemeral .rst files at builder-inited time, then re-parse
them). The directive looks up the dataset row from
:class:`eegdash.dataset.snapshot.DatasetSnapshot`, calls the section formatters
inline, and parses the assembled RST into docutils nodes directly -- no
80-line f-string skeleton, no per-page disk write of the rendered page body.

The companion ``builder-inited`` handler still emits 1-line shell ``.rst``
files (``eegdash.dataset.<NAME>.rst`` containing just ``.. dataset-page::
<NAME>``) so the Sphinx toctree machinery in ``docs/source/api/api.rst``
keeps working unchanged. That keeps URLs stable
(``/api/dataset/eegdash.dataset.<NAME>.html``) without re-implementing
toctree wiring as a virtual-document hook.

The implementation was originally a 2876-LOC monolith
(``docs/source/_extensions/dataset_page.py``); it is now split into:

* :mod:`._constants` -- regex, paths, license URL map.
* :mod:`.readme` -- Markdown -> RST conversion (importable without Sphinx).
* :mod:`.data_loaders` -- API/CSV loaders + context builder.
* :mod:`.sections` -- per-section RST formatters.
* :mod:`.directive` -- ``DatasetPageDirective`` + ``_build_page_rst``.
* :mod:`.shells` -- ``_write_dataset_shells`` builder-inited hook.

Sphinx loads this package by the string ``"dataset_page"`` (see
``docs/source/conf.py``'s ``extensions`` list) and calls :func:`setup`
on the package -- no caller changes were required by the split.

Top-level imports are deliberately kept light (constants + readme only)
so callers that just want the Markdown->RST helper do not pay the cost
of importing Sphinx and the eegdash dataset registry. The directive and
shell-writer modules are imported lazily inside :func:`setup`.
"""

from __future__ import annotations

from ._constants import AUTOGEN_NOTICE, DEFAULT_METADATA_FIELDS
from .readme import _convert_readme_to_rst, convert_readme_to_rst

__all__ = [
    "AUTOGEN_NOTICE",
    "DEFAULT_METADATA_FIELDS",
    "DatasetPageDirective",
    "_convert_readme_to_rst",
    "convert_readme_to_rst",
    "setup",
]


def __getattr__(name):
    """Lazily expose Sphinx-dependent symbols on first attribute access.

    Keeps ``import dataset_page`` and ``from dataset_page.readme import
    convert_readme_to_rst`` free of Sphinx / docutils / eegdash imports
    while still preserving the legacy public surface of the monolithic
    module (``dataset_page.DatasetPageDirective``,
    ``dataset_page._build_page_rst`` etc.) for any caller that imports
    them by name.
    """
    if name == "DatasetPageDirective":
        from .directive import DatasetPageDirective

        return DatasetPageDirective
    if name == "_build_page_rst":
        from .directive import _build_page_rst

        return _build_page_rst
    if name == "_snapshot_rows":
        from .directive import _snapshot_rows

        return _snapshot_rows
    if name == "_write_dataset_shells":
        from .shells import _write_dataset_shells

        return _write_dataset_shells
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def setup(app) -> dict:
    """Register the directive and the per-build shell-writer hook."""
    # Imported here (not at module top) so ``from dataset_page.readme
    # import convert_readme_to_rst`` does not transitively pull in
    # Sphinx, docutils, or eegdash.
    from .directive import DatasetPageDirective
    from .shells import _write_dataset_shells

    app.add_directive("dataset-page", DatasetPageDirective)
    app.connect("builder-inited", _write_dataset_shells)
    return {
        "version": "0.2",
        # The directive itself is read-safe (no shared mutable state
        # outside the module-level caches, which are populated under a
        # single GIL-protected first-access path). Write-safe because
        # each page is independent.
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
