"""Sphinx extension: rewrite the homepage entry in ``sitemap.xml``.

``sphinx-sitemap`` writes the homepage URL as ``…/index.html`` because
that's the page's filename, but our canonical override (set in
``seo_context``) points to the bare host. Ahrefs flags the mismatch as
"Non-canonical page in sitemap". This extension wires a ``build-finished``
hook that patches the emitted sitemap to keep both halves aligned.
"""

from __future__ import annotations

from pathlib import Path

from sphinx.util import logging

LOGGER = logging.getLogger(__name__)


def _rewrite_sitemap_index(app, exception) -> None:
    """Rewrite the homepage entry in ``sitemap.xml`` to the canonical
    bare-host URL.

    ``sphinx-sitemap`` emits ``https://eegdash.org/index.html`` for the
    homepage because that's the page's filename. We set a
    ``<link rel="canonical">`` override to ``https://eegdash.org/`` in
    ``_inject_seo_context``, which creates a mismatch Ahrefs flags as
    "Non-canonical page in sitemap". The two fixes must stay in sync —
    so we patch the emitted sitemap here at ``build-finished`` rather
    than fighting sphinx-sitemap's URL-construction logic.

    Originally shipped in #319 but silently dropped during the #318
    rebase conflict resolution (``git checkout --theirs`` on conf.py
    took the pre-#319 branch version). Re-adding here so the sitemap
    emitted on each deploy stays canonical.

    Safe no-op on: missing sitemap, non-HTML builder, already-canonical
    sitemap.
    """
    if exception is not None:
        return
    builder = getattr(app, "builder", None)
    if builder is None or builder.name != "html":
        return

    sitemap_path = Path(app.outdir) / "sitemap.xml"
    if not sitemap_path.exists():
        return

    try:
        text = sitemap_path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Unable to read %s: %s", sitemap_path, exc)
        return

    base = getattr(app.config, "html_baseurl", "") or ""
    if not base.endswith("/"):
        base += "/"
    index_url = f"{base}index.html"
    canonical = base

    if index_url not in text:
        return

    updated = text.replace(f"<loc>{index_url}</loc>", f"<loc>{canonical}</loc>")
    if updated == text:
        return
    try:
        sitemap_path.write_text(updated, encoding="utf-8")
        LOGGER.info(
            "sitemap.xml: rewrote %s -> %s for canonical alignment",
            index_url,
            canonical,
        )
    except OSError as exc:
        LOGGER.warning("Unable to write %s: %s", sitemap_path, exc)


def setup(app) -> dict:
    app.connect("build-finished", _rewrite_sitemap_index)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
