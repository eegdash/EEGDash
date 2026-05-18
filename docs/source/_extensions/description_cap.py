"""Sphinx extension: cap every meta description tag at the SERP budget.

Late-priority ``html-page-context`` handler that trims every
``<meta name="description">`` and ``<meta property="og:description">``
tag in the final ``context['metatags']`` string to ~155 characters
(Google's SERP display budget, Ahrefs' "too long" threshold).

Runs at priority 900 so it sees the metatags AFTER sphinxext-opengraph
(default priority 500) has appended its own description. Relies on the
regex stack and the per-tag cap helper that already live in
``seo_context``; importing here avoids duplicating those patterns.
"""

from __future__ import annotations

from seo_context import _cap_descriptions_in_metatags


def _cap_descriptions_hook(app, pagename, templatename, context, doctree):
    """Late-priority ``html-page-context`` handler that caps every
    description / og:description tag added by any earlier handler.

    Sphinx delivers events to connected callbacks in priority order
    (higher priority == later execution; default 500). sphinxext-
    opengraph registers at the default priority and writes its own
    description into ``context['metatags']`` during this phase, so
    any capping we do inside our own default-priority handler misses
    those insertions. Running at priority 900 guarantees we see the
    final value regardless of load order.
    """
    context["metatags"] = _cap_descriptions_in_metatags(context.get("metatags") or "")


def setup(app) -> dict:
    # Priority 900 -- higher = later -- so we run after every other
    # ``html-page-context`` listener (sphinxext-opengraph, seo_context).
    app.connect("html-page-context", _cap_descriptions_hook, priority=900)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
