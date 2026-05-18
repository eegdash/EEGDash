"""Sphinx extension: inject per-page SEO context.

Emits structured data (JSON-LD), canonical URL overrides, per-page meta
descriptions, and noindex hints during the ``html-page-context`` phase.
Coordinates with ``description_cap`` (a later-priority hook that caps
every description tag in the final ``context['metatags']``) and with
``sitemap_canonical`` (which patches the emitted sitemap.xml to keep
URLs aligned).
"""

from __future__ import annotations

import html
import json
import re
from typing import Mapping, Sequence

import eegdash

_HOMEPAGE_JSONLD = {
    "@context": "https://schema.org",
    "@type": "Organization",
    "name": "EEG Dash",
    "alternateName": "EEGDash",
    "url": "https://eegdash.org/",
    "logo": "https://eegdash.org/_static/eegdash_social_card.png",
    "description": (
        "EEGDash is a Python library and catalog for 700+ BIDS-first EEG, "
        "MEG, fNIRS, EMG, and iEEG datasets, providing PyTorch-ready data "
        "access for machine learning and reproducible neuroscience research."
    ),
    "sameAs": [
        "https://github.com/eegdash/EEGDash",
        "https://pypi.org/project/eegdash/",
        "https://registry.opendata.aws/eegdash/",
    ],
}

_SOFTWARE_JSONLD = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    "name": "EEGDash",
    "alternateName": "EEG-DaSh",
    "applicationCategory": "ScienceApplication",
    "applicationSubCategory": "DeveloperApplication",
    "operatingSystem": "Cross-platform",
    "url": "https://eegdash.org/",
    "softwareVersion": eegdash.__version__,
    "codeRepository": "https://github.com/eegdash/EEGDash",
    "programmingLanguage": "Python",
    "runtimePlatform": "Python 3.11+",
    "license": "https://opensource.org/licenses/BSD-3-Clause",
    "downloadUrl": "https://pypi.org/project/eegdash/",
    "installUrl": "https://pypi.org/project/eegdash/",
    "description": (
        "Python library for discovering, loading, and preprocessing 700+ "
        "BIDS-first EEG/MEG datasets. Integrates with MNE-Python, "
        "braindecode, and PyTorch for machine-learning workflows on "
        "open neuroelectromagnetic data."
    ),
    "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
}

_WEBSITE_JSONLD = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "name": "EEG Dash",
    "url": "https://eegdash.org/",
    "potentialAction": {
        "@type": "SearchAction",
        "target": {
            "@type": "EntryPoint",
            "urlTemplate": "https://eegdash.org/dataset_summary.html?q={search_term_string}",
        },
        "query-input": "required name=search_term_string",
    },
}


# Per-page descriptions for pages that sphinx-gallery regenerates (and thus
# wipe any inline `.. meta::` edits). Keyed by Sphinx pagename (no suffix).
_AUTO_PAGE_DESCRIPTIONS: dict[str, str] = {
    "generated/auto_examples/index": (
        "EEGDash tutorials and runnable examples — dataset loading, feature "
        "extraction, transfer learning, and the EEG2025 Competition challenges."
    ),
}


# schema.org/DataCatalog for the dataset catalog page. Lets Google Dataset
# Search treat eegdash.org as an aggregator. Per-dataset Dataset JSON-LD is
# still emitted on each /api/dataset/eegdash.dataset.*.html page, which
# Google reaches through the sitemap; this wrapper just declares the
# catalog identity.
_DATACATALOG_JSONLD = {
    "@context": "https://schema.org",
    "@type": "DataCatalog",
    "name": "EEGDash dataset catalog",
    "alternateName": "EEG-DaSh catalog",
    "url": "https://eegdash.org/dataset_summary.html",
    "description": (
        "Searchable catalog of 700+ BIDS-first EEG, MEG, fNIRS, EMG, and "
        "iEEG datasets aggregated from OpenNeuro, NEMAR, Zenodo, Figshare, "
        "SciDB, OSF, DataRN, and EEGManyLabs. Each row loads in Python via "
        "the EEGDash library (pip install eegdash)."
    ),
    "license": "https://opensource.org/licenses/BSD-3-Clause",
    "keywords": (
        "EEG, MEG, iEEG, fNIRS, EMG, BIDS, neuroscience, machine learning, "
        "Python, PyTorch, MNE-Python, braindecode, OpenNeuro, NEMAR"
    ),
    "provider": {
        "@type": "Organization",
        "name": "EEGDash",
        "url": "https://eegdash.org/",
    },
    "isAccessibleForFree": True,
    "inLanguage": "en",
}


# HowTo JSON-LD for install pages. Marks the install as a structured
# procedure so Google can surface it in rich results.
def _install_howto_jsonld(page_title: str, step_names: Sequence[str]) -> dict:
    return {
        "@context": "https://schema.org",
        "@type": "HowTo",
        "name": page_title,
        "description": (
            "Install the EEGDash Python library to load 700+ BIDS-first "
            "EEG/MEG datasets with PyTorch."
        ),
        "totalTime": "PT2M",
        "supply": [
            {"@type": "HowToSupply", "name": "Python 3.11+"},
            {"@type": "HowToSupply", "name": "pip or uv"},
        ],
        "step": [
            {
                "@type": "HowToStep",
                "position": i + 1,
                "name": name,
            }
            for i, name in enumerate(step_names)
        ],
    }


_INSTALL_HOWTO: Mapping[str, dict] = {
    "install/install": _install_howto_jsonld(
        "Install EEGDash",
        [
            "Check Python 3.11+ is available (python --version).",
            "Run pip install eegdash (or uv pip install eegdash).",
            "Import the library: from eegdash import EEGDashDataset.",
            "Load a dataset: EEGDashDataset(dataset='ds002718').",
        ],
    ),
    "install/install_pip": _install_howto_jsonld(
        "Install EEGDash with pip",
        [
            "Create or activate a Python 3.11+ environment.",
            "Run pip install eegdash (upgrade pip first on older Pythons).",
            "Verify with python -c 'import eegdash; print(eegdash.__version__)'.",
        ],
    ),
    "install/install_source": _install_howto_jsonld(
        "Install EEGDash from source",
        [
            "Clone https://github.com/eegdash/EEGDash.git.",
            "Create a Python 3.11+ virtual environment.",
            "Run pip install -e .[docs,tests] from the repository root.",
            "Run pytest to confirm the install works.",
        ],
    ),
}


# Article JSON-LD for narrative docs. Helps search engines treat the user
# guide and developer notes as primary reference content.
def _article_jsonld(title: str, description: str, url: str) -> dict:
    return {
        "@context": "https://schema.org",
        "@type": "TechArticle",
        "headline": title,
        "description": description,
        "url": url,
        "author": {
            "@type": "Organization",
            "name": "EEGDash",
            "url": "https://eegdash.org/",
        },
        "publisher": {
            "@type": "Organization",
            "name": "EEGDash",
            "logo": {
                "@type": "ImageObject",
                "url": "https://eegdash.org/_static/eegdash_social_card.png",
            },
        },
        "inLanguage": "en",
        "isAccessibleForFree": True,
    }


_ARTICLE_JSONLD: Mapping[str, dict] = {
    "quickstart": _article_jsonld(
        "EEGDash quick start guide",
        (
            "Quick start hub for the EEGDash Python library — the curated "
            "Cat A learning path, copy-paste recipes for query and filter, "
            "API configuration, and links to the full gallery and concepts."
        ),
        "https://eegdash.org/quickstart.html",
    ),
    "developer_notes": _article_jsonld(
        "EEGDash developer notes",
        (
            "Architecture, BIDS ingestion pipeline, test suite, and "
            "contributor workflows for the EEGDash Python library."
        ),
        "https://eegdash.org/developer_notes.html",
    ),
}


_MIN_META_DESC_CHARS = 50  # Ahrefs' "too short" threshold


def _cap_meta_description(text: str, limit: int = 160) -> str:
    """Trim a meta description to at most ``limit`` characters on a word
    boundary, appending an ellipsis. Google displays ~155-160 chars in
    the SERP and Ahrefs flags anything longer as "too long"; anything
    under 50 is flagged "too short" (we don't pad, caller is
    responsible for generating enough content).
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    trimmed = text[: limit - 1].rsplit(" ", 1)[0]
    return trimmed.rstrip(",. ") + "…"


# 8 narrow regexes (4 attribute orderings x 2 quote styles) covering
# every shape we've seen in the built HTML. One compound alternation
# with `[^"']*` fails when the content carries a different quote char
# (e.g. an apostrophe inside a double-quoted attribute — that silently
# broke a prior cap that shipped in #315 and was fixed in #317).
_META_DESC_PATTERNS = [
    # <meta name="description" … content="…">
    re.compile(
        r'<meta\s+(?:[^>]*?\s)?name="description"'
        r'\s+(?:[^>]*?\s)?content="(?P<v>[^"]*)"[^>]*>',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"<meta\s+(?:[^>]*?\s)?name='description'"
        r"\s+(?:[^>]*?\s)?content='(?P<v>[^']*)'[^>]*>",
        flags=re.IGNORECASE,
    ),
    # <meta content="…" … name="description">
    re.compile(
        r'<meta\s+(?:[^>]*?\s)?content="(?P<v>[^"]*)"'
        r'\s+(?:[^>]*?\s)?name="description"[^>]*>',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"<meta\s+(?:[^>]*?\s)?content='(?P<v>[^']*)'"
        r"\s+(?:[^>]*?\s)?name='description'[^>]*>",
        flags=re.IGNORECASE,
    ),
    # <meta property="og:description" … content="…">
    re.compile(
        r'<meta\s+(?:[^>]*?\s)?property="og:description"'
        r'\s+(?:[^>]*?\s)?content="(?P<v>[^"]*)"[^>]*>',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"<meta\s+(?:[^>]*?\s)?property='og:description'"
        r"\s+(?:[^>]*?\s)?content='(?P<v>[^']*)'[^>]*>",
        flags=re.IGNORECASE,
    ),
    # <meta content="…" … property="og:description">
    re.compile(
        r'<meta\s+(?:[^>]*?\s)?content="(?P<v>[^"]*)"'
        r'\s+(?:[^>]*?\s)?property="og:description"[^>]*>',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"<meta\s+(?:[^>]*?\s)?content='(?P<v>[^']*)'"
        r"\s+(?:[^>]*?\s)?property='og:description'[^>]*>",
        flags=re.IGNORECASE,
    ),
]


def _extract_first_description(metatags: str) -> str | None:
    """Return the first description content found in ``metatags``, or
    ``None`` if no description tag is present. Used to decide whether
    the backstop needs to fire (missing) or override (too short).
    """
    if not metatags:
        return None
    for pattern in _META_DESC_PATTERNS:
        m = pattern.search(metatags)
        if m:
            return html.unescape(m.group("v"))
    return None


def _replace_descriptions_in_metatags(metatags: str, new_text: str) -> str:
    """Rewrite the `content` attribute of every description /
    og:description tag in ``metatags`` to ``new_text``. The surrounding
    tag structure is preserved so extensions parsing the string later
    still see well-formed markup.
    """
    escaped = html.escape(new_text, quote=True)
    for pattern in _META_DESC_PATTERNS:
        metatags = pattern.sub(
            lambda m, escaped=escaped: m.group(0).replace(m.group("v"), escaped),
            metatags,
        )
    return metatags


def _trim_one_description(m: re.Match, limit: int) -> str:
    """Cap one description-tag match to ``limit`` chars (entity-aware)."""
    value = m.group("v")
    decoded = html.unescape(value)
    if len(decoded) <= limit:
        return m.group(0)
    capped = _cap_meta_description(decoded, limit=limit)
    return m.group(0).replace(value, html.escape(capped, quote=True))


def _cap_descriptions_in_metatags(metatags: str, limit: int = 155) -> str:
    """Cap every description / og:description content value in
    ``metatags`` at ``limit`` chars. HTML-entity-decodes before
    comparing so the visible budget is what the SERP renders.
    """
    if not metatags:
        return metatags

    for pattern in _META_DESC_PATTERNS:
        metatags = pattern.sub(lambda m: _trim_one_description(m, limit), metatags)
    return metatags


def _synthesize_description(pagename: str) -> str | None:
    """Generate a short, keyword-appropriate description for Sphinx
    auto-generated pages (per-dataset, per-module API reference).
    Returns ``None`` if no template matches — the page already has one.
    """
    ds_prefix = "api/dataset/eegdash.dataset."
    if pagename.startswith(ds_prefix) and pagename != ds_prefix.rstrip("."):
        ds_id = pagename[len(ds_prefix) :].upper()
        return (
            f"{ds_id} — BIDS-first EEG/MEG dataset accessible via the "
            f"EEGDash Python library. Load in a single line of code with "
            f"MNE-Python and braindecode. Full metadata, channels, and "
            f"citation on this page."
        )
    # Module pages that landed under api/dataset/eegdash.* (not
    # .dataset.*) — e.g. http_api_client, EEGDashDataset, bids_metadata.
    # Sphinx-apidoc puts them here but they lack their own descriptions.
    other_ds_prefix = "api/dataset/eegdash."
    if pagename.startswith(other_ds_prefix) and not pagename.startswith(ds_prefix):
        module = pagename[len(other_ds_prefix) :]
        return (
            f"EEGDash Python API reference for `eegdash.{module}` — "
            f"classes, functions, and schemas used to discover, load, "
            f"and preprocess BIDS-first EEG/MEG datasets for PyTorch "
            f"machine-learning workflows."
        )
    # Per-module API reference under api/generated/api-core/* or api-features/*
    if pagename.startswith("api/generated/api-core/"):
        module = pagename.rsplit("/", 1)[-1]
        return (
            f"EEGDash Python API reference for `{module}` — classes, "
            f"functions, and schemas used to load BIDS-first EEG/MEG "
            f"datasets, preprocess them, and feed them to PyTorch."
        )
    if pagename.startswith("api/generated/api-features/"):
        module = pagename.rsplit("/", 1)[-1]
        return (
            f"EEGDash feature-extraction API reference for `{module}` — "
            f"spectral, connectivity, complexity, and spatial feature "
            f"extractors for EEG/MEG machine-learning pipelines."
        )
    # Sphinx-gallery tutorial pages — sphinxext-opengraph picks the
    # first paragraph from the gallery-generated RST, which is usually
    # a one-liner ("This is a minimal tutorial demonstrating…") that
    # Ahrefs flags as too short. Pad with keywords that match the
    # tutorial's purpose.
    if pagename.startswith("generated/auto_examples/"):
        slug = pagename.rsplit("/", 1)[-1].replace("_", " ").removeprefix("noplot ")
        slug = slug.replace("tutorial ", "").strip() or "tutorial"
        return (
            f"EEGDash tutorial — {slug}. Runnable Python example showing "
            f"how to load BIDS-first EEG/MEG datasets, preprocess them "
            f"with MNE-Python, and train a model end-to-end."
        )
    # Source-aggregator pages under api/dataset/source_* (one per
    # upstream archive: OpenNeuro, NEMAR, Figshare, Zenodo, …). These
    # are generated by `_generate_dataset_docs` and don't carry their
    # own meta description.
    src_prefix = "api/dataset/source_"
    if pagename.startswith(src_prefix):
        source = pagename[len(src_prefix) :].replace("_", " ").title()
        return (
            f"EEG, MEG, and iEEG datasets from {source} wrapped by the "
            f"EEGDash Python library. Load any record with a single "
            f"call; preprocess with MNE-Python; train with braindecode."
        )
    return None


def _inject_seo_context(app, pagename, templatename, context, doctree) -> None:
    """Inject per-page SEO context (canonical override, JSON-LD, description).

    - Homepage: rewrite canonical to the bare host and emit Organization +
      SoftwareApplication JSON-LD.
    - Dataset pages: Dataset JSON-LD is written directly into the RST body
      via ``_format_jsonld_section`` (see ``DATASET_PAGE_TEMPLATE``).
    - Sphinx-gallery-regenerated pages (``auto_examples/index``): inject
      description via ``metatags`` since the RST is overwritten on every
      build.
    """
    if pagename == "index":
        # Keep canonical, OG URL, and sphinxext-opengraph's `og:url` all
        # pointing at `eegdash.org/` (bare host) — not `/index.html` —
        # so external scrapers and search engines see a consistent URL.
        context["pageurl"] = "https://eegdash.org/"
        # sphinxext-opengraph hardcodes og:url via
        # ``urljoin(ogp_site_url, builder.get_target_uri(pagename))``
        # which yields `/index.html` for the homepage. Its hook runs
        # before ours, so we fix the result in-place.
        metatags = context.get("metatags", "")
        context["metatags"] = metatags.replace(
            'content="https://eegdash.org/index.html"',
            'content="https://eegdash.org/"',
        )
        context["jsonld"] = json.dumps(
            [_HOMEPAGE_JSONLD, _SOFTWARE_JSONLD, _WEBSITE_JSONLD],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    # DataCatalog JSON-LD for the dataset_summary page. Google Dataset
    # Search uses this to recognize the page as a catalog aggregator; the
    # per-row Dataset JSON-LD lives on individual
    # `api/dataset/eegdash.dataset.*.html` pages and is reached via the
    # sitemap.
    if pagename == "dataset_summary":
        context["jsonld"] = json.dumps(
            _DATACATALOG_JSONLD, ensure_ascii=False, separators=(",", ":")
        )

    # HowTo JSON-LD on install pages. Eligible for Google's "How-to" rich
    # result and reinforces to LLM-driven assistants that pip install
    # eegdash is the canonical way to start using the library.
    howto = _INSTALL_HOWTO.get(pagename)
    if howto is not None:
        context["jsonld"] = json.dumps(howto, ensure_ascii=False, separators=(",", ":"))

    # TechArticle JSON-LD on the narrative docs. Signals "this is reference
    # content, not marketing" to search engines and LLM retrievers.
    article = _ARTICLE_JSONLD.get(pagename)
    if article is not None:
        context["jsonld"] = json.dumps(
            article, ensure_ascii=False, separators=(",", ":")
        )

    # BreadcrumbList JSON-LD on dataset pages. The visual breadcrumb nav
    # already exists (pydata-sphinx-theme's `<nav aria-label="Breadcrumb">`),
    # but without structured data Google can't use it for rich results.
    _ds_prefix = "api/dataset/eegdash.dataset."
    if pagename.startswith(_ds_prefix) and pagename != "api/dataset/eegdash.dataset":
        ds_name = pagename[len(_ds_prefix) :]
        breadcrumb = {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": 1,
                    "name": "Home",
                    "item": "https://eegdash.org/",
                },
                {
                    "@type": "ListItem",
                    "position": 2,
                    "name": "Datasets",
                    "item": "https://eegdash.org/api/dataset/api_dataset.html",
                },
                {
                    "@type": "ListItem",
                    "position": 3,
                    "name": ds_name,
                },
            ],
        }
        # Dataset JSON-LD (the `Dataset` schema) is already embedded in the
        # RST page body. BreadcrumbList goes in <head> via layout.html's
        # `{% if jsonld %}` block.
        context["jsonld"] = json.dumps(
            breadcrumb, ensure_ascii=False, separators=(",", ":")
        )

    description = _AUTO_PAGE_DESCRIPTIONS.get(pagename)
    if description:
        # Cap before HTML-escaping so the visible budget is what the
        # scanner counts.
        description = _cap_meta_description(description)
        # Escape attribute-unsafe chars (`"`, `<`, `&`) before interpolating
        # into the HTML. Current descriptions are trusted constants, but any
        # future addition containing a quote or ampersand would otherwise
        # silently produce broken markup.
        escaped = html.escape(description, quote=True)
        tag = (
            f'<meta name="description" content="{escaped}" />'
            f'<meta property="og:description" content="{escaped}" />'
        )
        existing = context.get("metatags") or ""
        context["metatags"] = existing + tag

    # Backstop meta descriptions for the auto-generated reference pages
    # (per-dataset, per-module). These pages ship either no description
    # at all or a very short first-paragraph excerpt that Ahrefs flags
    # as "too short" (< 50 chars). The hook fires in BOTH cases:
    # - no description  -> append our template
    # - short description (<50 chars) -> replace it with our template
    synth = _synthesize_description(pagename)
    if synth:
        existing = context.get("metatags") or ""
        current = _extract_first_description(existing)
        if current is None:
            # No description yet: append.
            escaped = html.escape(_cap_meta_description(synth), quote=True)
            tag = (
                f'<meta name="description" content="{escaped}" />'
                f'<meta property="og:description" content="{escaped}" />'
            )
            context["metatags"] = existing + tag
        elif len(current) < _MIN_META_DESC_CHARS:
            # Too short: swap in the synth text, preserving the surrounding
            # `<meta>` tag shape so nothing else in the pipeline gets
            # surprised by the edit.
            context["metatags"] = _replace_descriptions_in_metatags(
                existing, _cap_meta_description(synth)
            )

    # Dataset-summary chart fragments (`dataset_summary/table`,
    # `.../treemap`, etc.) are partial `.. include::` sources; Sphinx
    # builds them as standalone pages as a side-effect but they render
    # the same chart twice when someone lands on them directly. Tag
    # them `noindex` so search engines don't show them as orphan hits.
    # Also noindex Sphinx's own utility pages that ship without a
    # sitemap entry (genindex, search, sg_api_usage) — otherwise
    # scanners flag them as "indexable page with missing description".
    _NOINDEX_PAGENAMES = {"genindex", "search", "sg_api_usage"}
    noindex_needed = pagename in _NOINDEX_PAGENAMES or (
        pagename.startswith("dataset_summary/") and pagename != "dataset_summary"
    )
    if noindex_needed:
        existing = context.get("metatags") or ""
        if 'name="robots"' not in existing:
            context["metatags"] = (
                existing + '<meta name="robots" content="noindex,follow" />'
            )

    # NOTE: description capping runs from a separate late-priority hook
    # (``description_cap``). Doing it here would miss any descriptions
    # that sphinxext-opengraph adds later in the same ``html-page-context``
    # phase — its handler runs after ours at the default priority.


def setup(app) -> dict:
    app.connect("html-page-context", _inject_seo_context)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
