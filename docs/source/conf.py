import concurrent.futures
import csv
import html
import importlib
import inspect
import json
import os
import re
import shutil
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import quote

from sphinx.util import logging
from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath(".."))
# Local Sphinx extensions live under ``docs/source/_extensions``; make them
# importable before the ``extensions`` list below references them.
sys.path.insert(0, os.path.abspath("_extensions"))
if os.environ.get("SPHINX_BUILD", "") == "":
    os.environ["SPHINX_BUILD"] = "1"

import eegdash
import eegdash.dataset as dataset_module
from eegdash.dataset import EEGDashDataset
from eegdash.dataset.registry import fetch_datasets_from_api

# -- Project information -----------------------------------------------------

project = "EEG Dash"
copyright = f"2025-{datetime.now(tz=timezone.utc).year}, {project} Developers"
author = "Bruno Aristimunha and Arnaud Delorme"
release = eegdash.__version__
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx_time_estimation",
    "sphinxcontrib.bibtex",
    "dataset_explorer",
]

# Centralized bibliography (see docs/source/refs.bib + references.rst).
# Cite an entry from any RST or sphinx-gallery markdown cell with
# ``:cite:`<key>``` and the bibliography page renders the canonical list.
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

# -- Open Graph / Twitter Card configuration --------------------------------
# Populates <meta property="og:*"> and <meta name="twitter:*"> tags per page.
# Per-page overrides live in each RST file as `.. meta::` directives or via
# sphinxext-opengraph's `:og:...:` field lists.
ogp_site_url = "https://eegdash.org/"
ogp_site_name = "EEG Dash"
# PNG (1200x630) for social previews; X/Twitter and LinkedIn don't render
# SVG cards, so the earlier `eegdash_long.svg` was silently blank there.
# See `_static/eegdash_social_card.png` — generated from `eegdash_long.svg`
# with the logo centered on a white canvas.
ogp_image = "https://eegdash.org/_static/eegdash_social_card.png"
ogp_image_alt = "EEG Dash — data-sharing interface for M/EEG datasets"
ogp_description_length = 200
ogp_enable_meta_description = True
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary_large_image" />',
    '<meta name="twitter:site" content="@eegdash" />',
]

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# Autosummary: generate stub pages for documented items
autosummary_generate = True
# Include members that are imported into modules (e.g., re-exported dataset classes)
autosummary_imported_members = True
autosummary_ignore_module_all = False

# Suppress benign warnings
suppress_warnings = [
    # Sphinx-Gallery uses functions/classes in config which are not picklable
    "config.cache",
]

autodoc_type_aliases = {
    "FeatureExtractor": "eegdash.features.extractors.FeatureExtractor",
    "MultivariateFeature": "eegdash.features.extractors.MultivariateFeature",
    "FeaturesConcatDataset": "eegdash.features.datasets.FeaturesConcatDataset",
    "FeaturesDataset": "eegdash.features.datasets.FeaturesDataset",
    "TrainableFeature": "eegdash.features.extractors.TrainableFeature",
}

python_use_unqualified_type_names = False

# -- intersphinx ------------------------------------------------------------
# Resolve cross-references to external libraries so ``:class:`pandas.DataFrame```
# etc. become real hyperlinks in the rendered HTML instead of plain inline
# code. Inventories are cached locally by Sphinx; broken or unreachable
# upstream sites don't block local builds.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "mne": ("https://mne.tools/stable", None),
    "braindecode": ("https://braindecode.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/eegdash_image_only.svg"
html_favicon = "_static/favicon.ico"
html_title = "EEG Dash"
html_short_title = "EEG Dash"
# Only truly-global stylesheets are loaded here. The DataTables CSS bundle
# used by `dataset_summary.html` is already inlined into that page via the
# generated `_static/dataset_generated/dataset_summary_table.html` include,
# so keeping it in `html_css_files` would duplicate ~40 KB of CSS on every
# other page of the site for no benefit.
html_css_files = [
    "custom.css",
    "css/treemap.css",
    "css/custom.css",
]
# Only truly-global JS is loaded here; page-specific scripts (homepage hero
# search, dataset-summary DataTables stack) are gated in `_templates/layout.html`
# by `pagename`. The DataTables stack that dataset_summary depends on is
# inlined into `_static/dataset_generated/dataset_summary_table.html` by the
# generator, so we avoid double-loading by not listing it globally.
html_js_files = [
    ("js/tag-palette.js", {"defer": "defer"}),
    # Live search in PyData theme search modal (rendered on every page).
    ("js/search-as-you-type.js", {"defer": "defer"}),
    # Lazy-load the electrode-explorer iframe on <details> expansion.
    ("js/lazy-embed.js", {"defer": "defer"}),
]

# Required for sphinx-sitemap: set the canonical base URL of the site
# Make sure this matches the actual published docs URL and ends with '/'
html_baseurl = "https://eegdash.org/"

html_theme_options = {
    "icon_links_label": "External Links",  # for screen reader
    # Show an "Edit this page" button linking to GitHub
    "use_edit_page_button": True,
    "navigation_with_keys": False,
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    "navigation_depth": 6,
    "show_nav_level": 2,
    "show_toc_level": 1,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "logo": {
        "image_light": "_static/eegdash_long.svg",
        "image_dark": "_static/eegdash_long.svg",
        "alt_text": "EEG Dash Logo",
    },
    "external_links": [
        {"name": "EEG2025", "url": "https://eeg2025.github.io/"},
    ],
    # Local SVG icons instead of FontAwesome. Pydata-sphinx-theme only
    # loads fontawesome.js (~540 KiB) when at least one icon_links entry
    # has type="fontawesome"; switching to local SVGs removes that
    # dependency from every page load.
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/eegdash/EEGDash",
            "icon": "_static/icons/github.svg",
            "type": "local",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/eegdash/",
            "icon": "_static/icons/pypi.svg",
            "type": "local",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/8jd7nVKwsc",
            "icon": "_static/icons/discord.svg",
            "type": "local",
        },
    ],
}

html_sidebars = {
    "index": [],  # Remove sidebars on homepage
    "dataset_summary": [],
    "api": [],
    "installation": [],
    # sphinx-gallery pages render a large code preview; the sidebar adds
    # ~15 KB of chrome above the fold for no navigation gain (gallery
    # pages already have their own previous/next nav injected).
    "generated/auto_examples/*": [],
    "generated/auto_examples/tutorials/*": [],
    "generated/auto_examples/tutorials/*/*": [],
    "generated/auto_examples/applied/*": [],
    "generated/auto_examples/how_to/*": [],
    "generated/auto_examples/dev_scripts/*": [],
    "generated/auto_examples/eeg2025/*": [],
    "generated/auto_examples/hpc/*": [],
}

# Copy extra files (e.g., robots.txt) to the output root
html_extra_path = ["_extra"]

# Provide GitHub context so the edit button and custom templates
# (e.g., Sphinx-Gallery "Open in Colab") know where the source lives.
# These values should match the repository and docs location.
html_context = {
    "github_user": "eegdash",
    "github_repo": "EEGDash",
    # Branch used to build and host the docs
    "github_version": "develop",
    # Path to the documentation root within the repo
    "doc_path": "docs/source",
    "default_mode": "light",
}


# Linkcode configuration: map documented objects to GitHub source lines
def _linkcode_resolve_py_domain(info):
    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname:
        return None

    try:
        submod = sys.modules.get(modname)
        if submod is None:
            submod = importlib.import_module(modname)

        obj = submod
        for part in fullname.split("."):
            obj = getattr(obj, part)

        # Unwrap decorators to reach the actual implementation
        obj = inspect.unwrap(obj)
        fn = inspect.getsourcefile(obj) or inspect.getfile(obj)
        if not fn:
            return None
        fn = os.path.realpath(fn)

        # Compute line numbers
        try:
            source, start = inspect.getsourcelines(obj)
            end = start + len(source) - 1
            linespec = f"#L{start}-L{end}"
        except OSError:
            linespec = ""

        # Make path relative to repo root (parent of the installed package dir)
        pkg_dir = os.path.realpath(os.path.dirname(eegdash.__file__))
        repo_root = os.path.realpath(os.path.join(pkg_dir, os.pardir))
        rel_path = os.path.relpath(fn, start=repo_root)

        # Choose commit/branch for links; override via env if provided
        commit = os.environ.get(
            "LINKCODE_COMMIT", html_context.get("github_version", "main")
        )
        return f"https://github.com/{html_context['github_user']}/{html_context['github_repo']}/blob/{commit}/{rel_path}{linespec}"
    except Exception:
        return None


def linkcode_resolve(domain, info):
    if domain == "py":
        return _linkcode_resolve_py_domain(info)
    return None


# -- Extension configurations ------------------------------------------------
autoclass_content = "both"

# Numpydoc
numpydoc_show_class_members = False

# Sphinx Gallery
EX_DIR = "../../examples"  # relative to docs/source
# Sphinx-gallery 0.20 only walks one level: each entry in `examples_dirs` is
# treated as a gallery root, and its immediate subdirectories become
# subsections (with plot_*.py files collected at that depth only).
# `examples/tutorials/` itself is two levels deep (it has nested category
# subdirs like ``00_start_here/`` rather than ``plot_*.py`` files at the
# top), so we list each tutorial category as its own gallery root and pair
# it with a matching `gallery_dirs` entry. Same pattern for the leaf-only
# directories (`how_to`, `applied`, `eeg2025`, `hpc`, `dev_scripts`).
TUTORIAL_SUBDIRS = [
    "00_start_here",
    "10_core_workflow",
    "20_event_related",
    "30_resting_state",
    "40_features",
    "50_evaluation",
    "70_transfer_foundation",
]
LEAF_DIRS = ["how_to", "applied", "eeg2025", "hpc", "dev_scripts"]
EX_DIRS = [f"{EX_DIR}/tutorials/{name}" for name in TUTORIAL_SUBDIRS] + [
    f"{EX_DIR}/{name}" for name in LEAF_DIRS
]
GALLERY_DIRS = [
    f"generated/auto_examples/tutorials/{name}" for name in TUTORIAL_SUBDIRS
] + [f"generated/auto_examples/{name}" for name in LEAF_DIRS]
sphinx_gallery_conf = {
    "examples_dirs": EX_DIRS,
    "gallery_dirs": GALLERY_DIRS,
    # Execute examples by default for CI builds; use html-noplot target for local fast builds
    "plot_gallery": True,
    # Don't fail the build when examples error (e.g. missing cache, API down).
    # Failed examples show a traceback in the gallery instead of crashing CI.
    "abort_on_example_error": True,
    "only_warn_on_example_error": True,
    "binder": {
        "org": "eegdash",
        "repo": "EEGDash",
        "branch": "main",
        "binderhub_url": "https://mybinder.org",
        "dependencies": "binder/requirements.txt",
        "notebooks_dir": "notebooks",
        "use_jupyter_lab": True,
    },
    "capture_repr": ("_repr_html_", "__repr__"),
    # Each entry in `examples_dirs` above is a leaf gallery (its plot_*.py
    # files are immediate children), so `nested_sections=False` is correct
    # and avoids the `_replace_md5(None)` AssertionError that sphinx-gallery
    # 0.20 raises when a parent gallery dir contains only subdirectories.
    "nested_sections": False,
    "backreferences_dir": "gen_modules/backreferences",
    "inspect_global_variables": True,
    "show_memory": False,
    "show_api_usage": True,
    "doc_module": ("eegdash", "numpy", "scipy", "matplotlib"),
    "reference_url": {"eegdash": None},
    "filename_pattern": os.environ.get(
        "EEGDASH_GALLERY_FILENAME_PATTERN",
        r"/(?:plot|tutorial)_(?!_).*\.py",
    ),
    # Skip private helper modules (leading underscore) entirely. The
    # default ignore_pattern only matches __init__.py, which leaves files
    # like _pipeline_diagram.py rendered as standalone gallery pages
    # when they are imported by a sibling tutorial.
    "ignore_pattern": r"(?:^|/)_[^/]*\.py$",
    "matplotlib_animations": True,
    "reset_modules": ("matplotlib", "seaborn"),
    "first_notebook_cell": (
        "# For tips on running notebooks in Google Colab:\n"
        "# `pip install eegdash`\n"
        "%matplotlib inline"
    ),
    # `subsection_order` is no longer required because each tutorial
    # category and how-to/applied bucket is now its own gallery root (see
    # `examples_dirs` above). Order is therefore controlled by the order of
    # the entries in `examples_dirs`/`gallery_dirs`.
    "within_subsection_order": FileNameSortKey,
    # Polish: hide sub-1s timing rows, drop the noisy module signature line,
    # strip ``# sphinx-gallery-...`` config comments from the rendered output,
    # standardize card thumbnails at 320x224, ship a branded fallback when a
    # tutorial produces no figure, promote bare-string sentences to titles,
    # and pin the scraper list (we never use mayavi) to a single matplotlib
    # entry so docs builds don't import optional viz deps.
    "min_reported_time": 1,
    "show_signature": False,
    "remove_config_comments": True,
    "thumbnail_size": (320, 224),
    "default_thumb_file": str(Path(__file__).parent / "_static" / "eegdash_thumb.png"),
    # Note: ``promote_strings_to_titles`` is not a recognized key in
    # sphinx-gallery 0.20.x (raises ``ConfigError`` at startup), so it is
    # intentionally omitted; revisit when we bump to a release that ships it.
    "image_scrapers": ("matplotlib",),
}

# -- Custom Setup Function to fix the error -----------------------------------


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
CLONE_ROOT = REPO_ROOT / "ingestions" / "clone"
_DATASET_DETAILS_CACHE: dict[str, dict[str, object]] = {}
_DATASET_DETAILS_CACHE_LOCK = threading.Lock()
_DATASET_SUMMARY_CACHE = None


def _should_use_api_summary() -> bool:
    # Always try API first; set EEGDASH_NO_API=1 to disable
    return not bool(os.environ.get("EEGDASH_NO_API"))


# Shared User-Agent for outbound probes — many of the targets (NEMAR
# behind Cloudflare, HuggingFace) reject the bare urllib UA.
_PROBE_UA = "Mozilla/5.0 (compatible; EEGDashDocsBuild)"

# Match a "Paper DOI" shields.io badge inside a README and capture the
# wrapping URL — e.g. ``[![Paper DOI](...)](https://doi.org/10.1109/…)``.
# Used to recover paper DOIs that the structured ``external_links``
# block doesn't carry yet (NEMAR-ingested NM-series datasets in particular).
_README_PAPER_DOI_RE = re.compile(
    r"\[!\[Paper[^\]]*\]\([^)]+\)\]\((https?://[^)\s]+)\)",
    re.IGNORECASE,
)

# RST inline hyperlinks like `` `label <url>`__ `` or `` `label <url>`_ ``.
# Used in ``_is_prose`` to detect lede paragraphs that are nothing but
# README badge rows — those should never receive the drop-cap treatment.
_RST_HYPERLINK_RE = re.compile(r"`[^`<]+<[^>]+>`__?")


def _get_json(
    url: str,
    *,
    timeout: float = 10.0,
    extra_headers: Mapping[str, str] | None = None,
) -> dict | None:
    """GET ``url`` and return parsed JSON, or ``None`` on a recoverable
    error (network / HTTP error / bad JSON).

    Centralises the boilerplate that was duplicated across every probe
    helper: UA spoofing, timeout, narrow exception handling, JSON
    decode. Unexpected errors propagate so real bugs aren't silently
    swallowed.

    Use ``_head_ok(url, …)`` for HEAD-only checks.
    """
    headers = {"Accept": "application/json", "User-Agent": _PROBE_UA}
    if extra_headers:
        headers.update(extra_headers)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _head_content_length(
    url: str,
    *,
    timeout: float = 8.0,
) -> int | None:
    """HEAD ``url`` and return the integer ``Content-Length`` (or None
    when the request fails / header is missing). Used by probes that
    only need to know whether a resource has a meaningful body.
    """
    try:
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": _PROBE_UA}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not (200 <= resp.status < 400):
                return None
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None


def _load_dataset_summary_from_api():
    if not _should_use_api_summary():
        return None

    global _DATASET_SUMMARY_CACHE
    if _DATASET_SUMMARY_CACHE is not None:
        return _DATASET_SUMMARY_CACHE

    try:
        df = fetch_datasets_from_api()
    except Exception as exc:
        LOGGER.info("[dataset-docs] API summary fetch failed: %s", exc)
        df = None

    if df is None or df.empty:
        _DATASET_SUMMARY_CACHE = None
    else:
        _DATASET_SUMMARY_CACHE = df
    return _DATASET_SUMMARY_CACHE


DEFAULT_METADATA_FIELDS = [
    ("subject", "Subject identifier."),
    ("session", "Session identifier."),
    ("run", "Run identifier."),
    ("task", "Task label."),
    ("age", "Participant age (if available)."),
    ("gender", "Participant gender (if available)."),
    ("sex", "Participant sex (if available)."),
]


AUTOGEN_NOTICE = """..
   This documentation page is generated during the Sphinx build.
   The underlying code is manually maintained and not autogenerated.

"""


DATASET_PAGE_TEMPLATE = """{notice}:html_theme.sidebar_secondary.remove:
{og_description_field}

{meta_section}

{editorial_kicker_section}

{editorial_fieldcard_section}

{title}
{underline}

{jsonld_section}

{hero_section}

{editorial_layers_section}

{editorial_secnum_quickstart}

Quickstart
----------

{quickstart_section}

{editorial_secnum_about}

About This Dataset
------------------

{readme_section}

{editorial_secnum_info}

Cohort
------

{recording_stats_section}

{editorial_caveat_section}

{editorial_secnum_tech}

Signal · Electrodes & live trace
--------------------------------

{electrodes_traces_pair}

{nemar_analysis_section}

{editorial_secnum_manifest}

Manifest
--------

{explorer_section}

{dataset_info_dropdown_section}

{feedback_section}

{editorial_secnum_api}

API Reference
-------------

{api_section}

{editorial_access_modes_section}

{editorial_examples_gallery}

{editorial_footnotes_section}

{editorial_provenance_section}

See Also
--------

{see_also_section}

{editorial_colophon_section}

"""


DATASET_INDEX_TEMPLATE = """{notice}.. _api/dataset/api_dataset:

Datasets API
=======================

The :mod:`eegdash.dataset` package exposes dataset classes that are
registered dynamically at import time. See :doc:`eegdash.dataset` for the
module-level API, including :class:`~eegdash.dataset.EEGChallengeDataset`
and helper utilities.

What's in the registry
----------------------

EEGDash exposes **700+ OpenNeuro EEG datasets**, registered dynamically
from MongoDB. The table below summarizes the breakdown by experimental
type ({dataset_count} datasets in this build).

Base Dataset API
----------------

.. toctree::
   :maxdepth: 1

   eegdash.EEGDashDataset
   eegdash.dataset.EEGChallengeDataset

.. list-table:: Dataset counts by experimental type
   :widths: 60 20
   :header-rows: 1

   * - Experimental Type
     - Datasets
{experiment_rows}


All Datasets
------------

.. toctree::
   :maxdepth: 1
   :caption: Individual Datasets

{toctree_entries}

"""


BASE_DATASET_TEMPLATE = """{notice}.. _api_eegdash_challenge_dataset:

.. currentmodule:: eegdash.dataset

EEGChallengeDataset
===================

.. autoclass:: eegdash.dataset.EEGChallengeDataset
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

"""


PRIMARY_DATASET_TEMPLATE = """{notice}.. _api_eegdash_dataset:

.. currentmodule:: eegdash

EEGDashDataset
==============

.. autoclass:: eegdash.EEGDashDataset
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Usage Example
-------------

.. code-block:: python

   from eegdash import EEGDashDataset

   dataset = EEGDashDataset(cache_dir="./data", dataset="ds002718")
   print(f"Number of recordings: {{len(dataset)}}")

See Also
--------

* :mod:`eegdash.dataset`
* :class:`eegdash.dataset.EEGChallengeDataset`

"""


def _write_if_changed(path: Path, content: str) -> bool:
    """Write ``content`` to ``path`` if it differs from the current file."""
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _iter_dataset_classes() -> Sequence[str]:
    """Return the sorted dataset class names exported by ``eegdash.dataset``.

    Honours two env vars for fast iteration / preview builds:

    * ``EEGDASH_DOC_ONLY``: comma-separated list of dataset class names
      to keep (e.g. ``DS002893,DS001785``). Useful when iterating on the
      editorial template — limits the generator to just those datasets.
    * ``EEGDASH_DOC_LIMIT``: integer cap on the number of datasets.
      Applied after the ``DOC_ONLY`` filter (or to the full list when
      ``DOC_ONLY`` is unset).
    """
    class_names: list[str] = []
    for name in getattr(dataset_module, "__all__", []):
        if name == "EEGChallengeDataset":
            continue
        obj = getattr(dataset_module, name, None)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, EEGDashDataset):
            continue
        if getattr(obj, "_dataset", None) is None:
            continue
        class_names.append(name)

    sorted_names = sorted(class_names)

    only_env = (os.environ.get("EEGDASH_DOC_ONLY") or "").strip()
    if only_env:
        wanted = {n.strip().upper() for n in only_env.split(",") if n.strip()}
        sorted_names = [n for n in sorted_names if n.upper() in wanted]
        if not sorted_names:
            LOGGER.warning(
                "[dataset-docs] EEGDASH_DOC_ONLY=%r matched no datasets",
                only_env,
            )

    limit_env = (os.environ.get("EEGDASH_DOC_LIMIT") or "").strip()
    if limit_env:
        try:
            limit = max(0, int(limit_env))
            sorted_names = sorted_names[:limit]
        except ValueError:
            pass

    return tuple(sorted_names)


def _load_experiment_counts(dataset_names: Iterable[str]) -> list[tuple[str, int]]:
    """Return a sorted list of (experiment_type, count) pairs."""
    valid_names = {name.upper() for name in dataset_names}
    df = _load_dataset_summary_from_api()
    if df is not None and not df.empty:
        counter: Counter[str] = Counter()
        for _, row in df.iterrows():
            dataset_id = str(row.get("dataset", "")).strip().upper()
            if dataset_id not in valid_names:
                continue
            exp_type = str(row.get("type of exp") or "Unspecified").strip()
            counter[exp_type or "Unspecified"] += 1
        return sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    csv_path = Path(importlib.import_module("eegdash.dataset").__file__).with_name(
        "dataset_summary.csv"
    )
    if not csv_path.exists():
        return []

    counter: Counter[str] = Counter()

    with csv_path.open(encoding="utf-8") as handle:
        filtered = (
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        )
        reader = csv.DictReader(filtered)
        for row in reader:
            dataset_id = (row.get("dataset") or "").strip().upper()
            if dataset_id not in valid_names:
                continue
            exp_type = (row.get("type of exp") or "Unspecified").strip()
            counter[exp_type or "Unspecified"] += 1

    # Order by decreasing count then alphabetically for stable output
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _render_experiment_rows(pairs: Iterable[tuple[str, int]]) -> str:
    lines = []
    for exp_type, count in pairs:
        label = exp_type or "Unspecified"
        lines.append(f"   * - {label}\n     - {count}")
    if not lines:
        lines.append("   * - No experimental metadata available\n     - N/A")
    return "\n".join(lines)


def _render_toctree_entries(names: Sequence[str]) -> str:
    return "\n".join(f"   eegdash.dataset.{name}" for name in names)


def _load_dataset_rows(dataset_names: Sequence[str]) -> Mapping[str, Mapping[str, str]]:
    wanted = set(dataset_names)
    df = _load_dataset_summary_from_api()
    if df is not None and not df.empty:
        rows: dict[str, Mapping[str, str]] = {}
        for _, row in df.iterrows():
            dataset_id = str(row.get("dataset", "")).strip()
            if not dataset_id:
                continue
            class_name = dataset_id.upper()
            if class_name not in wanted:
                continue
            rows[class_name] = row.to_dict()
        if rows:
            return rows

    csv_path = Path(importlib.import_module("eegdash.dataset").__file__).with_name(
        "dataset_summary.csv"
    )
    if not csv_path.exists():
        return {}

    rows: dict[str, Mapping[str, str]] = {}

    with csv_path.open(encoding="utf-8") as handle:
        filtered = (
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        )
        reader = csv.DictReader(filtered)
        for row in reader:
            dataset_id = (row.get("dataset") or "").strip()
            if not dataset_id:
                continue
            class_name = dataset_id.upper()
            if class_name not in wanted:
                continue
            rows[class_name] = row

    return rows


def _clean_value(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown"}:
        return default
    return text


def _format_stat_counts(value: object, default: str = "") -> str:
    """Format JSON arrays of {val, count} objects into human-readable strings.

    Handles formats like: [{"val": 64, "count": 30}, {"val": 32, "count": 24}]
    Returns strings like: "64 (30), 32 (24)" or just "64" for single values.
    """
    if value is None:
        return default

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return default

    # Try to parse as JSON if it looks like a JSON array
    if text.startswith("["):
        try:
            items = json.loads(text)
            if not items:
                return default

            # Filter out null values and format
            valid_items = []
            for item in items:
                if isinstance(item, dict):
                    val = item.get("val")
                    count = item.get("count", 0)
                    if val is not None:
                        # Format as "value (count)" or just "value" if count is 1
                        if count > 1:
                            valid_items.append(f"{val} ({count})")
                        else:
                            valid_items.append(str(val))

            if not valid_items:
                return default

            # If all items have the same value, just show it once
            unique_vals = set(
                item.get("val") for item in items if isinstance(item, dict)
            )
            unique_vals.discard(None)
            if len(unique_vals) == 1:
                return str(unique_vals.pop())

            return ", ".join(valid_items)
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    # Fall back to regular cleaning
    return _clean_value(value, default)


def _collapse_whitespace(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _normalize_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [
            _collapse_whitespace(str(item).strip())
            for item in value
            if str(item).strip()
        ]
        return items
    text = _collapse_whitespace(str(value).strip())
    return [text] if text else []


def _value_or_unknown(value: str, field_type: str = "general") -> str:
    """Return value or a context-aware placeholder for missing data.

    Parameters
    ----------
    value : str
        The value to check.
    field_type : str
        The type of field, used to select an appropriate placeholder.
        Options: "n_channels", "pathology", "duration", "sampling_rate",
                 "subjects", "recordings", "tasks", "license", "general".

    """
    if value and value.strip() not in ("", "nan", "none", "null", "unknown", "—"):
        return value
    placeholders = {
        "n_channels": "Varies",
        "pathology": "Not specified",
        "duration": "Not calculated",
        "sampling_rate": "Varies",
        "subjects": "—",
        "recordings": "—",
        "tasks": "—",
        "license": "See source",
        "general": "—",
    }
    return placeholders.get(field_type, placeholders["general"])


def _normalize_doi(doi: str) -> str:
    if not doi:
        return ""
    return doi.replace("doi:", "").strip()


def _fetch_dataset_details_from_api(dataset_id: str) -> dict[str, object]:
    """Fetch detailed dataset information from the API.

    Uses the endpoint: /datasets/summary/{dataset_id}
    """
    if not _should_use_api_summary():
        return {}

    api_url = "https://data.eegdash.org/api/eegdash"

    # Try original ID first, then variants (API may be case-sensitive)
    ids_to_try = [dataset_id]
    # Also try with common case variations for known patterns
    if dataset_id.startswith("ds"):
        # OpenNeuro datasets are typically lowercase
        ids_to_try.append(dataset_id.lower())
    elif dataset_id.lower().startswith("eeg2025"):
        # EEG2025 datasets use mixed case in API
        ids_to_try.append(f"EEG2025r{dataset_id.lower().replace('eeg2025r', '')}")

    data = None
    for try_id in ids_to_try:
        url = f"{api_url}/datasets/summary/{try_id}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data.get("success"):
                    break
        except Exception as exc:
            LOGGER.debug("[dataset-docs] API fetch for %s failed: %s", try_id, exc)
            continue

    if not data or not data.get("success"):
        return {}

    ds = data.get("data", {})
    if not ds:
        return {}

    # Extract year from timestamps
    year = ""
    timestamps = ds.get("timestamps", {}) or {}
    created_at = timestamps.get("dataset_created_at", "")
    if created_at and len(created_at) >= 4:
        year = created_at[:4]

    # Map API fields to details structure
    # Use computed_title if available (populated by compute-stats endpoint)
    title = _clean_value(ds.get("computed_title")) or _clean_value(ds.get("name"))
    if title and (
        title.lower().endswith((".tsv", ".json", ".csv", ".md"))
        or title.lower() == "readme"
    ):
        # Try to find a better title in the readme
        readme = _clean_value(ds.get("readme", ""))
        if readme.startswith("# "):
            title = readme.split("\n")[0][2:].strip()
            # Still check for sub-H1 patterns
            if title.lower().startswith("wrist:"):
                title = title.split(":", 1)[1].strip()

    details: dict[str, object] = {
        "title": title,
        "authors": _normalize_list(ds.get("authors")),
        "license": _clean_value(ds.get("license")),
        "doi": _clean_value(ds.get("dataset_doi")),
        "year": year,
        "readme": _clean_value(ds.get("readme")),
        "funding": _normalize_list(ds.get("funding")),
        "senior_author": _clean_value(ds.get("senior_author")),
        "contact_info": _normalize_list(ds.get("contact_info")),
        "n_subjects": ds.get("demographics", {}).get("subjects_count"),
        "total_files": ds.get("total_files"),
        "n_tasks": len(ds.get("tasks", []) or []),
        "tasks": ds.get("tasks") or [],
        "recording_modality": ds.get("recording_modality", []),
        "datatypes": ds.get("datatypes") or [],
        "size_bytes": ds.get("size_bytes"),
        "source": _clean_value(ds.get("source")),
        "demographics": ds.get("demographics"),
        "nchans_counts": ds.get("nchans_counts"),
        "sfreq_counts": ds.get("sfreq_counts"),
        "total_duration_s": ds.get("total_duration_s"),
        "bad_channels_info": ds.get("bad_channels_info"),
        # Editorial Brief — fields the rich field-card and provenance strip
        # need to render real values instead of TODO placeholders.
        "bids_version": _clean_value(ds.get("bids_version")),
        "tags": ds.get("tags") or {},
        "dataset_storage": ds.get("storage") or {},
        "associated_paper_doi": _clean_value(ds.get("associated_paper_doi")),
        "stats_computed_at": _clean_value(ds.get("stats_computed_at")),
        "digested_at": _clean_value((ds.get("timestamps") or {}).get("digested_at")),
        # Additional fields surfaced from the summary endpoint:
        "sessions": ds.get("sessions") or [],
        "dataset_created_at": _clean_value(
            (ds.get("timestamps") or {}).get("dataset_created_at")
        ),
        "dataset_modified_at": _clean_value(
            (ds.get("timestamps") or {}).get("dataset_modified_at")
        ),
        "data_processed": bool(ds.get("data_processed")),
        "contributing_labs": ds.get("contributing_labs") or [],
        "n_contributing_labs": ds.get("n_contributing_labs"),
        "experimental_modalities": ds.get("experimental_modalities") or [],
        "study_design": _clean_value(ds.get("study_design")),
        "study_domain": _clean_value(ds.get("study_domain")),
    }

    # Extract source URL from external_links
    external_links = ds.get("external_links", {}) or {}
    details["source_url"] = _clean_value(external_links.get("source_url"))
    details["paper_url"] = _clean_value(external_links.get("paper_url"))
    details["github_url"] = _clean_value(external_links.get("github_url"))
    details["osf_url"] = _clean_value(external_links.get("osf_url"))

    # Some NEMAR-ingested datasets never populate ``external_links.paper_url``
    # even though the README ships a ``[![Paper DOI](...)](https://doi.org/…)``
    # badge. The lede block already turns that badge into a visible link, so
    # mirror it into the rail's quick-actions row when no structured field
    # exists yet.
    if not details["paper_url"]:
        readme = ds.get("readme") or ""
        if isinstance(readme, str) and "Paper" in readme:
            m = _README_PAPER_DOI_RE.search(readme)
            if m:
                details["paper_url"] = m.group(1)

    return details


def _load_dataset_details(dataset_id: str) -> dict[str, object]:
    """Aggregate per-dataset metadata from local files + API + probes.

    Thread-safe: the in-process cache is guarded by a lock so that
    duplicate workers don't waste 4 network probes on the same id.
    Network probes are issued concurrently because they hit different
    hosts (EEGDash API · NEMAR · HuggingFace) and don't depend on each
    other.
    """
    dataset_id = dataset_id.lower()

    # Fast read under the lock.
    with _DATASET_DETAILS_CACHE_LOCK:
        cached = _DATASET_DETAILS_CACHE.get(dataset_id)
        if cached is not None:
            return cached

    details: dict[str, object] = {}

    # --- Local files first --------------------------------------------
    dataset_dir = CLONE_ROOT / dataset_id
    desc_path = dataset_dir / "dataset_description.json"
    if desc_path.exists():
        try:
            data = json.loads(desc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        details["title"] = _clean_value(data.get("Name"))
        details["authors"] = _normalize_list(data.get("Authors"))
        details["license"] = _clean_value(data.get("License"))
        details["doi"] = _clean_value(data.get("DatasetDOI"))
        details["how_to_acknowledge"] = _clean_value(data.get("HowToAcknowledge"))
        details["references"] = _normalize_list(data.get("ReferencesAndLinks"))
        details["funding"] = _normalize_list(data.get("Funding"))

    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        details.setdefault("doi", _clean_value(data.get("dataset_doi")))
        details["source_url"] = _clean_value(data.get("source_url"))

    # --- API + parallel probes ----------------------------------------
    # `_fetch_dataset_details_from_api` must run first because the other
    # probes don't depend on it. The four downstream probes hit
    # different hosts (EEGDash, NEMAR, HuggingFace) and are network-bound,
    # so we run them concurrently — drops single-dataset latency from
    # ~5×timeout to ~max(timeouts).
    api_details = _fetch_dataset_details_from_api(dataset_id)
    for key, value in api_details.items():
        if value and not details.get(key):
            details[key] = value

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            "sidecars_detected": pool.submit(_detect_sidecars_for_dataset, dataset_id),
            "hed_annotated": pool.submit(_detect_hed_annotation, dataset_id),
            "huggingface": pool.submit(_detect_huggingface_mirror, dataset_id),
            "participants_rows": pool.submit(
                _fetch_participants_from_records, dataset_id
            ),
        }
        for key, future in futures.items():
            try:
                details[key] = future.result()
            except Exception as exc:  # noqa: BLE001 — probe results are best-effort
                LOGGER.warning(
                    "[dataset-docs] probe %s failed for %s: %s",
                    key,
                    dataset_id,
                    exc,
                )
                details[key] = (
                    False
                    if key == "hed_annotated"
                    else {}
                    if key == "huggingface"
                    else []
                )

    # Write under the lock so a racing worker that already finished
    # earlier doesn't get clobbered.
    with _DATASET_DETAILS_CACHE_LOCK:
        existing = _DATASET_DETAILS_CACHE.setdefault(dataset_id, details)
    return existing


# Suffixes the BIDS spec defines for the modality-specific sidecars the
# editorial field-card lists ("events · channels · electrodes · coordsystem").
# These match against the trailing path segment of each storage.dep_key.
_SIDECAR_SUFFIXES = {
    "events.tsv": "events",
    "events.json": "events.json",
    "channels.tsv": "channels",
    "electrodes.tsv": "electrodes",
    "coordsystem.json": "coordsystem",
    "eeg.json": "eeg.json",
    "meg.json": "meg.json",
    "ieeg.json": "ieeg.json",
    "physio.tsv": "physio",
    "stim.tsv": "stim",
}


# Spec-conventional order for sidecar labels we render in the field-card.
_SIDECAR_RENDER_ORDER = (
    "events",
    "events.json",
    "channels",
    "electrodes",
    "coordsystem",
    "eeg.json",
    "meg.json",
    "ieeg.json",
    "physio",
    "stim",
)


def _detect_sidecars_for_dataset(dataset_id: str) -> list[str]:
    """Return a sorted list of BIDS sidecar kinds present for ``dataset_id``.

    Probes ONE sample record via the eegdash records API and inspects
    ``storage.dep_keys`` for known BIDS sidecar suffixes. Returns an
    empty list if the probe fails (network error, dataset not yet
    ingested).
    """
    if not _should_use_api_summary():
        return []

    dataset_lower = dataset_id.lower()
    query = json.dumps(
        {"dataset": dataset_lower, "_has_missing_files": {"$ne": True}},
        separators=(",", ":"),
    )
    url = (
        "https://data.eegdash.org/api/eegdash/records"
        f"?{urllib.parse.urlencode({'limit': 1, 'filter': query})}"
    )
    body = _get_json(url)
    if not body or not body.get("success") or not body.get("data"):
        return []

    storage = body["data"][0].get("storage") or {}
    dep_keys = storage.get("dep_keys") or []
    if not isinstance(dep_keys, list):
        return []

    found: set[str] = set()
    for key in dep_keys:
        path = str(key)
        for suffix, label in _SIDECAR_SUFFIXES.items():
            if path.endswith(suffix):
                found.add(label)
                break
    return [k for k in _SIDECAR_RENDER_ORDER if k in found]


def _detect_huggingface_mirror(dataset_id: str) -> dict[str, object]:
    """Probe the EEGDash HuggingFace org for a mirror of ``dataset_id``.

    Returns a dict with ``available`` (bool), ``url`` (str, dataset-specific
    when available, org page otherwise), ``downloads`` (int when known),
    and ``last_modified`` (ISO-8601 str when known). The HF API returns
    200 with an ``{"error": "..."}`` body for missing datasets, so the
    discriminator is the presence of an ``id`` field in the JSON payload.
    """
    org_url = "https://huggingface.co/EEGDash"
    fallback: dict[str, object] = {
        "available": False,
        "url": org_url,
        "downloads": None,
        "last_modified": None,
    }
    if not _should_use_api_summary():
        return fallback

    dataset_lower = dataset_id.lower()
    body = _get_json(
        f"https://huggingface.co/api/datasets/EEGDash/{dataset_lower}",
        timeout=8.0,
    )
    if not isinstance(body, dict) or not body.get("id"):
        return fallback

    return {
        "available": True,
        "url": f"https://huggingface.co/datasets/EEGDash/{dataset_lower}",
        "downloads": body.get("downloads"),
        "last_modified": body.get("lastModified"),
    }


def _fetch_participants_from_records(dataset_id: str) -> list[dict[str, object]]:
    """Pull per-subject demographics for a dataset.

    Prefers the dedicated ``/api/eegdash/datasets/{dataset_id}/participants``
    endpoint, which deduplicates server-side via Mongo ``$group`` and
    returns one row per subject in a single request. Falls back to
    paginating ``/records`` for older server deployments that don't
    expose the participants endpoint yet.

    Returns a list of ``{subject, sex, age, group, …}`` dicts. Empty
    list on any failure.
    """
    if not _should_use_api_summary():
        return []

    dataset_lower = dataset_id.lower()

    # --- Primary path: dedicated participants endpoint ----------------
    body = _get_json(
        f"https://data.eegdash.org/api/eegdash/datasets/{dataset_lower}/participants",
        timeout=12.0,
    )
    if body is not None and body.get("success") and isinstance(body.get("data"), list):
        participants: list[dict[str, object]] = []
        for entry in body["data"]:
            subject = str(entry.get("subject") or "").strip()
            if not subject:
                continue
            tsv = entry.get("participant_tsv") or {}
            if not isinstance(tsv, dict):
                tsv = {}
            participants.append({"subject": subject, **tsv})
        return participants

    # --- Fallback: paginate /records ----------------------------------
    # Triggered when the server lacks /participants (404/405) or when the
    # primary request fails outright (network error). The records-based
    # path returns the same shape so callers don't branch.
    query = json.dumps(
        {
            "dataset": dataset_lower,
            "suffix": {"$in": ["eeg", "ieeg", "emg", "meg"]},
            "_has_missing_files": {"$ne": True},
        },
        separators=(",", ":"),
    )
    seen: set[str] = set()
    participants = []
    skip = 0
    page_size = 1000
    max_skip = 20000  # safety bound — no real dataset has 20k recordings
    while skip < max_skip:
        url = (
            "https://data.eegdash.org/api/eegdash/records"
            f"?{urllib.parse.urlencode({'limit': page_size, 'skip': skip, 'filter': query})}"
        )
        page = _get_json(url, timeout=12.0)
        records = page.get("data") if isinstance(page, dict) else None
        if not records:
            break
        for record in records:
            subject = str(record.get("subject") or "").strip()
            if not subject or subject in seen:
                continue
            seen.add(subject)
            tsv = record.get("participant_tsv") or {}
            if not isinstance(tsv, dict):
                continue
            participants.append({"subject": subject, **tsv})
        if len(records) < page_size:
            break
        skip += page_size
    return participants


def _detect_hed_annotation(dataset_id: str) -> bool:
    """Return True when NEMAR has published a HED word-cloud for this dataset.

    Cheap signal: NEMAR only generates per-dataset HED word clouds when
    the events sidecar carries valid HED tags. A HEAD request to the
    SVG URL with a ``Content-Length > 1 KB`` check is enough — NEMAR's
    download endpoint returns 200 for missing files but with empty body.

    TODO: replace with a ``hed_annotated`` flag at ingest time so we
    don't depend on a NEMAR HEAD round-trip.
    """
    if not _should_use_api_summary():
        return False

    url = (
        "https://nemar.org/dataexplorer/download"
        f"?filepath=/data/nemar/openneuro//processed/event_summaries/"
        f"{dataset_id.lower()}/word_cloud.svg&file_type=svg"
    )
    # NEMAR returns 200 even for non-HED datasets (its download endpoint
    # always succeeds) — but the body is empty. Use Content-Length > 1 KB
    # to discriminate from a real word-cloud SVG.
    length = _head_content_length(url, timeout=8.0)
    return length is not None and length > 1024


def _build_dataset_context(
    class_name: str, row: Mapping[str, str] | None
) -> dict[str, object]:
    dataset_id = _clean_value(row.get("dataset") if row else "")
    dataset_id = dataset_id.lower() if dataset_id else class_name.lower()
    details = _load_dataset_details(dataset_id)

    modality = _clean_value((row or {}).get("record_modality"))
    if not modality:
        modality_raw = details.get("recording_modality", [])
        if isinstance(modality_raw, list):
            modality = ", ".join(str(m) for m in modality_raw)
        else:
            modality = _clean_value(modality_raw)
    if not modality:
        modality = _clean_value((row or {}).get("modality of exp"))

    source = _clean_value((row or {}).get("source"))
    if not source:
        source = "OpenNeuro"

    # Fallback to row data for fields that might not be in local JSON files
    title = _collapse_whitespace(_clean_value(details.get("title")))
    if not title or title.lower().endswith((".tsv", ".json")):
        title = _collapse_whitespace(_clean_value((row or {}).get("dataset_title")))

    license_text = _clean_value(details.get("license"))
    if not license_text:
        license_text = _clean_value((row or {}).get("license"))

    doi = _clean_value(details.get("doi"))
    if not doi:
        doi = _clean_value((row or {}).get("doi"))

    # Get year from details
    year = _clean_value(details.get("year"))
    if not year or year == "—":
        # Try to find year in references
        refs = details.get("references", [])
        if not refs:
            # Try to find in authors/citations in details if references is empty
            readme = str(details.get("readme", ""))
            # Look for 4 digits in parentheses or after author name
            years = re.findall(r"\((\d{4})\)", readme)
            if not years:
                years = re.findall(r"\b(19|20)\d{2}\b", readme)
            if years:
                year = years[0]

    n_subjects = _clean_value((row or {}).get("n_subjects"))
    if not n_subjects or n_subjects == "0":
        n_subjects = _clean_value(details.get("n_subjects"))

    n_records = _clean_value((row or {}).get("n_records"))
    if not n_records or n_records == "0":
        n_records = _clean_value(details.get("total_files"))

    n_tasks = _clean_value((row or {}).get("n_tasks"))
    if not n_tasks or n_tasks == "0":
        n_tasks = _clean_value(details.get("n_tasks"))

    # Size on disk
    size = _clean_value((row or {}).get("size"))
    if not size or size == "Unknown":
        size_bytes = details.get("size_bytes")
        if size_bytes:
            # Simple human readable size if utils is not easily importable here
            # or just copy-paste the logic for simplicity in conf.py
            try:
                s = float(size_bytes)
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if s < 1024.0:
                        size = (
                            f"{s:.2f} {unit}"
                            if unit not in ["B", "KB"]
                            else f"{int(s)} {unit}"
                        )
                        break
                    s /= 1024.0
            except (ValueError, TypeError):
                pass

    # Format
    dataset_format = "—"
    if source.lower() in ["openneuro", "nemar"]:
        dataset_format = "BIDS"

    s3_item_count = _clean_value((row or {}).get("s3_item_count"))
    if not s3_item_count or s3_item_count == "0":
        s3_item_count = _clean_value(details.get("total_files"))

    # Canonical / author-year identifiers — populated from the CSV (or API
    # response) when the name_suggester pipeline has run, else empty.
    # Reuses the registry's parser so the rendered docs match the
    # aliases the runtime catalog registers.
    from eegdash.dataset.registry import _parse_canonical_names  # noqa: WPS433

    canonical_names = _parse_canonical_names((row or {}).get("canonical_name"))
    author_year_name = _clean_value((row or {}).get("author_year"))

    return {
        "class_name": class_name,
        "dataset_id": dataset_id,
        "dataset_upper": dataset_id.upper(),
        "title": title,
        "year": year,
        "authors": details.get("authors", []),
        "license": license_text,
        "doi": doi,
        "canonical_names": canonical_names,
        "author_year_name": author_year_name,
        "source_url": _clean_value(details.get("source_url")),
        "paper_url": _clean_value(details.get("paper_url")),
        "github_url": _clean_value(details.get("github_url")),
        "osf_url": _clean_value(details.get("osf_url")),
        "references": details.get("references", []),
        "how_to_acknowledge": _clean_value(details.get("how_to_acknowledge")),
        "n_subjects": n_subjects,
        "n_records": n_records,
        "n_tasks": n_tasks,
        "n_channels": _format_stat_counts((row or {}).get("nchans_set")),
        "sampling_freqs": _format_stat_counts((row or {}).get("sampling_freqs")),
        "duration_hours_total": _clean_value((row or {}).get("duration_hours_total")),
        "size": size,
        "s3_item_count": s3_item_count,
        "modality": modality,
        "pathology": _clean_value((row or {}).get("Type Subject")),
        "tag_modality": _clean_value((row or {}).get("modality of exp")),
        "tag_type": _clean_value((row or {}).get("type of exp")),
        "source": source,
        "openneuro_url": f"https://openneuro.org/datasets/{dataset_id}",
        "nemar_url": f"https://nemar.org/dataexplorer/detail?dataset_id={dataset_id}",
        "metadata_fields": DEFAULT_METADATA_FIELDS,
        "format": dataset_format,
        "readme": _clean_value(details.get("readme")),
        "nemar_citation_count": _clean_value((row or {}).get("nemar_citation_count")),
        "demographics": details.get("demographics") or {},
        "participants_rows": details.get("participants_rows") or [],
        "nchans_counts": details.get("nchans_counts") or [],
        "sfreq_counts": details.get("sfreq_counts") or [],
        "total_duration_s": details.get("total_duration_s"),
        "bad_channels_info": details.get("bad_channels_info"),
        # Editorial Brief fields (surfaced from API; populated by
        # _fetch_dataset_details_from_api + the sidecar/HED probes).
        "bids_version": _clean_value(details.get("bids_version")),
        "tags": details.get("tags") or {},
        "datatypes": details.get("datatypes") or [],
        "tasks": details.get("tasks") or [],
        "funding": details.get("funding") or [],
        "senior_author": _clean_value(details.get("senior_author")),
        "contact_info": details.get("contact_info") or [],
        "sidecars_detected": details.get("sidecars_detected") or [],
        "hed_annotated": bool(details.get("hed_annotated")),
        "huggingface": details.get("huggingface")
        or {
            "available": False,
            "url": "https://huggingface.co/EEGDash",
            "downloads": None,
            "last_modified": None,
        },
        "associated_paper_doi": _clean_value(details.get("associated_paper_doi")),
        "digested_at": _clean_value(details.get("digested_at")),
        # Storage descriptor (backend, base S3 url, dep_keys).
        "dataset_storage": details.get("dataset_storage") or {},
        # Newly-surfaced fields for the editorial layout (#30):
        "sessions": details.get("sessions") or [],
        "dataset_created_at": _clean_value(details.get("dataset_created_at")),
        "dataset_modified_at": _clean_value(details.get("dataset_modified_at")),
        "data_processed": bool(details.get("data_processed")),
        "contributing_labs": details.get("contributing_labs") or [],
        "n_contributing_labs": details.get("n_contributing_labs"),
        "experimental_modalities": details.get("experimental_modalities") or [],
        "study_design": _clean_value(details.get("study_design")),
        "study_domain": _clean_value(details.get("study_domain")),
    }


def _compute_quality_score(context: Mapping[str, object]) -> tuple[str, str, int]:
    """Compute a data quality score based on metadata completeness.

    Returns
    -------
    tuple[str, str, int]
        (label, badge_color, percentage) where:
        - label: "Complete", "Good", "Partial", or "Limited"
        - badge_color: "success", "primary", "warning", or "secondary"
        - percentage: 0-100 indicating completeness

    """
    # Key fields to check for completeness
    fields_to_check = [
        ("title", context.get("title")),
        ("authors", context.get("authors")),
        ("license", context.get("license")),
        ("doi", context.get("doi")),
        ("n_subjects", context.get("n_subjects")),
        ("n_records", context.get("n_records")),
        ("n_channels", context.get("n_channels")),
        ("sampling_freqs", context.get("sampling_freqs")),
        ("modality", context.get("modality")),
        ("readme", context.get("readme")),
    ]

    filled = 0
    for name, value in fields_to_check:
        if value:
            if isinstance(value, str):
                if value.strip() and value.strip() not in (
                    "—",
                    "Varies",
                    "Not specified",
                    "Not calculated",
                    "See source",
                ):
                    filled += 1
            elif isinstance(value, list) and len(value) > 0:
                filled += 1
            else:
                filled += 1

    percentage = int((filled / len(fields_to_check)) * 100)

    if percentage >= 90:
        return ("Complete", "success", percentage)
    elif percentage >= 70:
        return ("Good", "primary", percentage)
    elif percentage >= 50:
        return ("Partial", "warning", percentage)
    else:
        return ("Limited", "secondary", percentage)


def _format_badges(items: Sequence[tuple[str, str]], outline: bool = False) -> str:
    """Format badge items as RST badge directives.

    Parameters
    ----------
    items : Sequence[tuple[str, str]]
        List of (label, value) tuples for badges.
    outline : bool
        If True, use outline badge style for cleaner look.

    """
    # Map labels to badge colors
    color_map = {
        "Modality": "primary",
        "Tasks": "info",
        "Subjects": "secondary",
        "Recordings": "secondary",
        "License": "success",
        "Source": "warning",
        "Citations": "info",
    }
    badges = []
    for label, value in items:
        color = color_map.get(label, "light")
        val_text = _value_or_unknown(value)
        # Use outline style for cleaner, less overwhelming look
        style = f"{color}-line" if outline else color
        badges.append(f":bdg-{style}:`{label}: {val_text}`")

    badges_str = " ".join(badges)
    return "\n".join([".. rst-class:: sd-badges", "", badges_str]).rstrip()


def _format_hero_section(context: Mapping[str, object]) -> str:
    title = str(context.get("title", "")).strip()
    source = str(context.get("source", "")).strip() or "OpenNeuro"

    # Build subtitle based on available info
    if title:
        tagline = f"*{title}*"
    else:
        tagline = f"Dataset from {source}."
    tagline = f"{tagline}\n\nAccess recordings and metadata through EEGDash."

    # Format citation
    authors = context.get("authors") or []
    authors_text = (
        ", ".join(a.replace("*", r"\*") for a in authors) if authors else "Unknown"
    )
    year = _value_or_unknown(str(context.get("year", "")).strip())
    doi = str(context.get("doi", "")).strip()
    doi_clean = _normalize_doi(doi)
    if doi_clean:
        doi_link = f"`{doi_clean} <https://doi.org/{doi_clean}>`__"
    else:
        doi_link = ""

    citation_block = f"**Citation:** {authors_text} ({year}). *{title}*. {doi_link}"

    # Editorial Brief — drop the legacy sphinx-design `sd-badges` row +
    # the standalone "Metadata: …" quality badge. Their data is already
    # surfaced both as editorial pills (rendered below) and as rows in
    # the field-card aside. Keeping all three would triplicate the same
    # facts on every dataset page. Structured-data consumers (Google,
    # llms.txt, Croissant) read the JSON-LD <script> tag, not these
    # visual badges, so removing them is SEO-neutral.
    hero_block = f"{tagline}\n\n{citation_block}"

    # Editorial Brief — append the deck + byline directly under the
    # citation block. The kicker / issue strip rides separately above
    # the H1 via {editorial_kicker_section} in the page template.
    extras = _format_editorial_hero_extras(context)
    if extras:
        hero_block = f"{hero_block}\n\n{extras}"

    return hero_block


def _stat_line(
    label: str, value: object, suffix: str = "", field_type: str = "general"
) -> str:
    """Format a statistic line with label and value."""
    text = _value_or_unknown(_clean_value(value), field_type)
    if (
        text not in ("—", "Varies", "Not specified", "Not calculated", "See source")
        and suffix
    ):
        text = f"{text}{suffix}"
    return f"{label}: {text}"


def _format_highlights_section(context: Mapping[str, object]) -> str:
    openneuro_url = str(context.get("openneuro_url", ""))
    nemar_url = str(context.get("nemar_url", ""))
    dataset_id = str(context.get("dataset_id", ""))

    cards = [
        (
            "Subjects & recordings",
            "highlight-primary",
            [
                _stat_line(
                    "Subjects", context.get("n_subjects"), field_type="subjects"
                ),
                _stat_line(
                    "Recordings", context.get("n_records"), field_type="recordings"
                ),
                _stat_line("Tasks", context.get("n_tasks"), field_type="tasks"),
            ],
        ),
        (
            "Channels & sampling rate",
            "highlight-secondary",
            [
                _stat_line(
                    "Channels", context.get("n_channels"), field_type="n_channels"
                ),
                _stat_line(
                    "Sampling rate (Hz)",
                    context.get("sampling_freqs"),
                    field_type="sampling_rate",
                ),
                _stat_line(
                    "Duration (hours)",
                    context.get("duration_hours_total"),
                    field_type="duration",
                ),
            ],
        ),
        (
            "Tags",
            "highlight-tertiary",
            [
                _stat_line(
                    "Pathology", context.get("pathology"), field_type="pathology"
                ),
                _stat_line("Modality", context.get("tag_modality")),
                _stat_line("Type", context.get("tag_type")),
            ],
        ),
        (
            "Files & format",
            "",
            [
                _stat_line("Size on disk", context.get("size")),
                _stat_line("File count", context.get("s3_item_count")),
                _stat_line("Format", context.get("format")),
            ],
        ),
        (
            "License & citation",
            "",
            [
                _stat_line("License", context.get("license"), field_type="license"),
                _stat_line("DOI", context.get("doi")),
            ],
        ),
        (
            "Provenance",
            "",
            [
                _stat_line("Source", context.get("source")),
                f"OpenNeuro: `{dataset_id} <{openneuro_url}>`__",
                f"NeMAR: `{dataset_id} <{nemar_url}>`__",
            ],
        ),
    ]

    lines = [".. grid:: 1 2 3 3", "   :gutter: 2", ""]
    for title, css_class, items in cards:
        lines.append(f"   .. grid-item-card:: {title}")
        if css_class:
            lines.append(f"      :class-card: sd-border-1 {css_class}")
        else:
            lines.append("      :class-card: sd-border-1")
        lines.append("")
        for item in items:
            lines.append(f"      - {item}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _format_quickstart_section(context: Mapping[str, object]) -> str:
    class_name = str(context.get("class_name", ""))
    dataset_id = str(context.get("dataset_id", ""))
    title = str(context.get("title", ""))
    doi = _clean_value(context.get("doi"))
    doi_clean = _normalize_doi(doi)
    authors = context.get("authors") or []

    # Build BibTeX citation
    bibtex_key = dataset_id.replace("-", "_")
    bibtex_lines = [f"@dataset{{{bibtex_key},"]
    if title:
        bibtex_lines.append(f"  title = {{{title}}},")
    if authors:
        bibtex_lines.append(f"  author = {{{' and '.join(authors)}}},")
    if doi_clean:
        bibtex_lines.append(f"  doi = {{{doi_clean}}},")
        bibtex_lines.append(f"  url = {{https://doi.org/{doi_clean}}},")
    bibtex_lines.append("}")
    bibtex_block = "\n".join(f"         {line}" for line in bibtex_lines)

    return (
        ".. tab-set::\n\n"
        "   .. tab-item:: Get Started\n"
        "      :sync: start\n\n"
        "      **Install**\n\n"
        "      .. code-block:: bash\n\n"
        "         pip install eegdash\n\n"
        "      **Access the data**\n\n"
        "      .. code-block:: python\n\n"
        f"         from eegdash.dataset import {class_name}\n\n"
        f'         dataset = {class_name}(cache_dir="./data")\n'
        "         # Get the raw object of the first recording\n"
        "         raw = dataset.datasets[0].raw\n"
        "         print(raw.info)\n\n"
        "   .. tab-item:: Query & Filter\n"
        "      :sync: query\n\n"
        "      **Filter by subject**\n\n"
        "      .. code-block:: python\n\n"
        f'         dataset = {class_name}(cache_dir="./data", subject="01")\n\n'
        "      **Advanced query**\n\n"
        "      .. code-block:: python\n\n"
        f"         dataset = {class_name}(\n"
        '             cache_dir="./data",\n'
        '             query={"subject": {"$in": ["01", "02"]}},\n'
        "         )\n\n"
        "      **Iterate recordings**\n\n"
        "      .. code-block:: python\n\n"
        "         for rec in dataset:\n"
        "             print(rec.subject, rec.raw.info['sfreq'])\n\n"
        "   .. tab-item:: Cite This Dataset\n"
        "      :sync: cite\n\n"
        "      If you use this dataset in your research, please cite the original authors.\n\n"
        "      **BibTeX**\n\n"
        "      .. code-block:: bibtex\n\n"
        f"{bibtex_block}\n"
    )


def _format_dataset_info_section(context: Mapping[str, object]) -> str:
    dataset_id = str(context.get("dataset_id", ""))
    dataset_upper = str(context.get("dataset_upper", ""))
    title = _value_or_unknown(_clean_value(context.get("title")))
    authors = context.get("authors") or []
    # Escape asterisks for RST (they would otherwise be interpreted as emphasis)
    authors_text = (
        ", ".join(a.replace("*", r"\*") for a in authors) if authors else "Unknown"
    )
    license_text = _value_or_unknown(_clean_value(context.get("license")))
    doi = _clean_value(context.get("doi"))
    doi_clean = _normalize_doi(doi)
    doi_text = f"`{doi} <https://doi.org/{doi_clean}>`__" if doi_clean else "Unknown"
    openneuro_url = str(context.get("openneuro_url", ""))
    nemar_url = str(context.get("nemar_url", ""))
    source_url = _clean_value(context.get("source_url"))

    source_links = [
        f"`OpenNeuro <{openneuro_url}>`__",
        f"`NeMAR <{nemar_url}>`__",
    ]
    if source_url:
        source_links.append(f"`Source URL <{source_url}>`__")

    year = _value_or_unknown(_clean_value(context.get("year")))

    author_year_cell, canonical_cell, importable_cell = _render_identity_cells(
        dataset_upper=dataset_upper,
        author_year_name=_clean_value(context.get("author_year_name")),
        canonical_names=context.get("canonical_names") or [],
    )

    rows = [
        ("Dataset ID", f"``{dataset_upper}``"),
        ("Title", title),
        ("Author (year)", author_year_cell),
        ("Canonical", canonical_cell),
        ("Importable as", importable_cell),
        ("Year", year),
        ("Authors", authors_text),
        ("License", license_text),
        ("Citation / DOI", doi_text),
        ("Source links", " | ".join(source_links)),
    ]

    # Editorial Brief — wrap the list-table in a `eegdash-ed-fieldcard`
    # container so the editorial CSS can re-skin it as a field card.
    # `.. container::` is the standard sphinx way to add a wrapper class
    # to its contents without changing the data model.
    lines = [
        ".. container:: eegdash-ed-fieldcard",
        "",
        "   .. list-table::",
        "      :widths: 25 75",
        "      :header-rows: 0",
        "",
    ]
    for label, value in rows:
        lines.append(f"      * - {label}")
        lines.append(f"        - {value}")

    bibtex_dropdown = _format_bibtex_dropdown(dataset_id, context)
    if bibtex_dropdown:
        lines.append("")
        lines.append(bibtex_dropdown)

    return "\n".join(lines).rstrip()


def _render_identity_cells(
    *,
    dataset_upper: str,
    author_year_name: str,
    canonical_names: object,
) -> tuple[str, str, str]:
    """Build the three identity cells (Author year / Canonical / Importable as).

    Falls back to an em-dash for empty columns so the table stays
    regular. The ``author_year_name`` is stripped from the Canonical
    list so the same token isn't shown on two rows.
    """
    em_dash = "—"
    author_year_cell = f"``{author_year_name}``" if author_year_name else em_dash

    names_list = canonical_names if isinstance(canonical_names, (list, tuple)) else []
    canonical_display = [n for n in names_list if n and n != author_year_name]
    canonical_cell = (
        ", ".join(f"``{n}``" for n in canonical_display)
        if canonical_display
        else em_dash
    )

    importable = [dataset_upper] if dataset_upper else []
    if author_year_name and author_year_name not in importable:
        importable.append(author_year_name)
    for n in names_list:
        if n and n not in importable:
            importable.append(n)
    importable_cell = (
        ", ".join(f"``{n}``" for n in importable) if importable else em_dash
    )

    return author_year_cell, canonical_cell, importable_cell


def _format_bibtex_dropdown(dataset_id: str, context: Mapping[str, object]) -> str:
    doi = _clean_value(context.get("doi"))
    doi_clean = _normalize_doi(doi)
    if not doi_clean:
        return ""

    key = dataset_id.replace("-", "_")
    bibtex_lines = [f"@dataset{{{key},"]
    title = _clean_value(context.get("title"))
    if title:
        bibtex_lines.append(f"  title = {{{title}}},")
    authors = context.get("authors") or []
    if authors:
        bibtex_lines.append(f"  author = {{{' and '.join(authors)}}},")
    bibtex_lines.append(f"  doi = {{{doi_clean}}},")
    bibtex_lines.append(f"  url = {{https://doi.org/{doi_clean}}},")
    bibtex_lines.append("}")

    dropdown_lines = [
        ".. dropdown:: Copy-paste BibTeX",
        "   :class-container: sd-shadow-sm",
        "   :class-title: sd-bg-light",
        "",
        "   .. code-block:: bibtex",
        "",
    ]
    dropdown_lines.extend([f"      {line}" for line in bibtex_lines])
    return "\n".join(dropdown_lines)


def _is_decorative_line(s: str) -> bool:
    """Check if line is purely decorative (em-dashes, dashes, equals, etc.)."""
    s = s.strip()
    if len(s) < 3:
        return False
    # Check for lines made of repeated chars: —, -, =, _, *, #
    return bool(re.match(r"^[—\-=_*#~]+$", s)) and len(set(s)) <= 2


def _convert_readme_to_rst(text: str) -> str:
    """Convert README content to RST (headers become bold, not section headers).

    Handles markdown (#) headers, RST-style underline headers, and decorative
    box-style headers (em-dash lines) to avoid messing up the document structure.

    Also handles various Markdown constructs that conflict with RST syntax:
    - Markdown tables (wrapped in code blocks)
    - Directory trees (wrapped in code blocks)
    - Code fences (converted to RST code-block directives)
    - Reference-style links [text][1] with [1]: url
    - Trailing underscores (escaped to avoid hyperlink target errors)
    - Pipe characters (escaped to avoid substitution reference errors)
    - Orphan asterisks in file patterns (escaped)
    - Markdown checkboxes (converted to simple list items)
    """
    # Remove BOM if present
    text = text.lstrip("\ufeff")
    # Normalize HTML line breaks to newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Downgrade literal HTML headers so they don't collide with the
    # page's own `<h1>`. Upstream dataset READMEs (e.g. DS004100)
    # sometimes contain raw `<h1>HUP iEEG dataset</h1>` which Sphinx
    # passes through verbatim and Ahrefs then flags as "Multiple H1
    # tags". `<h2>` is the highest level that can safely sit inside the
    # "About this dataset" section which is already H2.
    text = re.sub(
        r"<(/?)h1(\b[^>]*)>",
        r"<\g<1>h3\g<2>>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"<(/?)h2(\b[^>]*)>",
        r"<\g<1>h4\g<2>>",
        text,
        flags=re.IGNORECASE,
    )
    # Replace leading tabs with spaces (tabs cause RST block-quote interpretation)
    text = re.sub(r"^\t+", lambda m: "  " * len(m.group(0)), text, flags=re.MULTILINE)

    # === Phase 1: Extract reference-style link definitions ===
    # Pattern: [1]: http://... or [name]: http://...
    ref_link_defs: dict[str, str] = {}
    ref_def_pattern = re.compile(r"^\s*\[([^\]]+)\]:\s*(.+?)\s*$", re.MULTILINE)
    for match in ref_def_pattern.finditer(text):
        ref_link_defs[match.group(1)] = match.group(2)
    # Remove reference definitions from text
    text = ref_def_pattern.sub("", text)

    def _sanitize_header_text(title: str) -> str:
        """Strip inline markdown that breaks bold headers in RST."""
        title = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", title)
        title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)
        title = re.sub(r"\[([^\]]+)\]\[([^\]]+)\]", r"\1", title)
        title = re.sub(r"`([^`]+)`", r"\1", title)
        title = re.sub(r"\*\*([^*]+)\*\*", r"\1", title)
        title = re.sub(r"\*([^*]+)\*", r"\1", title)
        title = re.sub(r"__([^_]+)__", r"\1", title)
        title = re.sub(r"_([^_]+)_", r"\1", title)
        title = re.sub(r"<(https?://[^>]+)>", r"\1", title)
        title = re.sub(r"\s+", " ", title).strip()
        # Escape characters with special meaning in RST
        title = re.sub(r"(\w)_(?=[\s.,;:!?\)\]\}]|$)", r"\1\\_", title)
        title = title.replace("|", "\\|")
        return title

    # === Phase 2: Convert markdown code fences to RST code blocks ===
    def convert_code_fence(match: re.Match) -> str:
        lang = match.group(1) or "text"
        code = match.group(2)
        # Indent the code content
        indented_code = "\n".join("   " + line for line in code.split("\n"))
        return f"\n.. code-block:: {lang}\n\n{indented_code}\n"

    # Match ```lang\n...\n``` (multiline) - with or without newline after opening
    code_fence_pattern = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)
    text = code_fence_pattern.sub(convert_code_fence, text)

    # Clean up any remaining orphan triple backticks (convert to RST literal)
    text = re.sub(r"``\s*`", "``", text)  # ``\` -> ``
    text = re.sub(r"`\s*``", "``", text)  # `\`` -> ``

    lines = text.split("\n")
    result: list[str] = []
    i = 0
    in_code_block = False  # Track if we're inside a code block directive

    # === Phase 3: Detect and wrap markdown tables ===
    def is_table_line(line: str) -> bool:
        """Check if line looks like a markdown table row."""
        stripped = line.strip()
        return (
            stripped.startswith("|")
            and stripped.endswith("|")
            and "|" in stripped[1:-1]
        )

    def is_table_separator(line: str) -> bool:
        """Check if line is a markdown table separator (|---|---|)."""
        stripped = line.strip()
        return bool(re.match(r"^\|[-:\s|]+\|$", stripped))

    def is_tree_line(line: str) -> bool:
        """Check if line is part of a directory tree."""
        return bool(re.match(r"^\s*[\|│├└][\s─\-\|│├└]*", line))

    # Pre-scan for table and tree regions
    table_regions: set[int] = set()
    tree_regions: set[int] = set()

    # Find markdown tables
    j = 0
    while j < len(lines):
        if is_table_line(lines[j]) or is_table_separator(lines[j]):
            # Found start of potential table
            while j < len(lines) and (
                is_table_line(lines[j])
                or is_table_separator(lines[j])
                or lines[j].strip() == ""
            ):
                if lines[j].strip():  # Skip empty lines from region
                    table_regions.add(j)
                j += 1
        else:
            j += 1

    # Find directory trees
    j = 0
    while j < len(lines):
        if is_tree_line(lines[j]):
            while j < len(lines) and (is_tree_line(lines[j]) or lines[j].strip() == ""):
                if lines[j].strip():
                    tree_regions.add(j)
                j += 1
        else:
            j += 1

    # === Phase 4: Process lines ===
    table_block: list[str] = []
    tree_block: list[str] = []
    in_table = False
    in_tree = False
    in_blockquote = False
    prev_was_list_item = False

    while i < len(lines):
        line = lines[i]

        # Track if we're in an RST code block (indented content after ::)
        if line.strip().endswith("::") or line.strip().startswith(".. code-block::"):
            in_code_block = True
        elif (
            in_code_block
            and line.strip()
            and not line.startswith(" ")
            and not line.startswith("\t")
        ):
            in_code_block = False

        is_blockquote_line = False
        if not in_code_block:
            blockquote_match = re.match(r"^\s*>\s?", line)
            if blockquote_match:
                is_blockquote_line = True
                line = line[blockquote_match.end() :]
            elif in_blockquote:
                if result and result[-1].strip():
                    result.append("")
                in_blockquote = False

        # Handle table regions
        if i in table_regions and not in_tree and not is_blockquote_line:
            if not in_table:
                # Start table block
                in_table = True
                table_block = []
                # Add blank line before code block
                if result and result[-1].strip():
                    result.append("")
                result.append(".. code-block:: text")
                result.append("")
            table_block.append("   " + line)
            result.append("   " + line)
            i += 1
            continue
        elif in_table:
            # End of table
            in_table = False
            result.append("")

        # Handle tree regions
        if i in tree_regions and not in_table and not is_blockquote_line:
            if not in_tree:
                # Start tree block
                in_tree = True
                tree_block = []
                if result and result[-1].strip():
                    result.append("")
                result.append(".. code-block:: text")
                result.append("")
            tree_block.append("   " + line)
            result.append("   " + line)
            i += 1
            continue
        elif in_tree:
            # End of tree
            in_tree = False
            result.append("")

        # Skip purely decorative lines (em-dashes, repeated dashes, etc.)
        if _is_decorative_line(line):
            # Check if this is a box-style header: decorative -> TITLE -> decorative
            if i + 2 < len(lines):
                potential_title = lines[i + 1].strip()
                next_decorative = lines[i + 2]
                if potential_title and _is_decorative_line(next_decorative):
                    # This is a box-style header
                    result.append("")
                    result.append(f"**{_sanitize_header_text(potential_title)}**")
                    result.append("")
                    i += 3  # Skip decorative, title, decorative
                    prev_was_list_item = False
                    continue
            # Just a decorative line by itself - skip it
            i += 1
            continue

        # Check for markdown headers (# Title)
        if not is_blockquote_line:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if header_match:
                title = _sanitize_header_text(header_match.group(2).strip())
                result.append("")
                result.append(f"**{title}**")
                result.append("")
                i += 1
                prev_was_list_item = False
                continue

        # Check for RST-style underline headers (Title followed by ==== or ----)
        # Only match if title is reasonably short (< 80 chars) to avoid matching paragraphs
        if not is_blockquote_line and i + 1 < len(lines) and len(line.strip()) < 80:
            next_line = lines[i + 1]
            # Check if next line is an underline (all same char, at least 3 chars)
            underline_match = re.match(r"^([=\-~^\"\'`—]+)$", next_line.strip())
            if (
                underline_match
                and len(next_line.strip()) >= 3
                and line.strip()
                and len(set(next_line.strip())) == 1
            ):
                title = _sanitize_header_text(line.strip())
                result.append("")
                result.append(f"**{title}**")
                result.append("")
                i += 2  # Skip both the title and underline
                prev_was_list_item = False
                continue

        # === Line-level transformations (skip if in code block) ===
        if not in_code_block:
            # Convert Unicode bullets to RST-compatible list items
            # • (U+2022 bullet), ⁃ (U+2043 hyphen bullet), ‣ (U+2023 triangular bullet)
            line = re.sub(r"^(\s*)[•⁃‣]\s*", r"\1- ", line)

            # Convert markdown checkboxes: - [ ] item -> - item
            line = re.sub(r"^(\s*)-\s*\[([ xX]?)\]\s*", r"\1- ", line)

            inline_code_spans: list[str] = []

            def stash_inline_code(m: re.Match) -> str:
                inline_code_spans.append(m.group(1))
                return f"\x00INLINE_CODE_{len(inline_code_spans) - 1}\x00"

            line = re.sub(r"(?<![\\`])`([^`]+)`(?!`)", stash_inline_code, line)

            # Convert linked images (badges): [![alt](img)](link) -> `alt <link>`__
            # Must run BEFORE individual image/link handlers to avoid mangling.
            line = re.sub(
                r"\[!\[([^\]]*)\]\([^)]+\)\]\(([^)]+)\)",
                lambda m: f"`{m.group(1).strip() or m.group(2).strip()} <{m.group(2).strip()}>`__",
                line,
            )

            # Convert markdown images: ![alt](url) -> `alt <url>`__
            def replace_md_image(m: re.Match) -> str:
                alt = m.group(1).strip()
                url = m.group(2).strip()
                text = alt or url
                return f"`{text} <{url}>`__"

            line = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_md_image, line)

            # Convert reference-style images: ![alt][ref]
            def replace_ref_image(m: re.Match) -> str:
                alt = m.group(1).strip()
                ref = m.group(2).strip()
                url = ref_link_defs.get(ref, "")
                if not url:
                    return m.group(0)
                text = alt or url
                return f"`{text} <{url}>`__"

            line = re.sub(r"!\[([^\]]*)\]\[([^\]]+)\]", replace_ref_image, line)

            # Stash already-converted RST links so the autolink handler
            # doesn't mangle the <url> inside them (e.g. `DOI <url>`__).
            rst_link_stash: list[str] = []

            def _stash_rst_link(m: re.Match) -> str:
                rst_link_stash.append(m.group(0))
                return f"\x00RST_LINK_{len(rst_link_stash) - 1}\x00"

            line = re.sub(r"`[^`]+\s<[^>]+>`__", _stash_rst_link, line)

            # Convert autolinks: <https://example.com>
            line = re.sub(r"<(https?://[^>]+)>", r"`\1 <\1>`__", line)

            # Restore stashed RST links
            for idx, stashed in enumerate(rst_link_stash):
                line = line.replace(f"\x00RST_LINK_{idx}\x00", stashed)

            # Convert reference-style links: [text][ref] -> `text <url>`__
            def replace_ref_link(m: re.Match) -> str:
                text = m.group(1)
                ref = m.group(2)
                url = ref_link_defs.get(ref, "")
                if url:
                    return f"`{text} <{url}>`__"
                return m.group(0)  # Keep original if ref not found

            line = re.sub(r"\[([^\]]+)\]\[([^\]]+)\]", replace_ref_link, line)

            # Convert markdown links [text](url) -> `text <url>`__
            line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"`\1 <\2>`__", line)

            # Escape trailing underscores on words (e.g., auricular_points_)
            # Uses [a-zA-Z0-9] instead of \w to avoid matching __ (RST link suffix).
            line = re.sub(r"([a-zA-Z0-9])_(?=[\s.,;:!?\)\]\}]|$)", r"\1\_", line)

            # Escape pipe characters that could be interpreted as substitution references
            # Don't escape pipes that are already in table-like structures we've processed
            # Escape |word| patterns and lone | characters
            line = re.sub(r"\|([^\|]+)\|", r"\\|\1\\|", line)
            # Also escape standalone | at start of line (except in tables we already handled)
            if (
                line.strip().startswith("|")
                and i not in table_regions
                and i not in tree_regions
            ):
                line = line.replace("|", "\\|")

            # === Handle escaped asterisks and malformed bold/emphasis ===

            # Fix \**text:** patterns (escaped asterisk before single-star emphasis)
            # \**Active:** -> *Active:* (convert to proper single emphasis)
            line = re.sub(r"\\?\*\*([^*:]+:)\*\*?(?=\s|$)", r"*\1*", line)

            # Fix patterns like \**MATLAB R2020b**using (escaped * before bold, no space after)
            # Convert to proper bold with space: **MATLAB R2020b** using
            line = re.sub(r"\\?\*(\*\*[^*]+\*\*)(?=[a-zA-Z])", r"\1 ", line)

            # Fix **text* \* patterns (bold opened with ** but closed with * \*)
            # Convert to proper bold: **text**
            line = re.sub(r"(\*\*[^*]+)\*\s*\\\*", r"\1**", line)

            # Normalize already-escaped asterisks: \* -> placeholder
            line = line.replace("\\*", "\x00ESCAPED_ASTERISK\x00")

            # Escape asterisks in file patterns and paths
            # Pattern: asterisk in file patterns (*.eeg, *_events.tsv, sub-*/eeg/)
            # Asterisk followed by dot, underscore, or /
            line = re.sub(r"\*(?=[._/])", r"\\*", line)
            # Asterisk preceded by / or - (path patterns like sub-* or /*)
            line = re.sub(r"(?<=[/\-])\*", r"\\*", line)

            # Count unescaped asterisks - if odd number, some are orphaned
            asterisk_count = line.count("*") - line.count("\\*")
            if asterisk_count % 2 == 1:
                # Escape lone asterisks not forming valid emphasis pairs
                # This catches asterisks at start of word not followed by closing asterisk
                line = re.sub(r"(?<=\s)\*(?=\S)", r"\\*", line)

            # Restore pre-escaped asterisks
            line = line.replace("\x00ESCAPED_ASTERISK\x00", "\\*")

            # Fix bold/emphasis that's immediately followed by text without space
            # **Protocol:**Data -> **Protocol:** Data
            line = re.sub(r"(\*\*[^*]+\*\*)(?=[a-zA-Z])", r"\1 ", line)

            # Fix text immediately before bold without space (only for double asterisks)
            # word**text** -> word **text**
            line = re.sub(r"([a-zA-Z])(\*\*[^*]+\*\*)", r"\1 \2", line)

            # Fix spaces inside bold/emphasis markers (RST doesn't allow this)
            # ** text** -> **text**
            line = re.sub(r"\*\*\s+([^*]+)\*\*", r"**\1**", line)
            # Also fix trailing spaces: **text ** -> **text**
            line = re.sub(r"\*\*([^*]+)\s+\*\*", r"**\1**", line)
            # Fix single emphasis with spaces: * text* -> *text*
            line = re.sub(r"\*\s+([^*]+)\*(?!\*)", r"*\1*", line)
            line = re.sub(r"\*([^*]+)\s+\*(?!\*)", r"*\1*", line)

            # Fix malformed emphasis patterns like **[text]* \* or * text*
            # These need to be either properly closed or escaped
            line = re.sub(r"\*\*\[([^\]]+)\]\*\s*\\?\*", r"**[\1]**", line)

            # Fix patterns where bold ends with \** (escaped closing)
            # **text\** -> **text** (the \* was meant to end the bold)
            line = re.sub(r"(\*\*[^*]+)\\(\*\*)$", r"\1\2", line)

            # Fix \**text** patterns at start - escaped asterisk before bold
            # \**text** should just be **text** (ignore leading escaped *)
            line = re.sub(r"(?<=\s)\\?\*(\*\*[^*]+\*\*)", r"\1", line)
            line = re.sub(r"^\\?\*(\*\*[^*]+\*\*)", r"\1", line)

            # Fix lone * markers that look like attempts at emphasis but have spaces
            # * real* -> *real* (if it looks like intended emphasis)
            line = re.sub(r"\*\s+(\w+)\*(?!\*)", r"*\1*", line)

            # Fix emphasis ending with trailing escaped asterisk: *text* \* -> *text*
            line = re.sub(r"(\*[^*]+\*)\s*\\\*$", r"\1", line)

            # Fix **text* \* patterns (bold opened, single * close with escaped *)
            # Convert to proper bold: **text**
            line = re.sub(r"(\*\*[^*]+)\*\s*\\\*$", r"\1**", line)

            if inline_code_spans:

                def restore_inline_code(m: re.Match) -> str:
                    code = inline_code_spans[int(m.group(1))]
                    return f"``{code}``"

                line = re.sub(r"\x00INLINE_CODE_(\d+)\x00", restore_inline_code, line)

            # Remove orphan triple backticks that weren't caught by code fence conversion
            # These appear as ``\` or `\`` after partial processing
            line = re.sub(r"``\\?`", "``", line)
            line = re.sub(r"`\\?``", "``", line)
            # Also remove standalone triple backticks
            line = re.sub(r"^```\s*$", "", line)

            # Escape orphan backticks (unbalanced inline code)
            backtick_count = line.count("`") - line.count("\\`") - 2 * line.count("``")
            if backtick_count % 2 == 1:
                # Find and escape lone backticks not forming pairs
                # Simple approach: escape backticks not followed by closing backtick
                parts = []
                in_backtick = False
                for j, char in enumerate(line):
                    if char == "`" and (j == 0 or line[j - 1] != "\\"):
                        if in_backtick:
                            in_backtick = False
                            parts.append(char)
                        else:
                            # Check if there's a closing backtick
                            remaining = line[j + 1 :]
                            if "`" in remaining and not remaining.startswith(" "):
                                in_backtick = True
                                parts.append(char)
                            else:
                                parts.append("\\`")
                    else:
                        parts.append(char)
                line = "".join(parts)

        # Check if current line is a list item
        is_list_item = bool(
            re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+[.)]\s+", line)
        )

        if is_blockquote_line and not in_blockquote:
            if result and result[-1].strip():
                result.append("")
            in_blockquote = True

        # Ensure blank line before text that follows a list item
        if (
            prev_was_list_item
            and not is_list_item
            and line.strip()
            and not line.startswith(" ")
        ):
            # Insert blank line if missing
            if result and result[-1].strip():
                result.append("")

        if is_blockquote_line:
            if line.strip():
                line = "   " + line
            else:
                line = "   "

        result.append(line)
        prev_was_list_item = is_list_item
        i += 1

    # Post-processing: insert blank lines before unindent transitions to avoid
    # "Block quote ends without a blank line" warnings in RST.
    final: list[str] = []
    for idx, line in enumerate(result):
        if idx > 0 and final:
            prev = final[-1]
            prev_indent = len(prev) - len(prev.lstrip())
            cur_indent = len(line) - len(line.lstrip()) if line.strip() else -1
            # Unindent transition: previous line was indented, current is less indented
            if (
                cur_indent >= 0
                and prev.strip()
                and prev_indent > cur_indent
                and final[-1] != ""
            ):
                final.append("")
        final.append(line)

    text = "\n".join(final)

    # --- Final sanitization pass ---

    # Convert RST citation directives to plain text to avoid cross-page
    # duplicate citation warnings (e.g. .. [Hinss2021] -> [Hinss2021])
    text = re.sub(r"^\.\.\s+(\[[^\]]+\])", r"\1", text, flags=re.MULTILINE)

    # Fix malformed RST links: ``<url>``_ -> `url <url>`__
    text = re.sub(
        r"``<(https?://[^>]+)>``_",
        r"`\1 <\1>`__",
        text,
    )

    # Escape orphan backticks that would cause "Inline literal start-string
    # without end-string" warnings in RST.
    lines = text.split("\n")
    for li, line in enumerate(lines):
        # Skip lines that are RST directives or already have proper markup
        stripped = line.lstrip()
        if stripped.startswith("..") or stripped.startswith(":"):
            continue

        # --- Orphan backtick escaping ---
        # Remove properly paired backtick patterns first to check
        cleaned = re.sub(r"``[^`]*``", "", line)  # remove ``code``
        cleaned = re.sub(r"`[^`]+ <[^>]+>`__", "", cleaned)  # remove `link`__
        cleaned = re.sub(r"`[^`]+`", "", cleaned)  # remove `ref`
        if cleaned.count("`") % 2 == 1:
            lines[li] = re.sub(r"(?<!`)`(?!`)", r"\`", line)
            line = lines[li]

        # --- Orphan bold/emphasis escaping ---
        # Remove properly paired **bold** and *emphasis* to check for orphans
        cleaned = re.sub(r"\*\*[^*]+\*\*", "", line)  # remove **bold**
        cleaned = re.sub(r"(?<!\*)\*(?!\*)[^*]+\*(?!\*)", "", cleaned)  # remove *em*
        cleaned = re.sub(r"\\\*", "", cleaned)  # remove already-escaped \*
        remaining_stars = cleaned.count("*")
        if remaining_stars > 0:
            # Escape orphan ** or * that aren't already escaped
            # First handle \*...*\* patterns (partially escaped)
            line = re.sub(r"\\\*([^*]+)\*(?!\*)", r"\\\*\1\\*", line)
            # Escape any remaining unmatched * that aren't part of proper pairs
            # by escaping * that are adjacent to non-space on only one side
            line = re.sub(r"(?<![\\*\s])\*(?=\s|$)", r"\\*", line)
            line = re.sub(r"(?:^|(?<=\s))\*(?![*\s])", r"\\*", line)
            lines[li] = line

    return "\n".join(lines)


def _format_readme_section(context: Mapping[str, object]) -> str:
    """Format the README content for RST display.

    The first two prose paragraphs are wrapped in an ``eegdash-ed-lede``
    container so the editorial CSS can render the two-column drop-cap
    intro from the v1-editorial-v2 design.
    """
    readme = _clean_value(context.get("readme"))

    if not readme:
        return "No README content is available for this dataset."

    # Convert README content to RST (headers become bold)
    content = _convert_readme_to_rst(readme)
    lines = content.split("\n")

    # Pull the first two non-empty paragraphs to render under the dropcap.
    lede_paragraphs, remainder_lines = _split_lede_paragraphs(lines, max_paragraphs=2)
    lede_block = ""
    if lede_paragraphs:
        # Wrap each paragraph in a `.. container::` so docutils renders
        # <p> elements, then put the lot inside the dropcap container.
        inner = []
        for para in lede_paragraphs:
            stripped = para.strip()
            if stripped:
                inner.append(stripped)
        if inner:
            paras_rst = "\n\n".join(f"   {p}" for p in inner)
            lede_block = f".. container:: eegdash-ed-lede\n\n{paras_rst}\n\n"

    # Reassemble the remainder.
    remainder = "\n".join(remainder_lines).strip("\n")

    if remainder:
        # For long READMEs (>30 lines remaining), wrap in dropdown
        rem_lines = remainder.split("\n")
        if len(rem_lines) > 30:
            preview = "\n".join(rem_lines[:10])
            indented = "\n".join(f"   {line}" for line in rem_lines)
            remainder = (
                f"{preview}\n\n"
                ".. dropdown:: View full README\n"
                "   :class-container: sd-shadow-sm\n"
                f"\n{indented}\n"
            )

    return f"{lede_block}\n{remainder}".strip()


def _split_lede_paragraphs(
    lines: Sequence[str], max_paragraphs: int = 2
) -> tuple[list[str], list[str]]:
    """Return (lede_paragraphs, remainder_lines).

    A paragraph is a run of consecutive non-blank lines that doesn't start
    with an RST directive marker (``..``), heading underline, or list
    bullet — i.e. ordinary prose. We collect at most ``max_paragraphs``
    such paragraphs and hand the rest back as ``remainder_lines``.
    """
    paragraphs: list[str] = []
    remainder_start = 0
    i = 0
    n = len(lines)

    def _is_prose(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        # Skip RST directives, transitions, code/quote markers, list items.
        if stripped.startswith(("..", "::", "#", "+", "|", ">")):
            return False
        if stripped.startswith(("- ", "+ ", ":")):  # bullet / field-list label
            return False
        # Skip bold-only "header" lines (e.g. ``**Audio-Visual Attention…**``)
        # so the dropcap lands on the first real prose paragraph instead.
        compact = stripped.rstrip(".:!?")
        if (
            compact.startswith("**")
            and compact.endswith("**")
            and compact.count("**") == 2
        ):
            return False
        if (
            compact.startswith("*")
            and compact.endswith("*")
            and compact.count("*") == 2
        ):
            return False
        # Skip lines that are nothing but RST hyperlinks — README badge rows
        # ("`DOI <...>`__ `Paper DOI <...>`__ `License <...>`__") would
        # otherwise be treated as prose and absorb the drop cap, leaving a
        # giant "D" floated next to a tiny "OI" subscript.
        without_links = _RST_HYPERLINK_RE.sub("", stripped).strip()
        if not without_links:
            return False
        return True

    # Anything before the first prose paragraph (badge rows, bold-only
    # titles, RST directives) is "preamble" we want to keep in the remainder
    # so it still renders below the lede — just without the drop cap.
    preamble_end = 0
    seen_prose = False

    while i < n and len(paragraphs) < max_paragraphs:
        # skip blanks
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        if not _is_prose(lines[i]):
            # Walk past the non-prose run so we can look further down for
            # the first real paragraph.
            while i < n and lines[i].strip() and not _is_prose(lines[i]):
                i += 1
            if not seen_prose:
                preamble_end = i
            continue
        seen_prose = True
        # collect this paragraph
        start = i
        while i < n and lines[i].strip():
            i += 1
        paragraphs.append("\n".join(lines[start:i]).strip())
        remainder_start = i

    if paragraphs:
        # Keep any preamble (badges/titles before the first prose) in front
        # of the remainder so the surrounding chrome still renders.
        remainder = list(lines[:preamble_end]) + list(lines[remainder_start:])
    else:
        remainder = list(lines)

    return paragraphs, remainder


def _format_schema_section(context: Mapping[str, object]) -> str:
    lines = [
        "``dataset[i]`` returns an :class:`eegdash.dataset.EEGDashRaw` recording.",
        "Recording-level metadata live in ``dataset.description`` (pandas DataFrame).",
    ]
    section = "\n\n".join(lines)

    fields = context.get("metadata_fields") or []
    if fields:
        dropdown_lines = [
            ".. dropdown:: Metadata fields",
            "   :class-container: sd-shadow-sm",
            "",
            "   Common fields (availability depends on the dataset):",
            "",
        ]
        for name, desc in fields:
            dropdown_lines.append(f"   - ``{name}``: {desc}")
        section = f"{section}\n\n" + "\n".join(dropdown_lines)

    return section


def _format_quality_section(context: Mapping[str, object]) -> str:
    caveats = context.get("caveats") or []
    if caveats:
        return "\n".join(f"- {note}" for note in caveats)
    return "- No dataset-specific caveats are listed in the available metadata."


def _format_api_section(
    class_name: str, context: Mapping[str, object] | None = None
) -> str:
    """Format the API section.

    Two parts:

    1. An editorial **Signature** card (raw HTML) that mirrors the
       v1-editorial-v2 §05 Access design: left-gutter ``SIGNATURE``
       label, a code-styled class signature, and an identifier table
       (Author/year, Canonical, Importable as, Source) backed by real
       registry data.
    2. The Sphinx ``autoclass`` block, restyled by editorial.css so
       the rendered Parameters / Attributes / Methods read as a
       single design-system block.
    """
    # No leading "Use the X class..." paragraph — the signature card
    # below already names the class, its bases, and how to import it.
    autoclass = (
        ".. currentmodule:: eegdash.dataset\n\n"
        f".. autoclass:: eegdash.dataset.{class_name}\n"
        "   :members: __init__, save\n"
        "   :member-order: bysource\n"
    )

    if context is None:
        return autoclass

    # --- Editorial signature card -----------------------------------------
    dataset_upper = class_name.upper()
    author_year_name = _clean_value(context.get("author_year_name"))
    canonical_names = context.get("canonical_names") or []
    if not isinstance(canonical_names, (list, tuple)):
        canonical_names = []

    importable = [dataset_upper]
    if author_year_name and author_year_name not in importable:
        importable.append(author_year_name)
    for n in canonical_names:
        if n and n not in importable:
            importable.append(n)

    importable_html = " · ".join(f"<code>{n}</code>" for n in importable)
    author_year_html = f"<b>{author_year_name}</b>" if author_year_name else "—"
    canonical_display = [n for n in canonical_names if n and n != author_year_name]
    canonical_html = (
        " · ".join(f"<code>{n}</code>" for n in canonical_display)
        if canonical_display
        else "—"
    )

    source_path = "eegdash/dataset/registry.py"
    github_url = f"https://github.com/eegdash/EEGDash/blob/develop/{source_path}"

    signature_card = (
        ".. raw:: html\n\n"
        '   <div class="eegdash-ed-apicard">\n'
        '     <div class="apicard-gutter">'
        '<div class="lbl">Signature</div>'
        '<div class="cls"><code>eegdash.dataset</code></div>'
        "</div>\n"
        '     <div class="apicard-body">\n'
        '       <div class="apicard-sig">\n'
        '         <div class="sig-kind">class</div>\n'
        f'         <div class="sig-line">'
        '<span class="ns">eegdash.dataset.</span>'
        f'<b class="cls-name">{class_name}</b>'
        '<span class="paren">(</span>'
        '<span class="arg">cache_dir</span>, '
        '<span class="arg">query</span>=<span class="lit">None</span>, '
        '<span class="arg">s3_bucket</span>=<span class="lit">None</span>, '
        '<span class="arg">**kwargs</span>'
        '<span class="paren">)</span>'
        "</div>\n"
        '         <div class="sig-base">Bases: <code>EEGDashDataset</code></div>\n'
        "       </div>\n"
        '       <div class="apicard-ids">\n'
        '         <div class="id-row">'
        '<span class="k">Author (year)</span>'
        f'<span class="v">{author_year_html}</span>'
        "</div>\n"
        '         <div class="id-row">'
        '<span class="k">Canonical</span>'
        f'<span class="v">{canonical_html}</span>'
        "</div>\n"
        '         <div class="id-row">'
        '<span class="k">Importable as</span>'
        f'<span class="v">{importable_html}</span>'
        "</div>\n"
        '         <div class="id-row">'
        '<span class="k">Source</span>'
        f'<span class="v"><code>{source_path}</code> · '
        f'<a href="{github_url}">[source ↗]</a></span>'
        "</div>\n"
        "       </div>\n"
        "     </div>\n"
        "   </div>\n"
    )

    return signature_card + "\n" + autoclass


# ---------------------------------------------------------------------------
# Electrode-explorer embed (Step 5 of the electrodes integration plan).
#
# `_static/dataset_generated/electrode-layouts.json` maps dataset_id →
# {label, n_channels, tsv_url, coords_url}. It is eventually populated
# by the eegdash backend montage registry; while that's being built we
# maintain a curated subset here as a fallback.
#
# Each dataset page gets a collapsed <details> block. Expanding it swaps
# the iframe's `data-src` onto `src` (see lazy-embed.js), so zero bytes
# are fetched from electrodes.eegdash.org until a reader opts in.
# ---------------------------------------------------------------------------

_ELECTRODE_EXPLORER_BASE = "https://electrodes.eegdash.org/"

_electrode_layouts_cache: dict[str, object] | None = None


def _load_electrode_layouts() -> Mapping[str, Mapping[str, object]]:
    """Read the curated electrode-layouts manifest (cached across calls).

    Missing file or malformed JSON degrades silently to empty — dataset
    pages then render a 'no scalp layout published' placeholder instead
    of the iframe.
    """
    global _electrode_layouts_cache
    if _electrode_layouts_cache is not None:
        return _electrode_layouts_cache  # type: ignore[return-value]
    path = (
        Path(__file__).parent
        / "_static"
        / "dataset_generated"
        / "electrode-layouts.json"
    )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        layouts = doc.get("layouts", {})
        if not isinstance(layouts, dict):
            layouts = {}
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        LOGGER.info(
            "[electrode-layouts] manifest unavailable (%s); placeholders only", exc
        )
        layouts = {}
    _electrode_layouts_cache = layouts
    return layouts


def _format_electrodes_section(context: Mapping[str, object]) -> str:
    """Render a lazy <details><iframe> block for this dataset's montage.

    If the manifest doesn't have an entry for this dataset_id, we still
    emit the section but with a short note — keeps the page layout
    consistent across the catalog.
    """
    dataset_id = str(context.get("dataset_id") or "").strip().lower()
    if not dataset_id:
        return ""

    layouts = _load_electrode_layouts()
    entry = layouts.get(dataset_id)

    heading = "Electrode Layout\n----------------\n\n"

    if not entry or not (entry.get("montage_id") or entry.get("tsv_url")):
        # Placeholder — keeps the section visible so readers know the
        # catalog intends to show it, just that this particular dataset
        # hasn't been indexed yet. Either shape of URL is acceptable:
        # schema v2 carries ``montage_id`` (the registry hash), schema
        # v1 carried ``tsv_url`` + optional ``coords_url``.
        body = (
            "No scalp electrode layout is currently indexed for this\n"
            "dataset. Once the eegdash montage registry ingests it,\n"
            "the interactive viewer will appear here automatically.\n"
        )
        return heading + body

    # Build the iframe URL. Prefer the registry id shape if present;
    # otherwise fall back to direct tsv/coords URLs so pages work even
    # before the registry endpoint is live.
    label = str(entry.get("label") or "Electrodes").strip()
    n_channels = entry.get("n_channels")
    montage_id = str(entry.get("montage_id") or "").strip()

    if montage_id:
        query = f"montage={montage_id}"
    else:
        from urllib.parse import quote

        tsv_q = quote(str(entry["tsv_url"]), safe="")
        parts = [f"tsv={tsv_q}"]
        coords_url = entry.get("coords_url")
        if coords_url:
            parts.append(f"coords={quote(str(coords_url), safe='')}")
        query = "&".join(parts)

    iframe_src = f"{_ELECTRODE_EXPLORER_BASE}?{query}&embed=1"

    title_bits = [label]
    if n_channels:
        title_bits.append(f"{n_channels} channels")
    summary_text = " — ".join(title_bits)

    # Keep the HTML block compact; Sphinx renders it as-is.
    html = (
        ".. raw:: html\n\n"
        '   <details class="electrode-explorer">\n'
        f"     <summary>Electrode layout — {summary_text}</summary>\n"
        "     <iframe\n"
        f'       data-src="{iframe_src}"\n'
        '       loading="lazy"\n'
        '       width="100%" height="640"\n'
        '       style="border: 1px solid var(--pst-color-border); border-radius: 8px; max-width: 900px; display: block;"\n'
        f'       title="Topomap of {label}"\n'
        '       referrerpolicy="no-referrer">\n'
        "     </iframe>\n"
        "   </details>\n"
    )
    return heading + html


def _make_count_bar_chart(
    entries: list,
    label: str,
    unit: str,
    bar_color: str = "#4472c4",
) -> str:
    """Render a compact vertical bar chart for a count distribution as RST raw HTML."""
    if not entries:
        return ""
    vals = [(e.get("val"), e.get("count")) for e in entries if e.get("val") is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return (
            ".. raw:: html\n\n"
            '   <div class="eegdash-stats-section" style="margin-bottom:1rem;">\n'
            f"     <p><strong>{label}</strong>: {vals[0][0]} {unit}"
            f" (n={vals[0][1]} recordings)</p>\n"
            "   </div>\n\n"
        )
    max_count = max(c for _, c in vals)
    bar_width = 28
    bars_html = ""
    labels_html = ""
    for val, count in sorted(vals, key=lambda x: x[0]):
        pct = int(count / max_count * 100) if max_count else 0
        val_label = str(int(val)) if float(val) == int(float(val)) else f"{val:.1f}"
        bars_html += (
            f'<div style="width:{bar_width}px; height:{pct}%; '
            f'background:{bar_color}; flex-shrink:0;" '
            f'title="{val_label} {unit}: {count}"></div>'
        )
        labels_html += (
            f'<span style="width:{bar_width}px; text-align:center; '
            f'overflow:hidden; white-space:nowrap; font-size:9px;">{val_label}</span>'
        )
    return (
        ".. raw:: html\n\n"
        '   <div class="eegdash-stats-section" style="margin-bottom:1rem;">\n'
        f"     <p><strong>{label}</strong> ({unit})</p>\n"
        '     <div class="eeg-chart-row" style="display:flex; align-items:flex-end; '
        'gap:2px; height:60px;">\n'
        f"       {bars_html}\n"
        "     </div>\n"
        '     <div class="eeg-chart-labels" style="display:flex; gap:2px; font-size:10px;">\n'
        f"       {labels_html}\n"
        "     </div>\n"
        "   </div>\n\n"
    )


# BIDS sex column (objects/columns.yaml) defines three phenotypical categories,
# each with a short and long form, all case-insensitive.
# Keys outside all three sets are treated as unknown and folded into "Other".
_BIDS_FEMALE_KEYS = {"f", "female", "fem", "w", "women", "girl", "1"}
_BIDS_MALE_KEYS = {"m", "male", "man", "men", "boy", "2"}
_BIDS_OTHER_KEYS = {"o", "other"}


def _is_positive_float(value: object) -> bool:
    """Truthy iff ``value`` can be coerced to a finite positive float."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f > 0


def _render_sex_donut(
    f_count: int,
    m_count: int,
    o_count: int,
    total: int,
    *,
    handedness: Mapping[str, object] | None = None,
) -> str:
    """Render the sex distribution as an SVG donut chart + side legend.

    Geometry matches the v1-editorial-v2 design: a 42×42 viewBox circle
    (r=15.9 → circumference ≈ 100, so each segment's stroke-dasharray
    reads as a percentage). Female draws first in EEGdash blue, male
    continues in orange, and any remaining 'other' fills the rest in
    neutral grey. Center label shows "{N} subjects" in serif.

    The side legend lists each non-empty group with its count and adds
    a Female : Male ratio row when both groups are present.

    When ``handedness`` is supplied (e.g. ``{"R": 38, "L": 8, "A": 3}``
    from the BIDS participants.tsv aggregate), a small chip-row
    summary is appended beneath the donut so the cohort panel
    surfaces handedness alongside sex without taking the donut's place.
    """
    pct_f = f_count / total * 100 if total else 0.0
    pct_m = m_count / total * 100 if total else 0.0
    pct_o = o_count / total * 100 if total else 0.0

    # Circumference of a r=15.9 circle is 2π·15.9 ≈ 99.9 → we treat the
    # stroke as carrying exactly 100 "percent units" so each dasharray
    # value is directly a percentage.
    C = 100.0

    arc_pieces: list[str] = []
    offset = 25.0  # Start at 12 o'clock — rotate(-90) makes the arc clockwise.

    def _arc(pct: float, color: str) -> str:
        nonlocal offset
        if pct <= 0:
            return ""
        length = pct / 100.0 * C
        gap = C - length
        piece = (
            f'<circle cx="21" cy="21" r="15.9" fill="none" stroke="{color}" '
            f'stroke-width="5" stroke-dasharray="{length:.3f} {gap:.3f}" '
            f'stroke-dashoffset="{offset:.3f}" transform="rotate(-90 21 21)" '
            'stroke-linecap="butt"/>'
        )
        offset -= length
        if offset < 0:
            offset += C
        return piece

    # Track ring underneath (covers any unaccounted slice when total > 0).
    arc_pieces.append(
        '<circle cx="21" cy="21" r="15.9" fill="none" stroke="#d0d6dc" '
        'stroke-width="5"/>'
    )
    arc_pieces.append(_arc(pct_f, "#006ca3"))
    arc_pieces.append(_arc(pct_m, "#f7941d"))
    arc_pieces.append(_arc(pct_o, "#6b7785"))

    arcs_svg = "".join(p for p in arc_pieces if p)

    # Build the side legend table.
    # Emit each row cell as a <div> (not <span>) — the global pydata
    # custom.css applies `position: absolute` to legend-flavoured spans,
    # which collapses the flex layout. Divs aren't targeted by those
    # rules, so the row stays a clean swatch / label / value triplet.
    legend_rows = []
    if f_count:
        legend_rows.append(("Female", f_count, "#006ca3"))
    if m_count:
        legend_rows.append(("Male", m_count, "#f7941d"))
    if o_count:
        legend_rows.append(("Other", o_count, "#6b7785"))
    rows_html = ""
    for label, count, color in legend_rows:
        rows_html += (
            '<div class="row">'
            f'<div class="sw" style="background:{color}"></div>'
            f'<div class="lbl">{label}</div>'
            f'<div class="v">{count}</div>'
            "</div>"
        )

    # F:M ratio row, only when both populated and m_count > 0
    if f_count and m_count:
        ratio = f_count / m_count
        rows_html += (
            '<div class="row ratio">'
            '<div class="sw" style="visibility:hidden"></div>'
            '<div class="lbl">F : M ratio</div>'
            f'<div class="v">{ratio:.2f} : 1</div>'
            "</div>"
        )

    # Center label inside the donut.
    center_html = (
        '<foreignObject x="0" y="0" width="42" height="42">'
        '<div xmlns="http://www.w3.org/1999/xhtml" '
        'style="width:100%;height:100%;display:flex;flex-direction:column;'
        'align-items:center;justify-content:center;font-family:Spectral,Georgia,serif;">'
        f'<div style="font-size:13px;line-height:1;letter-spacing:-.02em">{total}</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:2.8px;'
        "letter-spacing:.18em;color:#6a6e75;margin-top:1.5px;"
        'text-transform:uppercase">subjects</div>'
        "</div>"
        "</foreignObject>"
    )

    summary_note = ""
    if f_count and m_count:
        f_pct_int = round(pct_f)
        summary_note = (
            '<div class="sex-note" style="margin-top:14px; font-size:13px; '
            "color:#34404e; line-height:1.5; max-width:520px; "
            'font-family:Spectral,Georgia,serif;">'
            f"{f_pct_int}% female · n = {total} subjects with reported sex."
            "</div>"
        )

    handedness_chip = ""
    if handedness:
        # BIDS uses ``R``/``L``/``A``/``n/a`` keys (case-insensitive).
        # Aggregate into right/left/ambi/unknown so we can render a
        # compact chip row beneath the donut.
        h_buckets = {"right": 0, "left": 0, "ambi": 0, "unknown": 0}
        for k, v in handedness.items():
            try:
                count = int(v or 0)
            except (TypeError, ValueError):
                continue
            key = str(k).strip().lower()
            if key in ("r", "right"):
                h_buckets["right"] += count
            elif key in ("l", "left"):
                h_buckets["left"] += count
            elif key in ("a", "ambi", "ambidextrous"):
                h_buckets["ambi"] += count
            else:
                h_buckets["unknown"] += count
        h_total = sum(h_buckets.values())
        if h_total > 0:
            chips = []
            label_map = {
                "right": "Right",
                "left": "Left",
                "ambi": "Ambidextrous",
                "unknown": "Unknown",
            }
            for k, label in label_map.items():
                if h_buckets[k]:
                    chips.append(
                        '<span style="display:inline-block; padding:2px 8px; '
                        "margin-right:6px; font-family:Inter,sans-serif; "
                        "font-size:11px; letter-spacing:.04em; "
                        "background:rgba(0,108,163,0.08); color:#1a2532; "
                        'border-radius:3px;">'
                        f"{label} · {h_buckets[k]}"
                        "</span>"
                    )
            handedness_chip = (
                '<div class="hand-row" style="margin-top:10px; font-size:11px; '
                'color:#6b7785; font-family:Inter,sans-serif;">'
                '<span style="text-transform:uppercase; letter-spacing:.16em; '
                'margin-right:10px;">Handedness</span>' + "".join(chips) + "</div>"
            )

    return (
        ".. raw:: html\n\n"
        '   <div class="eegdash-stats-section eegdash-ed-sex" '
        'style="margin-bottom:1rem;">\n'
        "     <p><strong>Sex composition</strong></p>\n"
        '     <div class="sex-wrap" style="display:flex; align-items:center; '
        'gap:30px; flex-wrap:wrap;">\n'
        '       <svg class="sex-donut" viewBox="0 0 42 42" '
        'style="width:170px; height:170px; flex-shrink:0;">'
        f"{arcs_svg}"
        f"{center_html}"
        "</svg>\n"
        '       <div class="sex-legend" '
        'style="flex:1; font-family:JetBrains Mono,monospace; '
        'font-size:13px; min-width:220px;">'
        f"{rows_html}"
        "</div>\n"
        "     </div>\n"
        f"     {summary_note}\n"
        f"     {handedness_chip}\n"
        "   </div>\n\n"
    )


def _format_recording_stats_section(context: Mapping[str, object]) -> str:
    """Generate a Dataset Statistics section from EEGDash API data.

    Renders inline HTML bar charts and text stats for age distribution,
    sex distribution, channel counts, sampling frequencies, and total
    recording duration.  Returns an empty string when no useful data
    is present so the template placeholder collapses silently.
    """
    demographics: dict = context.get("demographics") or {}
    nchans_counts: list = context.get("nchans_counts") or []
    sfreq_counts: list = context.get("sfreq_counts") or []
    total_duration_s = context.get("total_duration_s")
    bad_channels_info: dict | None = context.get("bad_channels_info")

    ages: list = demographics.get("ages") or []
    sex_dist: dict = demographics.get("sex_distribution") or {}

    has_ages = bool(ages)
    has_sex = bool(sex_dist)
    has_nchans = bool(nchans_counts)
    has_sfreq = bool(sfreq_counts)
    has_duration = total_duration_s is not None
    has_bad_channels = bad_channels_info is not None

    if not any(
        (has_ages, has_sex, has_nchans, has_sfreq, has_duration, has_bad_channels)
    ):
        return ""

    heading = "Dataset Statistics\n------------------\n\n"
    parts: list[str] = []
    # Open a 2-column grid wrapper so the age histogram and the sex
    # donut render side-by-side (matching the v1-editorial-v2 design).
    # The wrapper is closed at the end of the function via a paired
    # raw-html block, regardless of how many sub-sections rendered.
    parts.append('.. raw:: html\n\n   <div class="eegdash-ed-cohort-grid">\n')

    # ------------------------------------------------------------------
    # A. Age distribution — vertical bars stacked by gender
    # ------------------------------------------------------------------
    # When per-subject (age, sex) pairs are available via the EEGDash
    # records endpoint (each record carries a `participant_tsv` blob),
    # stack the histogram by gender — Female in EEGdash blue, Male in
    # EEGdash orange. Other/unspecified renders in neutral grey on top.
    # Falls back to single-color bars when only aggregate `ages` are
    # known.
    participants_rows = context.get("participants_rows") or []

    pair_buckets_f: Counter[int] = Counter()
    pair_buckets_m: Counter[int] = Counter()
    pair_buckets_o: Counter[int] = Counter()
    bucket_size = 5
    paired_age_count = 0
    for p in participants_rows:
        try:
            a = float(p.get("age"))
        except (TypeError, ValueError):
            continue
        if a <= 0:
            continue
        sex = str(p.get("sex") or "").strip().lower()
        bucket_start = int(a // bucket_size) * bucket_size
        if sex in _BIDS_FEMALE_KEYS:
            pair_buckets_f[bucket_start] += 1
        elif sex in _BIDS_MALE_KEYS:
            pair_buckets_m[bucket_start] += 1
        else:
            pair_buckets_o[bucket_start] += 1
        paired_age_count += 1

    if pair_buckets_f or pair_buckets_m or pair_buckets_o:
        # Gender-stacked path — real per-subject data
        all_buckets = sorted(
            set(pair_buckets_f) | set(pair_buckets_m) | set(pair_buckets_o)
        )
        n_f = sum(pair_buckets_f.values())
        n_m = sum(pair_buckets_m.values())
        n_o = sum(pair_buckets_o.values())
        max_total = max(
            pair_buckets_f[b] + pair_buckets_m[b] + pair_buckets_o[b]
            for b in all_buckets
        )
        bar_width = 28
        chart_height = 80

        ages_used = [
            float(p.get("age"))
            for p in participants_rows
            if str(p.get("age")) not in ("", "None", "n/a")
            and _is_positive_float(p.get("age"))
        ]
        age_min_v = min(ages_used) if ages_used else 0
        age_max_v = max(ages_used) if ages_used else 0

        bars_html = ""
        labels_html = ""
        for start in all_buckets:
            f = pair_buckets_f.get(start, 0)
            m = pair_buckets_m.get(start, 0)
            o = pair_buckets_o.get(start, 0)
            tot = f + m + o
            # Stack bottom→top: Male orange, Female blue, Other grey
            col_pieces = ""
            if o:
                h = int(o / max_total * chart_height)
                col_pieces += (
                    f'<div style="width:{bar_width}px; height:{h}px; '
                    f'background:#6b7785; flex-shrink:0;" '
                    f'title="{start}-{start + bucket_size - 1}: other n={o}">'
                    "</div>"
                )
            if f:
                h = int(f / max_total * chart_height)
                col_pieces += (
                    f'<div style="width:{bar_width}px; height:{h}px; '
                    f'background:#006ca3; flex-shrink:0;" '
                    f'title="{start}-{start + bucket_size - 1}: female n={f}">'
                    "</div>"
                )
            if m:
                h = int(m / max_total * chart_height)
                col_pieces += (
                    f'<div style="width:{bar_width}px; height:{h}px; '
                    f'background:#f7941d; flex-shrink:0;" '
                    f'title="{start}-{start + bucket_size - 1}: male n={m}">'
                    "</div>"
                )
            bars_html += (
                f'<div style="display:flex; flex-direction:column-reverse; '
                f'justify-content:flex-start; gap:1px;" '
                f'title="{start}-{start + bucket_size - 1}: n={tot}">'
                f"{col_pieces}"
                "</div>"
            )
            labels_html += (
                f'<span style="width:{bar_width}px; text-align:center; '
                f'overflow:hidden; white-space:nowrap;">{start}</span>'
            )

        legend_pieces = []
        if n_f:
            legend_pieces.append(
                '<span style="display:inline-flex; align-items:center; gap:6px;">'
                '<i style="width:10px; height:10px; background:#006ca3; '
                'display:inline-block;"></i>'
                f"Female · {n_f}</span>"
            )
        if n_m:
            legend_pieces.append(
                '<span style="display:inline-flex; align-items:center; gap:6px;">'
                '<i style="width:10px; height:10px; background:#f7941d; '
                'display:inline-block;"></i>'
                f"Male · {n_m}</span>"
            )
        if n_o:
            legend_pieces.append(
                '<span style="display:inline-flex; align-items:center; gap:6px;">'
                '<i style="width:10px; height:10px; background:#6b7785; '
                'display:inline-block;"></i>'
                f"Other · {n_o}</span>"
            )
        legend_html = (
            (
                '<div style="display:flex; gap:18px; margin-top:8px; '
                'font-size:11px;">' + "".join(legend_pieces) + "</div>"
            )
            if legend_pieces
            else ""
        )

        # Caption: include mean age when the API reports it, so the
        # subheading is "n=45, range 20-78 yr, mean 49.9 yr" not just range.
        age_mean_v = demographics.get("age_mean")
        try:
            mean_str = (
                f", mean {float(age_mean_v):.1f} yr" if age_mean_v is not None else ""
            )
        except (TypeError, ValueError):
            mean_str = ""

        age_html = (
            ".. raw:: html\n\n"
            '   <div class="eegdash-stats-section" style="margin-bottom:1rem;">\n'
            "     <p><strong>Age distribution by gender</strong> "
            f"(n={paired_age_count}, range {age_min_v:.0f}–{age_max_v:.0f} yr"
            f"{mean_str})</p>\n"
            f'     <div class="eeg-chart-row" style="display:flex; align-items:flex-end; '
            f'gap:2px; height:{chart_height}px; border-bottom:1px solid #34404e;">\n'
            f"       {bars_html}\n"
            "     </div>\n"
            '     <div class="eeg-chart-labels" style="display:flex; gap:2px; font-size:10px;">\n'
            f"       {labels_html}\n"
            "     </div>\n"
            f"     {legend_html}\n"
            "   </div>\n\n"
        )
        parts.append(age_html)

    elif has_ages:
        # Fall back to single-color age histogram when per-subject
        # sex isn't available. Uses brand blue uniformly.
        valid_ages = [float(a) for a in ages if a is not None]
        if valid_ages:
            age_min = min(valid_ages)
            age_max = max(valid_ages)
            buckets: Counter[int] = Counter(
                int(float(a) // bucket_size) * bucket_size for a in valid_ages
            )
            max_count = max(buckets.values())
            bar_width = 28
            chart_height = 80
            bars_html = ""
            labels_html = ""
            for start in sorted(buckets):
                count = buckets[start]
                h = int(count / max_count * chart_height)
                bars_html += (
                    f'<div style="width:{bar_width}px; height:{h}px; '
                    f'background:#006ca3; flex-shrink:0;" '
                    f'title="{start}-{start + bucket_size - 1}: {count}"></div>'
                )
                labels_html += (
                    f'<span style="width:{bar_width}px; text-align:center; '
                    f'overflow:hidden; white-space:nowrap;">{start}</span>'
                )
            mean_v = demographics.get("age_mean")
            try:
                mean_str = (
                    f", mean {float(mean_v):.1f} yr" if mean_v is not None else ""
                )
            except (TypeError, ValueError):
                mean_str = ""
            age_html = (
                ".. raw:: html\n\n"
                '   <div class="eegdash-stats-section" style="margin-bottom:1rem;">\n'
                "     <p><strong>Age distribution</strong> "
                f"(n={len(valid_ages)}, range {age_min:.0f}–{age_max:.0f} yr"
                f"{mean_str} · sex per subject not reported)</p>\n"
                f'     <div class="eeg-chart-row" style="display:flex; align-items:flex-end; '
                f'gap:2px; height:{chart_height}px; border-bottom:1px solid #34404e;">\n'
                f"       {bars_html}\n"
                "     </div>\n"
                '     <div class="eeg-chart-labels" style="display:flex; gap:2px; font-size:10px;">\n'
                f"       {labels_html}\n"
                "     </div>\n"
                "   </div>\n\n"
            )
            parts.append(age_html)

    # ------------------------------------------------------------------
    # B. Sex distribution — SVG donut
    # ------------------------------------------------------------------
    # Replaces the older horizontal-bar treatment with a proper donut chart
    # matching the v1-editorial-v2 design: blue female / orange male arcs,
    # centered "N subjects" callout, side legend with female/male/other
    # counts and the F:M ratio.
    if has_sex:
        f_count = sum(
            int(v or 0) for k, v in sex_dist.items() if k.lower() in _BIDS_FEMALE_KEYS
        )
        m_count = sum(
            int(v or 0) for k, v in sex_dist.items() if k.lower() in _BIDS_MALE_KEYS
        )
        o_count = sum(
            int(v or 0)
            for k, v in sex_dist.items()
            if k.lower() not in _BIDS_FEMALE_KEYS | _BIDS_MALE_KEYS
        )
        total_sex = f_count + m_count + o_count

        if total_sex > 0:
            handedness = demographics.get("handedness_distribution") or {}
            parts.append(
                _render_sex_donut(
                    f_count,
                    m_count,
                    o_count,
                    total_sex,
                    handedness=handedness if isinstance(handedness, dict) else None,
                )
            )

    # ------------------------------------------------------------------
    # C. Channel count distribution
    # ------------------------------------------------------------------
    if has_nchans:
        parts.append(
            _make_count_bar_chart(
                nchans_counts, "Channel counts", "ch", bar_color="#009E73"
            )
        )

    # ------------------------------------------------------------------
    # D. Sampling frequency distribution
    # ------------------------------------------------------------------
    if has_sfreq:
        parts.append(
            _make_count_bar_chart(
                sfreq_counts, "Sampling frequencies", "Hz", bar_color="#D55E00"
            )
        )

    # ------------------------------------------------------------------
    # E. Total recording duration
    # ------------------------------------------------------------------
    if has_duration:
        try:
            total_s = float(total_duration_s)
            total_h = int(total_s // 3600)
            remaining_m = int((total_s % 3600) // 60)
            if total_h >= 24:
                duration_str = f"{total_h} h"
            else:
                duration_str = (
                    f"{total_h} h {remaining_m} min"
                    if total_h
                    else f"{remaining_m} min"
                )
            duration_html = (
                ".. raw:: html\n\n"
                '   <div class="eegdash-stats-section" style="margin-bottom:1rem;">\n'
                f"     <p><strong>Total recording duration</strong>: {duration_str}</p>\n"
                "   </div>\n\n"
            )
            parts.append(duration_html)
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # F. BIDS-annotated channel retention
    # ------------------------------------------------------------------
    if has_bad_channels:
        try:
            retained_pct = float(bad_channels_info["mean_retained_pct"])  # type: ignore[index]
            n_annotated = int(bad_channels_info["n_annotated"])  # type: ignore[index]
            bar_pct = int(retained_pct)
            # Colour: green (good retention) → red (poor retention)
            r = int(255 * (1 - retained_pct / 100))
            g = int(200 * retained_pct / 100)
            bar_color = f"rgb({r},{g},50)"
            bad_ch_html = (
                ".. raw:: html\n\n"
                '   <div class="eegdash-stats-section" style="margin-bottom:1rem;">\n'
                "     <p><strong>Channels retained (BIDS annotation)</strong>"
                f" — {retained_pct:.1f}% average across {n_annotated} annotated recording(s)</p>\n"
                '     <div style="display:flex; align-items:center; gap:8px;">\n'
                '       <div style="flex:1; max-width:300px; height:14px; '
                'background:#e0e0e0; border-radius:4px; overflow:hidden;">\n'
                f'         <div style="width:{bar_pct}%; height:100%; background:{bar_color};"></div>\n'
                "       </div>\n"
                f'       <span style="font-size:12px;">{retained_pct:.1f}%</span>\n'
                "     </div>\n"
                '     <p style="font-size:11px; color:var(--pst-color-muted, #666); margin-top:4px;">'
                "Based on <code>status: bad</code> in BIDS <code>channels.tsv</code>. "
                "This reflects researcher annotation, not automated pipeline rejection.</p>\n"
                "   </div>\n\n"
            )
            parts.append(bad_ch_html)
        except (TypeError, ValueError, KeyError):
            pass

    if len(parts) <= 1:
        # Only the wrapper open was added (no chart parts) — skip the
        # section entirely.
        return ""

    # Close the 2-column grid wrapper we opened at the top.
    parts.append(".. raw:: html\n\n   </div>\n")

    return heading + "".join(parts)


def _format_nemar_analysis_section(context: Mapping[str, object]) -> str:
    """Embed NEMAR pre-generated pipeline analysis plots for OpenNeuro/NEMAR datasets.

    NEMAR runs an automated EEG processing pipeline (via the Neuroscience Gateway)
    on all OpenNeuro datasets. The resulting static plots are publicly accessible
    at a consistent URL pattern and include:
      - Pipeline success: data-cleaning and ICA-decomposition pass/fail counts
      - Data frames retained (%) after cleaning
      - Line noise per channel (channel RMS, dB)
      - Data channels retained (%) after cleaning
      - Age and gender distribution of participants
      - HED event-descriptor word cloud (for HED-annotated datasets)
    """
    dataset_id = str(context.get("dataset_id") or "").strip().lower()
    source = str(context.get("source") or "").strip().lower()

    if not dataset_id or source not in ("openneuro", "nemar"):
        return ""

    nemar_url = str(
        context.get("nemar_url")
        or f"https://nemar.org/dataexplorer/detail?dataset_id={dataset_id}"
    )
    histogram_url = (
        "https://nemar.org/dataexplorer/download"
        f"?filepath=/data/nemar/openneuro//processed/{dataset_id}/code/{dataset_id}_histogram.png"
    )
    wordcloud_url = (
        "https://nemar.org/dataexplorer/download"
        f"?filepath=/data/nemar/openneuro//processed/event_summaries/{dataset_id}/word_cloud.svg"
        "&file_type=svg"
    )

    heading = "NEMAR Processing Statistics\n---------------------------\n\n"
    description = (
        f"The plots below are generated by `NEMAR's automated EEG pipeline <{nemar_url}>`_. "
        f"The histogram shows pipeline success for data cleaning and ICA decomposition, "
        "the percentage of data frames and EEG channels retained after artefact removal, "
        "line noise per channel (RMS, dB), and the age/gender distribution of participants.\n\n"
    )
    html_histogram = (
        ".. raw:: html\n\n"
        '   <div class="nemar-analysis-section">\n'
        f'     <a href="{nemar_url}" target="_blank" rel="noopener noreferrer">\n'
        "       <img\n"
        f'         src="{histogram_url}"\n'
        f'         alt="NEMAR pipeline statistics — {dataset_id.upper()}"\n'
        '         loading="lazy"\n'
        '         style="max-width: 100%; border: 1px solid var(--pst-color-border); border-radius: 8px; margin-bottom: 1rem;"\n'
        "       />\n"
        "     </a>\n"
        "   </div>\n\n"
    )
    html_wordcloud = (
        ".. raw:: html\n\n"
        '   <details class="nemar-wordcloud-details" style="margin-top: 0.5rem;">\n'
        "     <summary>HED event descriptors word cloud</summary>\n"
        "     <img\n"
        f'       src="{wordcloud_url}"\n'
        f'       alt="HED event descriptors word cloud — {dataset_id.upper()}"\n'
        '       loading="lazy"\n'
        '       style="max-width: 60%; display: block; margin: 0.5rem auto;"\n'
        "     />\n"
        "   </details>\n"
    )
    return heading + description + html_histogram + html_wordcloud


_EXPLORER_DATASET_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _format_explorer_section(name: str, context: Mapping[str, object]) -> str:
    """Render the BIDS file explorer for this dataset.

    Emits a ``.. dataset-explorer::`` directive for the catalog class
    name. The browser-side widget fetches records lazily from the
    EEGDash API the first time the user interacts with it, so there is
    no per-page build cost beyond the static directive output.

    Returns an empty string if ``name`` is not a safe identifier — this
    is the same gate the directive itself enforces, applied here so
    ``DATASET_PAGE_TEMPLATE`` collapses cleanly for any pathological
    class name rather than failing the whole build.
    """
    # The records collection stores dataset IDs in their original
    # ingested casing (e.g. ``ds001849``), while ``name`` is the Python
    # class identifier (e.g. ``DS001849``). The Mongo ``dataset`` filter
    # is case-sensitive, so prefer the lowercased ``dataset_id`` already
    # computed by the catalog row builder — falling back to ``name`` for
    # rows that have no separate id (and would have matched it anyway).
    candidate = str(context.get("dataset_id") or name or "")
    safe = candidate.strip()
    if not safe or not _EXPLORER_DATASET_RE.match(safe):
        return ""

    heading = "File Explorer\n-------------\n\n"
    description = (
        "Browse the BIDS file structure of this dataset. Records are "
        "fetched on demand from the EEGDash catalog the first time "
        "you open the explorer.\n\n"
    )
    directive = f".. dataset-explorer::\n   :dataset: {safe}\n"
    return heading + description + directive


# ---------------------------------------------------------------------------
# Trace viewer iframe: live signal preview from the eegdash-viewer
# (https://eegdash.github.io/eegdash-viewer/) embedded as a lazy iframe.
# Query the eegdash API for the first supported EEG record per dataset.
# ---------------------------------------------------------------------------

_TRACE_VIEWER_BASE = "https://eegdash.github.io/eegdash-viewer/"
_TRACE_API_URL = "https://data.eegdash.org/api/eegdash/records"
_TRACE_SUPPORTED_EXT = (".set", ".edf", ".bdf", ".vhdr", ".fif", ".fiff")


def _get_first_eeg_record(dataset_id: str) -> dict[str, object] | None:
    """Query eegdash API for the first supported electrophysiology record.

    Searches for any compatible modality (EEG, iEEG, EMG, MEG) that can be viewed
    with the eegdash-viewer, prioritizing EEG.
    """
    import urllib.parse
    import urllib.request

    query = {
        "dataset": dataset_id,
        "suffix": {"$in": ["eeg", "ieeg", "emg", "meg"]},
        "extension": {"$in": list(_TRACE_SUPPORTED_EXT)},
        "_has_missing_files": {"$ne": True},
    }
    params = {
        "limit": 1,
        "filter": json.dumps(query, separators=(",", ":")),
    }
    url = f"{_TRACE_API_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if body.get("success") and body.get("data"):
                return body["data"][0]
    except Exception:
        pass
    return None


def _format_traces_section(context: Mapping[str, object]) -> str:
    """Render an iframe for this dataset's signal preview.

    Query the API for the first supported electrophysiology record (EEG, iEEG, EMG)
    and build the viewer URL.
    """
    dataset_id = str(context.get("dataset_id") or "").strip().lower()
    if not dataset_id:
        return ""

    record = _get_first_eeg_record(dataset_id)
    if not record:
        return ""

    # Extract entity info from record or its entities_mne dict
    entities = (
        record.get("entities_mne", {})
        if isinstance(record.get("entities_mne"), dict)
        else {}
    )
    sub = str(record.get("subject") or entities.get("subject") or "").strip()
    task = str(record.get("task") or entities.get("task") or "").strip()
    ext = str(record.get("extension") or "").strip().lstrip(".")
    suffix = str(record.get("suffix") or "eeg").strip().lower()

    if not sub or not ext:
        return ""

    # Build the viewer URL
    from urllib.parse import urlencode

    qs_pairs = [("dataset", dataset_id), ("sub", sub)]
    ses = record.get("session") or entities.get("session")
    if ses:
        qs_pairs.append(("ses", str(ses)))
    if task:
        qs_pairs.append(("task", task))
    run = record.get("run") or entities.get("run")
    if run:
        qs_pairs.append(("run", str(run)))
    qs_pairs.append(("ext", ext))
    # Add suffix parameter if it's not the default EEG
    if suffix != "eeg":
        qs_pairs.append(("suffix", suffix))
    qs_pairs.append(("embed", "1"))
    iframe_src = f"{_TRACE_VIEWER_BASE}?{urlencode(qs_pairs)}"

    # Build entity label
    entity_bits = [f"sub-{sub}"]
    if ses:
        entity_bits.append(f"ses-{ses}")
    if task:
        entity_bits.append(f"task-{task}")
    if run:
        entity_bits.append(f"run-{run}")
    entity_label = " · ".join(entity_bits)

    # Caption above the iframe: explicit "one of many" so a reader
    # doesn't mistake the preview for the entire dataset. Pulls
    # n_subjects + n_records from context when available; otherwise
    # falls back to the generic "many recordings" phrasing.
    n_subjects = context.get("n_subjects")
    n_records = context.get("n_records")
    scope_bits = []
    if n_subjects:
        scope_bits.append(f"{n_subjects} subjects")
    if n_records:
        scope_bits.append(f"{n_records} recordings")
    scope_str = " and ".join(scope_bits) if scope_bits else "many recordings"

    openneuro_url = f"https://openneuro.org/datasets/{dataset_id}"

    # Map suffix to modality name for UI display
    modality_names = {
        "eeg": "EEG",
        "ieeg": "iEEG",
        "emg": "EMG",
        "meg": "MEG",
        "nirs": "fNIRS",
    }
    modality_display = modality_names.get(suffix, suffix.upper())

    heading = "Signal Preview\n--------------\n\n"
    html = (
        ".. raw:: html\n\n"
        '   <details class="trace-viewer">\n'
        f"     <summary>Live trace viewer — <strong>{entity_label}</strong></summary>\n"
        '     <p class="trace-viewer-caption">\n'
        "       Showing <strong>one</strong> representative recording out of\n"
        f"       <strong>{scope_str}</strong> in this dataset.\n"
        f'       Browse the full set on <a href="{openneuro_url}" target="_blank" rel="noopener">OpenNeuro</a>;\n'
        f"       drop any other <code>_{suffix}.{{set,edf,bdf,vhdr}}</code> file onto the\n"
        f"       viewer (or pass <code>?{suffix}=&lt;url&gt;</code>) to inspect it.\n"
        "     </p>\n"
        "     <iframe\n"
        f'       data-src="{iframe_src}"\n'
        '       loading="lazy"\n'
        '       width="100%" height="640"\n'
        '       style="border: 1px solid var(--pst-color-border); border-radius: 8px; max-width: 1200px; display: block; background: transparent;"\n'
        f'       title="Live {modality_display} trace viewer for {dataset_id} — {entity_label}"\n'
        '       referrerpolicy="no-referrer">\n'
        "     </iframe>\n"
        "   </details>\n"
    )
    return heading + html


def _format_see_also_section(
    dataset_id: str,
    class_name: str = "",
    related: Sequence[str] = (),
) -> str:
    dataset_lower = dataset_id.lower()
    nemar_url = f"https://nemar.org/dataexplorer/detail?dataset_id={dataset_lower}"
    openneuro_url = f"https://openneuro.org/datasets/{dataset_lower}"
    lines = [
        "* :class:`eegdash.dataset.EEGDashDataset`",
        "* :mod:`eegdash.dataset`",
        f"* `OpenNeuro dataset page <{openneuro_url}>`__",
        f"* `NeMAR dataset page <{nemar_url}>`__",
    ]
    # Cross-link up to 5 related datasets (same modality) to improve
    # internal link density across the 1,114 dataset pages.
    for rel_name in related[:5]:
        if rel_name != class_name:
            lines.append(
                f"* :doc:`eegdash.dataset.{rel_name} <eegdash.dataset.{rel_name}>`"
            )
    return "\n".join(lines)


def _format_feedback_section(dataset_id: str, title: str) -> str:
    """Generate a feedback section with a button to report issues on GitHub."""
    dataset_upper = dataset_id.upper()
    issue_title = quote(f"[Dataset] Issue with {dataset_upper}")
    issue_body = quote(
        f"## Dataset\n\n"
        f"- **Dataset ID:** {dataset_upper}\n"
        f"- **Title:** {title}\n\n"
        f"## Issue Description\n\n"
        f"Please describe the issue you encountered with this dataset:\n\n"
        f"## Steps to Reproduce\n\n"
        f"1. \n2. \n3. \n\n"
        f"## Expected Behavior\n\n\n"
        f"## Additional Context\n\n"
    )
    github_url = (
        f"https://github.com/eegdash/EEGDash/issues/new"
        f"?title={issue_title}&body={issue_body}&labels=dataset"
    )

    return f""".. admonition:: Found an issue with this dataset?
   :class: tip

   If you encounter any problems with this dataset (missing files, incorrect metadata,
   loading errors, etc.), please let us know!

   .. button-link:: {github_url}
      :color: primary
      :outline:

      Report an Issue on GitHub"""


# ---------------------------------------------------------------------------
# Editorial Brief — per-dataset chrome lifted from v1-editorial-v2.html.
#
# These helpers add the editorial cards (kicker, deck, 3-layer rail, caveat
# callout, provenance strip, footnotes, colophon) that wrap the existing
# Sphinx-rendered dataset content. They emit raw HTML blocks the
# accompanying `_static/css/dataset-editorial.css` styles. Fabricated
# placeholders (PSD curves, alpha-peak, frame retention) deliberately
# resolve to TODO labels — the backend doesn't compute those numbers yet,
# and shipping mock values site-wide would mislead readers.
# ---------------------------------------------------------------------------

_EDITORIAL_BRAND_GLYPH_SVG = (
    '<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<circle cx="16" cy="16" r="14" fill="none" stroke="currentColor" stroke-width="1.4"/>'
    '<path d="M2.5 16 Q4.5 12 6.5 16 Q8.5 20 10.5 14 Q12 12 13.5 18" '
    'stroke="#006ca3" stroke-width="1.6" fill="none" stroke-linecap="round" '
    'stroke-linejoin="round"/>'
    '<line x1="16" y1="2.5" x2="16" y2="29.5" stroke="#d0d6dc" stroke-width=".6" '
    'stroke-dasharray="1 2"/>'
    '<g stroke="#f7941d" stroke-width=".8" fill="none" opacity=".85">'
    '<path d="M19.5 10.5 L24.5 13 L21.5 19 L27 21.5 L19.5 10.5 L21.5 19 M24.5 13 L27 21.5"/>'
    "</g>"
    '<g fill="#f7941d">'
    '<circle cx="19.5" cy="10.5" r="1.6"/>'
    '<circle cx="24.5" cy="13" r="1.6"/>'
    '<circle cx="21.5" cy="19" r="1.6"/>'
    '<circle cx="27" cy="21.5" r="1.6"/>'
    "</g></svg>"
)


def _editorial_html(block: str) -> str:
    """Wrap an HTML block in a Sphinx ``.. raw:: html`` directive.

    Indents the payload by three spaces (required by the directive) and
    strips a trailing newline so successive editorial blocks don't
    accumulate empty paragraphs between them.
    """
    indented = "\n".join("   " + line if line else "" for line in block.split("\n"))
    return ".. raw:: html\n\n" + indented.rstrip()


def _short_study_label(context: Mapping[str, object]) -> str:
    """Trim the study title for use as a kicker / breadcrumb tail."""
    title = _clean_value(context.get("title")) or str(context.get("class_name", ""))
    short = title.split(":", 1)[0].strip() if ":" in title else title
    if len(short) > 64:
        short = short[:61].rstrip(" ,.;:") + "…"
    return short or str(context.get("class_name", "")) or "Dataset"


def _format_editorial_kicker_section(context: Mapping[str, object]) -> str:
    """Editorial kicker + issue strip rendered above the H1."""
    class_name = str(context.get("class_name", "")).strip()
    short_label = _short_study_label(context)
    source = _editorial_source_label(_clean_value(context.get("source")))
    n_subjects = _value_or_unknown(_clean_value(context.get("n_subjects")), "subjects")
    n_records = _value_or_unknown(_clean_value(context.get("n_records")), "recordings")
    license_text = _value_or_unknown(_clean_value(context.get("license")), "license")
    # Issue-line flavour: keep the actual dataset numeric tail so it
    # reads as a real "issue" without inventing a volume number.
    digits = "".join(c for c in class_name if c.isdigit())
    issue_num = digits.lstrip("0") or digits or "—"
    block = (
        '<div class="eegdash-ed-issue">'
        f'<div class="crumb">EEGdash'
        f'<span class="crumb-sep">›</span>{source}'
        f'<span class="crumb-sep">›</span><b>{class_name}</b></div>'
        f"<div>Iss. {issue_num} · {n_subjects} subjects · "
        f"{n_records} recordings · {license_text}</div>"
        "</div>"
        f'<div class="eegdash-ed-kicker">Dataset Brief · {short_label}</div>'
    )
    return _editorial_html(block)


# ---------------------------------------------------------------------------
# Editorial value formatters
# ---------------------------------------------------------------------------
# The CSV/API surface raw values that aren't display-ready (long fp seconds,
# JSON-stringified sample-rate counts, lowercase source tags). The helpers
# below normalise those to short, readable labels for the field card + pills.

_SOURCE_LABEL_MAP = {
    "openneuro": "OpenNeuro",
    "nemar": "NeMAR",
    "huggingface": "Hugging Face",
    "hf": "Hugging Face",
    "physionet": "PhysioNet",
    "github": "GitHub",
    "osf": "OSF",
}

_MODALITY_LABEL_MAP = {
    "eeg": "EEG",
    "ieeg": "iEEG",
    "meg": "MEG",
    "emg": "EMG",
    "ecog": "ECoG",
    "fnirs": "fNIRS",
    "nirs": "fNIRS",
}


def _editorial_source_label(value: str) -> str:
    """Title-case a source identifier (``openneuro`` → ``OpenNeuro``)."""
    if not value:
        return "OpenNeuro"
    key = value.strip().lower()
    return _SOURCE_LABEL_MAP.get(key, value.strip().title())


def _editorial_modality_label(value: str) -> str:
    """Convert a lowercase modality token to the canonical display form."""
    if not value:
        return "EEG"
    cleaned = value.strip()
    key = cleaned.lower()
    if key in _MODALITY_LABEL_MAP:
        return _MODALITY_LABEL_MAP[key]
    # Multi-modality list ("eeg, meg") — uppercase each token.
    parts = [
        _MODALITY_LABEL_MAP.get(p.strip().lower(), p.strip().upper())
        for p in cleaned.split(",")
        if p.strip()
    ]
    return ", ".join(parts) if parts else cleaned.upper()


def _editorial_sfreq_label(context: Mapping[str, object]) -> str:
    """Compact sampling-rate label like ``250 Hz`` (or ``250, 1000 Hz``).

    The CSV stores sampling-rate distributions as JSON arrays of
    ``{val, count}`` records (see _format_stat_counts). For the field
    card we want a single clean string — pick the modal value when
    one dominates, else list distinct round-Hz values.
    """
    counts = context.get("sfreq_counts") or []
    if isinstance(counts, list) and counts:
        try:
            valid = [
                (int(round(float(c.get("val")))), int(c.get("count") or 0))
                for c in counts
                if isinstance(c, dict) and c.get("val") is not None
            ]
        except (TypeError, ValueError):
            valid = []
        if valid:
            unique_vals = sorted({v for v, _ in valid})
            if len(unique_vals) == 1:
                return f"{unique_vals[0]} Hz"
            # Heavily-skewed: modal value handles ≥80% of recordings.
            total = sum(c for _, c in valid) or 1
            top_val, top_count = max(valid, key=lambda x: x[1])
            if top_count / total >= 0.8:
                return f"{top_val} Hz · mixed"
            return ", ".join(f"{v}" for v in unique_vals) + " Hz"

    # Fall back to the raw stat-counts string from the CSV row.
    raw = _value_or_unknown(
        _clean_value(context.get("sampling_freqs")), "sampling_rate"
    )
    if raw in ("Varies", "—"):
        return raw
    return f"{raw} Hz" if not raw.endswith("Hz") else raw


def _editorial_duration_label(context: Mapping[str, object]) -> str:
    """Round duration to a clean ``XX.X h`` value (or ``Y min`` when short)."""
    total_duration_s = context.get("total_duration_s")
    seconds: float | None = None
    if total_duration_s is not None:
        try:
            seconds = float(total_duration_s)
        except (TypeError, ValueError):
            seconds = None
    if seconds is None:
        # Fall back to the row's pre-computed hours field.
        raw = _clean_value(context.get("duration_hours_total"))
        if raw:
            try:
                seconds = float(raw) * 3600.0
            except (TypeError, ValueError):
                return raw  # last resort, return whatever was given
    if seconds is None or seconds <= 0:
        return "—"
    hours = seconds / 3600.0
    if hours < 1:
        return f"{int(round(seconds / 60))} min"
    if hours >= 100:
        return f"{int(round(hours))} h"
    return f"{hours:.1f} h"


def _editorial_citation_label(value: object) -> str:
    """Drop a stray .0 from NEMAR citation counts (``1.0`` → ``1``)."""
    if value is None:
        return "—"
    text = str(value).strip()
    if not text or text == "—":
        return "—"
    try:
        f = float(text)
    except (TypeError, ValueError):
        return text
    if f == int(f):
        return str(int(f))
    return f"{f:.1f}"


_BIDS_TASK_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _humanise_bids_task(token: str) -> str:
    """Turn a BIDS task entity (``neurCorrYoung``) into a sentence-case
    label (``Neur Corr Young``). Underscores and hyphens are also
    softened to spaces. The canonical token is preserved separately so
    callers can keep the original in ``<code>`` for copy-paste.
    """
    if not token:
        return ""
    parts = re.split(r"[_\-]+", token)
    words: list[str] = []
    for part in parts:
        if not part:
            continue
        # Split camelCase / PascalCase runs without breaking acronym sequences.
        for sub in _BIDS_TASK_CAMEL_RE.split(part):
            if sub:
                words.append(sub[0].upper() + sub[1:])
    return " ".join(words)


def _editorial_tasks_label(tasks: Sequence[str]) -> str:
    """Compact label for a tasks list.

    One task → just the name. Two or three → comma-joined. Four-plus
    → ``"<count> tasks · first, second, …"`` so the rail row stays
    readable even for HBN-style 10-task datasets. Tokens that look like
    camelCase BIDS task entities are softened into a human-readable
    label paired with the canonical ``<code>`` so users can still
    copy-paste the exact identifier.
    """
    cleaned = [str(t).strip() for t in (tasks or []) if str(t).strip()]
    if not cleaned:
        return "—"

    def render(tok: str) -> str:
        # Only annotate when humanising actually changes the token —
        # short lower-case names like "rest" need no wrap.
        nice = _humanise_bids_task(tok)
        if nice and nice.lower().replace(" ", "") != tok.lower():
            return f'<code title="{nice}">{tok}</code>'
        return f"<code>{tok}</code>"

    if len(cleaned) == 1:
        return render(cleaned[0])
    if len(cleaned) <= 3:
        return " · ".join(render(t) for t in cleaned)
    head = " · ".join(render(t) for t in cleaned[:2])
    return f"{len(cleaned)} tasks · {head} · …"


def _editorial_sessions_label(sessions: Sequence[str]) -> str:
    """Compact label for the BIDS sessions list."""
    cleaned = [str(s).strip() for s in (sessions or []) if str(s).strip()]
    if not cleaned:
        return ""  # caller suppresses the row entirely
    if len(cleaned) == 1:
        return f"<code>{cleaned[0]}</code>"
    if len(cleaned) <= 3:
        return f"{len(cleaned)} · " + " · ".join(f"<code>{s}</code>" for s in cleaned)
    return f"{len(cleaned)} · <code>{cleaned[0]}</code> … <code>{cleaned[-1]}</code>"


def _editorial_contact_label(contacts: Sequence[str]) -> str:
    """Comma-joined contact list, truncated to 3 names + ellipsis."""
    cleaned = [str(c).strip() for c in (contacts or []) if str(c).strip()]
    if not cleaned:
        return ""
    head = ", ".join(cleaned[:3])
    if len(cleaned) > 3:
        head += f" · +{len(cleaned) - 3}"
    return head


def _editorial_updated_label(iso_timestamp: str) -> str:
    """Format an ISO-8601 timestamp as ``YYYY-MM-DD`` for the field card.

    Empty / unparseable input yields an empty string so the caller can
    decide whether to suppress the row.
    """
    if not iso_timestamp:
        return ""
    # Tolerate both trailing-Z and ``+00:00`` suffixes that the API ships.
    cleaned = iso_timestamp.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return iso_timestamp[:10] if len(iso_timestamp) >= 10 else ""
    return dt.date().isoformat()


def _format_editorial_fieldcard_section(context: Mapping[str, object]) -> str:
    """Rich field-card aside emitted next to the hero.

    Pairs with the H1/deck/byline/pills via CSS float-right. Renders Identity
    / Signal / BIDS / ML sections backed by real API data — bids_version,
    sidecar inventory, HED annotation status, tags, license, etc. Fields
    with no real source resolve to "—" so the layout stays regular.
    """
    class_name = str(context.get("class_name", "")).strip()
    license_text = _value_or_unknown(_clean_value(context.get("license")), "license")
    source = _editorial_source_label(_clean_value(context.get("source")))
    doi_clean = _normalize_doi(_clean_value(context.get("doi")))

    # Identity: Dataset, Version, Source, License
    # The "version" tag is the BIDS dataset version when present (extracted
    # from the OpenNeuro DOI tail when it has the v1.0.0 suffix pattern).
    version_tag = "—"
    if doi_clean and "/" in doi_clean:
        tail = doi_clean.rsplit("/", 1)[-1]
        if ".v" in tail:
            version_tag = "v" + tail.split(".v", 1)[1]

    # Signal
    n_subjects = _value_or_unknown(_clean_value(context.get("n_subjects")), "subjects")
    n_records = _value_or_unknown(_clean_value(context.get("n_records")), "recordings")
    modality_raw = _editorial_modality_label(_clean_value(context.get("modality")))
    n_channels = _value_or_unknown(
        _clean_value(context.get("n_channels")), "n_channels"
    )
    sfreq = _editorial_sfreq_label(context)
    duration_label = _editorial_duration_label(context)
    size_label = _value_or_unknown(_clean_value(context.get("size")), "general")

    # BIDS
    bids_version = _clean_value(context.get("bids_version")) or "—"
    sidecars = context.get("sidecars_detected") or []
    sidecar_line = (
        " · ".join(sidecars) if sidecars else "<span class='dim'>not yet probed</span>"
    )
    hed_annotated = bool(context.get("hed_annotated"))
    hed_label = (
        '<a href="#nemar-processing-statistics">HED ✓</a>' if hed_annotated else "—"
    )

    # ML — citations from NEMAR + HuggingFace mirror status
    citations = _editorial_citation_label(context.get("nemar_citation_count"))
    hf_info = context.get("huggingface") or {}
    hf_available = bool(hf_info.get("available"))
    hf_url = str(hf_info.get("url") or "https://huggingface.co/EEGDash")
    hf_label = (
        f'<a href="{hf_url}">EEGDash/{context.get("dataset_id")}</a>'
        if hf_available
        else f'<a href="{hf_url}">org listing</a>'
    )

    # Tags
    tags = context.get("tags") or {}
    tag_pathology = _clean_value(
        tags.get("pathology") if isinstance(tags, dict) else ""
    )
    tag_type = _clean_value(tags.get("type") if isinstance(tags, dict) else "")
    tag_modality = _clean_value(tags.get("modality") if isinstance(tags, dict) else "")

    # Quality / completeness score (already computed elsewhere)
    quality_label, _quality_color, quality_pct = _compute_quality_score(context)
    metadata_line = f"{quality_pct}% · {quality_label}"

    # --- Newly-surfaced fields (#30) ---------------------------------
    tasks = context.get("tasks") or []
    tasks_line = _editorial_tasks_label(tasks)

    sessions = context.get("sessions") or []
    sessions_line = _editorial_sessions_label(sessions)

    contacts = context.get("contact_info") or []
    contact_line = _editorial_contact_label(contacts)

    storage = context.get("dataset_storage") or {}
    s3_base = _clean_value(storage.get("base")) if isinstance(storage, dict) else ""

    updated_line = _editorial_updated_label(
        _clean_value(context.get("dataset_modified_at"))
        or _clean_value(context.get("dataset_created_at"))
    )

    paper_doi = _normalize_doi(_clean_value(context.get("associated_paper_doi")))
    paper_url = _clean_value(context.get("paper_url"))
    paper_doi_html = (
        f'<a href="https://doi.org/{paper_doi}">{paper_doi}</a>' if paper_doi else ""
    )
    # Either a DOI or a free-form URL counts as a paper for the rail action.
    paper_action_href = f"https://doi.org/{paper_doi}" if paper_doi else paper_url

    # Build the action row at the bottom of the rail.
    openneuro_url = str(context.get("openneuro_url") or "")
    croissant_url = (
        f"../../_static/dataset_generated/croissant/{class_name}.croissant.json"
    )
    actions = f'<a href="{openneuro_url}">OpenNeuro</a>'
    if paper_action_href:
        actions += f'<a href="{paper_action_href}">Read paper</a>'
    actions += (
        f'<a href="{hf_url}">{"🤗 HF" if hf_available else "🤗 Org"}</a>'
        f'<a href="{croissant_url}" download>Croissant</a>'
    )

    block = (
        '<aside class="eegdash-ed-rail">'
        "<h4>Field card</h4>"
        "<dl>"
        '<dt class="hdr">Identity</dt><dd class="hdrpad"></dd>'
        f"<dt>Dataset</dt><dd>{class_name}</dd>"
        f"<dt>Version</dt><dd>{version_tag}</dd>"
        f"<dt>Source</dt><dd>{source}</dd>"
        f"<dt>License</dt><dd>{license_text}</dd>"
    )
    if updated_line:
        block += f"<dt>Updated</dt><dd>{updated_line}</dd>"
    if contact_line:
        block += f"<dt>Contact</dt><dd>{contact_line}</dd>"
    if paper_doi_html:
        block += f"<dt>Paper DOI</dt><dd>{paper_doi_html}</dd>"

    block += (
        '<dt class="hdr">Signal</dt><dd class="hdrpad"></dd>'
        f"<dt>Subjects</dt><dd>{n_subjects}</dd>"
        f"<dt>Recordings</dt><dd>{n_records}</dd>"
        f"<dt>Modality</dt><dd>{modality_raw}</dd>"
        f"<dt>Channels</dt><dd>{n_channels}</dd>"
        f"<dt>Sample rate</dt><dd>{sfreq}</dd>"
        f"<dt>Duration</dt><dd>{duration_label}</dd>"
        f"<dt>Size</dt><dd>{size_label}</dd>"
        f"<dt>Tasks</dt><dd>{tasks_line}</dd>"
    )
    if sessions_line:
        block += f"<dt>Sessions</dt><dd>{sessions_line}</dd>"

    block += (
        '<dt class="hdr">BIDS</dt><dd class="hdrpad"></dd>'
        f"<dt>BIDSVersion</dt><dd>{bids_version}</dd>"
        f"<dt>Sidecars</dt><dd>{sidecar_line}</dd>"
        f"<dt>Events ann.</dt><dd>{hed_label}</dd>"
        f"<dt>Metadata</dt><dd>{metadata_line}</dd>"
    )
    if s3_base:
        block += f"<dt>Storage</dt><dd><code>{s3_base}</code></dd>"

    # Tags section — pathology / paradigm / experiment type.
    # Rename the rail's "Modality" tag to "Paradigm" so it doesn't collide
    # with the recording-modality row in the Signal section above, and
    # drop "Type" when it duplicates "Paradigm" (NEMAR/OpenNeuro often
    # surfaces the same string on both `modality of exp` and `type of exp`).
    if (
        tag_modality
        and tag_type
        and tag_modality.strip().lower() == tag_type.strip().lower()
    ):
        tag_type = ""
    if tag_pathology or tag_type or tag_modality:
        block += '<dt class="hdr">Tags</dt><dd class="hdrpad"></dd>'
        if tag_pathology:
            block += f"<dt>Pathology</dt><dd>{tag_pathology}</dd>"
        if tag_modality:
            block += f"<dt>Paradigm</dt><dd>{tag_modality}</dd>"
        if tag_type:
            block += f"<dt>Type</dt><dd>{tag_type}</dd>"

    # ML / Reach section — only emits when at least one field is present.
    reach_rows: list[str] = []
    if hf_available or hf_url:
        reach_rows.append(f"<dt>HF mirror</dt><dd>{hf_label}</dd>")
    if citations and citations != "—":
        reach_rows.append(f"<dt>Citations</dt><dd>{citations}</dd>")
    if reach_rows:
        block += '<dt class="hdr">ML &amp; Reach</dt><dd class="hdrpad"></dd>'
        block += "".join(reach_rows)

    block += "</dl>"

    # DOI block + action row
    if doi_clean:
        block += (
            '<div class="doi">'
            "<span>Persistent identifier</span>"
            f'<a href="https://doi.org/{doi_clean}">{doi_clean}</a>'
            "</div>"
        )
    block += f'<div class="actions">{actions}</div>'
    block += "</aside>"

    return _editorial_html(block)


def _format_editorial_hero_extras(context: Mapping[str, object]) -> str:
    """Editorial deck + byline + signal pills emitted after the citation block."""
    title = _clean_value(context.get("title"))
    authors = context.get("authors") or []
    year = _clean_value(context.get("year"))
    source = _editorial_source_label(_clean_value(context.get("source")))
    n_subjects = _clean_value(context.get("n_subjects"))
    modality = _editorial_modality_label(_clean_value(context.get("modality")))
    senior = _clean_value(context.get("senior_author"))
    funding = context.get("funding") or []
    tags = context.get("tags") or {}

    # Build a deck line — falls back gracefully when fields are missing.
    parts = []
    if n_subjects and n_subjects not in ("—", "0"):
        parts.append(f"{n_subjects}-participant")
    parts.append(f"{modality} dataset")
    if title:
        parts.append(f"— {title}")
    deck_text = " ".join(parts).strip()
    if not deck_text or deck_text == "EEG dataset":
        deck_text = (
            f"A {modality} dataset distributed through EEGDash with "
            f"standardized BIDS metadata."
        )

    # Byline: first two authors as primary, rest folded.
    primary = []
    secondary = []
    for idx, author in enumerate(authors):
        cleaned = author.replace("*", "")
        if idx < 2:
            primary.append(f"<strong>{cleaned}</strong>")
        else:
            secondary.append(cleaned)
    if primary:
        byline_authors = " · ".join(primary)
        if secondary:
            byline_authors += " · " + " · ".join(secondary[:4])
            if len(secondary) > 4:
                byline_authors += " · …"
    else:
        byline_authors = "Authors unspecified"

    year_line = ""
    if year and year != "—":
        year_line = (
            f'<br/><span class="role">Year</span> {year} · Distributed via {source}'
        )

    senior_line = ""
    if senior:
        senior_line = (
            f'<br/><span class="role">Senior author</span> <strong>{senior}</strong>'
        )

    funding_line = ""
    if funding:
        # Keep funding line short — collapse list to a single string.
        fund_str = " · ".join(str(f).strip() for f in funding[:2] if str(f).strip())
        if len(funding) > 2:
            fund_str += f" · + {len(funding) - 2} more"
        if fund_str:
            funding_line = f'<br/><span class="role">Funding</span> {fund_str}'

    # Editorial pill row — signal/format pills the sphinx-design badges
    # don't already cover. These complement (not replace) the SEO badges.
    pills: list[str] = []
    n_channels = _clean_value(context.get("n_channels"))
    if n_channels and n_channels not in ("—", "Varies"):
        pills.append(f'<span class="pill">{modality} · {n_channels} ch</span>')
    sfreq = _editorial_sfreq_label(context)
    if sfreq and sfreq not in ("—", "Varies"):
        pills.append(f'<span class="pill">{sfreq}</span>')
    bids_version = _clean_value(context.get("bids_version"))
    if bids_version:
        pills.append(f'<span class="pill is-info">BIDS {bids_version}</span>')
    if bool(context.get("hed_annotated")):
        pills.append('<span class="pill is-warning">HED ✓</span>')
    # Tasks pill — single task surfaced verbatim, multi-task compressed
    # to a count for legibility (HBN-style datasets have 10+).
    task_list = context.get("tasks") or []
    if isinstance(task_list, (list, tuple)) and task_list:
        if len(task_list) == 1:
            pills.append(f'<span class="pill">Task · {str(task_list[0])[:32]}</span>')
        else:
            pills.append(f'<span class="pill">{len(task_list)} tasks</span>')
    # Sessions pill — only when the dataset is multi-session.
    sess_list = context.get("sessions") or []
    if isinstance(sess_list, (list, tuple)) and len(sess_list) > 1:
        pills.append(f'<span class="pill">{len(sess_list)} sessions</span>')
    if isinstance(tags, dict):
        tag_pathology = _clean_value(tags.get("pathology"))
        tag_modality = _clean_value(tags.get("modality"))
        tag_type = _clean_value(tags.get("type"))
        if tag_pathology and tag_pathology.lower() not in ("not specified", "—"):
            pills.append(f'<span class="pill">{tag_pathology}</span>')
        if tag_modality and tag_modality.lower() not in ("—",):
            pills.append(f'<span class="pill">{tag_modality}</span>')
        if tag_type and tag_type.lower() not in ("—",):
            pills.append(f'<span class="pill">{tag_type}</span>')
    pills_html = (
        f'<div class="eegdash-ed-pills">{"".join(pills)}</div>' if pills else ""
    )

    block = (
        f'<p class="eegdash-ed-deck">{deck_text}.</p>'
        f'<div class="eegdash-ed-byline">'
        f'<span class="role">Data &amp; curation</span> {byline_authors}'
        f"{senior_line}"
        f"{year_line}"
        f"{funding_line}"
        "</div>"
        f"{pills_html}"
    )
    return _editorial_html(block)


def _format_editorial_layers_section(context: Mapping[str, object]) -> str:
    """3-layer architecture rail — Study / Signal·BIDS / Training·ML."""
    block = (
        '<div class="eegdash-ed-layers">'
        "<div>"
        '<div class="ly-lbl"><span>Layer 01</span><b>Study</b></div>'
        '<div class="ly-tit">What was asked</div>'
        '<div class="ly-dsc">Hypothesis, independent &amp; dependent variables, '
        "paradigm, cohort, and the editorial caveats around what the "
        "recordings can and cannot answer.</div>"
        "</div>"
        "<div>"
        '<div class="ly-lbl"><span>Layer 02</span><b>Signal · BIDS</b></div>'
        '<div class="ly-tit">What was recorded</div>'
        '<div class="ly-dsc">Sidecars, channels &amp; electrodes, coordinate '
        "system, event semantics, and quality stats from the NEMAR pipeline "
        "when available.</div>"
        "</div>"
        "<div>"
        '<div class="ly-lbl"><span>Layer 03</span><b>Training · ML</b></div>'
        '<div class="ly-tit">What you can train on</div>'
        '<div class="ly-dsc">Recommended access modes — MNE Raw, '
        "braindecode windows, PyTorch DataLoader — plus the targets the "
        "metadata makes addressable.</div>"
        "</div>"
        "</div>"
    )
    return _editorial_html(block)


def _editorial_secnum(num: int, label: str) -> str:
    """Editorial § NN marker emitted before each major H2."""
    block = f'<div class="eegdash-ed-secnum">§ {num:02d}<b>{label}</b></div>'
    return _editorial_html(block)


def _format_editorial_caveat_section(context: Mapping[str, object]) -> str:
    """Conditional caveat callout — only fires for small cohorts (n < 50)."""
    n_sub_raw = _clean_value(context.get("n_subjects"))
    try:
        n_sub = int(n_sub_raw)
    except (TypeError, ValueError):
        return ""
    if n_sub <= 0 or n_sub >= 50:
        return ""
    modality = _clean_value(context.get("modality")) or "EEG"
    block = (
        '<div class="eegdash-ed-caveat">'
        '<div class="c-lbl">Editorial caveat · cohort size</div>'
        "<h4>Treat this as a features-first dataset, "
        "not a deep-learning playground.</h4>"
        f"<p>With <b>n = {n_sub}</b> {modality} participants, this dataset sits "
        "below the ~50-subject threshold where deep networks trained from scratch "
        "typically pay off. A well-tuned feature pipeline — band-power features, "
        "Riemannian geometry, linear classifier — is the recommended baseline. "
        "Use deep models only with transfer learning or pre-trained backbones.</p>"
        "<p>For splits, prefer <code>GroupShuffleSplit</code> with "
        "<code>groups=subject_id</code> so windows from the same recording do not "
        "leak between train and test.</p>"
        "</div>"
    )
    return _editorial_html(block)


def _format_electrodes_traces_pair(name: str, context: Mapping[str, object]) -> str:
    """Render electrode layout + signal-preview iframe as a paired figure.

    Outputs a 2-column figure block matching the v1-editorial-v2 Fig. 01:
    live trace viewer on the left, electrode topomap on the right, with
    shared meta strip on top. Falls back to whichever section is
    available when one is missing.
    """
    electrodes = _format_electrodes_section(context).strip()
    traces = _format_traces_section(context).strip()

    # Strip Sphinx headings — both helpers prepend their own H2 heading
    # text. We want the iframes only because the paired figure header
    # comes from the editorial `Signal · Electrodes & live trace` H2.
    def _strip_heading(s: str) -> str:
        if not s:
            return ""
        lines = s.splitlines()
        # Drop any leading "Title\n----" or "Title\n====" heading + blank.
        i = 0
        while i + 1 < len(lines):
            if (
                lines[i].strip()
                and i + 1 < len(lines)
                and (
                    set(lines[i + 1].strip()) <= set("-=")
                    and len(lines[i + 1].strip()) >= 3
                )
            ):
                i += 2
                continue
            break
        # Skip blank lines after the heading
        while i < len(lines) and not lines[i].strip():
            i += 1
        return "\n".join(lines[i:])

    electrodes_body = _strip_heading(electrodes)
    traces_body = _strip_heading(traces)

    if not electrodes_body and not traces_body:
        return ""

    n_subjects = _value_or_unknown(_clean_value(context.get("n_subjects")), "subjects")
    n_records = _value_or_unknown(_clean_value(context.get("n_records")), "recordings")
    n_channels = _clean_value(context.get("n_channels")) or "—"
    sfreq = _editorial_sfreq_label(context)
    modality = _editorial_modality_label(_clean_value(context.get("modality")))

    meta_line = (
        f"{n_channels} ch · {modality} · {sfreq} · "
        f"{n_subjects} subjects, {n_records} recordings"
    )

    # Open the meta strip + a CSS-grid wrapper. Both child sections are
    # already raw HTML (their respective helpers emit `.. raw:: html`
    # directives wrapping <details> elements), so we sandwich them
    # between two `.. raw:: html` blocks that provide the wrapper.
    wrapper_open = (
        ".. raw:: html\n\n"
        '   <div class="eegdash-ed-figpair">\n'
        '     <div class="figpair-meta">\n'
        f"       <b>Fig. 01</b> Signal &amp; montage\n"
        f'       <span class="right">{meta_line}</span>\n'
        "     </div>\n"
        '     <div class="figpair-grid">\n'
    )
    wrapper_close = ".. raw:: html\n\n   </div></div>\n"

    parts = [wrapper_open]
    if traces_body:
        parts.extend(
            [
                '.. raw:: html\n\n   <div class="figpair-cell figpair-trace">\n',
                traces_body,
                "\n.. raw:: html\n\n   </div>\n",
            ]
        )
    if electrodes_body:
        parts.extend(
            [
                '.. raw:: html\n\n   <div class="figpair-cell figpair-montage">\n',
                electrodes_body,
                "\n.. raw:: html\n\n   </div>\n",
            ]
        )
    parts.append(wrapper_close)
    return "\n\n".join(parts)


def _format_dataset_info_dropdown(context: Mapping[str, object]) -> str:
    """Wrap the Dataset Information table + BibTeX in a folded dropdown.

    The field-card rail already shows every value from this table; we
    keep the table itself accessible (for searchability / structured
    metadata) but tucked inside a `.. dropdown::` so the visible page
    stays compact.
    """
    inner = _format_dataset_info_section(context)
    if not inner.strip():
        return ""
    indented = "\n".join(f"   {line}" if line else "" for line in inner.split("\n"))
    return (
        ".. dropdown:: Full dataset metadata table\n"
        "   :class-container: sd-shadow-sm eegdash-ed-info-dropdown\n"
        "\n"
        f"{indented}\n"
    )


def _format_editorial_access_modes_section(
    context: Mapping[str, object],
) -> str:
    """Sidecar card listing the access modes available for this dataset.

    Generic across all datasets — each row maps a programmatic entry point
    (MNE Raw, braindecode BaseConcatDataset, PyTorch DataLoader, Zarr,
    HuggingFace, Croissant) to a one-line description. Rendered after the
    API Reference so readers see the path from API class to data loop.
    """
    class_name = str(context.get("class_name", "")).strip()
    croissant_url = (
        f"../../_static/dataset_generated/croissant/{class_name}.croissant.json"
    )
    hf_info = context.get("huggingface") or {}
    hf_available = bool(hf_info.get("available"))
    hf_url = str(hf_info.get("url") or "https://huggingface.co/EEGDash")

    if hf_available:
        dataset_id_lower = str(context.get("dataset_id") or "").lower()
        hf_blurb = (
            "Pre-bundled mirror at "
            f'<a href="{hf_url}">EEGDash/{dataset_id_lower}</a> · '
            'pull with <code>datasets.load_dataset("EEGDash/'
            f'{dataset_id_lower}")</code>.'
        )
    else:
        hf_blurb = (
            "No per-dataset mirror published yet — browse the "
            f'<a href="{hf_url}">EEGDash org listing</a> for sibling datasets.'
        )

    rows = [
        (
            ".raw",
            (
                "MNE <code>Raw</code> object — standard tools (filter, epoch, "
                "ICA, plot_psd)."
            ),
            "mne",
        ),
        (
            "BaseConcatDataset",
            (
                "Each record is a lazy <code>BaseDataset</code> from "
                "braindecode — windowed via <code>create_windows_from_events</code>."
            ),
            "braindecode",
        ),
        (
            "DataLoader",
            (
                "Wraps the windowed dataset into a PyTorch <code>DataLoader</code>; "
                "supports parallel workers and on-the-fly augmentations."
            ),
            "pytorch",
        ),
        (
            "Zarr cache",
            (
                "Optional braindecode Zarr mirror for fast resume; persisted to "
                "<code>cache_dir</code>."
            ),
            "zarr",
        ),
        (
            "Hugging Face",
            hf_blurb,
            "huggingface",
        ),
        (
            "Croissant 1.0",
            (
                f"Machine-readable JSON-LD descriptor — "
                f'<a href="{croissant_url}" download>{class_name}.croissant.json</a> '
                f"(MLCommons schema, ingestible by PyTorch / TensorFlow / JAX)."
            ),
            "mlcommons",
        ),
    ]

    rows_html = "".join(
        '<div class="am-row">'
        f'<span class="name">{name}</span>'
        f'<span class="what">{what}</span>'
        f'<span class="badge">{badge}</span>'
        "</div>"
        for name, what, badge in rows
    )

    block = (
        '<div class="eegdash-ed-access">'
        '<div class="sidecar-hdr">'
        "<span><b>Access modes</b></span>"
        '<span class="right">MNE → braindecode → PyTorch → ML</span>'
        "</div>"
        f'<div class="am-list">{rows_html}</div>'
        "</div>"
    )
    return _editorial_html(block)


def _format_editorial_provenance_section(context: Mapping[str, object]) -> str:
    """Provenance strip — five-column band placed before the See Also footer.

    All fields now carry real values where the API supplies them:
    * BIDS validity → BIDSVersion if known, otherwise "—".
    * Provenance → license + DOI (always real when both exist).
    * Sidecars → real list pulled from the per-record probe.
    * Machine-readable → schema.org/Dataset (inline JSON-LD on this page)
      and Croissant 1.0 (downloadable JSON-LD generated alongside the page).
    * Mirrors → OpenNeuro + NEMAR + optional paper URL.
    """
    class_name = str(context.get("class_name", "")).strip()
    openneuro_url = str(context.get("openneuro_url", ""))
    nemar_url = str(context.get("nemar_url", ""))
    paper_url = _clean_value(context.get("paper_url"))
    license_text = _value_or_unknown(_clean_value(context.get("license")), "license")
    doi_clean = _normalize_doi(_clean_value(context.get("doi")))
    doi_link = (
        f'<a href="https://doi.org/{doi_clean}">{doi_clean}</a>'
        if doi_clean
        else '<span class="todo">DOI not on file</span>'
    )

    bids_version = _clean_value(context.get("bids_version"))
    bids_cell = (
        f'<div class="v ok">BIDS {bids_version}</div>'
        if bids_version
        else '<div class="v todo">version not on file</div>'
    )

    sidecars = context.get("sidecars_detected") or []
    if sidecars:
        sidecars_cell = f'<div class="v">{" · ".join(sidecars)}</div>'
    else:
        sidecars_cell = '<div class="v todo">not yet probed</div>'

    croissant_url = (
        f"../../_static/dataset_generated/croissant/{class_name}.croissant.json"
    )

    hf_info = context.get("huggingface") or {}
    hf_url = str(hf_info.get("url") or "https://huggingface.co/EEGDash")
    hf_link_label = "HuggingFace" if hf_info.get("available") else "HF org"

    mirrors = [
        f'<a href="{openneuro_url}">OpenNeuro</a>',
        f'<a href="{nemar_url}">NEMAR</a>',
        f'<a href="{hf_url}">{hf_link_label}</a>',
    ]
    if paper_url:
        mirrors.append(f'<a href="{paper_url}">Paper</a>')

    block = (
        '<div class="eegdash-ed-prov">'
        "<div>"
        '<div class="lbl">BIDS</div>'
        f"{bids_cell}"
        "</div>"
        "<div>"
        '<div class="lbl">Sidecars</div>'
        f"{sidecars_cell}"
        "</div>"
        "<div>"
        '<div class="lbl">Provenance</div>'
        f'<div class="v">{license_text} · {doi_link}</div>'
        "</div>"
        "<div>"
        '<div class="lbl">Machine-readable</div>'
        '<div class="v">'
        '<a href="#dataset-information">schema.org/Dataset</a> · '
        f'<a href="{croissant_url}" download>Croissant</a>'
        "</div>"
        "</div>"
        "<div>"
        '<div class="lbl">Mirrors</div>'
        f'<div class="v">{" · ".join(mirrors)}</div>'
        "</div>"
        "</div>"
    )
    return _editorial_html(block)


def _format_editorial_footnotes_section(
    context: Mapping[str, object],
    related: Sequence[str] = (),
    related_meta: Sequence[Mapping[str, object]] = (),
) -> str:
    """Three-column footnotes block — Citation / Provenance / Related.

    ``related_meta`` carries optional per-neighbour metadata (modality,
    n_subjects, same-author flag) used to render compact cards. When
    empty the function falls back to a plain link list so the section
    still works for callers that haven't been updated.
    """
    title = _clean_value(context.get("title"))
    authors = context.get("authors") or []
    year = _clean_value(context.get("year"))
    doi_clean = _normalize_doi(_clean_value(context.get("doi")))
    source = _clean_value(context.get("source")) or "OpenNeuro"

    if authors:
        author_str = ", ".join(a.replace("*", "") for a in authors[:5])
        if len(authors) > 5:
            author_str += ", …"
    else:
        author_str = "Authors unspecified"
    year_str = year if year and year != "—" else "n.d."
    citation = (
        f"{author_str} ({year_str}). <em>{title or context.get('class_name')}</em>."
    )
    if doi_clean:
        citation += f" <code>{doi_clean}</code>"

    provenance_notes = [
        f'<span class="note-num">¹</span>Contributed to {source} in BIDS format.',
        '<span class="note-num">²</span>Curated &amp; ingested by the EEGDash '
        "catalog; see CITATION.cff for canonical reference.",
    ]
    if doi_clean:
        provenance_notes.append(
            f'<span class="note-num">³</span>Persistent identifier: '
            f"<code>{doi_clean}</code>."
        )
    provenance_html = "".join(f"<p>{n}</p>" for n in provenance_notes)

    if related_meta:
        cards = []
        for meta in list(related_meta)[:5]:
            rel_name = str(meta.get("name") or "").strip()
            if not rel_name:
                continue
            modality = (str(meta.get("modality") or "")).upper() or "—"
            n_sub = _clean_value(meta.get("n_subjects"))
            same_authors = bool(meta.get("same_authors"))
            meta_bits = [modality]
            if n_sub and n_sub != "0":
                meta_bits.append(f"{n_sub} subj")
            badge = (
                '<span class="rel-tag rel-same">Same authors</span>'
                if same_authors
                else ""
            )
            cards.append(
                f'<a class="rel-card" href="{rel_name}.html">'
                f'<span class="rel-id">{rel_name}</span>'
                f'<span class="rel-meta">{" · ".join(meta_bits)}</span>'
                f"{badge}"
                "</a>"
            )
        related_html = f'<div class="rel-grid">{"".join(cards)}</div>'
        if len(related_meta) > 5:
            related_html += (
                f'<p class="rel-more">+ {len(related_meta) - 5} more — '
                "see See Also below →</p>"
            )
    elif related:
        # Legacy fallback when no metadata was passed through.
        related_items = "<br/>".join(
            f'<a href="{rel}.html">{rel}</a>' for rel in related[:5]
        )
        related_html = f"<p>{related_items}</p>"
        if len(related) > 5:
            related_html += (
                f"<p><em>+ {len(related) - 5} more — see See Also below →</em></p>"
            )
    else:
        related_html = (
            "<p><em>No sibling datasets cross-linked for this modality yet.</em></p>"
        )

    block = (
        '<div class="eegdash-ed-footnotes">'
        "<div>"
        "<h5>Citation</h5>"
        f"<p>{citation}</p>"
        "</div>"
        "<div>"
        "<h5>Provenance</h5>"
        f"{provenance_html}"
        "</div>"
        "<div>"
        "<h5>Related &amp; sibling datasets</h5>"
        f"{related_html}"
        "</div>"
        "</div>"
    )
    return _editorial_html(block)


# Curated "start here" tutorials linked from every dataset page.
# Each tuple: (slug, generated-rst subpath, title, one-line blurb).
# Thumbnails live in docs/source/_static/thumbs/<slug>.png and pages render
# under docs/source/generated/auto_examples/<subpath>/<slug>.html.
_EDITORIAL_EXAMPLES = (
    (
        "plot_00_first_search",
        "tutorials/00_start_here",
        "Find datasets with the EEGDash API",
        "Query the catalogue, filter by task or modality, list candidates.",
    ),
    (
        "plot_01_first_recording",
        "tutorials/00_start_here",
        "Load one EEG recording",
        "Resolve a single record to an MNE Raw with channels and events.",
    ),
    (
        "plot_02_dataset_to_dataloader",
        "tutorials/00_start_here",
        "EEG recording to PyTorch DataLoader",
        "Wrap braindecode windows in a DataLoader for model training.",
    ),
    (
        "plot_10_preprocess_and_window",
        "tutorials/10_core_workflow",
        "Preprocess EEG and create windows",
        "Filter, resample, epoch — and persist the windowed dataset.",
    ),
    (
        "plot_13_save_and_reuse_prepared_data",
        "tutorials/10_core_workflow",
        "Save and reload prepared data",
        "Cache a windowed dataset to disk and reattach it without recompute.",
    ),
    (
        "how_to_download_a_dataset",
        "how_to",
        "Download a dataset locally",
        "Prefetch BIDS files to a local cache and validate the layout.",
    ),
)


def _format_editorial_examples_gallery(context: Mapping[str, object]) -> str:
    """Six-card thumbnail gallery linking the canonical "start here" tutorials.

    Borrowed from MOABB's dataset pages: instead of a wall of API text, give
    readers a few concrete entry points where the dataset is actually being
    used end to end. The list is shared across all dataset pages — the
    landing page links into ``EEGDash`` ergonomics, not per-dataset code.

    Returns an HTML block wrapped via ``_editorial_html`` so the surrounding
    Sphinx parser leaves it alone.
    """
    dataset_id = str(context.get("dataset_id") or "").strip()
    cards = []
    for slug, subpath, title, blurb in _EDITORIAL_EXAMPLES:
        thumb = f"../../_static/thumbs/{slug}.png"
        href = f"../../generated/auto_examples/{subpath}/{slug}.html"
        cards.append(
            '<a class="ex-card" '
            f'href="{href}">'
            f'<span class="ex-thumb"><img src="{thumb}" alt="" loading="lazy"></span>'
            '<span class="ex-body">'
            f'<span class="ex-title">{title}</span>'
            f'<span class="ex-blurb">{blurb}</span>'
            "</span>"
            "</a>"
        )
    hint = (
        f"Swap any <code>load_dataset(...)</code> call for "
        f"<code>{dataset_id}</code> to reproduce the tutorial on this dataset."
        if dataset_id
        else ""
    )
    block = (
        '<section class="eegdash-ed-examples">'
        '<div class="sidecar-hdr">'
        "<span><b>Examples using EEGDash</b></span>"
        '<span class="right">curated · start here</span>'
        "</div>"
        f'<div class="ex-grid">{"".join(cards)}</div>'
    )
    if hint:
        block += f'<p class="ex-hint">{hint}</p>'
    block += "</section>"
    return _editorial_html(block)


# ---------------------------------------------------------------------------
# Related-datasets kNN index — Gower distance over mixed metadata
# ---------------------------------------------------------------------------
# Gower distance is the canonical similarity measure for records whose
# features mix categorical and numerical types — it normalises numerical
# differences by per-feature range and treats categorical mismatch as a
# binary 0/1 contribution, then averages across features (weighted).
# References: Gower 1971 ("A General Coefficient of Similarity"); the
# `gower` / `gower-metric` Python packages implement the same recipe.
#
# We inline the algorithm so the docs build picks up no new runtime dep —
# numpy is already pulled in by sphinx-gallery, and the resulting matrix
# is tiny (~780² f32 ≈ 2.4 MB) even at full catalog size.

# (name, kind, weight) — weights are domain judgment: modality and
# paradigm dominate, source/license barely move the needle.
_RELATED_FEATURE_SPECS: tuple[tuple[str, str, float], ...] = (
    ("record_modality", "cat", 4.0),  # eeg / meg / emg / ieeg — primary axis
    ("modality of exp", "cat", 3.0),  # visual / motor / auditory / …
    ("type of exp", "cat", 2.5),  # perception / clinical / resting-state
    ("Type Subject", "cat", 2.0),  # healthy / clinical / development
    ("source", "cat", 0.5),  # openneuro / nemar — provenance
    ("n_subjects", "num", 1.5),
    ("n_records", "num", 1.0),
    ("n_channels", "num", 1.5),  # extracted from nchans_set top val
    ("sampling_rate", "num", 1.0),  # extracted from sampling_freqs top val
    ("duration_hours_total", "num", 0.8),
)


def _extract_top_numeric_val(value: object) -> float:
    """Return the most-common numeric ``val`` from a ``[{val, count}]`` array.

    Used to collapse channel / sampling-rate frequency lists to a single
    representative number for distance calculations. Returns ``nan`` when
    no usable value is present.
    """
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]", "[ ]"}:
        return float("nan")
    if text.startswith("["):
        try:
            items = json.loads(text)
        except (ValueError, json.JSONDecodeError):
            return float("nan")
        best_val: float | None = None
        best_count = -1
        for item in items:
            if not isinstance(item, dict):
                continue
            v = item.get("val")
            c = item.get("count", 0) or 0
            try:
                v_num = float(v)
            except (TypeError, ValueError):
                continue
            if c > best_count:
                best_val = v_num
                best_count = c
        return best_val if best_val is not None else float("nan")
    # Bare numeric column (n_subjects, n_records, …).
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _build_related_index_gower(
    dataset_names: Sequence[str],
    dataset_rows: Mapping[str, Mapping[str, object]],
    k: int = 6,
) -> dict[str, list[str]]:
    """Compute per-dataset top-K related neighbours via Gower distance.

    Returns ``{name: [neighbour_name, …]}``. An entry is empty when the
    dataset has no usable features. Falls back to ``{}`` (callers then
    use the legacy modality grouping) if numpy is unavailable.
    """
    try:
        import numpy as np
    except ImportError:
        LOGGER.warning(
            "[dataset-docs] numpy not available; related-datasets falls back "
            "to modality grouping"
        )
        return {}

    names = list(dataset_names)
    n = len(names)
    if n < 2:
        return {name: [] for name in names}

    specs = _RELATED_FEATURE_SPECS

    # Build typed columns once, indexed positionally so the distance loop
    # is plain numpy.
    cat_cols: dict[str, np.ndarray] = {}
    cat_missing: dict[str, np.ndarray] = {}
    num_cols: dict[str, np.ndarray] = {}
    num_missing: dict[str, np.ndarray] = {}
    num_ranges: dict[str, float] = {}

    for col, kind, _ in specs:
        if kind == "cat":
            arr = np.empty(n, dtype=object)
            miss = np.zeros(n, dtype=bool)
            for i, name in enumerate(names):
                row = dataset_rows.get(name) or {}
                raw = _clean_value(row.get(col))
                if not raw:
                    miss[i] = True
                    arr[i] = ""
                else:
                    arr[i] = raw.lower()
            cat_cols[col] = arr
            cat_missing[col] = miss
        else:  # numerical
            arr = np.full(n, np.nan, dtype=np.float64)
            for i, name in enumerate(names):
                row = dataset_rows.get(name) or {}
                if col == "n_channels":
                    val = _extract_top_numeric_val(row.get("nchans_set"))
                elif col == "sampling_rate":
                    val = _extract_top_numeric_val(row.get("sampling_freqs"))
                else:
                    val = _extract_top_numeric_val(row.get(col))
                arr[i] = val
            miss = np.isnan(arr)
            num_cols[col] = arr
            num_missing[col] = miss
            present = arr[~miss]
            if present.size:
                rng = float(present.max() - present.min())
                num_ranges[col] = rng if rng > 0 else 0.0
            else:
                num_ranges[col] = 0.0

    # Accumulate weighted pairwise distance + weight tracker.
    D = np.zeros((n, n), dtype=np.float32)
    W = np.zeros((n, n), dtype=np.float32)

    for col, kind, weight in specs:
        if kind == "cat":
            arr = cat_cols[col]
            miss = cat_missing[col]
            pair_d = (arr[:, None] != arr[None, :]).astype(np.float32)
            present = (~miss[:, None]) & (~miss[None, :])
        else:
            arr = num_cols[col]
            miss = num_missing[col]
            rng = num_ranges[col]
            if rng <= 0:
                # All present rows share one value → contributes nothing,
                # but the pair still "covers" this feature for normalisation.
                pair_d = np.zeros((n, n), dtype=np.float32)
            else:
                diff = np.abs(arr[:, None] - arr[None, :]) / rng
                pair_d = diff.astype(np.float32)
            present = (~miss[:, None]) & (~miss[None, :])
            # nan minus nan stayed nan in the diff above — clamp to 0 so
            # masking does the right thing.
            pair_d = np.where(present, pair_d, 0)
        D += weight * pair_d
        W += weight * present.astype(np.float32)

    # Normalise by per-pair weight coverage; pairs with zero shared
    # features get a distance of 1 (treated as "unrelated").
    no_overlap = W <= 0
    W = np.where(no_overlap, 1, W)
    D = D / W
    D = np.where(no_overlap, 1.0, D)
    np.fill_diagonal(D, 0.0)

    # Top-K nearest per row, excluding self and entries with zero distance
    # only when there is more than one such entry (perfect-match dupes are
    # still useful to surface).
    related: dict[str, list[str]] = {}
    for i, name in enumerate(names):
        order = np.argsort(D[i], kind="stable")
        out: list[str] = []
        for j in order:
            if j == i:
                continue
            if D[i, j] >= 1.0:  # no shared features
                continue
            out.append(names[j])
            if len(out) >= k:
                break
        related[name] = out
    return related


# ---------------------------------------------------------------------------
# Croissant 1.0 export
# ---------------------------------------------------------------------------
# Croissant is MLCommons' JSON-LD format for machine-readable dataset
# descriptions consumable by PyTorch, TensorFlow, JAX, and HuggingFace. We
# write one .croissant.json file per dataset alongside the .rst output;
# Sphinx copies the result to /_static/dataset_generated/croissant/ via the
# standard static asset pipeline.
#
# Spec: https://docs.mlcommons.org/croissant/docs/croissant-spec.html

CROISSANT_DIR = Path(__file__).parent / "_static" / "dataset_generated" / "croissant"

CROISSANT_CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileObject": "cr:fileObject",
    "fileProperty": "cr:fileProperty",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


def _build_croissant_export(context: Mapping[str, object]) -> dict[str, object]:
    """Build a Croissant 1.0 JSON-LD descriptor for one EEGDash dataset.

    Backed entirely by API fields surfaced in the dataset context. We do
    not synthesize a recordSet (the BIDS file layout already documents
    that) — the export sticks to the Dataset metadata + a few FileObject
    distributions so ML frameworks can discover the canonical S3 bucket
    and the EEGDash records endpoint.
    """
    class_name = str(context.get("class_name", "")).strip()
    dataset_id = str(context.get("dataset_id", "")).strip().lower()
    title = _clean_value(context.get("title")) or class_name
    description = (
        f"{title}. {context.get('modality') or 'EEG'} dataset accessible via "
        f"EEGDash as `{class_name}` with standardized BIDS metadata."
    )

    doi_clean = _normalize_doi(_clean_value(context.get("doi")))
    license_url = _resolve_license_url(_clean_value(context.get("license")))
    page_url = f"https://eegdash.org/api/dataset/eegdash.dataset.{class_name}.html"

    creators = []
    for author in context.get("authors") or []:
        cleaned = str(author).replace("*", "").strip()
        if cleaned:
            creators.append({"@type": "sc:Person", "name": cleaned})

    keywords = ["EEG", "BIDS", "neuroscience"]
    datatypes = context.get("datatypes") or []
    if isinstance(datatypes, list):
        for dt in datatypes:
            dt_s = str(dt).strip().upper()
            if dt_s and dt_s not in keywords:
                keywords.append(dt_s)
    tags = context.get("tags") or {}
    if isinstance(tags, dict):
        for key in ("modality", "pathology", "type"):
            value = _clean_value(tags.get(key))
            if value and value not in keywords:
                keywords.append(value)
    if bool(context.get("hed_annotated")):
        keywords.append("HED")

    distribution: list[dict[str, object]] = []

    # 1. Canonical S3 bucket (OpenNeuro / NEMAR).
    storage = context.get("dataset_storage") or {}
    s3_base = None
    if isinstance(storage, dict):
        s3_base = storage.get("base")
    if s3_base:
        distribution.append(
            {
                "@type": "cr:FileObject",
                "@id": f"{dataset_id}-bids-bucket",
                "name": f"{dataset_id} BIDS bucket",
                "description": (
                    f"Canonical BIDS-formatted dataset bucket for {class_name}."
                ),
                "contentUrl": str(s3_base),
                "encodingFormat": "application/vnd.bids+directory",
            }
        )

    # 2. OpenNeuro web entry (always present for openneuro source).
    openneuro_url = _clean_value(context.get("openneuro_url"))
    if openneuro_url:
        distribution.append(
            {
                "@type": "cr:FileObject",
                "@id": f"{dataset_id}-openneuro",
                "name": f"{dataset_id} on OpenNeuro",
                "description": "OpenNeuro dataset landing page.",
                "contentUrl": openneuro_url,
                "encodingFormat": "text/html",
            }
        )

    # 3. EEGDash MongoDB records endpoint — records describe individual
    # BIDS files, lazily fetchable for ML pipelines.
    records_query = json.dumps({"dataset": dataset_id}, separators=(",", ":"))
    distribution.append(
        {
            "@type": "cr:FileObject",
            "@id": f"{dataset_id}-eegdash-records",
            "name": f"{dataset_id} EEGDash records (live)",
            "description": (
                "Live JSON listing of every recording in this dataset, "
                "with BIDS metadata, channel counts, and S3 storage keys."
            ),
            "contentUrl": (
                "https://data.eegdash.org/api/eegdash/records"
                f"?filter={quote(records_query, safe='')}"
            ),
            "encodingFormat": "application/json",
            "isLiveDataset": True,
        }
    )

    # 4. NEMAR mirror.
    nemar_url = _clean_value(context.get("nemar_url"))
    if nemar_url:
        distribution.append(
            {
                "@type": "cr:FileObject",
                "@id": f"{dataset_id}-nemar",
                "name": f"{dataset_id} on NeMAR",
                "description": "NEMAR dataset landing page with pipeline plots.",
                "contentUrl": nemar_url,
                "encodingFormat": "text/html",
            }
        )

    # 5. HuggingFace mirror under the EEGDash org. Only emit the dataset-
    # specific entry when the probe confirmed the dataset exists; otherwise
    # skip — distributing the bare org page as a "distribution" would
    # confuse consumers.
    hf_info = context.get("huggingface") or {}
    hf_available = bool(hf_info.get("available"))
    hf_url = str(hf_info.get("url") or "") if hf_available else ""
    if hf_available and hf_url:
        distribution.append(
            {
                "@type": "cr:FileObject",
                "@id": f"{dataset_id}-huggingface",
                "name": f"EEGDash/{dataset_id} on Hugging Face",
                "description": (
                    "Hugging Face dataset mirror — ready for "
                    "`datasets.load_dataset(...)` pulls."
                ),
                "contentUrl": hf_url,
                "encodingFormat": "application/vnd.huggingface-dataset+json",
            }
        )

    document: dict[str, object] = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": dataset_id or class_name,
        "alternateName": [class_name],
        "description": description,
        "url": page_url,
        "keywords": keywords,
        "isAccessibleForFree": True,
        "includedInDataCatalog": {
            "@type": "sc:DataCatalog",
            "name": "EEG Dash",
            "url": "https://eegdash.org/",
        },
    }

    if license_url:
        document["license"] = license_url

    if creators:
        document["creator"] = creators

    senior = _clean_value(context.get("senior_author"))
    if senior:
        document["editor"] = {"@type": "sc:Person", "name": senior}

    funding = context.get("funding") or []
    if funding:
        document["funder"] = [
            {"@type": "sc:Organization", "name": str(f).strip()}
            for f in funding
            if str(f).strip()
        ]

    year = _clean_value(context.get("year"))
    if year and year != "—":
        document["datePublished"] = year

    if doi_clean:
        document["identifier"] = f"doi:{doi_clean}"
        same_as = [
            f"https://doi.org/{doi_clean}",
            openneuro_url or page_url,
        ]
        if hf_available and hf_url:
            same_as.append(hf_url)
        document["sameAs"] = same_as
        # Croissant prefers a citeAs property if a DOI is present.
        document["citeAs"] = (
            f"{', '.join(str(a) for a in (context.get('authors') or [])[:5]) or 'EEGDash'} "
            f"({year or 'n.d.'}). {title}. doi:{doi_clean}"
        )

    # Provide a BIDS version when known so consumers can dispatch on schema
    # generation. Falls back to omission rather than guessing.
    bids_version = _clean_value(context.get("bids_version"))
    if bids_version:
        document["version"] = bids_version

    # Created / modified timestamps from the OpenNeuro side (when present)
    # — useful for ML pipelines that cache by mtime.
    created_at = _clean_value(context.get("dataset_created_at"))
    modified_at = _clean_value(context.get("dataset_modified_at"))
    if created_at and not document.get("dateCreated"):
        document["dateCreated"] = created_at[:10]
    if modified_at:
        document["dateModified"] = modified_at[:10]

    # Associated paper DOI (when present) flows into `citation`, separate
    # from the dataset DOI carried by `identifier` / `sameAs`.
    paper_doi = _normalize_doi(_clean_value(context.get("associated_paper_doi")))
    if paper_doi:
        document["citation"] = {
            "@type": "sc:ScholarlyArticle",
            "identifier": f"doi:{paper_doi}",
            "url": f"https://doi.org/{paper_doi}",
        }

    # Tasks → mlcommons measurementTechnique slot so downstream ML
    # discovery surfaces (e.g. HuggingFace's Croissant ingestion) can
    # filter datasets by paradigm.
    tasks = context.get("tasks") or []
    if isinstance(tasks, (list, tuple)) and tasks:
        document["measurementTechnique"] = [
            f"BIDS task: {str(t).strip()}" for t in tasks if str(t).strip()
        ]

    # Contact maintainers — separate from authors. Maps cleanly to
    # schema.org `maintainer`.
    contacts = context.get("contact_info") or []
    if isinstance(contacts, (list, tuple)) and contacts:
        document["maintainer"] = [
            {"@type": "sc:Person", "name": str(c).strip()}
            for c in contacts
            if str(c).strip()
        ]

    if distribution:
        document["distribution"] = distribution

    return document


def _resolve_license_url(license_text: str) -> str:
    """Best-effort SPDX → URL mapping for the Croissant `license` field."""
    if not license_text:
        return ""
    lic = license_text.strip().lower()
    if lic in ("cc0", "cc0 1.0", "cc-0", "cc0-1.0"):
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    if lic.startswith("cc-by-sa") or lic.startswith("cc by-sa"):
        return "https://creativecommons.org/licenses/by-sa/4.0/"
    if lic.startswith("cc-by") or lic.startswith("cc by"):
        return "https://creativecommons.org/licenses/by/4.0/"
    if lic.startswith("mit"):
        return "https://opensource.org/license/mit/"
    if lic.startswith("apache"):
        return "https://www.apache.org/licenses/LICENSE-2.0"
    return license_text


def _write_croissant_export(class_name: str, context: Mapping[str, object]) -> Path:
    """Write the Croissant JSON-LD for ``class_name`` and return its path.

    Writes are atomic via ``os.replace`` so that an interrupted build
    cannot leave a reader observing a half-serialised file.
    """
    CROISSANT_DIR.mkdir(parents=True, exist_ok=True)
    path = CROISSANT_DIR / f"{class_name}.croissant.json"
    document = _build_croissant_export(context)
    payload = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == payload:
                return path
        except OSError:
            pass  # fall through to rewrite
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _format_editorial_colophon_section(context: Mapping[str, object]) -> str:
    """Footer band — typography credit, FAIR exports, DOI."""
    class_name = str(context.get("class_name", "")).strip()
    license_text = _value_or_unknown(_clean_value(context.get("license")), "license")
    doi_clean = _normalize_doi(_clean_value(context.get("doi")))
    doi_html = (
        f'<a href="https://doi.org/{doi_clean}">{doi_clean}</a>'
        if doi_clean
        else '<span style="opacity:.6">DOI not on file</span>'
    )

    croissant_url = (
        f"../../_static/dataset_generated/croissant/{class_name}.croissant.json"
    )
    hf_info = context.get("huggingface") or {}
    hf_url = str(hf_info.get("url") or "https://huggingface.co/EEGDash")
    hf_link_label = (
        "Hugging Face mirror" if hf_info.get("available") else "Hugging Face org"
    )
    block = (
        '<footer class="eegdash-ed-colophon">'
        f"<div>EEGdash · <b>The Dataset Brief — {class_name}</b></div>"
        "<div>FAIR exports · "
        '<a href="#dataset-information">schema.org/Dataset</a> · '
        f'<a href="{croissant_url}" download>Croissant 1.0</a> · '
        f'<a href="{hf_url}">{hf_link_label}</a></div>'
        f"<div>{license_text} · <b>{doi_html}</b></div>"
        "</footer>"
    )
    return _editorial_html(block)


def _format_dataset_meta_section(
    context: Mapping[str, object],
) -> tuple[str, str]:
    """Build per-dataset SEO markup for the dataset page template.

    Returns a ``(og_description_field, meta_directive)`` pair:

    * ``og_description_field`` is a top-of-file field list line
      (``:og:description: ...``) consumed by sphinxext-opengraph. It
      must sit in the same contiguous field list as other directives
      like ``:html_theme.sidebar_secondary.remove:`` — any blank line
      between them would demote it to regular text.
    * ``meta_directive`` is the ``.. meta::`` block with the
      ``:description:`` and ``:keywords:`` fields used by the
      ``<meta name="description">`` / ``<meta name="keywords">`` tags.
    """
    class_name = str(context.get("class_name", "")).strip()
    dataset_title = _collapse_whitespace(
        _clean_value(context.get("title")) or class_name
    )
    modality = _clean_value(context.get("modality")) or "EEG"
    n_subjects = _clean_value(context.get("n_subjects"))
    n_records = _clean_value(context.get("n_records"))
    source = _clean_value(context.get("source")) or "OpenNeuro"

    parts = [f"{modality} dataset"]
    if n_subjects and n_subjects not in ("—", "0"):
        parts.append(f"{n_subjects} subjects")
    if n_records and n_records not in ("—", "0"):
        parts.append(f"{n_records} recordings")
    stats = ", ".join(parts)

    description = (
        f"{dataset_title} — {stats}. Access via EEGDash with standardized "
        f"BIDS metadata. Source: {source}."
    )
    # Single-line for RST directive safety.
    description = _collapse_whitespace(description)

    keywords = ", ".join(
        filter(
            None,
            [
                class_name,
                modality,
                "BIDS",
                "EEG dataset",
                source,
            ],
        )
    )

    og_description_field = f":og:description: {description}"
    meta_directive = (
        f".. meta::\n   :description: {description}\n   :keywords: {keywords}"
    )
    return og_description_field, meta_directive


_LICENSE_URL_MAP: dict[str, str] = {
    # Keyed on the uppercased license string we receive from the dataset
    # registry. Mapping is intentionally narrow — anything not listed
    # falls back to the raw text (still valid per Google, just rendered
    # with a warning). See
    # https://developers.google.com/search/docs/appearance/structured-data/dataset
    "CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC BY": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY": "https://creativecommons.org/licenses/by/4.0/",
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC BY-SA": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-SA": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY-NC": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC-BY-NC": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC BY-NC 4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC-BY-NC 4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC BY-NC-SA": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC-BY-NC-SA": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC BY-NC-SA 4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC-BY-NC-SA 4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC-BY-NC-SA-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "ODC-BY": "https://opendatacommons.org/licenses/by/1-0/",
    "ODC-BY 1.0": "https://opendatacommons.org/licenses/by/1-0/",
    "ODBL": "https://opendatacommons.org/licenses/odbl/1-0/",
    "ODC-ODBL": "https://opendatacommons.org/licenses/odbl/1-0/",
    "PDDL": "https://opendatacommons.org/licenses/pddl/1-0/",
    "BSD-3-CLAUSE": "https://opensource.org/licenses/BSD-3-Clause",
    "MIT": "https://opensource.org/licenses/MIT",
    "APACHE-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "PUBLIC DOMAIN": "https://creativecommons.org/publicdomain/mark/1.0/",
}


def _license_text_to_url(text: str) -> str | None:
    """Map a free-form license string to a canonical URL if we can.

    Google's Rich Results validator prefers license URLs over names.
    We normalise by stripping whitespace and uppercasing; everything
    we can't match falls through so the caller keeps the raw text.
    """
    if not text:
        return None
    key = re.sub(r"\s+", " ", text.strip()).upper()
    return _LICENSE_URL_MAP.get(key)


def _format_dataset_jsonld_section(
    class_name: str, context: Mapping[str, object]
) -> str:
    """Emit a ``<script type="application/ld+json">`` Dataset block.

    The resulting block is placed into the page body (valid per the
    JSON-LD spec — Google Dataset Search reads either head or body).
    """
    dataset_upper = str(context.get("dataset_upper", "")).strip() or class_name
    title = _collapse_whitespace(_clean_value(context.get("title")) or dataset_upper)
    doi_clean = _normalize_doi(_clean_value(context.get("doi")))
    authors = [a for a in (context.get("authors") or []) if a]
    license_text = _clean_value(context.get("license"))
    year = _clean_value(context.get("year"))
    modality = _clean_value(context.get("modality")) or "EEG"
    source_url = _clean_value(context.get("source_url"))
    openneuro_url = str(context.get("openneuro_url", "")).strip()
    nemar_url = str(context.get("nemar_url", "")).strip()

    page_url = f"https://eegdash.org/api/dataset/eegdash.dataset.{class_name}.html"

    # Skip the leading title when it duplicates `dataset_upper` — the
    # trailing "...as ``class_name``..." already names the dataset, and
    # "{title}. {modality} dataset accessible via EEGDash as ``{title}``"
    # reads "ABSEQMEG. EEG dataset accessible via EEGDash as ``ABSEQMEG``"
    # which is triple-redundant for the common case where title is
    # just the uppercased class name.
    title_differs_from_class = title and title.strip().upper() not in {
        dataset_upper.strip().upper(),
        class_name.strip().upper(),
    }
    description_parts = [f"{title}."] if title_differs_from_class else []
    description_parts.append(
        f"{modality} dataset accessible via EEGDash as "
        f"``{class_name}`` with standardized BIDS metadata."
    )
    description = " ".join(description_parts)

    # Order-preserving dedupe: when `modality` is already "EEG" or "MEG",
    # it would otherwise duplicate entries in the published JSON-LD.
    keywords = list(dict.fromkeys([modality, "BIDS", "neuroscience", "EEG", "MEG"]))
    jsonld: dict[str, object] = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": title or dataset_upper,
        "alternateName": dataset_upper,
        "description": description,
        "url": page_url,
        "keywords": keywords,
        "isAccessibleForFree": True,
        "includedInDataCatalog": {
            "@type": "DataCatalog",
            "name": "EEG Dash",
            "url": "https://eegdash.org/",
        },
    }

    if doi_clean:
        jsonld["identifier"] = f"doi:{doi_clean}"
        jsonld["sameAs"] = f"https://doi.org/{doi_clean}"

    same_as_urls = [u for u in (openneuro_url, nemar_url, source_url) if u]
    if same_as_urls:
        existing_same_as = jsonld.get("sameAs")
        all_same_as = (
            [existing_same_as] if isinstance(existing_same_as, str) else []
        ) + same_as_urls
        # Deduplicate while preserving order
        seen: set[str] = set()
        jsonld["sameAs"] = [u for u in all_same_as if not (u in seen or seen.add(u))]

    if authors:
        jsonld["creator"] = [{"@type": "Person", "name": a} for a in authors]

    if license_text:
        jsonld["license"] = _license_text_to_url(license_text) or license_text

    # schema.org/Dataset expects `datePublished` as an ISO 8601 date.
    # Google's Rich Results validator emits a warning for bare years
    # like "2024". Pin to January 1st when we only have a year.
    if year and re.fullmatch(r"\d{4}", year):
        jsonld["datePublished"] = f"{year}-01-01"

    payload = json.dumps(jsonld, ensure_ascii=False, separators=(",", ":"))
    # HTML-parser safety: the only byte sequence that can terminate a
    # `<script>` block is `</` followed by any ASCII letter. `json.dumps`
    # escapes `"` and control chars, but lets `/` through, so a dataset
    # author whose metadata happens to contain `</script>` (or even
    # `</anything`) would escape the script element and inject arbitrary
    # HTML into the page. Escaping `</` → `<\/` is valid JSON *and* safe
    # inside `<script>` per the HTML spec.
    payload = payload.replace("</", "<\\/")

    return f'.. raw:: html\n\n   <script type="application/ld+json">{payload}</script>'


def _cleanup_stale_dataset_pages(dataset_dir: Path, expected: set[Path]) -> None:
    for path in dataset_dir.glob("eegdash.dataset.DS*.rst"):
        if path in expected:
            continue
        try:
            if not path.read_text(encoding="utf-8").startswith(AUTOGEN_NOTICE):
                continue
        except OSError:
            continue
        path.unlink()


def _process_dataset_item(
    name: str,
    dataset_dir: Path,
    row: Mapping[str, str] | None,
    srcdir: Path,
    related: Sequence[str] = (),
    related_meta: Sequence[Mapping[str, object]] = (),
) -> Path:
    context = _build_dataset_context(name, row)

    # Build a descriptive page title for better SERP snippets:
    #   "ABSeqMEG: EEG dataset, 20 subjects"  instead of bare  "ABSeqMEG"
    modality = _clean_value(context.get("modality")) or "EEG"
    n_sub = _clean_value(context.get("n_subjects"))
    title_parts = [name]
    suffix_parts = [f"{modality} dataset"]
    if n_sub and n_sub not in ("—", "0"):
        suffix_parts.append(f"{n_sub} subjects")
    title_parts.append(", ".join(suffix_parts))
    title = ": ".join(title_parts)
    dataset_id = str(context.get("dataset_id", ""))
    dataset_title = str(context.get("title", ""))
    og_description_field, meta_section = _format_dataset_meta_section(context)
    page_content = DATASET_PAGE_TEMPLATE.format(
        notice=AUTOGEN_NOTICE,
        title=title,
        underline="=" * len(title),
        og_description_field=og_description_field,
        meta_section=meta_section,
        jsonld_section=_format_dataset_jsonld_section(name, context),
        hero_section=_format_hero_section(context),
        readme_section=_format_readme_section(context),
        quickstart_section=_format_quickstart_section(context),
        recording_stats_section=_format_recording_stats_section(context),
        nemar_analysis_section=_format_nemar_analysis_section(context),
        explorer_section=_format_explorer_section(name, context),
        api_section=_format_api_section(name, context),
        see_also_section=_format_see_also_section(dataset_id, name, related),
        feedback_section=_format_feedback_section(dataset_id, dataset_title),
        # Combined electrodes + signal-preview figure block (replaces the
        # old separate "Electrode Layout" and "Signal Preview" sections).
        electrodes_traces_pair=_format_electrodes_traces_pair(name, context),
        # Bibtex + Dataset Information table relocated into a single
        # collapsible dropdown so the field-card rail stays the single
        # source of truth for the data card.
        dataset_info_dropdown_section=_format_dataset_info_dropdown(context),
        # Editorial Brief — re-skin chrome lifted from v1-editorial-v2.html
        editorial_kicker_section=_format_editorial_kicker_section(context),
        editorial_fieldcard_section=_format_editorial_fieldcard_section(context),
        editorial_layers_section=_format_editorial_layers_section(context),
        editorial_secnum_quickstart=_editorial_secnum(1, "Access · Get started"),
        editorial_secnum_about=_editorial_secnum(2, "Study · The README"),
        editorial_secnum_info=_editorial_secnum(3, "Cohort · Participants"),
        editorial_secnum_tech=_editorial_secnum(4, "Signal · Electrodes & trace"),
        editorial_secnum_manifest=_editorial_secnum(5, "Manifest · BIDS tree"),
        editorial_secnum_api=_editorial_secnum(6, "API · Programmatic access"),
        editorial_caveat_section=_format_editorial_caveat_section(context),
        editorial_access_modes_section=_format_editorial_access_modes_section(context),
        editorial_examples_gallery=_format_editorial_examples_gallery(context),
        editorial_provenance_section=_format_editorial_provenance_section(context),
        editorial_footnotes_section=_format_editorial_footnotes_section(
            context, related, related_meta
        ),
        editorial_colophon_section=_format_editorial_colophon_section(context),
    )
    # Keep the file name with full prefix for URL stability
    page_path = dataset_dir / f"eegdash.dataset.{name}.rst"
    if _write_if_changed(page_path, page_content):
        rel = page_path.relative_to(srcdir)
        LOGGER.info("[dataset-docs] Updated %s", rel)

    # Croissant JSON-LD export — written once per dataset under _static so
    # Sphinx serves it at /_static/dataset_generated/croissant/<name>.croissant.json
    # alongside the editorial dataset page.
    try:
        _write_croissant_export(name, context)
    except Exception as exc:  # never let Croissant block the build
        LOGGER.warning("[dataset-docs] Croissant export failed for %s: %s", name, exc)

    return page_path


def _generate_dataset_docs(app) -> None:
    dataset_dir = Path(app.srcdir) / "api" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = _iter_dataset_classes()
    dataset_rows = _load_dataset_rows(dataset_names)

    # Group datasets by source
    datasets_by_source = defaultdict(list)
    for name in dataset_names:
        row = dataset_rows.get(name) or {}
        source = _clean_value(row.get("source")) or "Other"
        datasets_by_source[source].append(name)

    # Generate group pages
    group_toctree_entries = []
    for source, names in datasets_by_source.items():
        safe_source = "".join(c if c.isalnum() else "_" for c in source).lower()
        if not safe_source:
            safe_source = "other"
        group_filename = f"source_{safe_source}.rst"
        group_path = dataset_dir / group_filename

        group_toctree = _render_toctree_entries(sorted(names))
        # Title case the source for the header
        source_title = source.title() if source.islower() else source

        group_content = f"""{AUTOGEN_NOTICE}
{source_title} Datasets
{"=" * (len(source_title) + 9)}

.. toctree::
   :maxdepth: 1

{group_toctree}
"""
        if _write_if_changed(group_path, group_content):
            LOGGER.info("[dataset-docs] Updated group page %s", group_filename)

        group_toctree_entries.append(
            group_filename
        )  # Just filename, relative to api/dataset

    toctree_entries_str = "\n".join(
        f"   {entry}" for entry in sorted(group_toctree_entries)
    )

    experiment_rows = _render_experiment_rows(_load_experiment_counts(dataset_names))
    index_content = DATASET_INDEX_TEMPLATE.format(
        notice=AUTOGEN_NOTICE,
        dataset_count=len(dataset_names),
        experiment_rows=experiment_rows,
        toctree_entries=toctree_entries_str,
    )

    index_path = dataset_dir / "api_dataset.rst"
    if _write_if_changed(index_path, index_content):
        LOGGER.info("[dataset-docs] Updated %s", index_path.relative_to(app.srcdir))

    base_content = BASE_DATASET_TEMPLATE.format(notice=AUTOGEN_NOTICE)
    base_path = dataset_dir / "eegdash.dataset.EEGChallengeDataset.rst"
    if _write_if_changed(base_path, base_content):
        LOGGER.info("[dataset-docs] Updated %s", base_path.relative_to(app.srcdir))

    primary_content = PRIMARY_DATASET_TEMPLATE.format(notice=AUTOGEN_NOTICE)
    primary_path = dataset_dir / "eegdash.EEGDashDataset.rst"
    if _write_if_changed(primary_path, primary_content):
        LOGGER.info("[dataset-docs] Updated %s", primary_path.relative_to(app.srcdir))

    # Compute related-datasets kNN index using Gower distance over mixed
    # categorical/numerical metadata (modality, paradigm, cohort, sample
    # rate, channel count, etc.). Falls back to a modality-only grouping
    # if numpy is unavailable or every dataset is missing features.
    related_index = _build_related_index_gower(dataset_names, dataset_rows, k=6)

    # Fallback modality index used when Gower returned no neighbours for a
    # given dataset (e.g. it had every feature missing). Read-only after
    # construction, so thread-safe for the parallel generator below.
    datasets_by_modality: dict[str, list[str]] = defaultdict(list)
    for name in dataset_names:
        row = dataset_rows.get(name) or {}
        mod = _clean_value(row.get("record_modality")) or _clean_value(
            row.get("modality of exp")
        )
        if mod:
            datasets_by_modality[mod.lower()].append(name)

    def _related_for(name: str) -> list[str]:
        """Return up to 6 related datasets by Gower kNN, else by modality."""
        nn = related_index.get(name) or []
        if nn:
            return nn[:6]
        row = dataset_rows.get(name) or {}
        mod = _clean_value(row.get("record_modality")) or _clean_value(
            row.get("modality of exp")
        )
        if not mod:
            return []
        siblings = datasets_by_modality.get(mod.lower(), [])
        return [s for s in siblings if s != name][:6]

    def _primary_author_key(value: object) -> str:
        """Extract the surname token from an ``author_year`` string."""
        text = _clean_value(value)
        m = re.match(r"^([A-Za-z][A-Za-z\-']*)", text)
        return m.group(1).lower() if m else ""

    def _related_meta_for(name: str) -> list[dict[str, object]]:
        """Return per-neighbour metadata for the footnotes "Related" card."""
        me_key = _primary_author_key((dataset_rows.get(name) or {}).get("author_year"))
        out: list[dict[str, object]] = []
        for nb in _related_for(name):
            nrow = dataset_rows.get(nb) or {}
            nb_key = _primary_author_key(nrow.get("author_year"))
            out.append(
                {
                    "name": nb,
                    "modality": _clean_value(nrow.get("record_modality"))
                    or _clean_value(nrow.get("modality of exp")),
                    "n_subjects": _clean_value(nrow.get("n_subjects")),
                    "same_authors": bool(me_key and nb_key and me_key == nb_key),
                }
            )
        return out

    generated_paths: set[Path] = set()
    srcdir = Path(app.srcdir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _process_dataset_item,
                name,
                dataset_dir,
                dataset_rows.get(name),
                srcdir,
                _related_for(name),
                _related_meta_for(name),
            ): name
            for name in dataset_names
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                path = future.result()
                generated_paths.add(path)
            except Exception as exc:
                LOGGER.warning(f"Failed to generate doc for {futures[future]}: {exc}")

    # Add group files to expected paths so they aren't deleted
    for source in datasets_by_source:
        safe_source = "".join(c if c.isalnum() else "_" for c in source).lower()
        if not safe_source:
            safe_source = "other"
        generated_paths.add(dataset_dir / f"source_{safe_source}.rst")

    _cleanup_stale_dataset_pages(dataset_dir, generated_paths)

    # Remove legacy pages that used the short filename convention
    for legacy in dataset_dir.glob("DS*.rst"):
        try:
            legacy.unlink()
        except OSError:
            continue


def _split_tokens(value: str | None) -> set[str]:
    tokens: set[str] = set()
    if not value:
        return tokens
    for part in value.split(","):
        cleaned = part.strip()
        if cleaned:
            tokens.add(cleaned)
    return tokens


def _compute_dataset_counter_defaults() -> dict[str, int]:
    # 1. Try to load from the generated JSON (produced by prepare_summary_tables.py)
    # Note: prepare_summary_tables.py runs before sphinx-build in the Makefile
    stats_path = (
        Path(__file__).parent / "_static" / "dataset_generated" / "summary_stats.json"
    )
    if stats_path.exists():
        try:
            with open(stats_path, "r") as f:
                data = json.load(f)
            return {
                "datasets": data.get("datasets_total", 0),
                "subjects": data.get("subjects_total", 0),
                "recording": data.get("recording_total", 0),
                "duration_hours": data.get("duration_hours", 0),
                "modalities": data.get("modalities_total", 0),
                "sources": data.get("sources_total", 0),
            }
        except Exception:
            pass

    # 2. Fallback to legacy CSV logic
    csv_path = Path(importlib.import_module("eegdash.dataset").__file__).with_name(
        "dataset_summary.csv"
    )
    if not csv_path.exists():
        return {}

    dataset_ids: set[str] = set()
    modalities: set[str] = set()
    sources: set[str] = set()
    subject_total = 0

    with csv_path.open(encoding="utf-8") as handle:
        filtered = (
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        )
        reader = csv.DictReader(filtered)
        for row in reader:
            dataset = (row.get("dataset") or row.get("Dataset") or "").strip()
            if dataset:
                dataset_ids.add(dataset)

            try:
                subject_total += int(float(row.get("n_subjects", "0") or 0))
            except (TypeError, ValueError):
                pass

            modalities.update(_split_tokens(row.get("record_modality")))
            sources.add((row.get("source") or "unknown").strip())

    return {
        "datasets": len(dataset_ids),
        "subjects": subject_total,
        "recording": 0,
        "modalities": len(modalities),
        "sources": len(sources),
    }


_DATASET_COUNTER_DEFAULTS = _compute_dataset_counter_defaults()


def _format_counter(key: str) -> str:
    value = _DATASET_COUNTER_DEFAULTS.get(key, 0)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            return f"{value:,.2f}"
        return f"{int(value):,}"
    return str(value)


_DATASET_COUNTER_PLACEHOLDERS = {
    "|datasets_total|": _format_counter("datasets"),
    "|subjects_total|": _format_counter("subjects"),
    "|recording_total|": _format_counter("recording"),
    "|duration_hours|": _format_counter("duration_hours"),
    "|modalities_total|": _format_counter("modalities"),
    "|sources_total|": _format_counter("sources"),
}


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


def _inject_counter_values(app, docname, source) -> None:
    if docname != "dataset_summary":
        return

    text = source[0]
    for token, value in _DATASET_COUNTER_PLACEHOLDERS.items():
        text = text.replace(token, value)
    source[0] = text


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
    # (``_cap_descriptions_hook`` below). Doing it here would miss any
    # descriptions that sphinxext-opengraph adds later in the same
    # ``html-page-context`` phase — its handler runs after ours at the
    # default priority.


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

    def _swap(m: re.Match) -> str:
        original = m.group(0)
        return original.replace(m.group("v"), escaped)

    for pattern in _META_DESC_PATTERNS:
        metatags = pattern.sub(_swap, metatags)
    return metatags


def _cap_descriptions_in_metatags(metatags: str, limit: int = 155) -> str:
    """Cap every description / og:description content value in
    ``metatags`` at ``limit`` chars. HTML-entity-decodes before
    comparing so the visible budget is what the SERP renders.
    """
    if not metatags:
        return metatags

    def _trim(m: re.Match) -> str:
        value = m.group("v")
        decoded = html.unescape(value)
        if len(decoded) <= limit:
            return m.group(0)
        capped = _cap_meta_description(decoded, limit=limit)
        return m.group(0).replace(value, html.escape(capped, quote=True))

    for pattern in _META_DESC_PATTERNS:
        metatags = pattern.sub(_trim, metatags)
    return metatags


def _page_still_lacks_description(context) -> bool:
    """True if neither a `.. meta::` description nor an auto-page hook
    has produced a `<meta name="description" …>` tag for this page.
    """
    current = context.get("metatags") or ""
    return 'name="description"' not in current


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


def _extract_sg_thumbnail_block(child_index_path: Path) -> str | None:
    """Return the sphinx-gallery thumbnail-grid block from a child gallery.

    Sphinx-gallery emits a self-contained card grid in each leaf
    ``index.rst`` between two ``raw:: html`` blocks: it opens with
    ``<div class="sphx-glr-thumbnails">`` and closes with the matching
    ``</div>``. Inside, every tutorial card is its own
    ``<div class="sphx-glr-thumbcontainer">`` with thumbnail image,
    cross-reference link, and caption.

    The grid block is everything from the first ``.. raw:: html`` line
    that opens the ``<div class="sphx-glr-thumbnails">`` container up to
    and including the closing ``raw:: html`` block whose payload is
    ``</div>`` and that is followed by a non-card structural element
    (download footer or hidden toctree). Returns the raw text ready to
    be inlined under any heading; returns ``None`` if the marker can't
    be found (for example, the gallery is empty).
    """
    try:
        text = child_index_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    open_marker = '<div class="sphx-glr-thumbnails">'
    open_idx = text.find(open_marker)
    if open_idx < 0:
        return None
    # Walk back to the directive header for the opening raw block so we
    # capture the whole ``.. raw:: html\n\n    <div...>`` chunk.
    block_start = text.rfind(".. raw:: html", 0, open_idx)
    if block_start < 0:
        return None
    # The grid ends at the ``.. toctree::`` (hidden) that sphinx-gallery
    # appends right after closing ``</div>``. That toctree is content we
    # render via our own toctree, so cut the slice just before it.
    toctree_idx = text.find(".. toctree::", open_idx)
    if toctree_idx < 0:
        return None
    return text[block_start:toctree_idx].rstrip() + "\n"


def _write_auto_examples_root_index(app):
    """Write a top-level ``generated/auto_examples/index.rst`` aggregator.

    Sphinx-gallery is configured with multiple gallery roots (one per
    tutorial category, plus how-to / applied / eeg2025 / hpc / dev_scripts).
    Each emits its own ``index.rst`` containing a card grid of tutorial
    thumbnails. This hook reads those per-leaf indexes and stitches them
    into a single parent page that mimics the SPDLearn theory aggregator:
    a hero intro, a "how to read this gallery" callout, and one section
    per gallery root with its full thumbnail card grid surfaced inline.

    Runs at priority 600 so that sphinx-gallery's own
    ``generate_gallery_rst`` (default priority 500) has already run and
    materialised every per-leaf ``index.rst``. ``dev_scripts`` is kept
    in the build (so internal links still resolve via a hidden toctree)
    but is omitted from the visible card grid: the plan calls it an
    internal-only catalogue.
    """
    src_root = Path(app.srcdir) / "generated" / "auto_examples"
    out_path = src_root / "index.rst"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-section metadata: gallery dir (relative to ``generated/auto_examples``),
    # H3 title, intro paragraph. Pulled from the per-folder README and the
    # tutorial restructure plan so the wording stays in lockstep with the
    # source-of-truth roster.
    tutorial_sections = [
        (
            "tutorials/00_start_here",
            "Start Here",
            "Difficulty 1. Three short lessons that take you from a fresh "
            "install to a working PyTorch ``DataLoader`` over real EEG "
            "records: find datasets and records, load one recording and "
            "inspect it, then turn an ``EEGDashDataset`` into windows and "
            "a dataloader. CPU-only, each runs in under a few minutes.",
        ),
        (
            "tutorials/10_core_workflow",
            "Core Decoding Workflow",
            "Difficulty 1-2. The canonical EEG decoding pipeline in four "
            "lessons: preprocess and window, split without subject leakage, "
            "train a baseline against chance, and persist prepared data for "
            "reuse. The leakage-safe split lesson is "
            "the rubric anchor for E3.27 invariants and Cisotto and Chicco "
            "2024's evaluation guidance.",
        ),
        (
            "tutorials/20_event_related",
            "Event-Related Decoding",
            "Difficulty 2. Two lessons that decode labels coming from "
            "events and annotations rather than continuous state: a P3 "
            "target-versus-standard classifier on a visual oddball "
            "paradigm, then the auditory oddball framed as a contrast "
            "with the visual case.",
        ),
        (
            "tutorials/30_resting_state",
            "Resting-State and State Decoding",
            "Difficulty 1. The canonical beginner decoding lesson: "
            "eyes-open versus eyes-closed classification on resting-state "
            "EEG, decoded from alpha-rhythm differences with a band-power "
            "baseline.",
        ),
        (
            "tutorials/40_features",
            "Feature Engineering",
            "Difficulty 1-2. EEGDash's feature extraction package as a "
            "first-class option, not an afterthought to deep learning. "
            "Three lessons cover feature tables from windows, preprocessor "
            "and dependency trees that avoid recomputation, and a "
            "scikit-learn / LightGBM baseline straight from the feature "
            "table.",
        ),
        (
            "tutorials/50_evaluation",
            "Evaluation and Benchmarking",
            "Difficulty 2-3. Five lessons that treat decoding evaluation "
            "as a core skill, drawing on MOABB (Chevallier, Aristimunha "
            "et al. 2024). Builds from a single split toward "
            "benchmark-grade pipeline comparison: within-subject, "
            "cross-subject, cross-session, learning curves, and a paired "
            "Wilcoxon comparison of two pipelines.",
        ),
        (
            "tutorials/70_transfer_foundation",
            "Transfer, Foundation Models, and EEG2025",
            "Difficulty 3. Four advanced lessons on transfer learning "
            "and foundation-model fine-tuning, framed around the EEG2025 "
            "Foundation Challenge: ``EEGChallengeDataset`` basics, "
            "cross-task transfer (Challenge 1), subject-invariant "
            "p-factor regression (Challenge 2), and fine-tuning a "
            "Braindecode pretrained model. Builds on Schirrmeister et al. 2017.",
        ),
    ]
    leaf_sections = [
        (
            "how_to",
            "How-to recipes",
            "Task-focused snippets that assume you already know the "
            "basics: how to download a dataset, run preprocessing on "
            "SLURM, parallelize feature extraction, use the HPC cache, "
            "and work offline. Each guide answers a single question; "
            "cross-link with the HPC track when relevant.",
        ),
        (
            "applied",
            "Applied research projects",
            "Project-style examples that target a concrete scientific "
            "question -- age regression, p-factor prediction, sex "
            "classification, P300 transfer, clinical-catalog summary -- "
            "with realistic data sizes, runtimes, and limitations. Treat "
            "them as starting points, not prescriptive recipes.",
        ),
        (
            "eeg2025",
            "EEG2025 Foundation Challenge",
            "End-to-end pipelines for the two EEG2025 Foundation "
            "Challenge tracks: cross-task transfer learning (passive to "
            "active), and subject-invariant representations for clinical "
            "factor prediction. Pre-trained weights ship alongside each "
            "tutorial.",
        ),
        (
            "hpc",
            "High-performance computing",
            "Reference setup for running EEGDash on shared HPC clusters: "
            "SLURM submission scripts (CPU and GPU), a Dockerfile, and a "
            "tutorial showing how to combine the on-disk cache with batch "
            "scheduling for an eyes-open / eyes-closed run.",
        ),
    ]

    def _section_block(rel_dir: str, title: str, intro: str, level: str) -> str:
        """Build one section: heading + intro paragraph + thumbnail grid."""
        underline = level * len(title)
        index_path = src_root / rel_dir / "index.rst"
        thumb_block = _extract_sg_thumbnail_block(index_path)
        if thumb_block is None:
            # Defensive fallback: no card grid available, defer to a
            # plain reference link so the section still renders.
            log = logging.getLogger(__name__)
            log.warning(
                "auto_examples aggregator: no thumbnail grid in %s; "
                "falling back to bare link",
                index_path,
            )
            thumb_block = (
                f":doc:`Browse {title} </generated/auto_examples/{rel_dir}/index>`\n"
            )
        return f"{title}\n{underline}\n\n{intro}\n\n{thumb_block}\n"

    parts: list[str] = [
        ":orphan:\n",
        ".. _sphx_glr_generated_auto_examples:\n",
        "Examples gallery",
        "================",
        "",
        "The EEGDash gallery is the runnable, narrative half of the docs: "
        "the **Concepts** chapter explains *why* a decision matters, the "
        "API reference enumerates every public symbol, and the gallery "
        "you're reading shows the choices in motion against real BIDS-"
        "curated EEG records. Every script under ``examples/`` is a "
        "sphinx-gallery tutorial -- meaning it executes top to bottom on "
        "every documentation build, and the captured first figure is the "
        "thumbnail you see below.",
        "",
        "The intended path: read the curated **Tutorials** in order, dip "
        "into **How-to recipes** when you have a specific question, then "
        "scale up using the **Applied research projects**, the **EEG2025 "
        "Foundation Challenge** pipelines, and the **High-performance "
        "computing** track.",
        "",
        ".. admonition:: How to read this gallery",
        "   :class: tip eegdash-gallery-howto",
        "",
        "   - **Reading order.** Tutorials are sorted by category and "
        "numbered (``plot_00_*``, ``plot_10_*``, ...). Inside a category "
        "they're sequenced beginner-first; the file numbers are the "
        "intended path.",
        "   - **Cards show the captured first figure.** Sphinx-gallery "
        "stores the first ``matplotlib`` figure as the thumbnail, so the "
        "card preview is the literal output of running the script. A "
        "branded fallback is shown when the tutorial produces no figure.",
        "   - **Difficulty.** Each section header states the difficulty "
        "range (1 = absolute beginner, 3 = advanced / foundation-model "
        "tier).",
        "",
        "Tutorials (curated learning path)",
        "---------------------------------",
        "",
        "Seven categories, ordered the way we would teach them: install, "
        "load, decode events, decode state, engineer features, evaluate "
        "rigorously, then scale to transfer and foundation models.",
        "",
        "Choose your path",
        "~~~~~~~~~~~~~~~~",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 30 35 35",
        "",
        "   * - Your goal",
        "     - Start with",
        "     - Then read",
        "   * - **Load my first dataset**",
        "     - :doc:`tutorials/00_start_here/index`",
        "     - :doc:`tutorials/10_core_workflow/index`",
        "   * - **Train a classifier safely**",
        "     - :doc:`tutorials/10_core_workflow/index`",
        "     - :doc:`tutorials/50_evaluation/index`",
        "   * - **Extract classical features**",
        "     - :doc:`tutorials/40_features/index`",
        "     - :doc:`how_to/index`",
        "   * - **Run on a cluster**",
        "     - :doc:`how_to/index`",
        "     - :doc:`hpc/index`",
        "   * - **Join EEG2025**",
        "     - :doc:`tutorials/70_transfer_foundation/index`",
        "     - :doc:`eeg2025/index`",
        "",
        ".. grid:: 1 2 2 4",
        "   :gutter: 3",
        "",
        "   .. grid-item-card:: 🚀 Learn the basics",
        "      :link: tutorials/00_start_here/index",
        "      :link-type: doc",
        "",
        "      Start with the absolute beginner tutorials.",
        "",
        "   .. grid-item-card:: 🔬 Run an applied project",
        "      :link: applied/index",
        "      :link-type: doc",
        "",
        "      Dive into real-world research case studies.",
        "",
        "   .. grid-item-card:: ⚡ Scale on HPC",
        "      :link: hpc/index",
        "      :link-type: doc",
        "",
        "      Move from local scripts to cluster-wide jobs.",
        "",
        "   .. grid-item-card:: 🏆 Join EEG2025",
        "      :link: eeg2025/index",
        "      :link-type: doc",
        "",
        "      Enter the official Foundation Challenge.",
        "",
    ]
    for rel, title, intro in tutorial_sections:
        parts.append(_section_block(rel, title, intro, level="~"))

    for rel, title, intro in leaf_sections:
        parts.append(_section_block(rel, title, intro, level="-"))

    # Toctrees keep the navigation tree wired up so individual tutorials
    # remain reachable from the sidebar. They render hidden because the
    # visible content is the card grid above; a visible toctree on top
    # of cards would duplicate the listing as a bullet list.
    parts.append(".. toctree::")
    parts.append("   :hidden:")
    parts.append("   :caption: Tutorials (curated learning path)")
    parts.append("")
    for rel, _title, _intro in tutorial_sections:
        parts.append(f"   {rel}/index")
    parts.append("")
    parts.append(".. toctree::")
    parts.append("   :hidden:")
    parts.append("   :caption: Recipes and applied work")
    parts.append("")
    for rel, _title, _intro in leaf_sections:
        parts.append(f"   {rel}/index")
    parts.append("")
    # ``dev_scripts`` is kept in the build (linked under a hidden toctree
    # so internal references stay valid) but is intentionally absent from
    # the visible gallery -- it's an internal debugging catalogue, not
    # public-facing tutorial content.
    parts.append(".. toctree::")
    parts.append("   :hidden:")
    parts.append("")
    parts.append("   dev_scripts/index")
    parts.append("")

    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def setup(app):
    """Create the back-references directory and setup Sphinx events."""
    backreferences_dir = os.path.join(
        app.srcdir, sphinx_gallery_conf["backreferences_dir"]
    )
    if not os.path.exists(backreferences_dir):
        os.makedirs(backreferences_dir)

    app.connect("builder-inited", _assert_dataset_table_inlines_datatables)
    app.connect("builder-inited", _generate_dataset_docs)
    # Run after sphinx-gallery's `generate_gallery_rst` (priority 500) so
    # the per-leaf index files exist before we link to them.
    app.connect("builder-inited", _write_auto_examples_root_index, priority=600)
    app.connect("build-finished", _copy_dataset_summary)
    # Align sitemap homepage entry with the canonical emitted in
    # `_inject_seo_context`. Must run after `sphinx-sitemap` writes
    # the file; using `build-finished` is the safest hook for that.
    app.connect("build-finished", _rewrite_sitemap_index)
    app.connect("source-read", _inject_counter_values)
    app.connect("html-page-context", _inject_seo_context)
    # Must run last — see docstring on ``_cap_descriptions_hook``.
    app.connect("html-page-context", _cap_descriptions_hook, priority=900)


# Configure sitemap URL format (omit .html where possible)
sitemap_url_scheme = "{link}"

# Exclude low-value / auto-generated pages that dilute the index.
sitemap_excludes = [
    # `index.html` is NOT excluded here: with the default
    # `sphinx-sitemap` URL format, excluding it drops the homepage
    # from the sitemap entirely (no `/` entry is emitted in its place).
    # Duplicate URLs (`/` vs `/index.html`) are instead deduped by the
    # canonical-link override in `_inject_seo_context`.
    "genindex.html",
    "search.html",
    "sg_execution_times.html",
    "sg_api_usage.html",
    "*/sg_execution_times.html",
    # Dataset-summary chart fragments (`dataset_summary/table.html`,
    # `.../treemap.html`, etc.). These are `.. include::` sources that
    # Sphinx builds as their own pages as a side-effect; the useful
    # rendered output is `dataset_summary.html`. Keeping them in the
    # sitemap would dilute crawl budget and Ahrefs was flagging them as
    # orphan pages. Paired with a noindex meta tag below so external
    # backlinks don't bring them into the search index anyway.
    "dataset_summary/*.html",
]
# Sphinx-gallery tutorial pages stay in the sitemap — they're narrative,
# keyword-rich content that matches our library-discoverability goal;
# excluding them was the source of Ahrefs' "indexable page not in
# sitemap" warning on `generated/auto_examples/hpc/tutorial_eoec.html`.

# Emit <lastmod> per URL so crawlers can prioritise recently updated pages.
sitemap_show_lastmod = True

# Copy button configuration: strip common interactive prompts when copying
copybutton_prompt_text = r">>> |\\$ |# "
copybutton_prompt_is_regexp = True
