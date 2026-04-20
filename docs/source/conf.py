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
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import quote

from sphinx.util import logging
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

sys.path.insert(0, os.path.abspath(".."))
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
]

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
    "generated/auto_examples/core/*": [],
    "generated/auto_examples/tutorials/*": [],
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
sphinx_gallery_conf = {
    "examples_dirs": [f"{EX_DIR}"],
    "gallery_dirs": ["generated/auto_examples"],
    # Execute examples by default for CI builds; use html-noplot target for local fast builds
    "plot_gallery": True,
    # Don't fail the build when examples error (e.g. missing cache, API down).
    # Failed examples show a traceback in the gallery instead of crashing CI.
    "abort_on_example_error": False,
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
    "nested_sections": False,
    "backreferences_dir": "gen_modules/backreferences",
    "inspect_global_variables": True,
    "show_memory": False,
    "show_api_usage": True,
    "doc_module": ("eegdash", "numpy", "scipy", "matplotlib"),
    "reference_url": {"eegdash": None},
    "filename_pattern": r"/(?:plot|tutorial)_(?!_).*\.py",
    "matplotlib_animations": True,
    "reset_modules": ("matplotlib", "seaborn"),
    "first_notebook_cell": (
        "# For tips on running notebooks in Google Colab:\n"
        "# `pip install eegdash`\n"
        "%matplotlib inline"
    ),
    "subsection_order": ExplicitOrder([f"{EX_DIR}/core", "*"]),
    "within_subsection_order": FileNameSortKey,
}

# -- Custom Setup Function to fix the error -----------------------------------


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
CLONE_ROOT = REPO_ROOT / "ingestions" / "clone"
_DATASET_DETAILS_CACHE: dict[str, dict[str, object]] = {}
_DATASET_SUMMARY_CACHE = None


def _should_use_api_summary() -> bool:
    # Always try API first; set EEGDASH_NO_API=1 to disable
    return not bool(os.environ.get("EEGDASH_NO_API"))


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

{title}
{underline}

{jsonld_section}

{hero_section}

Quickstart
----------

{quickstart_section}

About This Dataset
------------------

{readme_section}

Dataset Information
-------------------

{dataset_info_section}

{feedback_section}

Technical Details
-----------------

{highlights_section}

API Reference
-------------

{api_section}

See Also
--------

{see_also_section}

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
from MongoDB. The table below summarises the breakdown by experimental
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
    """Return the sorted dataset class names exported by ``eegdash.dataset``."""
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

    return tuple(sorted(class_names))


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
        "n_subjects": ds.get("demographics", {}).get("subjects_count"),
        "total_files": ds.get("total_files"),
        "n_tasks": len(ds.get("tasks", []) or []),
        "recording_modality": ds.get("recording_modality", []),
        "size_bytes": ds.get("size_bytes"),
        "source": _clean_value(ds.get("source")),
    }

    # Extract source URL from external_links
    external_links = ds.get("external_links", {}) or {}
    details["source_url"] = _clean_value(external_links.get("source_url"))

    return details


def _load_dataset_details(dataset_id: str) -> dict[str, object]:
    dataset_id = dataset_id.lower()
    cached = _DATASET_DETAILS_CACHE.get(dataset_id)
    if cached is not None:
        return cached

    details: dict[str, object] = {}

    # Try local files first
    dataset_dir = CLONE_ROOT / dataset_id
    desc_path = dataset_dir / "dataset_description.json"
    if desc_path.exists():
        try:
            data = json.loads(desc_path.read_text(encoding="utf-8"))
        except Exception:
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
        except Exception:
            data = {}
        details.setdefault("doi", _clean_value(data.get("dataset_doi")))
        details["source_url"] = _clean_value(data.get("source_url"))

    # Fallback to API for missing fields
    if not details.get("title") or not details.get("authors"):
        api_details = _fetch_dataset_details_from_api(dataset_id)
        for key, value in api_details.items():
            if value and not details.get(key):
                details[key] = value

    _DATASET_DETAILS_CACHE[dataset_id] = details
    return details


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

    # Consolidate all badges into a single line with outline style for cleaner look
    citation_count = context.get("nemar_citation_count", "")
    all_badges = [
        ("Modality", str(context.get("modality", ""))),
        ("Subjects", str(context.get("n_subjects", ""))),
        ("Recordings", str(context.get("n_records", ""))),
        ("License", str(context.get("license", ""))),
        ("Source", str(context.get("source", ""))),
    ]
    # Only add citations badge if there's a count
    if citation_count:
        all_badges.append(("Citations", str(citation_count)))

    badges_line = _format_badges(all_badges, outline=True)

    # Add quality indicator
    quality_label, quality_color, quality_pct = _compute_quality_score(context)
    quality_badge = f":bdg-{quality_color}:`Metadata: {quality_label} ({quality_pct}%)`"

    return f"{tagline}\n\n{citation_block}\n\n{badges_line}\n\n{quality_badge}"


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

    lines = [".. list-table::", "   :widths: 25 75", "   :header-rows: 0", ""]
    for label, value in rows:
        lines.append(f"   * - {label}")
        lines.append(f"     - {value}")

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
    """Format the README content for RST display."""
    readme = _clean_value(context.get("readme"))

    if not readme:
        return "No README content is available for this dataset."

    # Convert README content to RST (headers become bold)
    content = _convert_readme_to_rst(readme)
    lines = content.split("\n")

    # For long READMEs (>30 lines), wrap in dropdown
    if len(lines) > 30:
        preview_lines = lines[:10]
        preview = "\n".join(preview_lines)
        indented = "\n".join(f"   {line}" for line in lines)
        return f"""{preview}

.. dropdown:: View full README
   :class-container: sd-shadow-sm

{indented}
"""

    return content


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


def _format_api_section(class_name: str) -> str:
    """Format the API section with autodoc."""
    return (
        f"Use the ``{class_name}`` class to access this dataset programmatically.\n\n"
        ".. currentmodule:: eegdash.dataset\n\n"
        f".. autoclass:: eegdash.dataset.{class_name}\n"
        "   :members: __init__, save\n"
        "   :show-inheritance:\n"
        "   :member-order: bysource\n"
    )


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
        dataset_info_section=_format_dataset_info_section(context),
        readme_section=_format_readme_section(context),
        highlights_section=_format_highlights_section(context),
        quickstart_section=_format_quickstart_section(context),
        api_section=_format_api_section(name),
        see_also_section=_format_see_also_section(dataset_id, name, related),
        feedback_section=_format_feedback_section(dataset_id, dataset_title),
    )
    # Keep the file name with full prefix for URL stability
    page_path = dataset_dir / f"eegdash.dataset.{name}.rst"
    if _write_if_changed(page_path, page_content):
        rel = page_path.relative_to(srcdir)
        LOGGER.info("[dataset-docs] Updated %s", rel)
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

    # Build a modality index so each dataset page can cross-link to related
    # datasets (same recording modality). Read-only after construction, so
    # thread-safe for the parallel generator below.
    datasets_by_modality: dict[str, list[str]] = defaultdict(list)
    for name in dataset_names:
        row = dataset_rows.get(name) or {}
        mod = _clean_value(row.get("record_modality")) or _clean_value(
            row.get("modality of exp")
        )
        if mod:
            datasets_by_modality[mod.lower()].append(name)

    def _related_for(name: str) -> list[str]:
        """Return up to 6 sibling datasets sharing the same modality."""
        row = dataset_rows.get(name) or {}
        mod = _clean_value(row.get("record_modality")) or _clean_value(
            row.get("modality of exp")
        )
        if not mod:
            return []
        siblings = datasets_by_modality.get(mod.lower(), [])
        return [s for s in siblings if s != name][:6]

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
    "user_guide": _article_jsonld(
        "EEGDash user guide",
        (
            "Narrative walkthrough of the EEGDash Python library — query "
            "datasets, load BIDS records, and build reproducible ML pipelines."
        ),
        "https://eegdash.org/user_guide.html",
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
    # Search uses this to recognise the page as a catalog aggregator; the
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


def setup(app):
    """Create the back-references directory and setup Sphinx events."""
    backreferences_dir = os.path.join(
        app.srcdir, sphinx_gallery_conf["backreferences_dir"]
    )
    if not os.path.exists(backreferences_dir):
        os.makedirs(backreferences_dir)

    app.connect("builder-inited", _assert_dataset_table_inlines_datatables)
    app.connect("builder-inited", _generate_dataset_docs)
    app.connect("build-finished", _copy_dataset_summary)
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
