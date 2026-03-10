import concurrent.futures
import csv
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
    "sphinx_copybutton",
    "sphinx.ext.graphviz",
    "sphinx_time_estimation",
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
html_css_files = [
    "https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.css",
    "https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css",
    "https://cdn.datatables.net/select/1.7.0/css/select.dataTables.min.css",
    "https://cdn.datatables.net/searchpanes/2.3.1/css/searchPanes.dataTables.min.css",
    "custom.css",
    "css/treemap.css",
    "css/custom.css",
]
html_js_files = [
    "https://code.jquery.com/jquery-3.7.1.min.js",
    "https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.js",
    "https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js",
    "https://cdn.datatables.net/select/1.7.0/js/dataTables.select.min.js",
    "https://cdn.datatables.net/searchpanes/2.3.1/js/dataTables.searchPanes.min.js",
    # Fuse.js for fuzzy search autocomplete (24KB minified, 8KB gzipped)
    "https://cdn.jsdelivr.net/npm/fuse.js@7.0.0/dist/fuse.min.js",
    "js/tag-palette.js",
    "js/datatables-init.js",
    "js/hero-search.js",
    "js/search-as-you-type.js",  # Live search in PyData theme search modal
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
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/eegdash/EEGDash",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/eegdash/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
        {
            "name": "Docs (Stable)",
            "url": "https://eegdash.org/EEGDash",
            "icon": "fa-solid fa-book",
            "type": "fontawesome",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/8jd7nVKwsc",
            "icon": "fa-brands fa-discord",
            "type": "fontawesome",
        },
    ],
}

html_sidebars = {
    "index": [],  # Remove sidebars on homepage
    "dataset_summary": [],
    "api": [],
    "installation": [],
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
    "show_memory": True,
    "show_api_usage": True,
    "doc_module": ("eegdash", "numpy", "scipy", "matplotlib"),
    "reference_url": {"eegdash": None},
    "filename_pattern": r"/(?:plot|tutorial)_(?!_).*\.py",
    "matplotlib_animations": True,
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

{title}
{underline}

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

The :mod:`eegdash.dataset` package exposes dynamically registered dataset
classes. See :doc:`eegdash.dataset` for the module-level API, including
:class:`~eegdash.dataset.EEGChallengeDataset` and helper utilities.

Dataset Overview
----------------

EEGDash currently exposes **{dataset_count} OpenNeuro EEG datasets** that are
registered dynamically from mongo database. The table below summarises
the distribution by experimental type as tracked in the summary file.

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

    return {
        "class_name": class_name,
        "dataset_id": dataset_id,
        "dataset_upper": dataset_id.upper(),
        "title": title,
        "year": year,
        "authors": details.get("authors", []),
        "license": license_text,
        "doi": doi,
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
    rows = [
        ("Dataset ID", f"``{dataset_upper}``"),
        ("Title", title),
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

            # Convert autolinks: <https://example.com>
            line = re.sub(r"<(https?://[^>]+)>", r"`\1 <\1>`__", line)

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
            # Matches word_underscore followed by whitespace, punctuation, or end of line
            line = re.sub(r"(\w)_(?=[\s.,;:!?\)\]\}]|$)", r"\1\_", line)

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

    return "\n".join(result)


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


def _format_see_also_section(dataset_id: str) -> str:
    dataset_lower = dataset_id.lower()
    nemar_url = f"https://nemar.org/dataexplorer/detail?dataset_id={dataset_lower}"
    openneuro_url = f"https://openneuro.org/datasets/{dataset_lower}"
    return "\n".join(
        [
            "* :class:`eegdash.dataset.EEGDashDataset`",
            "* :mod:`eegdash.dataset`",
            f"* `OpenNeuro dataset page <{openneuro_url}>`__",
            f"* `NeMAR dataset page <{nemar_url}>`__",
        ]
    )


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
    name: str, dataset_dir: Path, row: Mapping[str, str] | None, srcdir: Path
) -> Path:
    # Use simplified title: just the dataset ID (e.g., "DS001787")
    title = name  # Dataset class name is already uppercase ID like DS001787
    context = _build_dataset_context(name, row)
    dataset_id = str(context.get("dataset_id", ""))
    dataset_title = str(context.get("title", ""))
    page_content = DATASET_PAGE_TEMPLATE.format(
        notice=AUTOGEN_NOTICE,
        title=title,
        underline="=" * len(title),
        hero_section=_format_hero_section(context),
        dataset_info_section=_format_dataset_info_section(context),
        readme_section=_format_readme_section(context),
        highlights_section=_format_highlights_section(context),
        quickstart_section=_format_quickstart_section(context),
        api_section=_format_api_section(name),
        see_also_section=_format_see_also_section(dataset_id),
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

    generated_paths: set[Path] = set()
    srcdir = Path(app.srcdir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _process_dataset_item, name, dataset_dir, dataset_rows.get(name), srcdir
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
    "|modalities_total|": _format_counter("modalities"),
    "|sources_total|": _format_counter("sources"),
}


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


def setup(app):
    """Create the back-references directory and setup Sphinx events."""
    backreferences_dir = os.path.join(
        app.srcdir, sphinx_gallery_conf["backreferences_dir"]
    )
    if not os.path.exists(backreferences_dir):
        os.makedirs(backreferences_dir)

    app.connect("builder-inited", _generate_dataset_docs)
    app.connect("build-finished", _copy_dataset_summary)
    app.connect("source-read", _inject_counter_values)


# Configure sitemap URL format (omit .html where possible)
sitemap_url_scheme = "{link}"

# Copy button configuration: strip common interactive prompts when copying
copybutton_prompt_text = r">>> |\\$ |# "
copybutton_prompt_is_regexp = True
