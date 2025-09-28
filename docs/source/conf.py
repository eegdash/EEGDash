import csv
import importlib
import inspect
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from sphinx.util import logging
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey
from tabulate import tabulate

import eegdash

# -- Project information -----------------------------------------------------

project = "EEG Dash"
copyright = f"2025–{datetime.now(tz=timezone.utc).year}, {project} Developers"
author = "Arnaud Delorme"
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

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/eegdash_icon.svg"
html_favicon = "_static/eegdash_icon.png"
html_title = "EEG Dash"
html_short_title = "EEG Dash"
html_css_files = [
    "https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.css",
    "https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css",
    "https://cdn.datatables.net/select/1.7.0/css/select.dataTables.min.css",
    "https://cdn.datatables.net/searchpanes/2.3.1/css/searchPanes.dataTables.min.css",
    "custom.css",
]
html_js_files = [
    "https://code.jquery.com/jquery-3.7.1.min.js",
    "https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.js",
    "https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js",
    "https://cdn.datatables.net/select/1.7.0/js/dataTables.select.min.js",
    "https://cdn.datatables.net/searchpanes/2.3.1/js/dataTables.searchPanes.min.js",
    "js/tag-palette.js",
    "js/datatables-init.js",
]

# Required for sphinx-sitemap: set the canonical base URL of the site
# Make sure this matches the actual published docs URL and ends with '/'
html_baseurl = "https://eegdash.org/"

html_theme_options = {
    "icon_links_label": "External Links",  # for screen reader
    # Show an "Edit this page" button linking to GitHub
    "use_edit_page_button": True,
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "header_links_before_dropdown": 6,
    "navigation_depth": 6,
    "show_nav_level": 2,
    "show_toc_level": 1,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "logo": {
        "image_light": "_static/eegdash_long.png",
        "image_dark": "_static/eegdash_long.png",
        "alt_text": "EEG Dash Logo",
    },
    "external_links": [
        {"name": "EEG2025", "url": "https://eeg2025.github.io/"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sccn/EEGDash",
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

html_sidebars = {"api": [], "dataset_summary": [], "installation": []}

# Copy extra files (e.g., robots.txt) to the output root
html_extra_path = ["_extra"]

# Provide GitHub context so the edit button and custom templates
# (e.g., Sphinx-Gallery "Open in Colab") know where the source lives.
# These values should match the repository and docs location.
html_context = {
    "github_user": "sccn",
    "github_repo": "EEGDash",
    # Branch used to build and host the docs
    "github_version": "main",
    # Path to the documentation root within the repo
    "doc_path": "docs/source",
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
    "nested_sections": False,
    "backreferences_dir": "gen_modules/backreferences",
    "inspect_global_variables": True,
    "show_memory": True,
    "show_api_usage": False,
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


AUTOGEN_NOTICE = """..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

"""


DATASET_PAGE_TEMPLATE = """{notice}{title}
{underline}

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.{class_name}
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

{metadata_section}

Usage Example
-------------

{usage_section}

See Also
--------

{see_also_section}

"""


DATASET_INDEX_TEMPLATE = """{notice}Datasets API
=======================

The :mod:`eegdash.dataset` package exposes dynamically registered dataset
classes. See :doc:`eegdash.dataset` for the module-level API, including
:class:`~eegdash.dataset.EEGChallengeDataset` and helper utilities.

Dataset Overview
----------------

EEGDash currently exposes **{dataset_count} OpenNeuro EEG datasets** that are
registered dynamically from mongo database. The table below summarises
the distribution by experimental type as tracked in the summary file.

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


def _write_if_changed(path: Path, content: str) -> bool:
    """Write ``content`` to ``path`` if it differs from the current file."""
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _iter_dataset_classes() -> Sequence[str]:
    """Return the sorted dataset class names exported by ``eegdash.dataset``."""
    import eegdash.dataset as dataset_module  # local import for clarity
    from eegdash.api import EEGDashDataset

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
    csv_path = Path(importlib.import_module("eegdash.dataset").__file__).with_name(
        "dataset_summary.csv"
    )
    if not csv_path.exists():
        return []

    counter: Counter[str] = Counter()
    valid_names = {name.upper() for name in dataset_names}

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
    csv_path = Path(importlib.import_module("eegdash.dataset").__file__).with_name(
        "dataset_summary.csv"
    )
    if not csv_path.exists():
        return {}

    wanted = set(dataset_names)
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


def _format_metadata_section(row: Mapping[str, str] | None, class_name: str) -> str:
    if not row:
        return (
            "Metadata for this dataset is not available in ``dataset_summary.csv``.\n"
        )

    def _clean(key: str, *, default: str | None = "") -> str:
        value = row.get(key, "")
        text = str(value).strip()
        if not text and default is not None:
            return default
        return text

    dataset_id = _clean("dataset", default="").lower() or class_name.lower()
    dataset_upper = dataset_id.upper()

    summary_bits: list[str] = []
    modality = _clean("modality of exp")
    if modality:
        summary_bits.append(f"Modality: {modality}")
    exp_type = _clean("type of exp")
    if exp_type:
        summary_bits.append(f"Type: {exp_type}")
    subject_type = _clean("Type Subject")
    if subject_type:
        summary_bits.append(f"Subjects: {subject_type}")

    lines = [f"- **Dataset ID:** ``{dataset_upper}``"]
    if summary_bits:
        lines.append(f"- **Summary:** {' | '.join(summary_bits)}")

    def _add_numeric(label: str, key: str) -> None:
        value = _clean(key)
        if value:
            lines.append(f"- **{label}:** {value}")

    _add_numeric("Number of Subjects", "n_subjects")
    _add_numeric("Number of Recordings", "n_records")
    _add_numeric("Number of Tasks", "n_tasks")
    _add_numeric("Number of Channels", "nchans_set")
    _add_numeric("Sampling Frequencies", "sampling_freqs")
    _add_numeric("Total Duration (hours)", "duration_hours_total")
    _add_numeric("Dataset Size", "size")

    nemar_url = f"https://nemar.org/dataexplorer/detail?dataset_id={dataset_id}"
    openneuro_url = f"https://openneuro.org/datasets/{dataset_id}"
    lines.append(f"- **OpenNeuro:** `{dataset_id} <{openneuro_url}>`__")
    lines.append(f"- **NeMAR:** `{dataset_id} <{nemar_url}>`__")

    table_row = {
        "dataset": dataset_id,
        "#Subj": _clean("n_subjects"),
        "#Chan": _clean("nchans_set"),
        "#Classes": _clean("n_tasks"),
        "Freq(Hz)": _clean("sampling_freqs"),
        "Duration(H)": _clean("duration_hours_total"),
        "Size": _clean("size"),
    }

    table = tabulate([table_row], headers="keys", tablefmt="rst", showindex=False)

    bullet_text = "\n".join(lines)
    return f"{bullet_text}\n\n{table}\n"


def _format_usage_section(class_name: str) -> str:
    return (
        ".. code-block:: python\n\n"
        f"   from eegdash.dataset import {class_name}\n\n"
        f'   dataset = {class_name}(cache_dir="./data")\n\n'
        '   print(f"Number of recordings: {len(dataset)}")\n\n'
        "   if len(dataset):\n"
        "       recording = dataset[0]\n"
        "       raw = recording.load()\n"
        "       print(f\"Sampling rate: {raw.info['sfreq']} Hz\")\n"
        '       print(f"Channels: {len(raw.ch_names)}")\n'
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


def _generate_dataset_docs(app) -> None:
    dataset_dir = Path(app.srcdir) / "api" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = _iter_dataset_classes()
    dataset_rows = _load_dataset_rows(dataset_names)
    toctree_entries = _render_toctree_entries(dataset_names)
    experiment_rows = _render_experiment_rows(_load_experiment_counts(dataset_names))
    index_content = DATASET_INDEX_TEMPLATE.format(
        notice=AUTOGEN_NOTICE,
        dataset_count=len(dataset_names),
        experiment_rows=experiment_rows,
        toctree_entries=toctree_entries,
    )

    index_path = dataset_dir / "api_dataset.rst"
    if _write_if_changed(index_path, index_content):
        LOGGER.info("[dataset-docs] Updated %s", index_path.relative_to(app.srcdir))

    generated_paths: set[Path] = set()
    for name in dataset_names:
        title = f"eegdash.dataset.{name}"
        row = dataset_rows.get(name)
        dataset_id = ((row.get("dataset") if row else "") or name.lower()).strip()
        dataset_id = dataset_id or name.lower()
        page_content = DATASET_PAGE_TEMPLATE.format(
            notice=AUTOGEN_NOTICE,
            title=title,
            underline="=" * len(title),
            class_name=name,
            metadata_section=_format_metadata_section(row, name),
            usage_section=_format_usage_section(name),
            see_also_section=_format_see_also_section(dataset_id),
        )
        page_path = dataset_dir / f"eegdash.dataset.{name}.rst"
        if _write_if_changed(page_path, page_content):
            LOGGER.info("[dataset-docs] Updated %s", page_path.relative_to(app.srcdir))
        generated_paths.add(page_path)

    _cleanup_stale_dataset_pages(dataset_dir, generated_paths)

    # Remove legacy pages that used the short filename convention
    for legacy in dataset_dir.glob("DS*.rst"):
        try:
            legacy.unlink()
        except OSError:
            continue


def setup(app):
    """Create the back-references directory if it doesn't exist."""
    backreferences_dir = os.path.join(
        app.srcdir, sphinx_gallery_conf["backreferences_dir"]
    )
    if not os.path.exists(backreferences_dir):
        os.makedirs(backreferences_dir)

    app.connect("builder-inited", _generate_dataset_docs)


# Configure sitemap URL format (omit .html where possible)
sitemap_url_scheme = "{link}"

# Copy button configuration: strip common interactive prompts when copying
copybutton_prompt_text = r">>> |\\$ |# "
copybutton_prompt_is_regexp = True
