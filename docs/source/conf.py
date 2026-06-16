import importlib
import inspect
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath(".."))
# Local Sphinx extensions live under ``docs/source/_extensions``; make them
# importable before the ``extensions`` list below references them.
sys.path.insert(0, os.path.abspath("_extensions"))

import eegdash
from eegdash.http_api_client import DEFAULT_API_URL

# The ONE API base for everything the docs build and its client-side
# JS talk to. Sourced from the official eegdash package default and the
# same EEGDASH_API_URL override the package honours, so docs and library
# can never disagree about which deployment they target.
EEGDASH_API_HOST = os.environ.get("EEGDASH_API_URL", DEFAULT_API_URL).rstrip("/")
EEGDASH_API_BASE = f"{EEGDASH_API_HOST}/api/eegdash"

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
    "dataset_figure",
    "dataset_page",
    "assert_dataset_table",
    "sitemap_canonical",
    "auto_examples_index",
    "copy_dataset_summary",
    "counter_values",
    "seo_context",
    "docs_search_index",
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
    # Global search palette (Ctrl+K). ~10 KB, used on every page.
    "css/eegdash-search.css",
]
# Per-page stylesheets are gated in `_templates/layout.html` so they only
# load on the routes that need them:
#   - css/dataset-explorer.css → /api/dataset/eegdash.dataset.* pages
#   - css/dataset-editorial.css → /api/dataset/eegdash.dataset.* pages
#     (Editorial Brief layout — see _extensions/dataset_page/)
# Keep that gating in layout.html rather than html_css_files so the bytes
# don't ride along on every site page.
# Only truly-global JS is loaded here; page-specific scripts (homepage hero
# search, dataset-summary DataTables stack) are gated in `_templates/layout.html`
# by `pagename`. The DataTables stack that dataset_summary depends on is
# inlined into `_static/dataset_generated/dataset_summary_table.html` by the
# generator, so we avoid double-loading by not listing it globally.
html_js_files = [
    ("js/tag-palette.js", {"defer": "defer"}),
    # Global search palette (Ctrl+K): datasets + docs + deep API search.
    # Intercepts the theme's search button and shortcut, and lazy-loads
    # its indexes on first open so pages don't pay the searchindex.js
    # (~2.3 MB) cost.
    ("js/eegdash-search.js", {"defer": "defer"}),
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
    # Single API base injected into every page (layout.html sets it as
    # <html data-eegdash-api>); eegdash-search.js and dataset-explorer.js
    # read it instead of hardcoding the host.
    "eegdash_api_base": EEGDASH_API_BASE,
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


# NOTE: The dataset-page rendering pipeline (catalog row lookup, page
# context build, section formatters, README/Markdown conversion,
# electrode embed, NEMAR analytics block, trace viewer, JSON-LD/SEO,
# and the per-build shell-writer hook) now lives in
# ``docs/source/_extensions/dataset_page.py`` -- registered via the
# ``dataset_page`` entry in ``extensions`` above. The legacy
# ``_generate_dataset_docs`` / ``DATASET_PAGE_TEMPLATE`` pair is gone:
# Sphinx no longer writes 700+ rendered .rst files at builder-inited
# time. Each dataset page is a 1-line ``.. dataset-page:: <NAME>``
# stub whose body is emitted directly as docutils nodes by the
# directive's ``run()``. See ``docs_pipeline_architecture_review.md``
# § 3 (C2) for the rationale.


def setup(app):
    """Create the back-references directory.

    The dataset-page rendering pipeline used to wire its own
    ``builder-inited`` hook here; that hook now lives in the
    ``dataset_page`` extension (see ``extensions = [...]`` above), so
    this ``setup`` only needs to make sure sphinx-gallery's
    backreferences directory exists before the first gallery run.
    """
    backreferences_dir = os.path.join(
        app.srcdir, sphinx_gallery_conf["backreferences_dir"]
    )
    if not os.path.exists(backreferences_dir):
        os.makedirs(backreferences_dir)


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
