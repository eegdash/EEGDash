import importlib
import inspect
import os
import sys
from datetime import datetime, timezone

from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

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
    "sphinx.ext.napoleon",
    "sphinx_design",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinx_sitemap",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

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
html_css_files = ["custom.css"]
html_js_files = []

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
    "show_toc_level": 1,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "logo": {
        "image_light": "_static/eegdash_long.png",
        "image_dark": "_static/eegdash_long.png",
        "alt_text": "EEG Dash Logo",
    },
    "external_links": [
        {"name": "EEG2025 competition", "url": "https://eeg2025.github.io/"},
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


def setup(app):
    """Create the back-references directory if it doesn't exist."""
    backreferences_dir = os.path.join(
        app.srcdir, sphinx_gallery_conf["backreferences_dir"]
    )
    if not os.path.exists(backreferences_dir):
        os.makedirs(backreferences_dir)


# Configure sitemap URL format (omit .html where possible)
sitemap_url_scheme = "{link}"
