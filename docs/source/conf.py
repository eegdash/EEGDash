import os
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
    "sphinx.ext.napoleon",
    "sphinx_design",
    # "autoapi.extension",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/eegdash_icon.svg"
html_favicon = "_static/eegdash_icon.png"
html_title = "EEG Dash"
html_short_title = "EEG Dash"
html_css_files = ["custom.css"]

html_theme_options = {
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": False,
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
            "url": "https://sccn.github.io/EEGDash",
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


# -- Extension configurations ------------------------------------------------
autoclass_content = "both"

# Numpydoc
numpydoc_show_class_members = False

# Sphinx Gallery
EX_DIR = "../../examples"  # relative to docs/source
sphinx_gallery_conf = {
    "examples_dirs": [EX_DIR],
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
    "subsection_order": ExplicitOrder(
        [
            f"{EX_DIR}/core",
            f"{EX_DIR}/eeg2025",
            "*",
        ]
    ),
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
