project = "eegdash"
copyright = "2025, Arnaud Delorme"
author = "Arnaud Delorme"

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
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]
templates_path = ["_templates"]
exclude_patterns = []


html_theme = "pydata_sphinx_theme"
html_static_path = ["_static/"]

html_sidebars = {
    "api": [],
    "examples": [],
}

html_logo = "_static/eegdash.png"

# -- Project information -----------------------------------------------------
from datetime import datetime, timezone

project = "EEG Dash"
html_title = "EEG Dash"
html_short_title = "EEG Dash"

td = datetime.now(tz=timezone.utc)

# We need to triage which date type we use so that incremental builds work
# (Sphinx looks at variable changes and rewrites all files if some change)
copyright = f"2025–{td.year}, {project} Developers"  # noqa: E501

author = f"{project} developers"

import eegdash

release = eegdash.__version__
# The full version, including alpha/beta/rc tags.
version = ".".join(release.split(".")[:2])

exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
switcher_version_match = "dev" if release.endswith("dev0") else version

autosummary_generate = True

html_theme_options = {
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "header_links_before_dropdown": 6,
    "navigation_depth": 6,
    "show_toc_level": 1,
    "navbar_end": ["theme-switcher"],
    "footer_start": ["copyright"],
    "logo": {
        "image_light": "_static/eegdash_long.png",
        "image_dark": "_static/eegdash_long.png",
        "alt_text": "EEG Dash Logo",
    },
}

html_favicon = "_static/eegdash_icon.png"

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
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
}

sphinx_gallery_conf["binder"] = dict(
    org="sccn",
    repo="sccn.github.io/eegdash",
    branch="main",
    binderhub_url="https://mybinder.org",
    dependencies="binder/requirements.txt",
    use_jupyter_lab=True,
)
