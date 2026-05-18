"""Registry of dataset-summary figure scaffolds.

Each entry describes one chart that is embedded in ``dataset_summary.rst`` via
``.. dataset-figure:: <key>``. The previous incarnation lived as eight
near-duplicate ``.rst`` files under ``docs/source/dataset_summary/`` (one per
chart), all repeating the same figure scaffold and only differing in title,
caption, and embedded HTML filename. Centralising them here means adding a
new chart is a single registry entry plus a ``.. dataset-figure::`` invocation
inside whichever tab-set should host it -- no new RST file required.

Fields per entry:

``title``
    The figure heading. Rendered either as an ``<h3>`` inside an
    ``<header class="eegdash-fig-title">`` (``title_style="header"``,
    the default) or as a Sphinx ``.. rubric::`` (``title_style="rubric"``).
    The two styles map to the two structural variants the original eight
    RSTs used; both are present elsewhere in ``custom.css`` so we preserve
    each chart's pre-refactor look-and-feel.

``chart_filename``
    Basename of the chart HTML under
    ``docs/source/_static/dataset_generated/``. Loaded verbatim via
    ``.. raw:: html :file:``.

``caption``
    Free HTML for the ``<figcaption class="eegdash-caption">`` body. May
    contain inline tags (``<strong>``, ``<em>``, ``<code>``, ``<br>``).

``title_style``
    ``"header"`` (default) or ``"rubric"``. Controls how the chart title
    is rendered. ``"header"`` produces the same ``<header class=
    "eegdash-fig-title"><h3>...</h3></header>`` markup the bubble/clinical/
    growth/sankey/treemap charts used; ``"rubric"`` uses Sphinx's native
    ``.. rubric::`` directive (matching the kde and moabb charts).

``prologue`` / ``epilogue``
    Optional reStructuredText blocks to render before / after the figure.
    Used by the ``table`` figure for its MeSH disease list and the trailing
    "Pathology, modality, ..." sentence. Multi-line strings; rendered with
    the same parser as any inline RST.
"""

from __future__ import annotations

from typing import Final, Mapping

# Type alias kept loose on purpose -- this module is consumed by a Sphinx
# directive at build time, not by application code.
FigureSpec = Mapping[str, str]


_TABLE_PROLOGUE = """\
EEG-DaSh aggregates M/EEG recordings (EEG, MEG, and combined setups) from both
healthy participants and clinical cohorts. Disease-bearing datasets span
`epilepsy <https://meshb.nlm.nih.gov/record/ui?ui=D004827>`__,
`Parkinson disease <https://meshb.nlm.nih.gov/record/ui?ui=D010300>`__,
`dementia <https://meshb.nlm.nih.gov/record/ui?ui=D003704>`__,
`depressive disorder <https://meshb.nlm.nih.gov/record/ui?ui=D003866>`__,
`schizophrenia <https://meshb.nlm.nih.gov/record/ui?ui=D012559>`__ and related
`psychotic disorders <https://meshb.nlm.nih.gov/record/ui?ui=D011618>`__,
`traumatic brain injury <https://meshb.nlm.nih.gov/record/ui?ui=D000070642>`__,
`alcohol use disorder <https://meshb.nlm.nih.gov/record/ui?ui=D000437>`__,
`dyslexia <https://meshb.nlm.nih.gov/record/ui?ui=D004410>`__, and
`obesity <https://meshb.nlm.nih.gov/record/ui?ui=D009765>`__, alongside
neurodevelopmental and surgical cohorts. The catalog also covers resting state,
sleep, and a range of cognitive, sensory, and motor tasks. Disease labels link
to the NLM `MeSH <https://www.nlm.nih.gov/mesh/meshhome.html>`__ vocabulary for
formal definitions.

A large share of the archive is converted from `NEMAR <https://nemar.org/>`__,
which contributes BIDS-formatted M/EEG datasets to the catalog.
"""

_TABLE_EPILOGUE = (
    "Pathology, modality, and dataset type appear as color-coded tags, so the "
    "table is quick to scan."
)


DATASET_FIGURES: Final[dict[str, FigureSpec]] = {
    "bubble": {
        "title": "Dataset map: subjects × records × duration",
        "title_style": "header",
        "chart_filename": "dataset_bubble.html",
        "caption": (
            "Figure: Dataset map. Each bubble represents a dataset: x-axis "
            "shows the number of subjects, y-axis the number of records, "
            "bubble size encodes recording duration per subject, and color "
            "indicates experiment modality. Hover for details, click to open "
            "a dataset page, and use the legend to filter."
        ),
    },
    "clinical": {
        "title": "Clinical breakdown by recording modality",
        "title_style": "header",
        "chart_filename": "dataset_clinical.html",
        "caption": (
            "Figure: Breakdown of datasets by clinical status and "
            "experimental modality. Use the toggle buttons to switch "
            "between the number of studies and the number of subjects."
        ),
    },
    "growth": {
        "title": "Cumulative growth of EEG-DaSh datasets",
        "title_style": "header",
        "chart_filename": "dataset_growth.html",
        "caption": (
            "Figure: Cumulative growth of open datasets indexed by EEG Dash "
            "over time. Use the toggle buttons to switch between cumulative "
            "datasets and cumulative subjects."
        ),
    },
    "kde": {
        "title": "Distribution of Sample Sizes Varies by Experimental Modality",
        "title_style": "rubric",
        "chart_filename": "dataset_kde_modalities.html",
        "caption": (
            "Figure: Participant distribution by modality. Kernel density "
            "estimates summarize how many participants are available for "
            "each experimental modality on a logarithmic scale. Individual "
            "points show dataset-level counts."
        ),
    },
    "moabb": {
        "title": "Subject Distribution Bubble Plot",
        "title_style": "rubric",
        "chart_filename": "dataset_moabb_bubble.html",
        "caption": (
            "Figure: Circle-packing overview of 700+ datasets (35 000+ "
            "subjects) catalogued in EEGDash. Each small circle represents "
            "one subject, grouped by dataset and colored by recording "
            "modality. Size encodes per-subject recording duration "
            "(log-scaled minutes); opacity encodes session count (fewer "
            "sessions = more opaque). Interactive: hover to inspect, scroll "
            "to zoom, click to navigate, search to filter."
        ),
    },
    "sankey": {
        "title": "Dataset flow by population, modality, and task",
        "title_style": "header",
        "chart_filename": "dataset_sankey.html",
        "caption": (
            "Figure: Dataset flow across population, modality, and "
            "cognitive domain. Link thickness is proportional to the total "
            "number of subjects, and the tooltip reports both subject and "
            "dataset counts. Hover and click legend entries to explore "
            "specific segments."
        ),
    },
    "table": {
        "title": "EEG Datasets Table",
        "title_style": "rubric",
        "chart_filename": "dataset_summary_table.html",
        "caption": (
            "Sortable catalogue of EEG‑DaSh datasets. Click any column "
            "header to sort, use the <strong>Filters</strong> chip to slice "
            "by recording / pathology / modality / type / source, or the "
            "<strong>Columns</strong> chip to show hidden metadata (Author, "
            "Canonical name, Source, Sessions). The <em>Total</em> row stays "
            "pinned at the bottom across filters.<br>Trailing <code>*</code> "
            "in Channels / Sampling rate marks a median across multiple "
            "recordings; em‑dashes mean the metadata hasn't been "
            "extracted yet."
        ),
        "prologue": _TABLE_PROLOGUE,
        "epilogue": _TABLE_EPILOGUE,
    },
    "treemap": {
        "title": "Dataset treemap",
        "title_style": "header",
        "chart_filename": "dataset_treemap.html",
        "caption": (
            "Figure: Treemap of EEG Dash datasets. The top level groups "
            "population type, the second level breaks down experimental "
            "modality, and leaves list individual datasets. Tile area "
            "encodes the total number of subjects; hover to view aggregated "
            "hours (or records when unavailable)."
        ),
    },
}


def get(key: str) -> FigureSpec:
    """Look up a figure spec by key.

    Raises ``KeyError`` if ``key`` is unknown; callers (the directive) turn
    this into a Sphinx-friendly error.
    """
    return DATASET_FIGURES[key]


def keys() -> list[str]:
    """Return all registered figure keys (stable, sorted)."""
    return sorted(DATASET_FIGURES)
