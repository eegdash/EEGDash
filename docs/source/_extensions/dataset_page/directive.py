"""The ``dataset-page`` directive plus its page-assembly helpers.

``_build_page_rst`` glues the section formatters together; ``DatasetPageDirective``
parses the assembled RST via ``nested_parse_with_titles`` and returns docutils
nodes -- replacing the legacy ``DATASET_PAGE_TEMPLATE`` f-string + per-page
disk write.

Module-level caches (``_SNAPSHOT_ROWS_CACHE``, ``_MODALITY_INDEX_CACHE``)
populate on first access and survive across the 700+ directive invocations
inside one Sphinx build.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles

from ._constants import AUTOGEN_NOTICE, DATASET_NAME_RE
from .data_loaders import (
    _build_dataset_context,
    _clean_value,
    _iter_dataset_classes,
    _load_dataset_rows,
)
from .related import _build_related_index_gower, _related_meta_for
from .sections import (
    _editorial_secnum,
    _format_api_section,
    _format_dataset_info_dropdown,
    _format_dataset_jsonld_section,
    _format_dataset_meta_section,
    _format_editorial_access_modes_section,
    _format_editorial_caveat_section,
    _format_editorial_colophon_section,
    _format_editorial_examples_gallery,
    _format_editorial_fieldcard_section,
    _format_editorial_footnotes_section,
    _format_editorial_hero_extras,
    _format_editorial_kicker_section,
    _format_editorial_layers_section,
    _format_editorial_provenance_section,
    _format_electrodes_traces_pair,
    _format_explorer_section,
    _format_feedback_section,
    _format_hero_section,
    _format_nemar_analysis_section,
    _format_nemar_metadata_section,
    _format_quickstart_section,
    _format_readme_section,
    _format_recording_stats_section,
    _format_see_also_section,
)

LOGGER = logging.getLogger(__name__)


# Module-level snapshot cache so a directive invocation does not refetch
# the snapshot for every dataset page (700+ calls). Keyed implicitly by
# ``DatasetSnapshot.build()``'s own LRU; we just remember the rows.
_SNAPSHOT_ROWS_CACHE: dict[str, Mapping[str, str]] | None = None


def _snapshot_rows() -> Mapping[str, Mapping[str, str]]:
    """Return the dataset_name -> row mapping, populated on first call.

    Reuses ``_load_dataset_rows`` -- the underlying ``DatasetSnapshot``
    keeps an in-process cache so repeated builds inside one process do
    not refetch the chart-data payload.
    """
    global _SNAPSHOT_ROWS_CACHE
    if _SNAPSHOT_ROWS_CACHE is None:
        _SNAPSHOT_ROWS_CACHE = dict(_load_dataset_rows(_iter_dataset_classes()))
    return _SNAPSHOT_ROWS_CACHE


_MODALITY_INDEX_CACHE: dict[str, list[str]] | None = None
_GOWER_INDEX_CACHE: dict[str, list[str]] | None = None


def _related_for_modality(
    name: str, rows: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """Modality-grouping fallback when Gower kNN is unavailable."""
    global _MODALITY_INDEX_CACHE
    if _MODALITY_INDEX_CACHE is None:
        index: dict[str, list[str]] = defaultdict(list)
        for other_name, other_row in rows.items():
            row = other_row or {}
            mod = _clean_value(row.get("record_modality")) or _clean_value(
                row.get("modality of exp")
            )
            if mod:
                index[mod.lower()].append(other_name)
        _MODALITY_INDEX_CACHE = index

    row = rows.get(name) or {}
    mod = _clean_value(row.get("record_modality")) or _clean_value(
        row.get("modality of exp")
    )
    if not mod:
        return []
    siblings = _MODALITY_INDEX_CACHE.get(mod.lower(), [])
    return [s for s in siblings if s != name][:6]


def _related_for(name: str, rows: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Return up to 6 sibling datasets via Gower-distance kNN.

    Falls back to modality grouping when numpy is unavailable.
    """
    global _GOWER_INDEX_CACHE
    if _GOWER_INDEX_CACHE is None:
        try:
            _GOWER_INDEX_CACHE = _build_related_index_gower(
                list(rows.keys()), rows, k=6
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[dataset-page] Gower kNN failed (%s); falling back", exc)
            _GOWER_INDEX_CACHE = {}

    if _GOWER_INDEX_CACHE:
        siblings = _GOWER_INDEX_CACHE.get(name)
        if siblings:
            return siblings[:6]

    return _related_for_modality(name, rows)


def _build_page_rst(class_name: str) -> str:
    """Assemble the full page RST for ``class_name``.

    Replaces the legacy ``DATASET_PAGE_TEMPLATE`` f-string skeleton.
    Emits the editorial "Dataset Brief" layout: kicker + field-card,
    H1, hero, layers rail, then six numbered sections (Quickstart /
    About / Cohort / Signal / Manifest / API) flanked by editorial
    chrome (caveat callout, access-modes sidecar, examples gallery,
    provenance strip, footnotes, colophon).
    """
    rows = _snapshot_rows()
    row = rows.get(class_name)
    context = _build_dataset_context(class_name, row)

    # Page title -- matches the old hand-rolled "ABSEQMEG: EEG dataset,
    # 20 subjects" prefix used by the template generator.
    modality = _clean_value(context.get("modality")) or "EEG"
    n_sub = _clean_value(context.get("n_subjects"))
    title_parts = [class_name]
    suffix_parts = [f"{modality} dataset"]
    if n_sub and n_sub not in ("—", "0"):
        suffix_parts.append(f"{n_sub} subjects")
    title_parts.append(", ".join(suffix_parts))
    title = ": ".join(title_parts)

    dataset_id = str(context.get("dataset_id", ""))
    dataset_title = str(context.get("title", ""))
    og_description_field, meta_section = _format_dataset_meta_section(context)
    related = _related_for(class_name, rows)
    related_meta = _related_meta_for(class_name, related, rows)

    # Top-of-page field list. Stays in a single contiguous block --
    # blank lines between entries would demote them to body text.
    header_block = (
        f"{AUTOGEN_NOTICE}"
        ":html_theme.sidebar_secondary.remove:\n"
        f"{og_description_field}\n"
    )

    sections: list[str] = [
        header_block,
        meta_section,
        # Editorial kicker (breadcrumb + issue line) and field-card aside
        # are emitted BEFORE the H1 so the rail floats next to the hero.
        _format_editorial_kicker_section(context),
        _format_editorial_fieldcard_section(context),
        f"{title}\n{'=' * len(title)}\n",
        _format_dataset_jsonld_section(class_name, context),
        _format_hero_section(context),
        # Editorial deck + byline + signal pills sit just under the hero.
        _format_editorial_hero_extras(context),
        _format_editorial_layers_section(context),
        # §01 Access · Get started
        _editorial_secnum(1, "Access · Get started"),
        "Quickstart\n----------\n",
        _format_quickstart_section(context),
        # §02 Study · The README
        _editorial_secnum(2, "Study · The README"),
        "About This Dataset\n------------------\n",
        _format_readme_section(context),
        # NEMAR-specific metadata (authors with ORCID, MeSH keywords,
        # DOI-stamped version history) — returns "" for non-NEMAR.
        _format_nemar_metadata_section(context),
        # §03 Cohort · Participants
        _editorial_secnum(3, "Cohort · Participants"),
        "Cohort\n------\n",
        _format_recording_stats_section(context),
        # Editorial caveat fires only when n_subjects < 50.
        _format_editorial_caveat_section(context),
        # §04 Signal · Electrodes & trace
        _editorial_secnum(4, "Signal · Electrodes & trace"),
        "Signal · Electrodes & live trace\n--------------------------------\n",
        _format_electrodes_traces_pair(class_name, context),
        _format_nemar_analysis_section(context),
        # §05 Manifest · BIDS tree
        _editorial_secnum(5, "Manifest · BIDS tree"),
        "Manifest\n--------\n",
        _format_explorer_section(class_name, context),
        _format_dataset_info_dropdown(context),
        _format_feedback_section(dataset_id, dataset_title),
        # §06 API · Programmatic access
        _editorial_secnum(6, "API · Programmatic access"),
        "API Reference\n-------------\n",
        _format_api_section(class_name, context),
        _format_editorial_access_modes_section(context),
        _format_editorial_examples_gallery(context),
        _format_editorial_footnotes_section(context, related, related_meta),
        _format_editorial_provenance_section(context),
        "See Also\n--------\n",
        _format_see_also_section(dataset_id, class_name, related),
        _format_editorial_colophon_section(context),
    ]

    # Filter out empty sections, then join with blank lines so adjacent
    # RST blocks aren't collapsed into a single paragraph.
    return "\n\n".join(s for s in sections if s and s.strip())


class DatasetPageDirective(Directive):
    """Emit one dataset catalog page from a class name.

    Usage::

        .. dataset-page:: DS002718

    The single positional argument is a Python class name in
    ``eegdash.dataset.__all__``. The directive looks up the row from
    :class:`eegdash.dataset.snapshot.DatasetSnapshot`, calls the section
    formatters in this module, and parses the assembled RST into
    docutils nodes via ``state.nested_parse``. No string-template
    interpolation across an 80-line skeleton; no per-page disk write of
    the rendered body.

    The companion ``builder-inited`` handler still writes a 1-line
    ``.rst`` shell per dataset so the existing 3-level toctree at
    ``docs/source/api/api.rst`` -> ``api_dataset.rst`` ->
    ``source_<group>.rst`` works unchanged.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self) -> list[nodes.Node]:
        class_name: str = self.arguments[0].strip()

        if not DATASET_NAME_RE.match(class_name):
            raise self.error(
                f"dataset-page: invalid class name {class_name!r}. "
                "Must match [A-Za-z0-9_-]{1,64}."
            )

        try:
            page_rst = _build_page_rst(class_name)
        except Exception as exc:  # noqa: BLE001 -- surface as a doc-build warning
            LOGGER.warning("[dataset-page] failed to build %s: %s", class_name, exc)
            return [
                nodes.paragraph(
                    text=(
                        f"Failed to render dataset page for {class_name}: {exc}. "
                        "See build log for details."
                    )
                )
            ]

        # ``nested_parse_with_titles`` is the Sphinx-blessed wrapper
        # around ``state.nested_parse`` that allows the parsed content
        # to introduce its own section titles -- exactly what a
        # full-page directive needs. The plain ``state.nested_parse``
        # path would emit "Unexpected section title" errors because the
        # surrounding shell .rst is a non-section context.
        container = nodes.section()
        source = self.state.document.current_source or f"<dataset-page:{class_name}>"
        lines = StringList(page_rst.splitlines(), source=source)
        nested_parse_with_titles(self.state, lines, container)
        return list(container.children)
