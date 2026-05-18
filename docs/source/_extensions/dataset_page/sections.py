"""Section formatters for the dataset_page directive.

Each ``_format_*_section`` function returns an RST fragment that the
directive concatenates into the full page body. Functions read from a
shared ``context: Mapping[str, object]`` slice produced by
:func:`data_loaders._build_dataset_context`.

The sections are grouped in this file by the page region they belong to:

* **Hero / quickstart / dataset-info** -- the page header.
* **README** -- delegates to :mod:`.readme` for the Markdown->RST work.
* **Schema / quality / electrodes / recording-stats / NEMAR-analysis /
  explorer / traces** -- the technical details block.
* **API / see-also / feedback / dataset-meta / dataset-jsonld** --
  trailing reference + SEO blocks.

The split out of the monolithic ``dataset_page.py`` is mechanical:
sections are byte-identical, just reattached to the package's helper
imports.
"""

from __future__ import annotations

import json
import re
import urllib.request  # noqa: F401 -- used by _get_first_eeg_record (urllib.request.Request/urlopen); ruff misses it because the function also does a local `import urllib.parse`
from collections import Counter
from typing import Mapping, Sequence
from urllib.parse import quote

from sphinx.util import logging

from eegdash.dataset.nemar import NemarClient, NemarMetadata

from ._constants import _DOCS_SOURCE_ROOT, _LICENSE_URL_MAP
from .data_loaders import (
    _clean_value,
    _collapse_whitespace,
    _compute_quality_score,
    _normalize_doi,
    _value_or_unknown,
)
from .readme import convert_readme_to_rst

LOGGER = logging.getLogger(__name__)


def _format_badges(items: Sequence[tuple[str, str]], outline: bool = False) -> str:
    """Format badge items as RST badge directives."""
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
        style = f"{color}-line" if outline else color
        badges.append(f":bdg-{style}:`{label}: {val_text}`")

    badges_str = " ".join(badges)
    return "\n".join([".. rst-class:: sd-badges", "", badges_str]).rstrip()


# ---------------------------------------------------------------------------
# Section formatters -- each returns an RST fragment. The directive
# concatenates them into the full page body and parses the result.
# ---------------------------------------------------------------------------


def _format_hero_section(context: Mapping[str, object]) -> str:
    title = str(context.get("title", "")).strip()
    source = str(context.get("source", "")).strip() or "OpenNeuro"

    if title:
        tagline = f"*{title}*"
    else:
        tagline = f"Dataset from {source}."
    tagline = f"{tagline}\n\nAccess recordings and metadata through EEGDash."

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

    citation_count = context.get("nemar_citation_count", "")
    all_badges = [
        ("Modality", str(context.get("modality", ""))),
        ("Subjects", str(context.get("n_subjects", ""))),
        ("Recordings", str(context.get("n_records", ""))),
        ("License", str(context.get("license", ""))),
        ("Source", str(context.get("source", ""))),
    ]
    if citation_count:
        all_badges.append(("Citations", str(citation_count)))

    badges_line = _format_badges(all_badges, outline=True)

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


def _render_identity_cells(
    *,
    dataset_upper: str,
    author_year_name: str,
    canonical_names: object,
) -> tuple[str, str, str]:
    """Build the three identity cells (Author year / Canonical / Importable as)."""
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


def _format_dataset_info_section(context: Mapping[str, object]) -> str:
    dataset_id = str(context.get("dataset_id", ""))
    dataset_upper = str(context.get("dataset_upper", ""))
    title = _value_or_unknown(_clean_value(context.get("title")))
    authors = context.get("authors") or []
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


# ---------------------------------------------------------------------------
# README section -- delegates to .readme.convert_readme_to_rst for the
# Markdown -> RST text transformation, then wraps in a preview/dropdown
# when the converted body is longer than ~30 lines.
# ---------------------------------------------------------------------------


def _format_readme_section(context: Mapping[str, object]) -> str:
    """Format the README content for RST display."""
    readme = _clean_value(context.get("readme"))

    if not readme:
        return "No README content is available for this dataset."

    content = convert_readme_to_rst(readme)
    lines = content.split("\n")

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


# ---------------------------------------------------------------------------
# Electrode-explorer embed (Step 5 of the electrodes integration plan).
#
# `_static/dataset_generated/electrode-layouts.json` maps dataset_id ->
# {label, n_channels, tsv_url, coords_url}. It is eventually populated by
# the eegdash backend montage registry; while that's being built we
# maintain a curated subset as a fallback.
#
# Each dataset page gets a collapsed <details> block. Expanding it swaps
# the iframe's `data-src` onto `src` (see lazy-embed.js), so zero bytes
# are fetched from electrodes.eegdash.org until a reader opts in.
# ---------------------------------------------------------------------------

_ELECTRODE_EXPLORER_BASE = "https://electrodes.eegdash.org/"

_electrode_layouts_cache: dict[str, object] | None = None


def _load_electrode_layouts() -> Mapping[str, Mapping[str, object]]:
    """Read the curated electrode-layouts manifest (cached across calls)."""
    global _electrode_layouts_cache
    if _electrode_layouts_cache is not None:
        return _electrode_layouts_cache  # type: ignore[return-value]
    path = (
        _DOCS_SOURCE_ROOT / "_static" / "dataset_generated" / "electrode-layouts.json"
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
    """Render a lazy <details><iframe> block for this dataset's montage."""
    dataset_id = str(context.get("dataset_id") or "").strip().lower()
    if not dataset_id:
        return ""

    layouts = _load_electrode_layouts()
    entry = layouts.get(dataset_id)

    heading = "Electrode Layout\n----------------\n\n"

    if not entry or not (entry.get("montage_id") or entry.get("tsv_url")):
        body = (
            "No scalp electrode layout is currently indexed for this\n"
            "dataset. Once the eegdash montage registry ingests it,\n"
            "the interactive viewer will appear here automatically.\n"
        )
        return heading + body

    label = str(entry.get("label") or "Electrodes").strip()
    n_channels = entry.get("n_channels")
    montage_id = str(entry.get("montage_id") or "").strip()

    if montage_id:
        query = f"montage={montage_id}"
    else:
        from urllib.parse import quote as _quote

        tsv_q = _quote(str(entry["tsv_url"]), safe="")
        parts = [f"tsv={tsv_q}"]
        coords_url = entry.get("coords_url")
        if coords_url:
            parts.append(f"coords={_quote(str(coords_url), safe='')}")
        query = "&".join(parts)

    iframe_src = f"{_ELECTRODE_EXPLORER_BASE}?{query}&embed=1"

    title_bits = [label]
    if n_channels:
        title_bits.append(f"{n_channels} channels")
    summary_text = " — ".join(title_bits)

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


# BIDS sex column (objects/columns.yaml) defines three phenotypical
# categories, each with a short and long form, all case-insensitive.
# Keys outside all three sets are treated as unknown and folded into
# "Other".
_BIDS_FEMALE_KEYS = {"f", "female"}
_BIDS_MALE_KEYS = {"m", "male"}
_BIDS_OTHER_KEYS = {"o", "other"}


def _format_recording_stats_section(context: Mapping[str, object]) -> str:
    """Generate a Dataset Statistics section from EEGDash API data."""
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

    if has_ages:
        valid_ages = [float(a) for a in ages if a is not None]
        if valid_ages:
            age_min = min(valid_ages)
            age_max = max(valid_ages)
            bucket_size = 5
            buckets: Counter[int] = Counter(
                int(float(a) // bucket_size) * bucket_size for a in valid_ages
            )
            max_count = max(buckets.values())
            bar_width = 28

            bars_html = ""
            labels_html = ""
            for start in sorted(buckets):
                count = buckets[start]
                pct = int(count / max_count * 100)
                label = f"{start}-{start + bucket_size - 1}"
                bars_html += (
                    f'<div style="width:{bar_width}px; height:{pct}%; '
                    f'background:#4472c4; flex-shrink:0;" '
                    f'title="{label}: {count}"></div>'
                )
                labels_html += (
                    f'<span style="width:{bar_width}px; text-align:center; '
                    f'overflow:hidden; white-space:nowrap;">{start}</span>'
                )

            age_html = (
                ".. raw:: html\n\n"
                '   <div class="eegdash-stats-section" style="margin-bottom:1rem;">\n'
                "     <p><strong>Age distribution</strong> "
                f"(n={len(valid_ages)}, range {age_min:.0f}–{age_max:.0f} yr)</p>\n"
                '     <div class="eeg-chart-row" style="display:flex; align-items:flex-end; '
                'gap:2px; height:60px;">\n'
                f"       {bars_html}\n"
                "     </div>\n"
                '     <div class="eeg-chart-labels" style="display:flex; gap:2px; font-size:10px;">\n'
                f"       {labels_html}\n"
                "     </div>\n"
                "   </div>\n\n"
            )
            parts.append(age_html)

    if has_sex:
        f_count = sum(
            int(v or 0) for k, v in sex_dist.items() if k.lower() in _BIDS_FEMALE_KEYS
        )
        m_count = sum(
            int(v or 0) for k, v in sex_dist.items() if k.lower() in _BIDS_MALE_KEYS
        )
        # Explicit BIDS "other" (o/other) plus any non-spec keys
        # (n/a, unknown, ...).
        o_count = sum(
            int(v or 0)
            for k, v in sex_dist.items()
            if k.lower() not in _BIDS_FEMALE_KEYS | _BIDS_MALE_KEYS
        )
        total_sex = f_count + m_count + o_count

        if total_sex > 0:
            f_pct = f_count / total_sex * 100
            m_pct = m_count / total_sex * 100
            o_pct = o_count / total_sex * 100

            bar_segments = ""
            if f_count:
                bar_segments += (
                    f'<div style="width:{f_pct:.1f}%; background:#e07ab5; '
                    "display:inline-flex; align-items:center; justify-content:center; "
                    f'color:#fff; font-size:11px; min-width:2px;" title="Female: {f_count}">'
                    f"{f_count if f_pct >= 8 else ''}</div>"
                )
            if m_count:
                bar_segments += (
                    f'<div style="width:{m_pct:.1f}%; background:#4472c4; '
                    "display:inline-flex; align-items:center; justify-content:center; "
                    f'color:#fff; font-size:11px; min-width:2px;" title="Male: {m_count}">'
                    f"{m_count if m_pct >= 8 else ''}</div>"
                )
            if o_count:
                bar_segments += (
                    f'<div style="width:{o_pct:.1f}%; background:#999; '
                    "display:inline-flex; align-items:center; justify-content:center; "
                    f'color:#fff; font-size:11px; min-width:2px;" title="Other: {o_count}">'
                    f"{o_count if o_pct >= 8 else ''}</div>"
                )

            legend = []
            if f_count:
                legend.append(
                    '<span style="display:inline-block;width:12px;height:12px;'
                    'background:#e07ab5;border-radius:2px;margin-right:4px;"></span>Female'
                )
            if m_count:
                legend.append(
                    '<span style="display:inline-block;width:12px;height:12px;'
                    'background:#4472c4;border-radius:2px;margin-right:4px;"></span>Male'
                )
            if o_count:
                legend.append(
                    '<span style="display:inline-block;width:12px;height:12px;'
                    'background:#999;border-radius:2px;margin-right:4px;"></span>Other'
                )
            legend_html = (
                '<div style="font-size:11px;margin-top:4px;">'
                + "&nbsp;&nbsp;".join(legend)
                + f"&nbsp;&nbsp;<strong>Total: {total_sex}</strong></div>"
            )

            sex_html = (
                ".. raw:: html\n\n"
                '   <div class="eegdash-stats-section" style="margin-bottom:1rem;">\n'
                "     <p><strong>Sex distribution</strong></p>\n"
                '     <div style="display:flex; height:22px; width:100%; max-width:400px; '
                'border-radius:4px; overflow:hidden;">\n'
                f"       {bar_segments}\n"
                "     </div>\n"
                f"     {legend_html}\n"
                "   </div>\n\n"
            )
            parts.append(sex_html)

    if has_nchans:
        parts.append(
            _make_count_bar_chart(
                nchans_counts, "Channel counts", "ch", bar_color="#009E73"
            )
        )

    if has_sfreq:
        parts.append(
            _make_count_bar_chart(
                sfreq_counts, "Sampling frequencies", "Hz", bar_color="#D55E00"
            )
        )

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

    if has_bad_channels:
        try:
            retained_pct = float(bad_channels_info["mean_retained_pct"])  # type: ignore[index]
            n_annotated = int(bad_channels_info["n_annotated"])  # type: ignore[index]
            bar_pct = int(retained_pct)
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

    if not parts:
        return ""

    return heading + "".join(parts)


# ---------------------------------------------------------------------------
# NEMAR per-dataset metadata section
#
# Reuses :class:`eegdash.dataset.nemar.NemarClient` for the network +
# cache work. The client is built once per process (module-level
# memoisation) so its disk cache is hit across the 700+ directive
# invocations in one Sphinx build.
# ---------------------------------------------------------------------------


_nemar_client: NemarClient | None = None


def _get_nemar_client() -> NemarClient:
    """Return the module-shared NEMAR client (built lazily)."""
    global _nemar_client
    if _nemar_client is None:
        _nemar_client = NemarClient()
    return _nemar_client


def _human_readable_size(num_bytes: int) -> str:
    """Bytes → human-readable string (e.g. ``"17.5 GB"``)."""
    if not num_bytes:
        return "0 B"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _format_authors_list(meta: NemarMetadata, *, max_visible: int = 5) -> list[str]:
    """Render the authors block as an RST bullet list.

    Each author becomes one bullet ``- Name (ORCID: 0000-...)`` with an
    embedded external link when an ORCID is present. When the dataset
    has more than ``max_visible`` authors, the trailing tail is collapsed
    into a ``"... and N more"`` line so the section stays readable.
    """
    visible = meta.authors[:max_visible]
    hidden = len(meta.authors) - len(visible)
    lines: list[str] = []
    for author in visible:
        name = author.name.rstrip(".").rstrip()
        if author.orcid:
            lines.append(
                f"- {name} (`ORCID: {author.orcid} "
                f"<https://orcid.org/{author.orcid}>`__)"
            )
        else:
            lines.append(f"- {name}")
    if hidden > 0:
        lines.append(f"- ... and {hidden} more")
    return lines


def _format_keywords_inline(meta: NemarMetadata) -> str:
    """Render keywords as a comma-separated inline list.

    MeSH-tagged keywords become RST external links into the controlled
    vocabulary; plain keywords render as text. Returns ``""`` when no
    keyword survives the cleaning pass.
    """
    parts: list[str] = []
    for kw in meta.keywords:
        term = kw.term.strip()
        if not term:
            continue
        if kw.scheme and kw.scheme.upper() == "MESH" and kw.value_uri:
            parts.append(f"`{term} <{kw.value_uri}>`__")
        else:
            parts.append(term)
    return ", ".join(parts)


def _render_version_rows(rows) -> list[str]:
    """Render an RST list-table for a slice of :class:`NemarVersion`.

    Extracted from :func:`_format_versions_table` so the visible /
    overflow branches share one implementation -- and so the pre-commit
    ``no-nested-functions`` hook stays happy.
    """
    lines = [
        ".. list-table::",
        "   :widths: 20 50 30",
        "   :header-rows: 1",
        "",
        "   * - Version",
        "     - DOI",
        "     - Released",
    ]
    for v in rows:
        doi_link = f"`{v.doi} <https://doi.org/{v.doi}>`__"
        date_str = v.created_at.strftime("%Y-%m-%d")
        lines.append(f"   * - ``{v.version}``")
        lines.append(f"     - {doi_link}")
        lines.append(f"     - {date_str}")
    return lines


def _format_versions_table(meta: NemarMetadata, *, visible_cap: int = 5) -> str:
    """Render the versions history as an RST list-table.

    Columns: ``Version``, ``DOI`` (linked), ``Date`` (UTC, ISO-8601).
    Newest first, capped at ``visible_cap`` visible rows; anything past
    that goes into a collapsed ``dropdown`` so long histories do not
    swamp the page.
    """
    if not meta.versions:
        return ""

    visible = meta.versions[:visible_cap]
    overflow = meta.versions[visible_cap:]

    lines = _render_version_rows(visible)
    if overflow:
        lines.append("")
        lines.append(f".. dropdown:: Older versions ({len(overflow)})")
        lines.append("   :class-container: sd-shadow-sm")
        lines.append("")
        for line in _render_version_rows(overflow):
            lines.append(f"   {line}")
    return "\n".join(lines)


def _format_license_line(meta: NemarMetadata) -> str:
    """Render the license as ``License: <link>`` when SPDX-mappable."""
    if not meta.license:
        return ""
    spdx_key = re.sub(r"\s+", " ", meta.license.strip()).upper()
    url = _LICENSE_URL_MAP.get(spdx_key)
    if url:
        return f"**License**: `{meta.license} <{url}>`__"
    return f"**License**: {meta.license}"


def _maybe_format_manifest_summary(meta: NemarMetadata, *, dataset_id: str) -> str:
    """Optionally fetch the latest manifest and emit a one-line summary.

    Only fetched for EEG datasets and only for the latest version
    (NEMAR manifests can be ~1k entries). Returns ``""`` on any
    failure or empty manifest so the section degrades gracefully.
    """
    if not meta.versions:
        return ""
    has_eeg = any(m.upper() == "EEG" for m in meta.recording_modality)
    if not has_eeg:
        return ""
    try:
        entries = _get_nemar_client().manifest(dataset_id, version=meta.latest_version)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug(
            "[nemar-section] manifest fetch failed for %s: %s", dataset_id, exc
        )
        return ""
    if not entries:
        return ""
    total_size = sum(e.size for e in entries)
    return (
        f"**Manifest** ({meta.latest_version}): {len(entries):,} files, "
        f"{_human_readable_size(total_size)} total"
    )


def _format_nemar_metadata_section(context: Mapping[str, object]) -> str:
    """Render the "NEMAR Metadata" section for NEMAR-sourced datasets.

    The section is emitted only when:

    * the context carries a ``nemar_id`` (always present for NEMAR
      datasets; derived from ``dataset_id`` when it starts with ``nm``),
    * the NEMAR client returns a non-``None`` :class:`NemarMetadata`.

    Otherwise an empty string is returned so the page builder can drop
    the section silently.
    """
    nemar_id = str(context.get("nemar_id") or "").strip()
    if not nemar_id:
        # Allow the section formatter to be wired in unconditionally:
        # non-NEMAR datasets just skip it.
        dataset_id = str(context.get("dataset_id") or "").strip().lower()
        if not dataset_id.startswith("nm"):
            return ""
        nemar_id = dataset_id

    try:
        meta = _get_nemar_client().metadata(nemar_id)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[nemar-section] metadata fetch failed for %s: %s", nemar_id, exc)
        return ""
    if meta is None:
        return ""

    heading = "NEMAR Metadata\n--------------\n\n"
    parts: list[str] = []

    # Prefer NEMAR's description when it is longer than what we already
    # show on the page (the EEGDash chart-data response often carries a
    # shorter summary or none at all).
    page_desc = str(context.get("title") or "").strip()
    nemar_desc = (meta.description or "").strip()
    if nemar_desc and len(nemar_desc) > len(page_desc) + 20:
        parts.append(nemar_desc)

    license_line = _format_license_line(meta)
    if license_line:
        parts.append(license_line)

    if meta.authors:
        author_lines = _format_authors_list(meta)
        parts.append("**Authors**:\n\n" + "\n".join(author_lines))

    keywords_inline = _format_keywords_inline(meta)
    if keywords_inline:
        parts.append(f"**Keywords**: {keywords_inline}")

    versions_block = _format_versions_table(meta)
    if versions_block:
        parts.append("**Versions**:\n\n" + versions_block)

    manifest_summary = _maybe_format_manifest_summary(meta, dataset_id=nemar_id)
    if manifest_summary:
        parts.append(manifest_summary)

    if not parts:
        return ""

    return heading + "\n\n".join(parts) + "\n"


def _format_nemar_analysis_section(context: Mapping[str, object]) -> str:
    """Embed NEMAR pre-generated pipeline analysis plots for OpenNeuro/NEMAR datasets."""
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


# Used by ``_format_explorer_section`` to gate the dataset_id passed
# through to the dataset-explorer directive. Kept here (not in
# ``_constants``) because it is the *explorer*'s grammar, not the
# directive's: it intentionally mirrors but does not share the directive
# argument regex.
from ._constants import DATASET_NAME_RE as _EXPLORER_DATASET_RE  # noqa: E402


def _format_explorer_section(name: str, context: Mapping[str, object]) -> str:
    """Render the BIDS file explorer directive for this dataset."""
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

    Searches for any compatible modality (EEG, iEEG, EMG, MEG) that can
    be viewed with the eegdash-viewer, prioritizing EEG.
    """
    import urllib.parse

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
    """Render an iframe for this dataset's signal preview."""
    dataset_id = str(context.get("dataset_id") or "").strip().lower()
    if not dataset_id:
        return ""

    record = _get_first_eeg_record(dataset_id)
    if not record:
        return ""

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
    if suffix != "eeg":
        qs_pairs.append(("suffix", suffix))
    qs_pairs.append(("embed", "1"))
    iframe_src = f"{_TRACE_VIEWER_BASE}?{urlencode(qs_pairs)}"

    entity_bits = [f"sub-{sub}"]
    if ses:
        entity_bits.append(f"ses-{ses}")
    if task:
        entity_bits.append(f"task-{task}")
    if run:
        entity_bits.append(f"run-{run}")
    entity_label = " · ".join(entity_bits)

    n_subjects = context.get("n_subjects")
    n_records = context.get("n_records")
    scope_bits = []
    if n_subjects:
        scope_bits.append(f"{n_subjects} subjects")
    if n_records:
        scope_bits.append(f"{n_records} recordings")
    scope_str = " and ".join(scope_bits) if scope_bits else "many recordings"

    openneuro_url = f"https://openneuro.org/datasets/{dataset_id}"

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
    # internal link density across the dataset pages.
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
      like ``:html_theme.sidebar_secondary.remove:`` -- any blank line
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


def _license_text_to_url(text: str) -> str | None:
    """Map a free-form license string to a canonical URL if we can.

    Google's Rich Results validator prefers license URLs over names.
    We normalise by stripping whitespace and uppercasing; everything we
    can't match falls through so the caller keeps the raw text.
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
    JSON-LD spec -- Google Dataset Search reads either head or body).
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

    # Skip the leading title when it duplicates `dataset_upper` -- the
    # trailing "...as ``class_name``..." already names the dataset, and
    # "{title}. {modality} dataset accessible via EEGDash as ``{title}``"
    # reads "ABSEQMEG. EEG dataset accessible via EEGDash as ``ABSEQMEG``"
    # which is triple-redundant for the common case where title is just
    # the uppercased class name.
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
    # author whose metadata happens to contain `</script>` would escape
    # the script element and inject arbitrary HTML into the page.
    # Escaping `</` -> `<\/` is valid JSON *and* safe inside `<script>`
    # per the HTML spec.
    payload = payload.replace("</", "<\\/")

    return f'.. raw:: html\n\n   <script type="application/ld+json">{payload}</script>'
