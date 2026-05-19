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
from datetime import datetime
from typing import Mapping, Sequence
from urllib.parse import quote

from sphinx.util import logging

from eegdash.dataset.nemar import NemarClient, NemarMetadata

from ._constants import (
    _BIDS_FEMALE_KEYS,
    _BIDS_MALE_KEYS,
    _BIDS_TASK_CAMEL_RE,
    _LICENSE_URL_MAP,
    _MODALITY_LABEL_MAP,
    _RST_HYPERLINK_RE,
    _SOURCE_LABEL_MAP,
)
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


def _is_lede_prose(text: str) -> bool:
    """Return True iff ``text`` is a plain prose line (not an RST directive,
    bullet, badge row, or bold-only header). Used by
    :func:`_split_lede_paragraphs` to find the first real paragraph in a
    README so it can carry the drop-cap.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(("..", "::", "#", "+", "|", ">")):
        return False
    if stripped.startswith(("- ", "+ ", ":")):
        return False
    compact = stripped.rstrip(".:!?")
    if compact.startswith("**") and compact.endswith("**") and compact.count("**") == 2:
        return False
    if compact.startswith("*") and compact.endswith("*") and compact.count("*") == 2:
        return False
    without_links = _RST_HYPERLINK_RE.sub("", stripped).strip()
    if not without_links:
        return False
    return True


def _split_lede_paragraphs(
    lines: Sequence[str], max_paragraphs: int = 2
) -> tuple[list[str], list[str]]:
    """Return (lede_paragraphs, remainder_lines).

    A paragraph is a run of consecutive non-blank lines that doesn't start
    with an RST directive marker (``..``), heading underline, or list
    bullet — i.e. ordinary prose. We collect at most ``max_paragraphs``
    such paragraphs and hand the rest back as ``remainder_lines``.
    """
    paragraphs: list[str] = []
    remainder_start = 0
    i = 0
    n = len(lines)

    preamble_end = 0
    seen_prose = False

    while i < n and len(paragraphs) < max_paragraphs:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        if not _is_lede_prose(lines[i]):
            while i < n and lines[i].strip() and not _is_lede_prose(lines[i]):
                i += 1
            if not seen_prose:
                preamble_end = i
            continue
        seen_prose = True
        start = i
        while i < n and lines[i].strip():
            i += 1
        paragraphs.append("\n".join(lines[start:i]).strip())
        remainder_start = i

    if paragraphs:
        remainder = list(lines[:preamble_end]) + list(lines[remainder_start:])
    else:
        remainder = list(lines)

    return paragraphs, remainder


def _format_readme_section(context: Mapping[str, object]) -> str:
    """Format the README content for RST display.

    The first two prose paragraphs are wrapped in an ``eegdash-ed-lede``
    container so the editorial CSS can render the two-column drop-cap
    intro from the editorial design.
    """
    readme = _clean_value(context.get("readme"))

    if not readme:
        return "No README content is available for this dataset."

    content = convert_readme_to_rst(readme)
    lines = content.split("\n")

    # Pull the first two non-empty paragraphs to render under the dropcap.
    lede_paragraphs, remainder_lines = _split_lede_paragraphs(lines, max_paragraphs=2)
    lede_block = ""
    if lede_paragraphs:
        inner = []
        for para in lede_paragraphs:
            stripped = para.strip()
            if stripped:
                inner.append(stripped)
        if inner:
            paras_rst = "\n\n".join(f"   {p}" for p in inner)
            lede_block = f".. container:: eegdash-ed-lede\n\n{paras_rst}\n\n"

    remainder = "\n".join(remainder_lines).strip("\n")

    if remainder:
        rem_lines = remainder.split("\n")
        if len(rem_lines) > 30:
            preview = "\n".join(rem_lines[:10])
            indented = "\n".join(f"   {line}" for line in rem_lines)
            remainder = (
                f"{preview}\n\n"
                ".. dropdown:: View full README\n"
                "   :class-container: sd-shadow-sm\n"
                f"\n{indented}\n"
            )

    return f"{lede_block}\n{remainder}".strip()


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


def _format_api_section(
    class_name: str, context: Mapping[str, object] | None = None
) -> str:
    """Format the API section.

    Two parts:

    1. An editorial **Signature** card (raw HTML) that mirrors the
       editorial §05 Access design: left-gutter ``SIGNATURE`` label,
       a code-styled class signature, and an identifier table backed
       by real registry data.
    2. The Sphinx ``autoclass`` block, restyled by editorial.css.
    """
    autoclass = (
        ".. currentmodule:: eegdash.dataset\n\n"
        f".. autoclass:: eegdash.dataset.{class_name}\n"
        "   :members: __init__, save\n"
        "   :member-order: bysource\n"
    )

    if context is None:
        return autoclass

    # --- Editorial signature card -----------------------------------------
    dataset_upper = class_name.upper()
    author_year_name = _clean_value(context.get("author_year_name"))
    canonical_names = context.get("canonical_names") or []
    if not isinstance(canonical_names, (list, tuple)):
        canonical_names = []

    importable = [dataset_upper]
    if author_year_name and author_year_name not in importable:
        importable.append(author_year_name)
    for n in canonical_names:
        if n and n not in importable:
            importable.append(n)

    importable_html = " · ".join(f"<code>{n}</code>" for n in importable)
    author_year_html = f"<b>{author_year_name}</b>" if author_year_name else "—"
    canonical_display = [n for n in canonical_names if n and n != author_year_name]
    canonical_html = (
        " · ".join(f"<code>{n}</code>" for n in canonical_display)
        if canonical_display
        else "—"
    )

    source_path = "eegdash/dataset/registry.py"
    github_url = f"https://github.com/eegdash/EEGDash/blob/develop/{source_path}"

    signature_card = (
        ".. raw:: html\n\n"
        '   <div class="eegdash-ed-apicard">\n'
        '     <div class="apicard-gutter">'
        '<div class="lbl">Signature</div>'
        '<div class="cls"><code>eegdash.dataset</code></div>'
        "</div>\n"
        '     <div class="apicard-body">\n'
        '       <div class="apicard-sig">\n'
        '         <div class="sig-kind">class</div>\n'
        f'         <div class="sig-line">'
        '<span class="ns">eegdash.dataset.</span>'
        f'<b class="cls-name">{class_name}</b>'
        '<span class="paren">(</span>'
        '<span class="arg">cache_dir</span>, '
        '<span class="arg">query</span>=<span class="lit">None</span>, '
        '<span class="arg">s3_bucket</span>=<span class="lit">None</span>, '
        '<span class="arg">**kwargs</span>'
        '<span class="paren">)</span>'
        "</div>\n"
        '         <div class="sig-base">Bases: <code>EEGDashDataset</code></div>\n'
        "       </div>\n"
        '       <div class="apicard-ids">\n'
        '         <div class="id-row">'
        '<span class="k">Author (year)</span>'
        f'<span class="v">{author_year_html}</span>'
        "</div>\n"
        '         <div class="id-row">'
        '<span class="k">Canonical</span>'
        f'<span class="v">{canonical_html}</span>'
        "</div>\n"
        '         <div class="id-row">'
        '<span class="k">Importable as</span>'
        f'<span class="v">{importable_html}</span>'
        "</div>\n"
        '         <div class="id-row">'
        '<span class="k">Source</span>'
        f'<span class="v"><code>{source_path}</code> · '
        f'<a href="{github_url}">[source ↗]</a></span>'
        "</div>\n"
        "       </div>\n"
        "     </div>\n"
        "   </div>\n"
    )

    return signature_card + "\n" + autoclass


# ---------------------------------------------------------------------------
# Electrode-explorer embed (Step 5 of the electrodes integration plan).
#
# Previously fed by ``docs/build_electrode_layouts.py`` writing
# ``_static/dataset_generated/electrode-layouts.json`` from ~700
# per-dataset GETs. Arch #5 retired that script: the server now joins
# the top montage onto each ``/datasets/chart-data`` row when called
# with ``?include=montages``, and ``DatasetSnapshot`` exposes it
# through :meth:`DatasetSnapshot.montage`. The lookup shape is
# unchanged (``label``, ``n_channels``, ``montage_id``) so this
# section formatter is a straight rewire.
#
# Each dataset page gets a collapsed <details> block. Expanding it
# swaps the iframe's ``data-src`` onto ``src`` (see lazy-embed.js),
# so zero bytes are fetched from electrodes.eegdash.org until a
# reader opts in.
# ---------------------------------------------------------------------------

_ELECTRODE_EXPLORER_BASE = "https://electrodes.eegdash.org/"


def _load_electrode_layout(dataset_id: str) -> Mapping[str, object] | None:
    """Return the top montage for ``dataset_id`` via the snapshot.

    Returns ``None`` when:

    * the dataset id is empty / falsy,
    * the snapshot fell back to disk-cache / package-CSV without a
      cached montages sidecar (live API unreachable for this build),
    * the server did not join a montage onto that dataset row, or
    * the snapshot's montage payload is structurally unusable.

    The caller renders an empty placeholder for ``None``, so each of
    these states degrades gracefully into "no layout indexed".
    """
    if not dataset_id:
        return None
    try:
        from eegdash.dataset.snapshot import DatasetSnapshot

        snapshot = DatasetSnapshot.build()
        return snapshot.montage(dataset_id)
    except Exception as exc:  # noqa: BLE001 — must never break the docs build
        LOGGER.info(
            "[electrode-layouts] snapshot lookup failed for %s (%s); placeholder",
            dataset_id,
            exc,
        )
        return None


def _format_electrodes_section(context: Mapping[str, object]) -> str:
    """Render a lazy <details><iframe> block for this dataset's montage."""
    dataset_id = str(context.get("dataset_id") or "").strip().lower()
    if not dataset_id:
        return ""

    entry = _load_electrode_layout(dataset_id)

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


def _is_positive_float(value: object) -> bool:
    """Truthy iff ``value`` can be coerced to a finite positive float."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f > 0


def _donut_arc(
    pct: float, color: str, offset: float, circumference: float
) -> tuple[str, float]:
    """Return the SVG ``<circle>`` for one arc of a donut chart plus the
    advanced stroke-dashoffset for the next arc. ``circumference`` is
    treated as 100% so each arc's stroke-dasharray reads as a percentage.
    """
    if pct <= 0:
        return "", offset
    length = pct / 100.0 * circumference
    gap = circumference - length
    piece = (
        f'<circle cx="21" cy="21" r="15.9" fill="none" stroke="{color}" '
        f'stroke-width="5" stroke-dasharray="{length:.3f} {gap:.3f}" '
        f'stroke-dashoffset="{offset:.3f}" transform="rotate(-90 21 21)" '
        'stroke-linecap="butt"/>'
    )
    next_offset = offset - length
    if next_offset < 0:
        next_offset += circumference
    return piece, next_offset


def _render_sex_donut(
    f_count: int,
    m_count: int,
    o_count: int,
    total: int,
    *,
    handedness: Mapping[str, object] | None = None,
) -> str:
    """Render the sex distribution as an SVG donut chart + side legend."""
    pct_f = f_count / total * 100 if total else 0.0
    pct_m = m_count / total * 100 if total else 0.0
    pct_o = o_count / total * 100 if total else 0.0

    C = 100.0

    arc_pieces: list[str] = []
    # ``offset`` tracks the stroke-dashoffset for the next arc; each call
    # to ``_donut_arc`` returns (svg_piece, new_offset) so we can iterate
    # without closure state.
    offset = 25.0

    arc_pieces.append(
        '<circle cx="21" cy="21" r="15.9" fill="none" stroke="#d0d6dc" '
        'stroke-width="5"/>'
    )
    for pct, color in ((pct_f, "#006ca3"), (pct_m, "#f7941d"), (pct_o, "#6b7785")):
        piece, offset = _donut_arc(pct, color, offset, C)
        arc_pieces.append(piece)

    arcs_svg = "".join(p for p in arc_pieces if p)

    legend_rows = []
    if f_count:
        legend_rows.append(("Female", f_count, "#006ca3"))
    if m_count:
        legend_rows.append(("Male", m_count, "#f7941d"))
    if o_count:
        legend_rows.append(("Other", o_count, "#6b7785"))
    rows_html = ""
    for label, count, color in legend_rows:
        rows_html += (
            '<div class="row">'
            f'<div class="sw" style="background:{color}"></div>'
            f'<div class="lbl">{label}</div>'
            f'<div class="v">{count}</div>'
            "</div>"
        )

    if f_count and m_count:
        ratio = f_count / m_count
        rows_html += (
            '<div class="row ratio">'
            '<div class="sw" style="visibility:hidden"></div>'
            '<div class="lbl">F : M ratio</div>'
            f'<div class="v">{ratio:.2f} : 1</div>'
            "</div>"
        )

    center_html = (
        '<foreignObject x="0" y="0" width="42" height="42">'
        '<div xmlns="http://www.w3.org/1999/xhtml" '
        'style="width:100%;height:100%;display:flex;flex-direction:column;'
        'align-items:center;justify-content:center;font-family:Spectral,Georgia,serif;">'
        f'<div style="font-size:13px;line-height:1;letter-spacing:-.02em">{total}</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:2.8px;'
        "letter-spacing:.18em;color:#6a6e75;margin-top:1.5px;"
        'text-transform:uppercase">subjects</div>'
        "</div>"
        "</foreignObject>"
    )

    summary_note = ""
    if f_count and m_count:
        f_pct_int = round(pct_f)
        summary_note = (
            '<div class="sex-note" style="margin-top:14px; font-size:13px; '
            "color:#34404e; line-height:1.5; max-width:520px; "
            'font-family:Spectral,Georgia,serif;">'
            f"{f_pct_int}% female · n = {total} subjects with reported sex."
            "</div>"
        )

    handedness_chip = ""
    if handedness:
        h_buckets = {"right": 0, "left": 0, "ambi": 0, "unknown": 0}
        for k, v in handedness.items():
            try:
                count = int(v or 0)
            except (TypeError, ValueError):
                continue
            key = str(k).strip().lower()
            if key in ("r", "right"):
                h_buckets["right"] += count
            elif key in ("l", "left"):
                h_buckets["left"] += count
            elif key in ("a", "ambi", "ambidextrous"):
                h_buckets["ambi"] += count
            else:
                h_buckets["unknown"] += count
        h_total = sum(h_buckets.values())
        if h_total > 0:
            chips = []
            label_map = {
                "right": "Right",
                "left": "Left",
                "ambi": "Ambidextrous",
                "unknown": "Unknown",
            }
            for k, label in label_map.items():
                if h_buckets[k]:
                    chips.append(
                        '<span style="display:inline-block; padding:2px 8px; '
                        "margin-right:6px; font-family:Inter,sans-serif; "
                        "font-size:11px; letter-spacing:.04em; "
                        "background:rgba(0,108,163,0.08); color:#1a2532; "
                        'border-radius:3px;">'
                        f"{label} · {h_buckets[k]}"
                        "</span>"
                    )
            handedness_chip = (
                '<div class="hand-row" style="margin-top:10px; font-size:11px; '
                'color:#6b7785; font-family:Inter,sans-serif;">'
                '<span style="text-transform:uppercase; letter-spacing:.16em; '
                'margin-right:10px;">Handedness</span>' + "".join(chips) + "</div>"
            )

    return (
        ".. raw:: html\n\n"
        '   <div class="eegdash-stats-section eegdash-ed-sex" '
        'style="margin-bottom:1rem;">\n'
        "     <p><strong>Sex composition</strong></p>\n"
        '     <div class="sex-wrap" style="display:flex; align-items:center; '
        'gap:30px; flex-wrap:wrap;">\n'
        '       <svg class="sex-donut" viewBox="0 0 42 42" '
        'style="width:170px; height:170px; flex-shrink:0;">'
        f"{arcs_svg}"
        f"{center_html}"
        "</svg>\n"
        '       <div class="sex-legend" '
        'style="flex:1; font-family:JetBrains Mono,monospace; '
        'font-size:13px; min-width:220px;">'
        f"{rows_html}"
        "</div>\n"
        "     </div>\n"
        f"     {summary_note}\n"
        f"     {handedness_chip}\n"
        "   </div>\n\n"
    )


def _format_recording_stats_section(context: Mapping[str, object]) -> str:
    """Generate a Dataset Statistics section from EEGDash API data.

    Renders inline HTML bar charts and text stats for age distribution,
    sex distribution, channel counts, sampling frequencies, and total
    recording duration. Returns an empty string when no useful data
    is present so the template placeholder collapses silently.
    """
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
    # Open a 2-column grid wrapper so the age histogram and the sex
    # donut render side-by-side (matching the editorial design).
    parts.append('.. raw:: html\n\n   <div class="eegdash-ed-cohort-grid">\n')

    # ------------------------------------------------------------------
    # A. Age distribution — vertical bars stacked by gender
    # ------------------------------------------------------------------
    participants_rows = context.get("participants_rows") or []

    pair_buckets_f: Counter[int] = Counter()
    pair_buckets_m: Counter[int] = Counter()
    pair_buckets_o: Counter[int] = Counter()
    bucket_size = 5
    paired_age_count = 0
    for p in participants_rows:
        try:
            a = float(p.get("age"))
        except (TypeError, ValueError):
            continue
        if a <= 0:
            continue
        sex = str(p.get("sex") or "").strip().lower()
        bucket_start = int(a // bucket_size) * bucket_size
        if sex in _BIDS_FEMALE_KEYS:
            pair_buckets_f[bucket_start] += 1
        elif sex in _BIDS_MALE_KEYS:
            pair_buckets_m[bucket_start] += 1
        else:
            pair_buckets_o[bucket_start] += 1
        paired_age_count += 1

    if pair_buckets_f or pair_buckets_m or pair_buckets_o:
        all_buckets = sorted(
            set(pair_buckets_f) | set(pair_buckets_m) | set(pair_buckets_o)
        )
        n_f = sum(pair_buckets_f.values())
        n_m = sum(pair_buckets_m.values())
        n_o = sum(pair_buckets_o.values())
        max_total = max(
            pair_buckets_f[b] + pair_buckets_m[b] + pair_buckets_o[b]
            for b in all_buckets
        )
        bar_width = 28
        chart_height = 80

        ages_used = [
            float(p.get("age"))
            for p in participants_rows
            if str(p.get("age")) not in ("", "None", "n/a")
            and _is_positive_float(p.get("age"))
        ]
        age_min_v = min(ages_used) if ages_used else 0
        age_max_v = max(ages_used) if ages_used else 0

        bars_html = ""
        labels_html = ""
        for start in all_buckets:
            f = pair_buckets_f.get(start, 0)
            m = pair_buckets_m.get(start, 0)
            o = pair_buckets_o.get(start, 0)
            tot = f + m + o
            col_pieces = ""
            if o:
                h = int(o / max_total * chart_height)
                col_pieces += (
                    f'<div style="width:{bar_width}px; height:{h}px; '
                    f'background:#6b7785; flex-shrink:0;" '
                    f'title="{start}-{start + bucket_size - 1}: other n={o}">'
                    "</div>"
                )
            if f:
                h = int(f / max_total * chart_height)
                col_pieces += (
                    f'<div style="width:{bar_width}px; height:{h}px; '
                    f'background:#006ca3; flex-shrink:0;" '
                    f'title="{start}-{start + bucket_size - 1}: female n={f}">'
                    "</div>"
                )
            if m:
                h = int(m / max_total * chart_height)
                col_pieces += (
                    f'<div style="width:{bar_width}px; height:{h}px; '
                    f'background:#f7941d; flex-shrink:0;" '
                    f'title="{start}-{start + bucket_size - 1}: male n={m}">'
                    "</div>"
                )
            bars_html += (
                f'<div style="display:flex; flex-direction:column-reverse; '
                f'justify-content:flex-start; gap:1px;" '
                f'title="{start}-{start + bucket_size - 1}: n={tot}">'
                f"{col_pieces}"
                "</div>"
            )
            labels_html += (
                f'<span style="width:{bar_width}px; text-align:center; '
                f'overflow:hidden; white-space:nowrap;">{start}</span>'
            )

        legend_pieces = []
        if n_f:
            legend_pieces.append(
                '<span style="display:inline-flex; align-items:center; gap:6px;">'
                '<i style="width:10px; height:10px; background:#006ca3; '
                'display:inline-block;"></i>'
                f"Female · {n_f}</span>"
            )
        if n_m:
            legend_pieces.append(
                '<span style="display:inline-flex; align-items:center; gap:6px;">'
                '<i style="width:10px; height:10px; background:#f7941d; '
                'display:inline-block;"></i>'
                f"Male · {n_m}</span>"
            )
        if n_o:
            legend_pieces.append(
                '<span style="display:inline-flex; align-items:center; gap:6px;">'
                '<i style="width:10px; height:10px; background:#6b7785; '
                'display:inline-block;"></i>'
                f"Other · {n_o}</span>"
            )
        legend_html = (
            (
                '<div style="display:flex; gap:18px; margin-top:8px; '
                'font-size:11px;">' + "".join(legend_pieces) + "</div>"
            )
            if legend_pieces
            else ""
        )

        age_mean_v = demographics.get("age_mean")
        try:
            mean_str = (
                f", mean {float(age_mean_v):.1f} yr" if age_mean_v is not None else ""
            )
        except (TypeError, ValueError):
            mean_str = ""

        age_html = (
            ".. raw:: html\n\n"
            '   <div class="eegdash-stats-section" style="margin-bottom:1rem;">\n'
            "     <p><strong>Age distribution by gender</strong> "
            f"(n={paired_age_count}, range {age_min_v:.0f}–{age_max_v:.0f} yr"
            f"{mean_str})</p>\n"
            f'     <div class="eeg-chart-row" style="display:flex; align-items:flex-end; '
            f'gap:2px; height:{chart_height}px; border-bottom:1px solid #34404e;">\n'
            f"       {bars_html}\n"
            "     </div>\n"
            '     <div class="eeg-chart-labels" style="display:flex; gap:2px; font-size:10px;">\n'
            f"       {labels_html}\n"
            "     </div>\n"
            f"     {legend_html}\n"
            "   </div>\n\n"
        )
        parts.append(age_html)

    elif has_ages:
        # Fall back to single-color age histogram when per-subject
        # demographics aren't available.
        valid_ages = [float(a) for a in ages if a is not None]
        if valid_ages:
            age_min = min(valid_ages)
            age_max = max(valid_ages)
            buckets: Counter[int] = Counter(
                int(float(a) // bucket_size) * bucket_size for a in valid_ages
            )
            max_count = max(buckets.values())
            bar_width = 28
            chart_height = 80
            bars_html = ""
            labels_html = ""
            for start in sorted(buckets):
                count = buckets[start]
                h = int(count / max_count * chart_height)
                bars_html += (
                    f'<div style="width:{bar_width}px; height:{h}px; '
                    f'background:#006ca3; flex-shrink:0;" '
                    f'title="{start}-{start + bucket_size - 1}: {count}"></div>'
                )
                labels_html += (
                    f'<span style="width:{bar_width}px; text-align:center; '
                    f'overflow:hidden; white-space:nowrap;">{start}</span>'
                )
            mean_v = demographics.get("age_mean")
            try:
                mean_str = (
                    f", mean {float(mean_v):.1f} yr" if mean_v is not None else ""
                )
            except (TypeError, ValueError):
                mean_str = ""
            age_html = (
                ".. raw:: html\n\n"
                '   <div class="eegdash-stats-section" style="margin-bottom:1rem;">\n'
                "     <p><strong>Age distribution</strong> "
                f"(n={len(valid_ages)}, range {age_min:.0f}–{age_max:.0f} yr"
                f"{mean_str} · sex per subject not reported)</p>\n"
                f'     <div class="eeg-chart-row" style="display:flex; align-items:flex-end; '
                f'gap:2px; height:{chart_height}px; border-bottom:1px solid #34404e;">\n'
                f"       {bars_html}\n"
                "     </div>\n"
                '     <div class="eeg-chart-labels" style="display:flex; gap:2px; font-size:10px;">\n'
                f"       {labels_html}\n"
                "     </div>\n"
                "   </div>\n\n"
            )
            parts.append(age_html)

    # ------------------------------------------------------------------
    # B. Sex distribution — SVG donut
    # ------------------------------------------------------------------
    if has_sex:
        f_count = sum(
            int(v or 0) for k, v in sex_dist.items() if k.lower() in _BIDS_FEMALE_KEYS
        )
        m_count = sum(
            int(v or 0) for k, v in sex_dist.items() if k.lower() in _BIDS_MALE_KEYS
        )
        o_count = sum(
            int(v or 0)
            for k, v in sex_dist.items()
            if k.lower() not in _BIDS_FEMALE_KEYS | _BIDS_MALE_KEYS
        )
        total_sex = f_count + m_count + o_count

        if total_sex > 0:
            handedness = demographics.get("handedness_distribution") or {}
            parts.append(
                _render_sex_donut(
                    f_count,
                    m_count,
                    o_count,
                    total_sex,
                    handedness=handedness if isinstance(handedness, dict) else None,
                )
            )

    # ------------------------------------------------------------------
    # C. Channel count distribution
    # ------------------------------------------------------------------
    if has_nchans:
        parts.append(
            _make_count_bar_chart(
                nchans_counts, "Channel counts", "ch", bar_color="#009E73"
            )
        )

    # ------------------------------------------------------------------
    # D. Sampling frequency distribution
    # ------------------------------------------------------------------
    if has_sfreq:
        parts.append(
            _make_count_bar_chart(
                sfreq_counts, "Sampling frequencies", "Hz", bar_color="#D55E00"
            )
        )

    # ------------------------------------------------------------------
    # E. Total recording duration
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # F. BIDS-annotated channel retention
    # ------------------------------------------------------------------
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

    if len(parts) <= 1:
        # Only the wrapper open was added (no chart parts) — skip the
        # section entirely.
        return ""

    # Close the 2-column grid wrapper we opened at the top.
    parts.append(".. raw:: html\n\n   </div>\n")

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


# ===========================================================================
# Editorial Brief — section formatters and value helpers
# ===========================================================================
# Re-skin chrome lifted from the v1-editorial-v2 design. Each
# ``_format_editorial_*_section`` returns an RST fragment that the
# directive concatenates into the full page body. The functions read
# from the same ``context`` slice as the rest of the section formatters.


def _editorial_html(block: str) -> str:
    """Wrap an HTML block in a Sphinx ``.. raw:: html`` directive.

    Indents the payload by three spaces (required by the directive) and
    strips a trailing newline so successive editorial blocks don't
    accumulate empty paragraphs between them.
    """
    indented = "\n".join("   " + line if line else "" for line in block.split("\n"))
    return ".. raw:: html\n\n" + indented.rstrip()


def _short_study_label(context: Mapping[str, object]) -> str:
    """Trim the study title for use as a kicker / breadcrumb tail."""
    title = _clean_value(context.get("title")) or str(context.get("class_name", ""))
    short = title.split(":", 1)[0].strip() if ":" in title else title
    if len(short) > 64:
        short = short[:61].rstrip(" ,.;:") + "…"
    return short or str(context.get("class_name", "")) or "Dataset"


# ---------------------------------------------------------------------------
# Editorial value formatters
# ---------------------------------------------------------------------------


def _editorial_source_label(value: str) -> str:
    """Title-case a source identifier (``openneuro`` → ``OpenNeuro``)."""
    if not value:
        return "OpenNeuro"
    key = value.strip().lower()
    return _SOURCE_LABEL_MAP.get(key, value.strip().title())


def _editorial_modality_label(value: str) -> str:
    """Convert a lowercase modality token to the canonical display form."""
    if not value:
        return "EEG"
    cleaned = value.strip()
    key = cleaned.lower()
    if key in _MODALITY_LABEL_MAP:
        return _MODALITY_LABEL_MAP[key]
    parts = [
        _MODALITY_LABEL_MAP.get(p.strip().lower(), p.strip().upper())
        for p in cleaned.split(",")
        if p.strip()
    ]
    return ", ".join(parts) if parts else cleaned.upper()


def _editorial_sfreq_label(context: Mapping[str, object]) -> str:
    """Compact sampling-rate label like ``250 Hz`` (or ``250, 1000 Hz``)."""
    counts = context.get("sfreq_counts") or []
    if isinstance(counts, list) and counts:
        try:
            valid = [
                (int(round(float(c.get("val")))), int(c.get("count") or 0))
                for c in counts
                if isinstance(c, dict) and c.get("val") is not None
            ]
        except (TypeError, ValueError):
            valid = []
        if valid:
            unique_vals = sorted({v for v, _ in valid})
            if len(unique_vals) == 1:
                return f"{unique_vals[0]} Hz"
            total = sum(c for _, c in valid) or 1
            top_val, top_count = max(valid, key=lambda x: x[1])
            if top_count / total >= 0.8:
                return f"{top_val} Hz · mixed"
            return ", ".join(f"{v}" for v in unique_vals) + " Hz"

    raw = _value_or_unknown(
        _clean_value(context.get("sampling_freqs")), "sampling_rate"
    )
    if raw in ("Varies", "—"):
        return raw
    return f"{raw} Hz" if not raw.endswith("Hz") else raw


def _editorial_duration_label(context: Mapping[str, object]) -> str:
    """Round duration to a clean ``XX.X h`` value (or ``Y min`` when short)."""
    total_duration_s = context.get("total_duration_s")
    seconds: float | None = None
    if total_duration_s is not None:
        try:
            seconds = float(total_duration_s)
        except (TypeError, ValueError):
            seconds = None
    if seconds is None:
        raw = _clean_value(context.get("duration_hours_total"))
        if raw:
            try:
                seconds = float(raw) * 3600.0
            except (TypeError, ValueError):
                return raw
    if seconds is None or seconds <= 0:
        return "—"
    hours = seconds / 3600.0
    if hours < 1:
        return f"{int(round(seconds / 60))} min"
    if hours >= 100:
        return f"{int(round(hours))} h"
    return f"{hours:.1f} h"


def _editorial_citation_label(value: object) -> str:
    """Drop a stray .0 from NEMAR citation counts (``1.0`` → ``1``)."""
    if value is None:
        return "—"
    text = str(value).strip()
    if not text or text == "—":
        return "—"
    try:
        f = float(text)
    except (TypeError, ValueError):
        return text
    if f == int(f):
        return str(int(f))
    return f"{f:.1f}"


def _humanise_bids_task(token: str) -> str:
    """Turn a BIDS task entity (``neurCorrYoung``) into a sentence-case
    label (``Neur Corr Young``). Underscores and hyphens are also
    softened to spaces.
    """
    if not token:
        return ""
    parts = re.split(r"[_\-]+", token)
    words: list[str] = []
    for part in parts:
        if not part:
            continue
        for sub in _BIDS_TASK_CAMEL_RE.split(part):
            if sub:
                words.append(sub[0].upper() + sub[1:])
    return " ".join(words)


def _render_task_token(tok: str) -> str:
    """Wrap a BIDS task token in ``<code>``, annotating with a humanised
    title attribute when the camelCase split actually changes the text.
    """
    nice = _humanise_bids_task(tok)
    if nice and nice.lower().replace(" ", "") != tok.lower():
        return f'<code title="{nice}">{tok}</code>'
    return f"<code>{tok}</code>"


def _editorial_tasks_label(tasks: Sequence[str]) -> str:
    """Compact label for a tasks list."""
    cleaned = [str(t).strip() for t in (tasks or []) if str(t).strip()]
    if not cleaned:
        return "—"

    if len(cleaned) == 1:
        return _render_task_token(cleaned[0])
    if len(cleaned) <= 3:
        return " · ".join(_render_task_token(t) for t in cleaned)
    head = " · ".join(_render_task_token(t) for t in cleaned[:2])
    return f"{len(cleaned)} tasks · {head} · …"


def _editorial_sessions_label(sessions: Sequence[str]) -> str:
    """Compact label for the BIDS sessions list."""
    cleaned = [str(s).strip() for s in (sessions or []) if str(s).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return f"<code>{cleaned[0]}</code>"
    if len(cleaned) <= 3:
        return f"{len(cleaned)} · " + " · ".join(f"<code>{s}</code>" for s in cleaned)
    return f"{len(cleaned)} · <code>{cleaned[0]}</code> … <code>{cleaned[-1]}</code>"


def _editorial_contact_label(contacts: Sequence[str]) -> str:
    """Comma-joined contact list, truncated to 3 names + ellipsis."""
    cleaned = [str(c).strip() for c in (contacts or []) if str(c).strip()]
    if not cleaned:
        return ""
    head = ", ".join(cleaned[:3])
    if len(cleaned) > 3:
        head += f" · +{len(cleaned) - 3}"
    return head


def _editorial_updated_label(iso_timestamp: str) -> str:
    """Format an ISO-8601 timestamp as ``YYYY-MM-DD`` for the field card."""
    if not iso_timestamp:
        return ""
    cleaned = iso_timestamp.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return iso_timestamp[:10] if len(iso_timestamp) >= 10 else ""
    return dt.date().isoformat()


# ---------------------------------------------------------------------------
# Editorial sections
# ---------------------------------------------------------------------------


def _format_editorial_kicker_section(context: Mapping[str, object]) -> str:
    """Editorial kicker + issue strip rendered above the H1."""
    class_name = str(context.get("class_name", "")).strip()
    short_label = _short_study_label(context)
    source = _editorial_source_label(_clean_value(context.get("source")))
    n_subjects = _value_or_unknown(_clean_value(context.get("n_subjects")), "subjects")
    n_records = _value_or_unknown(_clean_value(context.get("n_records")), "recordings")
    license_text = _value_or_unknown(_clean_value(context.get("license")), "license")
    digits = "".join(c for c in class_name if c.isdigit())
    issue_num = digits.lstrip("0") or digits or "—"
    block = (
        '<div class="eegdash-ed-issue">'
        f'<div class="crumb">EEGdash'
        f'<span class="crumb-sep">›</span>{source}'
        f'<span class="crumb-sep">›</span><b>{class_name}</b></div>'
        f"<div>Iss. {issue_num} · {n_subjects} subjects · "
        f"{n_records} recordings · {license_text}</div>"
        "</div>"
        f'<div class="eegdash-ed-kicker">Dataset Brief · {short_label}</div>'
    )
    return _editorial_html(block)


def _format_editorial_fieldcard_section(context: Mapping[str, object]) -> str:
    """Rich field-card aside emitted next to the hero."""
    class_name = str(context.get("class_name", "")).strip()
    license_text = _value_or_unknown(_clean_value(context.get("license")), "license")
    source = _editorial_source_label(_clean_value(context.get("source")))
    doi_clean = _normalize_doi(_clean_value(context.get("doi")))

    version_tag = "—"
    if doi_clean and "/" in doi_clean:
        tail = doi_clean.rsplit("/", 1)[-1]
        if ".v" in tail:
            version_tag = "v" + tail.split(".v", 1)[1]

    n_subjects = _value_or_unknown(_clean_value(context.get("n_subjects")), "subjects")
    n_records = _value_or_unknown(_clean_value(context.get("n_records")), "recordings")
    modality_raw = _editorial_modality_label(_clean_value(context.get("modality")))
    n_channels = _value_or_unknown(
        _clean_value(context.get("n_channels")), "n_channels"
    )
    sfreq = _editorial_sfreq_label(context)
    duration_label = _editorial_duration_label(context)
    size_label = _value_or_unknown(_clean_value(context.get("size")), "general")

    bids_version = _clean_value(context.get("bids_version")) or "—"
    sidecars = context.get("sidecars_detected") or []
    sidecar_line = (
        " · ".join(sidecars) if sidecars else "<span class='dim'>not yet probed</span>"
    )
    hed_annotated = bool(context.get("hed_annotated"))
    hed_label = (
        '<a href="#nemar-processing-statistics">HED ✓</a>' if hed_annotated else "—"
    )

    citations = _editorial_citation_label(context.get("nemar_citation_count"))
    hf_info = context.get("huggingface") or {}
    hf_available = bool(hf_info.get("available"))
    hf_url = str(hf_info.get("url") or "https://huggingface.co/EEGDash")
    hf_label = (
        f'<a href="{hf_url}">EEGDash/{context.get("dataset_id")}</a>'
        if hf_available
        else f'<a href="{hf_url}">org listing</a>'
    )

    tags = context.get("tags") or {}
    tag_pathology = _clean_value(
        tags.get("pathology") if isinstance(tags, dict) else ""
    )
    tag_type = _clean_value(tags.get("type") if isinstance(tags, dict) else "")
    tag_modality = _clean_value(tags.get("modality") if isinstance(tags, dict) else "")

    quality_label, _quality_color, quality_pct = _compute_quality_score(context)
    metadata_line = f"{quality_pct}% · {quality_label}"

    tasks = context.get("tasks") or []
    tasks_line = _editorial_tasks_label(tasks)

    sessions = context.get("sessions") or []
    sessions_line = _editorial_sessions_label(sessions)

    contacts = context.get("contact_info") or []
    contact_line = _editorial_contact_label(contacts)

    storage = context.get("dataset_storage") or {}
    s3_base = _clean_value(storage.get("base")) if isinstance(storage, dict) else ""

    updated_line = _editorial_updated_label(
        _clean_value(context.get("dataset_modified_at"))
        or _clean_value(context.get("dataset_created_at"))
    )

    paper_doi = _normalize_doi(_clean_value(context.get("associated_paper_doi")))
    paper_url = _clean_value(context.get("paper_url"))
    paper_doi_html = (
        f'<a href="https://doi.org/{paper_doi}">{paper_doi}</a>' if paper_doi else ""
    )
    paper_action_href = f"https://doi.org/{paper_doi}" if paper_doi else paper_url

    openneuro_url = str(context.get("openneuro_url") or "")
    croissant_url = (
        f"../../_static/dataset_generated/croissant/{class_name}.croissant.json"
    )
    actions = f'<a href="{openneuro_url}">OpenNeuro</a>'
    if paper_action_href:
        actions += f'<a href="{paper_action_href}">Read paper</a>'
    actions += (
        f'<a href="{hf_url}">{"🤗 HF" if hf_available else "🤗 Org"}</a>'
        f'<a href="{croissant_url}" download>Croissant</a>'
    )

    block = (
        '<aside class="eegdash-ed-rail">'
        "<h4>Field card</h4>"
        "<dl>"
        '<dt class="hdr">Identity</dt><dd class="hdrpad"></dd>'
        f"<dt>Dataset</dt><dd>{class_name}</dd>"
        f"<dt>Version</dt><dd>{version_tag}</dd>"
        f"<dt>Source</dt><dd>{source}</dd>"
        f"<dt>License</dt><dd>{license_text}</dd>"
    )
    if updated_line:
        block += f"<dt>Updated</dt><dd>{updated_line}</dd>"
    if contact_line:
        block += f"<dt>Contact</dt><dd>{contact_line}</dd>"
    if paper_doi_html:
        block += f"<dt>Paper DOI</dt><dd>{paper_doi_html}</dd>"

    block += (
        '<dt class="hdr">Signal</dt><dd class="hdrpad"></dd>'
        f"<dt>Subjects</dt><dd>{n_subjects}</dd>"
        f"<dt>Recordings</dt><dd>{n_records}</dd>"
        f"<dt>Modality</dt><dd>{modality_raw}</dd>"
        f"<dt>Channels</dt><dd>{n_channels}</dd>"
        f"<dt>Sample rate</dt><dd>{sfreq}</dd>"
        f"<dt>Duration</dt><dd>{duration_label}</dd>"
        f"<dt>Size</dt><dd>{size_label}</dd>"
        f"<dt>Tasks</dt><dd>{tasks_line}</dd>"
    )
    if sessions_line:
        block += f"<dt>Sessions</dt><dd>{sessions_line}</dd>"

    block += (
        '<dt class="hdr">BIDS</dt><dd class="hdrpad"></dd>'
        f"<dt>BIDSVersion</dt><dd>{bids_version}</dd>"
        f"<dt>Sidecars</dt><dd>{sidecar_line}</dd>"
        f"<dt>Events ann.</dt><dd>{hed_label}</dd>"
        f"<dt>Metadata</dt><dd>{metadata_line}</dd>"
    )
    if s3_base:
        block += f"<dt>Storage</dt><dd><code>{s3_base}</code></dd>"

    if (
        tag_modality
        and tag_type
        and tag_modality.strip().lower() == tag_type.strip().lower()
    ):
        tag_type = ""
    if tag_pathology or tag_type or tag_modality:
        block += '<dt class="hdr">Tags</dt><dd class="hdrpad"></dd>'
        if tag_pathology:
            block += f"<dt>Pathology</dt><dd>{tag_pathology}</dd>"
        if tag_modality:
            block += f"<dt>Paradigm</dt><dd>{tag_modality}</dd>"
        if tag_type:
            block += f"<dt>Type</dt><dd>{tag_type}</dd>"

    reach_rows: list[str] = []
    if hf_available or hf_url:
        reach_rows.append(f"<dt>HF mirror</dt><dd>{hf_label}</dd>")
    if citations and citations != "—":
        reach_rows.append(f"<dt>Citations</dt><dd>{citations}</dd>")
    if reach_rows:
        block += '<dt class="hdr">ML &amp; Reach</dt><dd class="hdrpad"></dd>'
        block += "".join(reach_rows)

    block += "</dl>"

    if doi_clean:
        block += (
            '<div class="doi">'
            "<span>Persistent identifier</span>"
            f'<a href="https://doi.org/{doi_clean}">{doi_clean}</a>'
            "</div>"
        )
    block += f'<div class="actions">{actions}</div>'
    block += "</aside>"

    return _editorial_html(block)


def _format_editorial_hero_extras(context: Mapping[str, object]) -> str:
    """Editorial deck + byline + signal pills emitted after the citation block."""
    title = _clean_value(context.get("title"))
    authors = context.get("authors") or []
    year = _clean_value(context.get("year"))
    source = _editorial_source_label(_clean_value(context.get("source")))
    n_subjects = _clean_value(context.get("n_subjects"))
    modality = _editorial_modality_label(_clean_value(context.get("modality")))
    senior = _clean_value(context.get("senior_author"))
    funding = context.get("funding") or []
    tags = context.get("tags") or {}

    parts = []
    if n_subjects and n_subjects not in ("—", "0"):
        parts.append(f"{n_subjects}-participant")
    parts.append(f"{modality} dataset")
    if title:
        parts.append(f"— {title}")
    deck_text = " ".join(parts).strip()
    if not deck_text or deck_text == "EEG dataset":
        deck_text = (
            f"A {modality} dataset distributed through EEGDash with "
            f"standardized BIDS metadata."
        )

    primary = []
    secondary = []
    for idx, author in enumerate(authors):
        cleaned = author.replace("*", "")
        if idx < 2:
            primary.append(f"<strong>{cleaned}</strong>")
        else:
            secondary.append(cleaned)
    if primary:
        byline_authors = " · ".join(primary)
        if secondary:
            byline_authors += " · " + " · ".join(secondary[:4])
            if len(secondary) > 4:
                byline_authors += " · …"
    else:
        byline_authors = "Authors unspecified"

    year_line = ""
    if year and year != "—":
        year_line = (
            f'<br/><span class="role">Year</span> {year} · Distributed via {source}'
        )

    senior_line = ""
    if senior:
        senior_line = (
            f'<br/><span class="role">Senior author</span> <strong>{senior}</strong>'
        )

    funding_line = ""
    if funding:
        fund_str = " · ".join(str(f).strip() for f in funding[:2] if str(f).strip())
        if len(funding) > 2:
            fund_str += f" · + {len(funding) - 2} more"
        if fund_str:
            funding_line = f'<br/><span class="role">Funding</span> {fund_str}'

    pills: list[str] = []
    n_channels = _clean_value(context.get("n_channels"))
    if n_channels and n_channels not in ("—", "Varies"):
        pills.append(f'<span class="pill">{modality} · {n_channels} ch</span>')
    sfreq = _editorial_sfreq_label(context)
    if sfreq and sfreq not in ("—", "Varies"):
        pills.append(f'<span class="pill">{sfreq}</span>')
    bids_version = _clean_value(context.get("bids_version"))
    if bids_version:
        pills.append(f'<span class="pill is-info">BIDS {bids_version}</span>')
    if bool(context.get("hed_annotated")):
        pills.append('<span class="pill is-warning">HED ✓</span>')
    task_list = context.get("tasks") or []
    if isinstance(task_list, (list, tuple)) and task_list:
        if len(task_list) == 1:
            pills.append(f'<span class="pill">Task · {str(task_list[0])[:32]}</span>')
        else:
            pills.append(f'<span class="pill">{len(task_list)} tasks</span>')
    sess_list = context.get("sessions") or []
    if isinstance(sess_list, (list, tuple)) and len(sess_list) > 1:
        pills.append(f'<span class="pill">{len(sess_list)} sessions</span>')
    if isinstance(tags, dict):
        tag_pathology = _clean_value(tags.get("pathology"))
        tag_modality = _clean_value(tags.get("modality"))
        tag_type = _clean_value(tags.get("type"))
        if tag_pathology and tag_pathology.lower() not in ("not specified", "—"):
            pills.append(f'<span class="pill">{tag_pathology}</span>')
        if tag_modality and tag_modality.lower() not in ("—",):
            pills.append(f'<span class="pill">{tag_modality}</span>')
        if tag_type and tag_type.lower() not in ("—",):
            pills.append(f'<span class="pill">{tag_type}</span>')
    pills_html = (
        f'<div class="eegdash-ed-pills">{"".join(pills)}</div>' if pills else ""
    )

    block = (
        f'<p class="eegdash-ed-deck">{deck_text}.</p>'
        f'<div class="eegdash-ed-byline">'
        f'<span class="role">Data &amp; curation</span> {byline_authors}'
        f"{senior_line}"
        f"{year_line}"
        f"{funding_line}"
        "</div>"
        f"{pills_html}"
    )
    return _editorial_html(block)


def _format_editorial_layers_section(context: Mapping[str, object]) -> str:
    """3-layer architecture rail — Study / Signal·BIDS / Training·ML."""
    block = (
        '<div class="eegdash-ed-layers">'
        "<div>"
        '<div class="ly-lbl"><span>Layer 01</span><b>Study</b></div>'
        '<div class="ly-tit">What was asked</div>'
        '<div class="ly-dsc">Hypothesis, independent &amp; dependent variables, '
        "paradigm, cohort, and the editorial caveats around what the "
        "recordings can and cannot answer.</div>"
        "</div>"
        "<div>"
        '<div class="ly-lbl"><span>Layer 02</span><b>Signal · BIDS</b></div>'
        '<div class="ly-tit">What was recorded</div>'
        '<div class="ly-dsc">Sidecars, channels &amp; electrodes, coordinate '
        "system, event semantics, and quality stats from the NEMAR pipeline "
        "when available.</div>"
        "</div>"
        "<div>"
        '<div class="ly-lbl"><span>Layer 03</span><b>Training · ML</b></div>'
        '<div class="ly-tit">What you can train on</div>'
        '<div class="ly-dsc">Recommended access modes — MNE Raw, '
        "braindecode windows, PyTorch DataLoader — plus the targets the "
        "metadata makes addressable.</div>"
        "</div>"
        "</div>"
    )
    return _editorial_html(block)


def _editorial_secnum(num: int, label: str) -> str:
    """Editorial § NN marker emitted before each major H2."""
    block = f'<div class="eegdash-ed-secnum">§ {num:02d}<b>{label}</b></div>'
    return _editorial_html(block)


def _format_editorial_caveat_section(context: Mapping[str, object]) -> str:
    """Conditional caveat callout — only fires for small cohorts (n < 50)."""
    n_sub_raw = _clean_value(context.get("n_subjects"))
    try:
        n_sub = int(n_sub_raw)
    except (TypeError, ValueError):
        return ""
    if n_sub <= 0 or n_sub >= 50:
        return ""
    modality = _clean_value(context.get("modality")) or "EEG"
    block = (
        '<div class="eegdash-ed-caveat">'
        '<div class="c-lbl">Editorial caveat · cohort size</div>'
        "<h4>Treat this as a features-first dataset, "
        "not a deep-learning playground.</h4>"
        f"<p>With <b>n = {n_sub}</b> {modality} participants, this dataset sits "
        "below the ~50-subject threshold where deep networks trained from scratch "
        "typically pay off. A well-tuned feature pipeline — band-power features, "
        "Riemannian geometry, linear classifier — is the recommended baseline. "
        "Use deep models only with transfer learning or pre-trained backbones.</p>"
        "<p>For splits, prefer <code>GroupShuffleSplit</code> with "
        "<code>groups=subject_id</code> so windows from the same recording do not "
        "leak between train and test.</p>"
        "</div>"
    )
    return _editorial_html(block)


def _strip_rst_heading(s: str) -> str:
    r"""Drop a leading ``Title\n-----`` (or ``====``) heading from ``s``.

    The paired electrodes+traces figure replaces the per-section H2 with
    its own editorial header, so each child section's heading has to be
    peeled off before we wrap them in the figure grid.
    """
    if not s:
        return ""
    lines = s.splitlines()
    i = 0
    while i + 1 < len(lines):
        if (
            lines[i].strip()
            and i + 1 < len(lines)
            and (
                set(lines[i + 1].strip()) <= set("-=")
                and len(lines[i + 1].strip()) >= 3
            )
        ):
            i += 2
            continue
        break
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:])


def _format_electrodes_traces_pair(name: str, context: Mapping[str, object]) -> str:
    """Render electrode layout + signal-preview iframe as a paired figure."""
    electrodes = _format_electrodes_section(context).strip()
    traces = _format_traces_section(context).strip()

    electrodes_body = _strip_rst_heading(electrodes)
    traces_body = _strip_rst_heading(traces)

    if not electrodes_body and not traces_body:
        return ""

    n_subjects = _value_or_unknown(_clean_value(context.get("n_subjects")), "subjects")
    n_records = _value_or_unknown(_clean_value(context.get("n_records")), "recordings")
    n_channels = _clean_value(context.get("n_channels")) or "—"
    sfreq = _editorial_sfreq_label(context)
    modality = _editorial_modality_label(_clean_value(context.get("modality")))

    meta_line = (
        f"{n_channels} ch · {modality} · {sfreq} · "
        f"{n_subjects} subjects, {n_records} recordings"
    )

    wrapper_open = (
        ".. raw:: html\n\n"
        '   <div class="eegdash-ed-figpair">\n'
        '     <div class="figpair-meta">\n'
        f"       <b>Fig. 01</b> Signal &amp; montage\n"
        f'       <span class="right">{meta_line}</span>\n'
        "     </div>\n"
        '     <div class="figpair-grid">\n'
    )
    wrapper_close = ".. raw:: html\n\n   </div></div>\n"

    parts = [wrapper_open]
    if traces_body:
        parts.extend(
            [
                '.. raw:: html\n\n   <div class="figpair-cell figpair-trace">\n',
                traces_body,
                "\n.. raw:: html\n\n   </div>\n",
            ]
        )
    if electrodes_body:
        parts.extend(
            [
                '.. raw:: html\n\n   <div class="figpair-cell figpair-montage">\n',
                electrodes_body,
                "\n.. raw:: html\n\n   </div>\n",
            ]
        )
    parts.append(wrapper_close)
    return "\n\n".join(parts)


def _format_dataset_info_dropdown(context: Mapping[str, object]) -> str:
    """Wrap the Dataset Information table + BibTeX in a folded dropdown."""
    inner = _format_dataset_info_section(context)
    if not inner.strip():
        return ""
    indented = "\n".join(f"   {line}" if line else "" for line in inner.split("\n"))
    return (
        ".. dropdown:: Full dataset metadata table\n"
        "   :class-container: sd-shadow-sm eegdash-ed-info-dropdown\n"
        "\n"
        f"{indented}\n"
    )


def _format_editorial_access_modes_section(
    context: Mapping[str, object],
) -> str:
    """Sidecar card listing the access modes available for this dataset."""
    class_name = str(context.get("class_name", "")).strip()
    croissant_url = (
        f"../../_static/dataset_generated/croissant/{class_name}.croissant.json"
    )
    hf_info = context.get("huggingface") or {}
    hf_available = bool(hf_info.get("available"))
    hf_url = str(hf_info.get("url") or "https://huggingface.co/EEGDash")

    if hf_available:
        dataset_id_lower = str(context.get("dataset_id") or "").lower()
        hf_blurb = (
            "Pre-bundled mirror at "
            f'<a href="{hf_url}">EEGDash/{dataset_id_lower}</a> · '
            'pull with <code>datasets.load_dataset("EEGDash/'
            f'{dataset_id_lower}")</code>.'
        )
    else:
        hf_blurb = (
            "No per-dataset mirror published yet — browse the "
            f'<a href="{hf_url}">EEGDash org listing</a> for sibling datasets.'
        )

    rows = [
        (
            ".raw",
            (
                "MNE <code>Raw</code> object — standard tools (filter, epoch, "
                "ICA, plot_psd)."
            ),
            "mne",
        ),
        (
            "BaseConcatDataset",
            (
                "Each record is a lazy <code>BaseDataset</code> from "
                "braindecode — windowed via <code>create_windows_from_events</code>."
            ),
            "braindecode",
        ),
        (
            "DataLoader",
            (
                "Wraps the windowed dataset into a PyTorch <code>DataLoader</code>; "
                "supports parallel workers and on-the-fly augmentations."
            ),
            "pytorch",
        ),
        (
            "Zarr cache",
            (
                "Optional braindecode Zarr mirror for fast resume; persisted to "
                "<code>cache_dir</code>."
            ),
            "zarr",
        ),
        (
            "Hugging Face",
            hf_blurb,
            "huggingface",
        ),
        (
            "Croissant 1.0",
            (
                f"Machine-readable JSON-LD descriptor — "
                f'<a href="{croissant_url}" download>{class_name}.croissant.json</a> '
                f"(MLCommons schema, ingestible by PyTorch / TensorFlow / JAX)."
            ),
            "mlcommons",
        ),
    ]

    rows_html = "".join(
        '<div class="am-row">'
        f'<span class="name">{name}</span>'
        f'<span class="what">{what}</span>'
        f'<span class="badge">{badge}</span>'
        "</div>"
        for name, what, badge in rows
    )

    block = (
        '<div class="eegdash-ed-access">'
        '<div class="sidecar-hdr">'
        "<span><b>Access modes</b></span>"
        '<span class="right">MNE → braindecode → PyTorch → ML</span>'
        "</div>"
        f'<div class="am-list">{rows_html}</div>'
        "</div>"
    )
    return _editorial_html(block)


def _format_editorial_provenance_section(context: Mapping[str, object]) -> str:
    """Provenance strip — five-column band placed before the See Also footer."""
    class_name = str(context.get("class_name", "")).strip()
    openneuro_url = str(context.get("openneuro_url", ""))
    nemar_url = str(context.get("nemar_url", ""))
    paper_url = _clean_value(context.get("paper_url"))
    license_text = _value_or_unknown(_clean_value(context.get("license")), "license")
    doi_clean = _normalize_doi(_clean_value(context.get("doi")))
    doi_link = (
        f'<a href="https://doi.org/{doi_clean}">{doi_clean}</a>'
        if doi_clean
        else '<span class="todo">DOI not on file</span>'
    )

    bids_version = _clean_value(context.get("bids_version"))
    bids_cell = (
        f'<div class="v ok">BIDS {bids_version}</div>'
        if bids_version
        else '<div class="v todo">version not on file</div>'
    )

    sidecars = context.get("sidecars_detected") or []
    if sidecars:
        sidecars_cell = f'<div class="v">{" · ".join(sidecars)}</div>'
    else:
        sidecars_cell = '<div class="v todo">not yet probed</div>'

    croissant_url = (
        f"../../_static/dataset_generated/croissant/{class_name}.croissant.json"
    )

    hf_info = context.get("huggingface") or {}
    hf_url = str(hf_info.get("url") or "https://huggingface.co/EEGDash")
    hf_link_label = "HuggingFace" if hf_info.get("available") else "HF org"

    mirrors = [
        f'<a href="{openneuro_url}">OpenNeuro</a>',
        f'<a href="{nemar_url}">NEMAR</a>',
        f'<a href="{hf_url}">{hf_link_label}</a>',
    ]
    if paper_url:
        mirrors.append(f'<a href="{paper_url}">Paper</a>')

    block = (
        '<div class="eegdash-ed-prov">'
        "<div>"
        '<div class="lbl">BIDS</div>'
        f"{bids_cell}"
        "</div>"
        "<div>"
        '<div class="lbl">Sidecars</div>'
        f"{sidecars_cell}"
        "</div>"
        "<div>"
        '<div class="lbl">Provenance</div>'
        f'<div class="v">{license_text} · {doi_link}</div>'
        "</div>"
        "<div>"
        '<div class="lbl">Machine-readable</div>'
        '<div class="v">'
        '<a href="#dataset-information">schema.org/Dataset</a> · '
        f'<a href="{croissant_url}" download>Croissant</a>'
        "</div>"
        "</div>"
        "<div>"
        '<div class="lbl">Mirrors</div>'
        f'<div class="v">{" · ".join(mirrors)}</div>'
        "</div>"
        "</div>"
    )
    return _editorial_html(block)


def _format_editorial_footnotes_section(
    context: Mapping[str, object],
    related: Sequence[str] = (),
    related_meta: Sequence[Mapping[str, object]] = (),
) -> str:
    """Three-column footnotes block — Citation / Provenance / Related."""
    title = _clean_value(context.get("title"))
    authors = context.get("authors") or []
    year = _clean_value(context.get("year"))
    doi_clean = _normalize_doi(_clean_value(context.get("doi")))
    source = _clean_value(context.get("source")) or "OpenNeuro"

    if authors:
        author_str = ", ".join(a.replace("*", "") for a in authors[:5])
        if len(authors) > 5:
            author_str += ", …"
    else:
        author_str = "Authors unspecified"
    year_str = year if year and year != "—" else "n.d."
    citation = (
        f"{author_str} ({year_str}). <em>{title or context.get('class_name')}</em>."
    )
    if doi_clean:
        citation += f" <code>{doi_clean}</code>"

    provenance_notes = [
        f'<span class="note-num">¹</span>Contributed to {source} in BIDS format.',
        '<span class="note-num">²</span>Curated &amp; ingested by the EEGDash '
        "catalog; see CITATION.cff for canonical reference.",
    ]
    if doi_clean:
        provenance_notes.append(
            f'<span class="note-num">³</span>Persistent identifier: '
            f"<code>{doi_clean}</code>."
        )
    provenance_html = "".join(f"<p>{n}</p>" for n in provenance_notes)

    if related_meta:
        cards = []
        for meta in list(related_meta)[:5]:
            rel_name = str(meta.get("name") or "").strip()
            if not rel_name:
                continue
            modality = (str(meta.get("modality") or "")).upper() or "—"
            n_sub = _clean_value(meta.get("n_subjects"))
            same_authors = bool(meta.get("same_authors"))
            meta_bits = [modality]
            if n_sub and n_sub != "0":
                meta_bits.append(f"{n_sub} subj")
            badge = (
                '<span class="rel-tag rel-same">Same authors</span>'
                if same_authors
                else ""
            )
            cards.append(
                f'<a class="rel-card" href="{rel_name}.html">'
                f'<span class="rel-id">{rel_name}</span>'
                f'<span class="rel-meta">{" · ".join(meta_bits)}</span>'
                f"{badge}"
                "</a>"
            )
        related_html = f'<div class="rel-grid">{"".join(cards)}</div>'
        if len(related_meta) > 5:
            related_html += (
                f'<p class="rel-more">+ {len(related_meta) - 5} more — '
                "see See Also below →</p>"
            )
    elif related:
        related_items = "<br/>".join(
            f'<a href="{rel}.html">{rel}</a>' for rel in related[:5]
        )
        related_html = f"<p>{related_items}</p>"
        if len(related) > 5:
            related_html += (
                f"<p><em>+ {len(related) - 5} more — see See Also below →</em></p>"
            )
    else:
        related_html = (
            "<p><em>No sibling datasets cross-linked for this modality yet.</em></p>"
        )

    block = (
        '<div class="eegdash-ed-footnotes">'
        "<div>"
        "<h5>Citation</h5>"
        f"<p>{citation}</p>"
        "</div>"
        "<div>"
        "<h5>Provenance</h5>"
        f"{provenance_html}"
        "</div>"
        "<div>"
        "<h5>Related &amp; sibling datasets</h5>"
        f"{related_html}"
        "</div>"
        "</div>"
    )
    return _editorial_html(block)


# Curated "start here" tutorials linked from every dataset page.
_EDITORIAL_EXAMPLES = (
    (
        "plot_00_first_search",
        "tutorials/00_start_here",
        "Find datasets with the EEGDash API",
        "Query the catalogue, filter by task or modality, list candidates.",
    ),
    (
        "plot_01_first_recording",
        "tutorials/00_start_here",
        "Load one EEG recording",
        "Resolve a single record to an MNE Raw with channels and events.",
    ),
    (
        "plot_02_dataset_to_dataloader",
        "tutorials/00_start_here",
        "EEG recording to PyTorch DataLoader",
        "Wrap braindecode windows in a DataLoader for model training.",
    ),
    (
        "plot_10_preprocess_and_window",
        "tutorials/10_core_workflow",
        "Preprocess EEG and create windows",
        "Filter, resample, epoch — and persist the windowed dataset.",
    ),
    (
        "plot_13_save_and_reuse_prepared_data",
        "tutorials/10_core_workflow",
        "Save and reload prepared data",
        "Cache a windowed dataset to disk and reattach it without recompute.",
    ),
    (
        "how_to_download_a_dataset",
        "how_to",
        "Download a dataset locally",
        "Prefetch BIDS files to a local cache and validate the layout.",
    ),
)


def _format_editorial_examples_gallery(context: Mapping[str, object]) -> str:
    """Six-card thumbnail gallery linking the canonical "start here" tutorials."""
    dataset_id = str(context.get("dataset_id") or "").strip()
    cards = []
    for slug, subpath, title, blurb in _EDITORIAL_EXAMPLES:
        thumb = f"../../_static/thumbs/{slug}.png"
        href = f"../../generated/auto_examples/{subpath}/{slug}.html"
        cards.append(
            '<a class="ex-card" '
            f'href="{href}">'
            f'<span class="ex-thumb"><img src="{thumb}" alt="" loading="lazy"></span>'
            '<span class="ex-body">'
            f'<span class="ex-title">{title}</span>'
            f'<span class="ex-blurb">{blurb}</span>'
            "</span>"
            "</a>"
        )
    hint = (
        f"Swap any <code>load_dataset(...)</code> call for "
        f"<code>{dataset_id}</code> to reproduce the tutorial on this dataset."
        if dataset_id
        else ""
    )
    block = (
        '<section class="eegdash-ed-examples">'
        '<div class="sidecar-hdr">'
        "<span><b>Examples using EEGDash</b></span>"
        '<span class="right">curated · start here</span>'
        "</div>"
        f'<div class="ex-grid">{"".join(cards)}</div>'
    )
    if hint:
        block += f'<p class="ex-hint">{hint}</p>'
    block += "</section>"
    return _editorial_html(block)


def _format_editorial_colophon_section(context: Mapping[str, object]) -> str:
    """Footer band — typography credit, FAIR exports, DOI."""
    class_name = str(context.get("class_name", "")).strip()
    license_text = _value_or_unknown(_clean_value(context.get("license")), "license")
    doi_clean = _normalize_doi(_clean_value(context.get("doi")))
    doi_html = (
        f'<a href="https://doi.org/{doi_clean}">{doi_clean}</a>'
        if doi_clean
        else '<span style="opacity:.6">DOI not on file</span>'
    )

    croissant_url = (
        f"../../_static/dataset_generated/croissant/{class_name}.croissant.json"
    )
    hf_info = context.get("huggingface") or {}
    hf_url = str(hf_info.get("url") or "https://huggingface.co/EEGDash")
    hf_link_label = (
        "Hugging Face mirror" if hf_info.get("available") else "Hugging Face org"
    )
    block = (
        '<footer class="eegdash-ed-colophon">'
        f"<div>EEGdash · <b>The Dataset Brief — {class_name}</b></div>"
        "<div>FAIR exports · "
        '<a href="#dataset-information">schema.org/Dataset</a> · '
        f'<a href="{croissant_url}" download>Croissant 1.0</a> · '
        f'<a href="{hf_url}">{hf_link_label}</a></div>'
        f"<div>{license_text} · <b>{doi_html}</b></div>"
        "</footer>"
    )
    return _editorial_html(block)
