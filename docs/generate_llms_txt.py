"""Generate `docs/source/_extra/llms.txt` from the Sphinx source tree.

The static curated `llms.txt` we previously shipped listed ~12 pages
(under `llmstxt.org`'s "curated index" spirit) but covered <1 % of our
sitemap, so agent-readiness scanners (buildwithfern Agent Score in
particular) flag it as stale.

This script produces a hybrid file:

1. Handwritten narrative index kept up-front (installation, tutorials,
   API reference) so humans and LLMs hitting the first few KB of the
   file get the high-signal content.
2. A compact `Dataset pages` section appended afterwards that links to
   every `api/dataset/eegdash.dataset.*.rst` source page we can find,
   truncated to stay under Fern's 50 K cap.

Run as a build-time step before `make html`. Its output is read-only
by Sphinx (`html_extra_path = ["_extra"]` copies it to the build root).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

# Keep well under Fern's 50_000 cap so lint-style scanners don't flag.
SIZE_BUDGET = 45_000

SITE_URL = "https://eegdash.org"


CURATED_HEADER = """\
# EEGDash

> EEGDash is an open catalog and Python library for finding, loading, and
> preprocessing publicly available EEG, MEG, and iEEG datasets in BIDS
> format. It aggregates recordings from OpenNeuro, NEMAR, Zenodo,
> Figshare, SciDB, OSF, DataRN, and EEGManyLabs into a single searchable
> catalog with a uniform `EEGDashDataset` Python interface compatible
> with MNE-Python and braindecode.

The project is maintained by the BrainIAK / EEGDash contributors and is
released under an open-source license. Dataset licenses are inherited
from each upstream source and must be checked independently.

Automated clients: prefer the structured resources listed under
"Machine-readable" before parsing HTML pages.

## Getting started

- [Project homepage]({site}/index.html): what EEGDash is and why.
- [Install with pip]({site}/install/install_pip.html): `pip install eegdash`.
- [Install from source]({site}/install/install_source.html): developer setup.
- [User guide]({site}/user_guide.html): narrative walkthrough of the core workflow.

## Catalog

- [Dataset summary]({site}/dataset_summary.html): interactive catalog of every dataset with counts, modalities, tasks, and links to upstream archives.

## Python API

- [API overview]({site}/api/api.html): top-level entry points.
- [Core API reference]({site}/api/api_core.html): `eegdash.api`, schemas, downloader, HTTP client.
- [Dataset class reference]({site}/api/dataset/eegdash.EEGDashDataset.html): filters, lazy loading, BIDS metadata.
- [Feature-extraction overview]({site}/api/features_overview.html): spectral, connectivity, bivariate, and complexity features.
- [Features API reference]({site}/api/api_features.html): `eegdash.features` module listings.

## Tutorials and examples

- [All tutorials index]({site}/generated/auto_examples/index.html)
- [Minimal tutorial]({site}/generated/auto_examples/core/tutorial_minimal.html): load a dataset and train a tiny model end-to-end.
- [Eyes-open / eyes-closed tutorial]({site}/generated/auto_examples/core/tutorial_eoec.html): classic EEG classification.
- [P300 transfer learning]({site}/generated/auto_examples/core/p300_transfer_learning.html): cross-subject transfer on the P300 paradigm.
- [Feature extraction on EOEC]({site}/generated/auto_examples/core/tutorial_feature_extractor_open_close_eye.html): using the feature API on resting-state data.
- [Age prediction tutorial]({site}/generated/auto_examples/tutorials/noplot_tutorial_age_prediction.html): regression from raw EEG.
- [p-factor regression]({site}/generated/auto_examples/tutorials/noplot_tutorial_pfactor_regression.html): clinical outcome regression.
- [Auditory oddball]({site}/generated/auto_examples/tutorials/noplot_tutorial_audi_oddball.html): event-related paradigm.

## Project info

- [Developer notes]({site}/developer_notes.html): contribution, build, and release notes.
- [GitHub repository](https://github.com/eegdash/EEGDash): source code and issue tracker.

## Machine-readable

- [Agent Skills manifest]({site}/.well-known/agent-skills/index.json): structured skills (find datasets, get metadata, load BIDS records, count records, list features).
- [API catalog (RFC 9727)]({site}/.well-known/api-catalog): linkset pointing at the public EEGDash HTTP API.
- [Full markdown corpus]({site}/llms-full.txt): concatenation of every rendered markdown page (larger, for full-corpus retrieval).
- [OpenAPI specification](https://data.eegdash.org/openapi.json): full OpenAPI 3.1 spec for the `data.eegdash.org` catalog API.
- [Swagger UI](https://data.eegdash.org/docs) and [ReDoc](https://data.eegdash.org/redoc): human-readable API documentation.
- [Sitemap]({site}/sitemap.xml): every indexable page on this site.
- [robots.txt]({site}/robots.txt): crawl rules and Content Signals (`search=yes, ai-input=yes, ai-train=no`).

## Optional

- [BIDS specification](https://bids-specification.readthedocs.io/): the data format EEGDash speaks natively.
- [MNE-Python](https://mne.tools/): the numerical backbone used by `EEGDashDataset`.
- [braindecode](https://braindecode.org/): downstream deep-learning library compatible with EEGDash outputs.
"""


def _discover_api_pages(source: Path) -> list[tuple[str, str]]:
    """Top-level API reference pages (stable, not per-dataset)."""
    pages: list[tuple[str, str]] = []
    core = source / "api" / "generated" / "api-core"
    features = source / "api" / "generated" / "api-features"
    for folder, url_prefix in (
        (core, "api/generated/api-core"),
        (features, "api/generated/api-features"),
    ):
        if not folder.is_dir():
            continue
        for rst in sorted(folder.glob("*.rst")):
            stem = rst.stem
            pages.append((stem, f"{url_prefix}/{stem}.html"))
    return pages


def _discover_dataset_pages(source: Path) -> list[tuple[str, str]]:
    """Per-dataset pages. One compact entry per `eegdash.dataset.DSXXXXXX` stub."""
    folder = source / "api" / "dataset"
    if not folder.is_dir():
        return []
    prefix = "eegdash.dataset."
    entries: list[tuple[str, str]] = []
    for rst in sorted(folder.glob(f"{prefix}*.rst")):
        stem = rst.stem  # e.g. eegdash.dataset.DS001234
        ds_id = stem[len(prefix) :]
        if not ds_id:
            continue
        url = f"api/dataset/{stem}.html"
        entries.append((ds_id, url))
    return entries


def _render_section(heading: str, entries: Iterable[tuple[str, str]]) -> str:
    lines = [f"## {heading}", ""]
    for label, rel_url in entries:
        lines.append(f"- [{label}]({SITE_URL}/{rel_url})")
    lines.append("")
    return "\n".join(lines)


def _truncate_to_budget(section: str, budget: int, tail: str) -> str:
    """Drop trailing entries until section + tail fits in budget."""
    if len(section) + len(tail) <= budget:
        return section
    lines = section.split("\n")
    header = lines[:2]
    body = lines[2:]
    while body and len("\n".join(header + body)) + len(tail) > budget:
        body.pop()
    return "\n".join(header + body) + "\n"


def build(source: Path, output: Path, site_url: str = SITE_URL) -> int:
    curated = CURATED_HEADER.format(site=site_url)

    api_pages = _discover_api_pages(source)
    dataset_pages = _discover_dataset_pages(source)

    api_section = _render_section("API reference pages (per-module)", api_pages)

    dataset_header = (
        f"## Dataset pages (N={len(dataset_pages)})\n\n"
        f"Every EEGDash dataset has a dedicated Sphinx page with its "
        f"BIDS metadata, upstream citation, and load-in-Python snippet. "
        f"The full interactive catalog lives at "
        f"<{site_url}/dataset_summary.html>; the list below is what "
        f"sitemap / agent crawlers should enumerate.\n\n"
    )
    dataset_body_lines = [
        f"- [{ds_id}]({site_url}/{url})" for ds_id, url in dataset_pages
    ]
    dataset_full = dataset_header + "\n".join(dataset_body_lines) + "\n"

    head = curated + "\n" + api_section + "\n"
    tail = "\n---\n"

    remaining_budget = SIZE_BUDGET - len(head) - len(tail)
    dataset_section = _truncate_to_budget(dataset_full, remaining_budget, "")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(head + dataset_section + tail, encoding="utf-8")
    return output.stat().st_size


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=here / "source",
        help="Sphinx source directory (default: docs/source).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "source" / "_extra" / "llms.txt",
        help="Target llms.txt path (default: docs/source/_extra/llms.txt).",
    )
    parser.add_argument(
        "--site-url",
        default=SITE_URL,
        help=f"Canonical site URL used for absolute links (default: {SITE_URL}).",
    )
    args = parser.parse_args()

    size = build(args.source, args.output, args.site_url)
    print(
        f"[generate_llms_txt] wrote {args.output} ({size:,} bytes, "
        f"budget {SIZE_BUDGET:,})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
