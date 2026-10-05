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

Registered as a Sphinx extension: generate the index at build-finished,
after dataset shells and HTML pages exist. The CLI also supports offline
source inspection and validation of the final deployed tree.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlsplit

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
- [Quickstart]({site}/quickstart.html): walkthrough of the core workflow.

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
- [Dataset to DataLoader]({site}/generated/auto_examples/tutorials/00_start_here/plot_02_dataset_to_dataloader.html): turn a dataset into training batches.
- [Eyes-open / eyes-closed tutorial]({site}/generated/auto_examples/tutorials/30_resting_state/plot_30_eyes_open_closed.html): classic EEG classification.
- [P300 transfer learning]({site}/generated/auto_examples/applied/project_p300_transfer.html): cross-subject transfer on the P300 paradigm.
- [First feature extraction]({site}/generated/auto_examples/tutorials/40_features/plot_40_first_features.html): using the feature API.
- [Age prediction tutorial]({site}/generated/auto_examples/applied/project_age_regression.html): regression from raw EEG.
- [p-factor regression]({site}/generated/auto_examples/applied/project_pfactor_deep.html): clinical outcome regression.
- [Auditory oddball]({site}/generated/auto_examples/tutorials/20_event_related/plot_21_auditory_oddball.html): event-related paradigm.

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
    folder = source / "api" / "dataset"
    for rst in sorted(folder.glob("*.rst")):
        if re.search(r"^\.\. automodule::\s+", rst.read_text(encoding="utf-8"), re.M):
            pages.append((rst.stem, f"api/dataset/{rst.stem}.html"))
    return pages


def _discover_dataset_pages(source: Path) -> list[tuple[str, str]]:
    """Use generated dataset-page metadata, not a module-name prefix or ID format."""
    folder = source / "api" / "dataset"
    if not folder.is_dir():
        return []
    prefix = "eegdash.dataset."
    entries: list[tuple[str, str]] = []
    for rst in sorted(folder.glob(f"{prefix}*.rst")):
        stem = rst.stem  # e.g. eegdash.dataset.DS001234
        ds_id = stem[len(prefix) :]
        directives = re.findall(
            r"^\.\. dataset-page::[ \t]+(\S+)[ \t]*$",
            rst.read_text(encoding="utf-8"),
            re.M,
        )
        if directives != [ds_id]:
            continue
        url = f"api/dataset/{stem}.html"
        entries.append((ds_id, url))
    return entries


def _render_section(
    heading: str, entries: Iterable[tuple[str, str]], site_url: str = SITE_URL
) -> str:
    lines = [f"## {heading}", ""]
    for label, rel_url in entries:
        lines.append(f"- [{label}]({site_url.rstrip('/')}/{rel_url})")
    lines.append("")
    return "\n".join(lines)


def build(source: Path, output: Path, site_url: str = SITE_URL) -> int:
    site_url = site_url.rstrip("/")
    curated = CURATED_HEADER.format(site=site_url)
    api_pages = _discover_api_pages(source)
    dataset_pages = _discover_dataset_pages(source)
    head = (
        curated
        + "\n"
        + _render_section("API reference pages (per-module)", api_pages, site_url)
        + "\n"
    )
    tail = "\n---\n"
    entries = [f"- [{name}]({site_url}/{url})" for name, url in dataset_pages]
    total = len(entries)
    while True:
        count = len(entries)
        summary = (
            f"Showing {count} of {total} generated dataset pages"
            + (" (truncated to the index size budget)." if count < total else ".")
            if total
            else "No generated dataset pages are available in this build."
        )
        section = (
            f"## Dataset pages (N={count})\n\n{summary} "
            f"See the full interactive catalog at <{site_url}/dataset_summary.html>.\n\n"
            + "\n".join(entries)
            + "\n"
        )
        content = head + section + tail
        if len(content.encode("utf-8")) <= SIZE_BUDGET:
            break
        if not entries:
            raise ValueError("Curated/API index exceeds the llms.txt byte budget")
        entries.pop()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output.stat().st_size


def validate_links(index: Path, html_root: Path, site_url: str | None = None) -> None:
    """Fail publication if any local index link lacks a built target.

    Unless explicitly supplied, recover the effective Sphinx origin/path from
    the generated homepage entry so preview builds cannot pass with zero checks.
    """
    text = index.read_text(encoding="utf-8")
    if site_url is None:
        homepage = re.search(
            r"\[Project homepage\]\((https?://[^)]+/index\.html)\)", text
        )
        if homepage is None:
            raise ValueError("Missing canonical homepage in llms.txt")
        site_url = homepage.group(1).removesuffix("/index.html")
    prefix = site_url.rstrip("/") + "/"
    missing = []
    for url in re.findall(r"\]\(([^)]+)\)|<([^>]+)>", text):
        target = url[0] or url[1]
        if not target.startswith(prefix):
            continue
        relative = unquote(urlsplit(target[len(prefix) :]).path)
        if not (html_root / relative).is_file():
            missing.append(relative)
    if missing:
        raise ValueError("Missing llms.txt targets: " + ", ".join(sorted(set(missing))))


def _build_finished(app, exception) -> None:
    if exception is None and app.builder.format == "html":
        build(
            Path(app.srcdir),
            Path(app.outdir) / "llms.txt",
            app.config.html_baseurl or SITE_URL,
        )


def setup(app) -> dict:
    # build-finished runs after builder-inited shell generation, autosummary,
    # gallery generation and html_extra_path copying (which contains an old index).
    app.connect("build-finished", _build_finished)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}


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
        default=None,
        help=(
            f"Canonical site URL for generation (default: {SITE_URL}); "
            "validation defaults to the generated homepage origin/path."
        ),
    )
    parser.add_argument(
        "--check-html",
        type=Path,
        help="Validate local links in --output against this final HTML tree (no generation).",
    )
    args = parser.parse_args()
    if args.check_html is not None:
        validate_links(args.output, args.check_html, args.site_url)
        print("[generate_llms_txt] all local index targets exist")
        return 0

    size = build(args.source, args.output, args.site_url or SITE_URL)
    print(
        f"[generate_llms_txt] wrote {args.output} ({size:,} bytes, "
        f"budget {SIZE_BUDGET:,})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
