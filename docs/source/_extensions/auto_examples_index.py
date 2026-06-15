"""Sphinx extension: aggregate sphinx-gallery leaf indexes into one page.

Sphinx-gallery is configured with multiple gallery roots (one per
tutorial category, plus how-to / applied / eeg2025 / hpc / dev_scripts).
Each emits its own ``index.rst`` with a card grid of tutorial thumbnails.
This extension stitches those per-leaf indexes into a single parent
``generated/auto_examples/index.rst`` aggregator page that mimics the
SPDLearn theory aggregator: a hero intro, a "how to read this gallery"
callout, and one section per gallery root with the thumbnail card grid
surfaced inline.

The hook runs at ``builder-inited`` with priority 600 so that
sphinx-gallery's own ``generate_gallery_rst`` (default priority 500) has
already produced the per-leaf ``index.rst`` files.
"""

from __future__ import annotations

from pathlib import Path

from sphinx.util import logging


def _extract_sg_thumbnail_block(child_index_path: Path) -> str | None:
    """Return the sphinx-gallery thumbnail-grid block from a child gallery.

    Sphinx-gallery emits a self-contained card grid in each leaf
    ``index.rst`` between two ``raw:: html`` blocks: it opens with
    ``<div class="sphx-glr-thumbnails">`` and closes with the matching
    ``</div>``. Inside, every tutorial card is its own
    ``<div class="sphx-glr-thumbcontainer">`` with thumbnail image,
    cross-reference link, and caption.

    The grid block is everything from the first ``.. raw:: html`` line
    that opens the ``<div class="sphx-glr-thumbnails">`` container up to
    and including the closing ``raw:: html`` block whose payload is
    ``</div>`` and that is followed by a non-card structural element
    (download footer or hidden toctree). Returns the raw text ready to
    be inlined under any heading; returns ``None`` if the marker can't
    be found (for example, the gallery is empty).
    """
    try:
        text = child_index_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    open_marker = '<div class="sphx-glr-thumbnails">'
    open_idx = text.find(open_marker)
    if open_idx < 0:
        return None
    # Walk back to the directive header for the opening raw block so we
    # capture the whole ``.. raw:: html\n\n    <div...>`` chunk.
    block_start = text.rfind(".. raw:: html", 0, open_idx)
    if block_start < 0:
        return None
    # The grid ends at the ``.. toctree::`` (hidden) that sphinx-gallery
    # appends right after closing ``</div>``. That toctree is content we
    # render via our own toctree, so cut the slice just before it.
    toctree_idx = text.find(".. toctree::", open_idx)
    if toctree_idx < 0:
        return None
    return text[block_start:toctree_idx].rstrip() + "\n"


def _section_block(
    src_root: Path, rel_dir: str, title: str, intro: str, level: str
) -> str:
    """Build one section: heading + intro paragraph + thumbnail grid."""
    underline = level * len(title)
    index_path = src_root / rel_dir / "index.rst"
    thumb_block = _extract_sg_thumbnail_block(index_path)
    if thumb_block is None:
        # Defensive fallback: no card grid available, defer to a
        # plain reference link so the section still renders.
        log = logging.getLogger(__name__)
        log.warning(
            "auto_examples aggregator: no thumbnail grid in %s; "
            "falling back to bare link",
            index_path,
        )
        thumb_block = (
            f":doc:`Browse {title} </generated/auto_examples/{rel_dir}/index>`\n"
        )
    return f"{title}\n{underline}\n\n{intro}\n\n{thumb_block}\n"


def _write_auto_examples_root_index(app):
    """Write a top-level ``generated/auto_examples/index.rst`` aggregator.

    Sphinx-gallery is configured with multiple gallery roots (one per
    tutorial category, plus how-to / applied / eeg2025 / hpc / dev_scripts).
    Each emits its own ``index.rst`` containing a card grid of tutorial
    thumbnails. This hook reads those per-leaf indexes and stitches them
    into a single parent page that mimics the SPDLearn theory aggregator:
    a hero intro, a "how to read this gallery" callout, and one section
    per gallery root with its full thumbnail card grid surfaced inline.

    Runs at priority 600 so that sphinx-gallery's own
    ``generate_gallery_rst`` (default priority 500) has already run and
    materialised every per-leaf ``index.rst``. ``dev_scripts`` is kept
    in the build (so internal links still resolve via a hidden toctree)
    but is omitted from the visible card grid: the plan calls it an
    internal-only catalogue.
    """
    src_root = Path(app.srcdir) / "generated" / "auto_examples"
    out_path = src_root / "index.rst"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-section metadata: gallery dir (relative to ``generated/auto_examples``),
    # H3 title, intro paragraph. Pulled from the per-folder README and the
    # tutorial restructure plan so the wording stays in lockstep with the
    # source-of-truth roster.
    tutorial_sections = [
        (
            "tutorials/00_start_here",
            "Start Here",
            "Difficulty 1. Three short lessons that take you from a fresh "
            "install to a working PyTorch ``DataLoader`` over real EEG "
            "records: find datasets and records, load one recording and "
            "inspect it, then turn an ``EEGDashDataset`` into windows and "
            "a dataloader. CPU-only, each runs in under a few minutes.",
        ),
        (
            "tutorials/10_core_workflow",
            "Core Decoding Workflow",
            "Difficulty 1-2. The canonical EEG decoding pipeline in four "
            "lessons: preprocess and window, split without subject leakage, "
            "train a baseline against chance, and persist prepared data for "
            "reuse. The leakage-safe split lesson is "
            "the rubric anchor for E3.27 invariants and Cisotto and Chicco "
            "2024's evaluation guidance.",
        ),
        (
            "tutorials/20_event_related",
            "Event-Related Decoding",
            "Difficulty 2. Two lessons that decode labels coming from "
            "events and annotations rather than continuous state: a P3 "
            "target-versus-standard classifier on a visual oddball "
            "paradigm, then the auditory oddball framed as a contrast "
            "with the visual case.",
        ),
        (
            "tutorials/30_resting_state",
            "Resting-State and State Decoding",
            "Difficulty 1. The canonical beginner decoding lesson: "
            "eyes-open versus eyes-closed classification on resting-state "
            "EEG, decoded from alpha-rhythm differences with a band-power "
            "baseline.",
        ),
        (
            "tutorials/40_features",
            "Feature Engineering",
            "Difficulty 1-2. EEGDash's feature extraction package as a "
            "first-class option, not an afterthought to deep learning. "
            "Three lessons cover feature tables from windows, preprocessor "
            "and dependency trees that avoid recomputation, and a "
            "scikit-learn / LightGBM baseline straight from the feature "
            "table.",
        ),
        (
            "tutorials/50_evaluation",
            "Evaluation and Benchmarking",
            "Difficulty 2-3. Five lessons that treat decoding evaluation "
            "as a core skill, drawing on MOABB (Chevallier, Aristimunha "
            "et al. 2024). Builds from a single split toward "
            "benchmark-grade pipeline comparison: within-subject, "
            "cross-subject, cross-session, learning curves, and a paired "
            "Wilcoxon comparison of two pipelines.",
        ),
        (
            "tutorials/70_transfer_foundation",
            "Transfer, Foundation Models, and EEG2025",
            "Difficulty 3. Four advanced lessons on transfer learning "
            "and foundation-model fine-tuning, framed around the EEG2025 "
            "Foundation Challenge: ``EEGChallengeDataset`` basics, "
            "cross-task transfer (Challenge 1), subject-invariant "
            "p-factor regression (Challenge 2), and fine-tuning a "
            "Braindecode pretrained model. Builds on Schirrmeister et al. 2017.",
        ),
    ]
    leaf_sections = [
        (
            "how_to",
            "How-to recipes",
            "Task-focused snippets that assume you already know the "
            "basics: how to download a dataset, run preprocessing on "
            "SLURM, parallelize feature extraction, use the HPC cache, "
            "and work offline. Each guide answers a single question; "
            "cross-link with the HPC track when relevant.",
        ),
        (
            "applied",
            "Applied research projects",
            "Project-style examples that target a concrete scientific "
            "question -- age regression, p-factor prediction, sex "
            "classification, P300 transfer, clinical-catalog summary -- "
            "with realistic data sizes, runtimes, and limitations. Treat "
            "them as starting points, not prescriptive recipes.",
        ),
        (
            "eeg2025",
            "EEG2025 Foundation Challenge",
            "End-to-end pipelines for the two EEG2025 Foundation "
            "Challenge tracks: cross-task transfer learning (passive to "
            "active), and subject-invariant representations for clinical "
            "factor prediction. Pre-trained weights ship alongside each "
            "tutorial.",
        ),
        (
            "hpc",
            "High-performance computing",
            "Reference setup for running EEGDash on shared HPC clusters: "
            "SLURM submission scripts (CPU and GPU), a Dockerfile, and a "
            "tutorial showing how to combine the on-disk cache with batch "
            "scheduling for an eyes-open / eyes-closed run.",
        ),
    ]

    parts: list[str] = [
        ":orphan:\n",
        ".. _sphx_glr_generated_auto_examples:\n",
        "Examples gallery",
        "================",
        "",
        "The EEGDash gallery is the runnable, narrative half of the docs: "
        "the **Concepts** chapter explains *why* a decision matters, the "
        "API reference enumerates every public symbol, and the gallery "
        "you're reading shows the choices in motion against real BIDS-"
        "curated EEG records. Every script under ``examples/`` is a "
        "sphinx-gallery tutorial -- meaning it executes top to bottom on "
        "every documentation build, and the captured first figure is the "
        "thumbnail you see below.",
        "",
        "The intended path: read the curated **Tutorials** in order, dip "
        "into **How-to recipes** when you have a specific question, then "
        "scale up using the **Applied research projects**, the **EEG2025 "
        "Foundation Challenge** pipelines, and the **High-performance "
        "computing** track.",
        "",
        ".. admonition:: How to read this gallery",
        "   :class: tip eegdash-gallery-howto",
        "",
        "   - **Reading order.** Tutorials are sorted by category and "
        "numbered (``plot_00_*``, ``plot_10_*``, ...). Inside a category "
        "they're sequenced beginner-first; the file numbers are the "
        "intended path.",
        "   - **Cards show the captured first figure.** Sphinx-gallery "
        "stores the first ``matplotlib`` figure as the thumbnail, so the "
        "card preview is the literal output of running the script. A "
        "branded fallback is shown when the tutorial produces no figure.",
        "   - **Difficulty.** Each section header states the difficulty "
        "range (1 = absolute beginner, 3 = advanced / foundation-model "
        "tier).",
        "",
        "Tutorials (curated learning path)",
        "---------------------------------",
        "",
        "Seven categories, ordered the way we would teach them: install, "
        "load, decode events, decode state, engineer features, evaluate "
        "rigorously, then scale to transfer and foundation models.",
        "",
        "Choose your path",
        "~~~~~~~~~~~~~~~~",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 30 35 35",
        "",
        "   * - Your goal",
        "     - Start with",
        "     - Then read",
        "   * - **Load my first dataset**",
        "     - :doc:`tutorials/00_start_here/index`",
        "     - :doc:`tutorials/10_core_workflow/index`",
        "   * - **Train a classifier safely**",
        "     - :doc:`tutorials/10_core_workflow/index`",
        "     - :doc:`tutorials/50_evaluation/index`",
        "   * - **Extract classical features**",
        "     - :doc:`tutorials/40_features/index`",
        "     - :doc:`how_to/index`",
        "   * - **Run on a cluster**",
        "     - :doc:`how_to/index`",
        "     - :doc:`hpc/index`",
        "   * - **Join EEG2025**",
        "     - :doc:`tutorials/70_transfer_foundation/index`",
        "     - :doc:`eeg2025/index`",
        "",
        ".. grid:: 1 2 2 4",
        "   :gutter: 3",
        "",
        "   .. grid-item-card:: 🚀 Learn the basics",
        "      :link: tutorials/00_start_here/index",
        "      :link-type: doc",
        "",
        "      Start with the absolute beginner tutorials.",
        "",
        "   .. grid-item-card:: 🔬 Run an applied project",
        "      :link: applied/index",
        "      :link-type: doc",
        "",
        "      Dive into real-world research case studies.",
        "",
        "   .. grid-item-card:: ⚡ Scale on HPC",
        "      :link: hpc/index",
        "      :link-type: doc",
        "",
        "      Move from local scripts to cluster-wide jobs.",
        "",
        "   .. grid-item-card:: 🏆 Join EEG2025",
        "      :link: eeg2025/index",
        "      :link-type: doc",
        "",
        "      Enter the official Foundation Challenge.",
        "",
    ]
    for rel, title, intro in tutorial_sections:
        parts.append(_section_block(src_root, rel, title, intro, level="~"))

    for rel, title, intro in leaf_sections:
        parts.append(_section_block(src_root, rel, title, intro, level="-"))

    # Toctrees keep the navigation tree wired up so individual tutorials
    # remain reachable from the sidebar. They render hidden because the
    # visible content is the card grid above; a visible toctree on top
    # of cards would duplicate the listing as a bullet list.
    parts.append(".. toctree::")
    parts.append("   :hidden:")
    parts.append("   :caption: Tutorials (curated learning path)")
    parts.append("")
    for rel, _title, _intro in tutorial_sections:
        parts.append(f"   {rel}/index")
    parts.append("")
    parts.append(".. toctree::")
    parts.append("   :hidden:")
    parts.append("   :caption: Recipes and applied work")
    parts.append("")
    for rel, _title, _intro in leaf_sections:
        parts.append(f"   {rel}/index")
    parts.append("")
    # ``dev_scripts`` is kept in the build (linked under a hidden toctree
    # so internal references stay valid) but is intentionally absent from
    # the visible gallery -- it's an internal debugging catalogue, not
    # public-facing tutorial content.
    parts.append(".. toctree::")
    parts.append("   :hidden:")
    parts.append("")
    parts.append("   dev_scripts/index")
    parts.append("")

    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def setup(app) -> dict:
    # Priority 600 so sphinx-gallery (default 500) has already emitted
    # the per-leaf index.rst files before we read them.
    app.connect("builder-inited", _write_auto_examples_root_index, priority=600)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
