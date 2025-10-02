"""
=========================================
Plotly dataset summary grouped bubbles
=========================================

Phase 5 demo illustrating the streamlined Plotly workflow for EEG-DaSh dataset
summaries. The script loads packaged summary metadata, generates grouped bubble
plots sized by record counts, and writes a self-contained HTML artefact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eegdash.logging import logger
from eegdash.plotting.dataset_summary_adapter import (
    BubbleSizeMetric,
    SummaryBubbleConfig,
    load_dataset_summary,
)
from eegdash.plotting.plotly_dataset_bubbles import generate_plotly_dataset_bubble

DEFAULT_OUTPUT_HTML = Path("dataset_summary_bubbles.html")
DEFAULT_LIMIT = 12
DEFAULT_LAYOUT_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render interactive Plotly grouped bubble charts from EEG-DaSh summaries."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum number of datasets to include (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--include-type-subject",
        action="store_true",
        help="Render Type Subject badge legends when metadata is available.",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=DEFAULT_LAYOUT_SEED,
        help=f"Deterministic seed applied to bubble layout (default: {DEFAULT_LAYOUT_SEED}).",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=DEFAULT_OUTPUT_HTML,
        help=f"Destination HTML file (default: {DEFAULT_OUTPUT_HTML}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    records = load_dataset_summary()
    if not records:
        raise SystemExit("No dataset summary records available for plotting.")

    config = SummaryBubbleConfig(
        limit=args.limit,
        include_type_subject_badges=args.include_type_subject,
        layout_seed=args.layout_seed,
        legend_heading="Record volume (log scale)",
    )

    generate_plotly_dataset_bubble(
        records,
        config,
        BubbleSizeMetric.N_RECORDS,
        outfile=args.output_html,
    )
    logger.info("Plotly grouped bubble chart written to %s", args.output_html.resolve())


if __name__ == "__main__":
    main()
