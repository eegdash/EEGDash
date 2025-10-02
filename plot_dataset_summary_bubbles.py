"""
==============================
Dataset summary grouped bubbles
==============================

Phase 2 demo illustrating the enriched EEG-DaSh dataset summary workflow. The script
loads canonical summary metadata, converts it into grouped bubble payloads via the
adapter utilities, and renders deterministic cluster plots for one or more metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from dataset_bubble_methods import _ClusterDatasetPlotter
from eegdash.logging import logger
from eegdash.plotting.dataset_summary_adapter import (
    BubbleSizeMetric,
    SummaryBubbleArtifacts,
    SummaryBubbleConfig,
    load_dataset_summary,
    summary_to_bubble_kwargs,
)

DEFAULT_METRICS: tuple[BubbleSizeMetric, ...] = (BubbleSizeMetric.N_RECORDS,)
DEFAULT_LIMIT = 12
DEFAULT_LAYOUT_SEED = 42
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "demo_outputs"

_METRIC_ALIASES = {
    "records": BubbleSizeMetric.N_RECORDS,
    "n_records": BubbleSizeMetric.N_RECORDS,
    BubbleSizeMetric.N_RECORDS.value: BubbleSizeMetric.N_RECORDS,
}


def _metric_argument(value: str) -> BubbleSizeMetric:
    """Normalise CLI arguments into :class:`BubbleSizeMetric` instances."""
    key = value.lower()
    metric = _METRIC_ALIASES.get(key)
    if metric is None:
        raise argparse.ArgumentTypeError(
            f"Unknown metric '{value}'. Valid choices: {', '.join(sorted(_METRIC_ALIASES))}"
        )
    return metric


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render deterministic grouped bubble charts from EEG-DaSh summaries."
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        type=_metric_argument,
        default=None,
        help="Bubble size metric to render. Only 'records' is currently supported.",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum number of datasets to include (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--include-type-subject",
        dest="include_type_subject",
        action="store_true",
        help="Render Type Subject badge legends when metadata is available.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where figures are written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display the generated figures using pyplot after saving.",
    )
    return parser.parse_args()


def _render_cluster(artifacts: SummaryBubbleArtifacts) -> plt.Figure:
    cfg = artifacts.config or SummaryBubbleConfig()
    plotter = _ClusterDatasetPlotter(
        datasets=artifacts.bubble_kwargs,
        meta_gap=cfg.meta_gap,
        collapse_iterations=cfg.collapse_iterations,
        kwargs={
            "color_map": artifacts.color_map,
            "scale": cfg.scale,
            "shape": cfg.shape,
            "legend": True,
            "legend_config": artifacts.legend_config,
        },
    )
    return plotter.plot()


def _resolve_metrics(
    raw_metrics: Iterable[BubbleSizeMetric] | None,
) -> list[BubbleSizeMetric]:
    if not raw_metrics:
        return list(DEFAULT_METRICS)
    seen: list[BubbleSizeMetric] = []
    for metric in raw_metrics:
        if metric not in seen:
            seen.append(metric)
    return seen


def main() -> None:
    args = _parse_args()
    metrics = _resolve_metrics(args.metrics)

    records = load_dataset_summary()
    if not records:
        raise SystemExit("No dataset summary records available for plotting.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        cfg = SummaryBubbleConfig(
            size_metric=metric,
            limit=args.limit,
            include_type_subject_badges=args.include_type_subject,
            layout_seed=DEFAULT_LAYOUT_SEED,
            legend_heading=f"{metric.heading} (log scale)",
        )
        artifacts = summary_to_bubble_kwargs(records, config=cfg)
        if not artifacts.bubble_kwargs:
            logger.warning(
                "Skipping %s bubble rendering because no payloads were produced.",
                metric.value,
            )
            continue

        fig = _render_cluster(artifacts)
        fig.suptitle(
            "EEG-DaSh Dataset Summary – Grouped Bubbles (record volume)",
            fontsize=14,
        )
        fig.tight_layout()

        filename = f"dataset_summary_bubbles_{metric.value}.png"
        output_path = args.output_dir / filename
        fig.savefig(output_path, dpi=150)
        if args.show:
            plt.show(block=False)
        else:
            plt.close(fig)
        logger.info("Saved %s bubble figure to %s", metric.value, output_path)


if __name__ == "__main__":
    main()
