"""Drawing helpers for the ``plot_55`` EEGDash + MOABB interop figure."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_SURFACE,
    chance_line,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


def _per_subject_table(
    per_subject_results: pd.DataFrame,
    *,
    pipeline_a: str,
    pipeline_b: str | None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    df = per_subject_results.copy()
    df["subject"] = df["subject"].astype(str)
    pivot = df.pivot_table(
        index="subject", columns="pipeline", values="score", aggfunc="mean"
    ).sort_index()
    if pipeline_a not in pivot.columns:
        raise ValueError(
            f"pipeline_a={pipeline_a!r} missing from per_subject_results "
            f"(found {list(pivot.columns)})"
        )
    subjects = list(pivot.index)
    scores_a = pivot[pipeline_a].to_numpy(dtype=float)
    scores_b = (
        pivot[pipeline_b].to_numpy(dtype=float)
        if pipeline_b is not None and pipeline_b in pivot.columns
        else None
    )
    return scores_a, scores_b, subjects


def _draw_per_subject_bars(
    ax,
    *,
    scores: np.ndarray,
    subjects: Sequence[str],
    pipeline_name: str,
    dataset_name: str,
    paradigm_name: str,
    chance_level: float,
) -> None:
    n = scores.size
    positions = np.arange(n)
    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores, ddof=0))

    ax.axhspan(
        max(0.0, mean_acc - std_acc),
        min(1.0, mean_acc + std_acc),
        color=EEGDASH_BLUE,
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )
    ax.axhline(
        mean_acc,
        color=EEGDASH_BLUE_DARK,
        linestyle=":",
        linewidth=1.0,
        zorder=2,
        label=f"mean = {mean_acc:.2f}",
    )

    bars = ax.bar(
        positions,
        scores,
        width=0.62,
        color=EEGDASH_BLUE,
        edgecolor=EEGDASH_INK,
        linewidth=0.7,
        zorder=3,
    )
    for bar, acc in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.018,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color=EEGDASH_INK,
            fontweight="bold",
        )

    ax.set_xlim(positions[0] - 0.7, positions[-1] + 1.6)
    chance_line(ax, level=float(chance_level), label="chance")

    short_ids = [str(s).replace("sub-", "") for s in subjects]
    ax.set_xticks(positions)
    ax.set_xticklabels(short_ids, fontsize=7.8, family="monospace")
    ax.set_xlabel("subject")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.text(
        0.0,
        1.06,
        f"MOABB per-subject: {dataset_name}",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.0,
        1.005,
        f"{paradigm_name} | {pipeline_name}",
        transform=ax.transAxes,
        fontsize=8.2,
        color=EEGDASH_MUTED,
        ha="left",
        va="bottom",
    )
    ax.legend(
        loc="lower right",
        fontsize=7.4,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
    )


def _draw_pipeline_comparison(
    ax,
    *,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    subjects: Sequence[str],
    pipeline_names: Sequence[str],
    chance_level: float,
) -> None:
    n = scores_a.size
    positions = np.arange(n)
    bar_w = 0.40
    deltas = scores_a - scores_b
    mean_delta = float(np.mean(deltas))
    n_wins = int(np.sum(scores_a > scores_b))

    bars_a = ax.bar(
        positions - bar_w / 2,
        scores_a,
        width=bar_w,
        color=EEGDASH_BLUE,
        edgecolor=EEGDASH_INK,
        linewidth=0.6,
        label=pipeline_names[0],
        zorder=3,
    )
    bars_b = ax.bar(
        positions + bar_w / 2,
        scores_b,
        width=bar_w,
        color=EEGDASH_ORANGE,
        edgecolor=EEGDASH_INK,
        linewidth=0.6,
        label=pipeline_names[1],
        zorder=3,
    )

    top_envelope = np.maximum(scores_a, scores_b)
    for x, top, delta in zip(positions, top_envelope, deltas):
        if delta > 0:
            color = EEGDASH_BLUE_DARK
        elif delta < 0:
            color = EEGDASH_ORANGE
        else:
            color = EEGDASH_MUTED
        ax.text(
            x,
            top + 0.025,
            f"{delta:+.02f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=color,
            family="monospace",
        )

    ax.set_xlim(positions[0] - 0.7, positions[-1] + 1.8)
    chance_line(ax, level=float(chance_level), label="chance")

    short_ids = [str(s).replace("sub-", "") for s in subjects]
    ax.set_xticks(positions)
    ax.set_xticklabels(short_ids, fontsize=7.8, family="monospace")
    ax.set_xlabel("subject")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.18)
    ax.text(
        0.0,
        1.06,
        "Pipeline A vs Pipeline B (paired by subject)",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    readout = f"mean delta(A-B) = {mean_delta:+.03f}\nA wins on {n_wins} / {n} subjects"
    ax.text(
        0.98,
        0.93,
        readout,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        family="monospace",
        color=EEGDASH_INK,
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": EEGDASH_MUTED,
            "linewidth": 0.6,
        },
    )
    ax.legend(
        handles=[bars_a, bars_b],
        loc="lower right",
        fontsize=7.4,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
    )


def _draw_node(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    title: str,
    sub: str,
    foot: str,
) -> tuple[float, float]:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.020",
            linewidth=1.5,
            edgecolor=color,
            facecolor=EEGDASH_SURFACE,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.045,
        title,
        ha="center",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=EEGDASH_INK,
    )
    ax.text(
        x + w / 2,
        y + h - 0.130,
        sub,
        ha="center",
        va="top",
        fontsize=7.6,
        color=color,
    )
    ax.text(
        x + w / 2,
        y + 0.050,
        foot,
        ha="center",
        va="bottom",
        fontsize=7.0,
        family="monospace",
        color=EEGDASH_MUTED,
    )
    return x, x + w


def _draw_flow_arrow(
    ax,
    *,
    x_from: float,
    x_to: float,
    y: float,
    label_top: str,
    label_bot: str,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x_from, y),
            (x_to, y),
            arrowstyle="-|>,head_length=8,head_width=6",
            linewidth=1.6,
            color=EEGDASH_GRID,
            zorder=4,
        )
    )
    xc = (x_from + x_to) / 2
    ax.text(
        xc,
        y + 0.075,
        label_top,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=EEGDASH_INK,
        fontweight="bold",
    )
    ax.text(
        xc,
        y + 0.020,
        label_bot,
        ha="center",
        va="bottom",
        fontsize=6.6,
        color=EEGDASH_MUTED,
    )


def _draw_integration_diagram(
    ax,
    *,
    dataset_name: str,
    paradigm_name: str,
    n_subjects: int,
    n_pipelines: int,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.set_axis_off()

    ax.text(
        0.0,
        1.06,
        "Integration flow: EEGDash <-> MOABB",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    # Node geometry. Reserve enough horizontal gap between nodes so the
    # short-form arrow labels render in the gutter, not on top of the box.
    node_y = 0.34
    node_h = 0.54
    node_w = 0.180
    n_nodes = 4
    interior_gap = 0.075
    edge_pad = (1.0 - n_nodes * node_w - (n_nodes - 1) * interior_gap) / 2
    xs = [edge_pad + i * (node_w + interior_gap) for i in range(n_nodes)]

    paradigm_short = paradigm_name.split(" ")[0] if paradigm_name else "Paradigm"
    nodes = [
        {
            "x": xs[0],
            "color": EEGDASH_BLUE,
            "title": "EEGDashDataset",
            "sub": "BIDS catalog",
            "foot": str(dataset_name)[:18],
        },
        {
            "x": xs[1],
            "color": EEGDASH_BLUE_DARK,
            "title": "mne.Raw",
            "sub": "loaded recording",
            "foot": f"{n_subjects} subjects",
        },
        {
            "x": xs[2],
            "color": EEGDASH_ORANGE,
            "title": "MOABB Paradigm",
            "sub": ".get_data()",
            "foot": paradigm_short,
        },
        {
            "x": xs[3],
            "color": EEGDASH_INK,
            "title": "Evaluation",
            "sub": "CrossSession",
            "foot": f"{n_pipelines} pipelines",
        },
    ]

    anchors = []
    for node in nodes:
        anchors.append(
            _draw_node(
                ax,
                x=node["x"],
                y=node_y,
                w=node_w,
                h=node_h,
                color=node["color"],
                title=node["title"],
                sub=node["sub"],
                foot=node["foot"],
            )
        )

    arrow_y = node_y + node_h * 0.55
    arrow_labels = [
        ("load", "subject"),
        ("epochs", "+ labels"),
        ("fit/score", "per fold"),
    ]
    for i, (top, bot) in enumerate(arrow_labels):
        _draw_flow_arrow(
            ax,
            x_from=anchors[i][1] + 0.003,
            x_to=anchors[i + 1][0] - 0.003,
            y=arrow_y,
            label_top=top,
            label_bot=bot,
        )

    # Footnote band: name the eegdash <-> moabb bridge function.
    ax.text(
        0.5,
        0.13,
        "bridge: eegdash.splits.to_moabb_split_inputs(dataset, target=...) -> (y, metadata)",
        ha="center",
        va="center",
        fontsize=7.8,
        family="monospace",
        color=EEGDASH_INK,
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": EEGDASH_MUTED,
            "linewidth": 0.6,
        },
    )


def draw_moabb_interop_figure(
    *,
    per_subject_results: pd.DataFrame,
    dataset_name: str,
    paradigm_name: str,
    pipeline_a: str,
    pipeline_b: str | None = None,
    chance_level: float = 0.5,
    used_moabb: bool = True,
    plot_id: str = "plot_55",
) -> plt.Figure:
    """Render the three-panel EEGDash + MOABB benchmark figure.

    Parameters
    ----------
    per_subject_results : pandas.DataFrame
        Long-format frame with columns ``subject``, ``pipeline``, ``score``.
        Rows for subjects without a value for a given pipeline are tolerated:
        the pivot fills missing entries with NaN, and the comparison panel
        falls back to the first pipeline only.
    dataset_name : str
        Display name for the source dataset (e.g. ``"BNCI2014_001"``).
    paradigm_name : str
        Display name for the MOABB paradigm (e.g. ``"MotorImagery"``).
    pipeline_a : str
        Pipeline whose per-subject scores carry the headline panel and the
        first set of bars in the comparison panel.
    pipeline_b : str, optional
        Second pipeline for the paired comparison panel. When ``None`` the
        comparison panel falls back to a duplicate of the headline bars
        with a single legend entry.
    chance_level : float, default 0.5
        Reference chance accuracy. For a balanced binary task this is 0.5.
    used_moabb : bool, default True
        Whether the underlying numbers came from a real MOABB evaluation
        (``True``) or from the synthetic-results fallback (``False``).
        Surfaced in the subtitle so the rendered figure is self-describing.
    plot_id : str, default ``"plot_55"``
        Tutorial id forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure

    """
    if not isinstance(per_subject_results, pd.DataFrame):
        raise TypeError(
            "per_subject_results must be a pandas.DataFrame in long format with "
            "columns subject / pipeline / score."
        )
    expected_cols = {"subject", "pipeline", "score"}
    missing = expected_cols - set(per_subject_results.columns)
    if missing:
        raise ValueError(
            f"per_subject_results missing columns: {sorted(missing)} "
            f"(got {list(per_subject_results.columns)})"
        )

    scores_a, scores_b, subjects = _per_subject_table(
        per_subject_results, pipeline_a=pipeline_a, pipeline_b=pipeline_b
    )
    n_subjects = len(subjects)
    if n_subjects == 0:
        raise ValueError("per_subject_results is empty after pivot.")
    if scores_b is None:
        # Fallback: paint the same bars in both colors so the layout stays
        # legible even when the caller only ran one pipeline.
        scores_b = scores_a.copy()
        pipeline_b_name = pipeline_a
    else:
        pipeline_b_name = pipeline_b  # type: ignore[assignment]

    fig = plt.figure(figsize=(17.4, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.00, 1.05, 1.45),
        wspace=0.36,
    )
    ax_bars = fig.add_subplot(gs[0, 0])
    ax_compare = fig.add_subplot(gs[0, 1])
    ax_flow = fig.add_subplot(gs[0, 2])

    _draw_per_subject_bars(
        ax_bars,
        scores=scores_a,
        subjects=subjects,
        pipeline_name=pipeline_a,
        dataset_name=dataset_name,
        paradigm_name=paradigm_name,
        chance_level=chance_level,
    )
    _draw_pipeline_comparison(
        ax_compare,
        scores_a=scores_a,
        scores_b=scores_b,
        subjects=subjects,
        pipeline_names=(pipeline_a, pipeline_b_name),
        chance_level=chance_level,
    )
    _draw_integration_diagram(
        ax_flow,
        dataset_name=dataset_name,
        paradigm_name=paradigm_name,
        n_subjects=n_subjects,
        n_pipelines=int(per_subject_results["pipeline"].nunique()),
    )

    mean_acc = float(np.mean(scores_a))
    std_acc = float(np.std(scores_a, ddof=0))
    backend = "moabb" if used_moabb else "synthetic fallback"
    subtitle = (
        f"n_subjects={n_subjects} | dataset={dataset_name} | "
        f"paradigm={paradigm_name} | mean_acc={mean_acc:.2f} +/- {std_acc:.2f} "
        f"| backend={backend}"
    )
    citation = (
        "BNCI2014_001 (Tangermann et al. 2012)"
        if dataset_name.startswith("BNCI")
        else f"{dataset_name}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=None,
        citation=citation,
    )
    style_figure(
        fig,
        title="How do I benchmark an EEGDash dataset with MOABB?",
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )
    fig.subplots_adjust(top=0.74, bottom=0.18, left=0.06, right=0.98)
    return fig


__all__ = ["draw_moabb_interop_figure"]
