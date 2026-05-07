"""Drawing helpers for the ``plot_54`` paired-pipelines figure."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    chance_line,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


def _draw_paired_bars_panel(
    ax,
    *,
    pipeline_a_scores: np.ndarray,
    pipeline_b_scores: np.ndarray,
    subject_ids: Sequence[str],
    pipeline_names: Sequence[str],
    chance_level: float,
) -> np.ndarray:
    a_scores = np.asarray(pipeline_a_scores, dtype=float)
    b_scores = np.asarray(pipeline_b_scores, dtype=float)
    order = np.argsort(a_scores, kind="stable")
    a_sorted = a_scores[order]
    b_sorted = b_scores[order]
    deltas = a_sorted - b_sorted
    labels = [str(subject_ids[i]) for i in order]
    n = len(labels)

    positions = np.arange(n)
    bar_width = 0.40
    bars_a = ax.bar(
        positions - bar_width / 2,
        a_sorted,
        width=bar_width,
        color=EEGDASH_BLUE,
        edgecolor=EEGDASH_INK,
        linewidth=0.6,
        label=pipeline_names[0],
        zorder=3,
    )
    ax.bar(
        positions + bar_width / 2,
        b_sorted,
        width=bar_width,
        color=EEGDASH_ORANGE,
        edgecolor=EEGDASH_INK,
        linewidth=0.6,
        label=pipeline_names[1],
        zorder=3,
    )

    # Pre-set x-limits so the chance-line annotation drawn by
    # ``chance_line`` lands inside this panel rather than spilling into
    # the next gridspec slot.
    ax.set_xlim(positions[0] - 0.7, positions[-1] + 1.8)
    chance_line(ax, level=float(chance_level), label="chance")

    # Per-subject delta annotation above the taller bar in each pair.
    top_envelope = np.maximum(a_sorted, b_sorted)
    ymax = max(1.0, float(top_envelope.max()) + 0.18)
    for x, top, delta in zip(positions, top_envelope, deltas):
        if delta > 0:
            color = EEGDASH_BLUE
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
            fontsize=7.6,
            color=color,
            family="monospace",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.8, rotation=35, ha="right", color=EEGDASH_INK)
    ax.set_ylim(0.0, ymax)
    ax.set_ylabel("accuracy")
    ax.text(
        0.0,
        1.06,
        "Per-subject paired accuracy",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    # Caption indicating sort order so the reader knows where the
    # left-most subject came from without reading the helper docstring.
    ax.text(
        1.0,
        1.06,
        f"sorted by {pipeline_names[0]} accuracy",
        transform=ax.transAxes,
        fontsize=8.0,
        color=EEGDASH_MUTED,
        ha="right",
        va="bottom",
    )

    # Use only the bar handles (chance_line annotates itself in-axes).
    ax.legend(
        handles=[bars_a, ax.containers[1]],
        loc="lower right",
        fontsize=7.8,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
    )
    return order


def _draw_difference_panel(
    ax,
    *,
    deltas: np.ndarray,
    p_value: float,
    cohens_d: float,
    pipeline_names: Sequence[str],
) -> tuple[float, float, float]:
    diffs = np.asarray(deltas, dtype=float)
    n = diffs.size
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1)) if n > 1 else 0.0
    se_diff = sd_diff / np.sqrt(max(n, 1))
    # Use a normal-approx 95 % CI; with n_subjects in the 8-20 range
    # this is the band an EEG paper would print, and the formula is
    # short enough to keep next to the panel.
    half = 1.96 * se_diff
    ci_lo = mean_diff - half
    ci_hi = mean_diff + half

    # Histogram bins symmetric around zero for an honest readout of the
    # win/loss balance.
    span = float(np.max(np.abs(diffs))) if diffs.size else 0.05
    span = max(span, 0.05)
    bins = np.linspace(-span - 0.02, span + 0.02, 11)

    ax.hist(
        diffs,
        bins=bins,
        color=EEGDASH_BLUE,
        edgecolor=EEGDASH_INK,
        linewidth=0.6,
        alpha=0.85,
        zorder=3,
    )
    ax.axvline(
        0.0,
        color=EEGDASH_INK,
        linestyle=(0, (1, 1)),
        linewidth=1.0,
        zorder=4,
        label="no difference",
    )
    ax.axvline(
        mean_diff,
        color=EEGDASH_ORANGE,
        linestyle="-",
        linewidth=1.6,
        zorder=5,
        label=f"mean = {mean_diff:+.03f}",
    )
    ax.axvspan(
        ci_lo,
        ci_hi,
        color=EEGDASH_ORANGE,
        alpha=0.16,
        linewidth=0,
        zorder=2,
        label=f"95% CI [{ci_lo:+.03f}, {ci_hi:+.03f}]",
    )

    ax.set_xlabel(f"{pipeline_names[0]} - {pipeline_names[1]} accuracy")
    ax.set_ylabel("subjects")
    ax.text(
        0.0,
        1.06,
        "Paired-difference distribution",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    # Inferential readout printed inside the panel so the figure stands
    # alone in the rendered gallery thumbnail. Upper-right keeps it
    # away from the legend in the upper-left and from the bars below.
    readout = f"p = {p_value:.03f}\nCohen's d = {cohens_d:+.02f}\nn = {n} subjects"
    ax.text(
        0.98,
        0.97,
        readout,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        family="monospace",
        color=EEGDASH_INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": EEGDASH_MUTED,
            "linewidth": 0.6,
        },
    )

    # Headroom for the readout so it does not collide with bars when
    # most differences cluster near zero (peaky histogram).
    if diffs.size:
        ymax_default = ax.get_ylim()[1]
        ax.set_ylim(0.0, ymax_default * 1.40)
    # Legend in the centre-left so it stays clear of the upper-right
    # readout and the histogram peak (which usually sits around zero).
    ax.legend(loc="center left", fontsize=7.2, frameon=False)
    return mean_diff, ci_lo, ci_hi


def _draw_cumulative_wins_panel(
    ax,
    *,
    pipeline_a_scores: np.ndarray,
    pipeline_b_scores: np.ndarray,
    subject_ids: Sequence[str],
    sort_order: np.ndarray,
    pipeline_names: Sequence[str],
) -> int:
    a_sorted = np.asarray(pipeline_a_scores, dtype=float)[sort_order]
    b_sorted = np.asarray(pipeline_b_scores, dtype=float)[sort_order]
    n = a_sorted.size
    wins = (a_sorted > b_sorted).astype(int)
    cum_wins = np.cumsum(wins)
    xs = np.arange(1, n + 1)

    # Bounds first so the observed curve sits on top.
    ax.plot(
        xs,
        xs,
        color=EEGDASH_MUTED,
        linestyle="--",
        linewidth=1.0,
        label=f"unanimous {pipeline_names[0]} wins",
        zorder=2,
    )
    ax.plot(
        xs,
        np.zeros_like(xs),
        color=EEGDASH_MUTED,
        linestyle=":",
        linewidth=1.0,
        label=f"no {pipeline_names[0]} wins",
        zorder=2,
    )
    ax.step(
        xs,
        cum_wins,
        where="post",
        color=EEGDASH_BLUE,
        linewidth=1.8,
        label=f"observed ({pipeline_names[0]} > {pipeline_names[1]})",
        zorder=3,
    )
    # Final marker so the endpoint is visible against the dashed
    # diagonal even when wins == n_subjects.
    ax.scatter(
        [xs[-1]],
        [cum_wins[-1]],
        color=EEGDASH_ORANGE,
        edgecolor=EEGDASH_INK,
        linewidth=0.6,
        zorder=4,
        s=42,
    )

    win_rate = cum_wins[-1] / n if n else 0.0
    # Lower-right is the calmest corner: above it the dashed diagonal
    # rises into the legend, below sits the x-axis.
    ax.text(
        0.98,
        0.45,
        f"final wins = {int(cum_wins[-1])} / {n} ({win_rate:.0%})",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=8.4,
        family="monospace",
        color=EEGDASH_INK,
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": EEGDASH_MUTED,
            "linewidth": 0.6,
        },
    )

    ax.set_xlabel("subjects scanned (in left-panel order)")
    ax.set_ylabel(f"cumulative {pipeline_names[0]} wins")
    ax.set_xlim(1, n)
    ax.set_ylim(-0.5, n + 0.5)
    ax.set_xticks(xs)
    ax.set_yticks(np.arange(0, n + 1))
    ax.text(
        0.0,
        1.06,
        "Cumulative wins curve",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.legend(loc="upper left", fontsize=7.2, frameon=False)
    return int(cum_wins[-1])


def draw_compare_pipelines_figure(
    *,
    pipeline_a_scores: Sequence[float],
    pipeline_b_scores: Sequence[float],
    pipeline_names: Sequence[str],
    subject_ids: Sequence[str],
    p_value: float,
    cohens_d: float,
    chance_level: float = 0.5,
    plot_id: str = "plot_54",
) -> plt.Figure:
    """Render the three-panel paired-pipeline comparison figure.

    Parameters
    ----------
    pipeline_a_scores, pipeline_b_scores : sequence of float
        Per-subject test accuracies. Both arrays must align element-wise
        on ``subject_ids`` (this is what makes the comparison paired).
    pipeline_names : sequence of two str
        Display names for Pipeline A (blue) and Pipeline B (orange),
        used in legends and axis labels.
    subject_ids : sequence of str
        Held-out subject identifier per fold; one entry per element of
        ``pipeline_a_scores``.
    p_value : float
        Two-sided p-value from a paired test (Wilcoxon signed-rank or
        :func:`scipy.stats.ttest_rel`). Printed inside the centre panel.
    cohens_d : float
        Standardised effect size on the paired differences. Printed
        next to the p-value.
    chance_level : float, default ``0.5``
        Chance-accuracy reference rendered on the bar panel.
    plot_id : str, default ``"plot_54"``
        Tutorial id forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure

    """
    a_scores = np.asarray(pipeline_a_scores, dtype=float)
    b_scores = np.asarray(pipeline_b_scores, dtype=float)
    if a_scores.shape != b_scores.shape:
        raise ValueError(
            "pipeline_a_scores and pipeline_b_scores must have matching shapes; "
            f"got {a_scores.shape} vs {b_scores.shape}"
        )
    if len(subject_ids) != a_scores.size:
        raise ValueError(
            "subject_ids length must equal the number of paired scores; "
            f"got {len(subject_ids)} ids for {a_scores.size} scores"
        )
    if len(pipeline_names) != 2:
        raise ValueError(
            f"pipeline_names must have exactly two entries, got {len(pipeline_names)}"
        )

    fig = plt.figure(figsize=(15.0, 5.6))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.65, 1.00, 1.10),
        wspace=0.42,
    )
    ax_bars = fig.add_subplot(gs[0, 0])
    ax_diff = fig.add_subplot(gs[0, 1])
    ax_cum = fig.add_subplot(gs[0, 2])

    sort_order = _draw_paired_bars_panel(
        ax_bars,
        pipeline_a_scores=a_scores,
        pipeline_b_scores=b_scores,
        subject_ids=subject_ids,
        pipeline_names=pipeline_names,
        chance_level=chance_level,
    )

    deltas = a_scores - b_scores
    mean_diff, _, _ = _draw_difference_panel(
        ax_diff,
        deltas=deltas,
        p_value=p_value,
        cohens_d=cohens_d,
        pipeline_names=pipeline_names,
    )

    _draw_cumulative_wins_panel(
        ax_cum,
        pipeline_a_scores=a_scores,
        pipeline_b_scores=b_scores,
        subject_ids=subject_ids,
        sort_order=sort_order,
        pipeline_names=pipeline_names,
    )

    n_subjects = a_scores.size
    sd_diff = float(deltas.std(ddof=1)) if n_subjects > 1 else 0.0
    se_diff = sd_diff / np.sqrt(max(n_subjects, 1))
    subtitle = (
        f"n_subjects={n_subjects}  |  A={pipeline_names[0]}, "
        f"B={pipeline_names[1]}  |  mean_diff(A-B)={mean_diff:+.03f} "
        f"+/- {se_diff:.03f} (SE)  |  p={p_value:.03f}  "
        f"|  Cohen's d={cohens_d:+.02f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=None,
        citation="synthetic cohort | shared cross-subject manifest",
    )
    style_figure(
        fig,
        title="Does Pipeline A really beat Pipeline B, or is the gap noise?",
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )
    # ``style_figure`` reserves the default top/bottom band; the
    # three-panel layout needs a touch more breathing room for the
    # rotated subject ticks and the in-panel readouts.
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.06, right=0.97)
    return fig


__all__ = ["draw_compare_pipelines_figure"]
