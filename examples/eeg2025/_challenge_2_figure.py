"""Drawing helpers for the EEG2025 Challenge 2 starter kit.

Sibling module to ``tutorial_challenge_2.py``. The leading underscore
keeps sphinx-gallery from rendering the file (its ``ignore_pattern``
matches ``_*.py``), so the plotting plumbing stays out of the rendered
tutorial. The tutorial imports the public
:func:`draw_challenge_2_figure` entry point and feeds it three live
inputs: the train-cohort p-factor distribution, a vector of held-out
predictions, and a leaderboard table.

The figure has three panels arranged left to right and reads as the
classic baseline-vs-target story for a regression challenge:

1. **Target distribution.** Histogram of p-factor across the training
   cohort. Mean and median guides annotate the location and skew of the
   target so reviewers can see how far the leaderboard scores can
   plausibly travel.
2. **Predicted vs true scatter.** One marker per held-out subject (mean
   prediction across that subject's windows). The diagonal ``y = x``
   reference is the only line a perfect model would hit; corner box
   reports Pearson r, R^2, and MAE on the subject-level vectors.
3. **Leaderboard card.** Three rows: median baseline (chance), the
   starter-kit baseline produced by this notebook, and the public top
   score from the EEG2025 challenge dashboard. Bars are colored by
   regime so the gap between chance, starter, and target reads at a
   glance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

from eegdash.viz import (
    EEGDASH_AMBER,
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_SURFACE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# Regime -> color map for the leaderboard card. ``chance`` (amber) is
# the floor any honest submission must clear; ``starter`` (blue) is the
# notebook baseline; ``target`` (orange) is the public top score. A
# fourth fallback color is reserved for custom rows callers may add.
_REGIME_COLORS = {
    "chance": EEGDASH_AMBER,
    "starter": EEGDASH_BLUE,
    "target": EEGDASH_ORANGE,
    "custom": EEGDASH_BLUE_DARK,
}


def _aggregate_per_subject(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Reduce window-level vectors to one mean prediction per subject.

    The p-factor is a subject-level score: every window of a given
    subject carries the same ``y_true``. So the only meaningful x point
    per subject is its single true score, and the only meaningful y is
    the mean prediction across that subject's held-out windows.
    """
    subjects = np.asarray(list(subject_ids))
    unique = list(dict.fromkeys(subjects.tolist()))
    yt_subj = np.empty(len(unique), dtype=float)
    yp_subj = np.empty(len(unique), dtype=float)
    for idx, sid in enumerate(unique):
        mask = subjects == sid
        yt_subj[idx] = float(np.mean(y_true[mask]))
        yp_subj[idx] = float(np.mean(y_pred[mask]))
    return yt_subj, yp_subj, unique


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute Pearson r, R^2, MAE on subject-level vectors."""
    if y_true.size < 2:
        return {"pearson_r": np.nan, "r2": np.nan, "mae": np.nan}
    pearson_r = float(pearsonr(y_true, y_pred).statistic)
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"pearson_r": pearson_r, "r2": r2, "mae": mae}


def _draw_distribution_panel(
    ax,
    *,
    p_factor_distribution: np.ndarray,
) -> None:
    """Render the train-cohort p-factor histogram with mean / median guides."""
    values = np.asarray(p_factor_distribution, dtype=float).ravel()
    values = values[~np.isnan(values)]
    if values.size == 0:
        ax.text(
            0.5,
            0.5,
            "no p_factor values supplied",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=EEGDASH_MUTED,
        )
        return

    n_bins = int(min(24, max(8, np.sqrt(values.size))))
    ax.hist(
        values,
        bins=n_bins,
        color=EEGDASH_BLUE,
        edgecolor=EEGDASH_INK,
        linewidth=0.5,
        alpha=0.82,
        zorder=2,
    )

    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    ax.axvline(
        mean_val,
        color=EEGDASH_ORANGE,
        linestyle="-",
        linewidth=1.3,
        zorder=4,
        label=f"mean = {mean_val:+.2f}",
    )
    ax.axvline(
        median_val,
        color=EEGDASH_BLUE_DARK,
        linestyle="--",
        linewidth=1.1,
        zorder=4,
        label=f"median = {median_val:+.2f}",
    )

    ax.set_xlabel("p-factor (CBCL-derived, z-scored)")
    ax.set_ylabel("count (subjects)")
    ax.text(
        0.0,
        1.06,
        "Target distribution (training cohort)",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.legend(
        loc="upper left",
        fontsize=7.6,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
    )


def _draw_scatter_panel(
    ax,
    *,
    y_true_subj: np.ndarray,
    y_pred_subj: np.ndarray,
    metrics: Mapping[str, float],
) -> None:
    """Render predicted vs true scatter with a y=x reference and a metric box."""
    if y_true_subj.size == 0:
        ax.text(
            0.5,
            0.5,
            "no held-out predictions supplied",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=EEGDASH_MUTED,
        )
        return

    lo = float(min(y_true_subj.min(), y_pred_subj.min()))
    hi = float(max(y_true_subj.max(), y_pred_subj.max()))
    pad = 0.10 * max(hi - lo, 1e-3)
    lo -= pad
    hi += pad

    ax.plot(
        [lo, hi],
        [lo, hi],
        color=EEGDASH_MUTED,
        linestyle="--",
        linewidth=0.9,
        zorder=1,
        label="y = x (perfect)",
    )
    ax.scatter(
        y_true_subj,
        y_pred_subj,
        s=70,
        facecolor=EEGDASH_BLUE,
        edgecolor=EEGDASH_INK,
        linewidth=0.7,
        alpha=0.85,
        zorder=3,
        label="held-out subject",
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("true p-factor")
    ax.set_ylabel("predicted p-factor")
    ax.text(
        0.0,
        1.06,
        "Predicted vs true (held-out subjects)",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    metric_box = (
        f"r    = {metrics['pearson_r']:+.3f}\n"
        f"R^2  = {metrics['r2']:+.3f}\n"
        f"MAE  = {metrics['mae']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        metric_box,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        family="monospace",
        color=EEGDASH_INK,
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": EEGDASH_MUTED,
            "linewidth": 0.6,
            "alpha": 0.95,
        },
    )
    ax.legend(loc="lower right", fontsize=7.6, frameon=False)


def _draw_leaderboard_panel(
    ax,
    *,
    leaderboard_rows: Sequence[Mapping[str, object]],
) -> None:
    """Render a leaderboard-style result card.

    Each row is rendered as a horizontal bar from 0 to ``score`` with a
    monospace label on the right and the team / regime name on the
    left. Rows are drawn top to bottom in input order so the caller
    decides the reading sequence (typically chance, starter, target).
    """
    if len(leaderboard_rows) == 0:
        ax.text(
            0.5,
            0.5,
            "no leaderboard rows supplied",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=EEGDASH_MUTED,
        )
        return

    rows = list(leaderboard_rows)
    n_rows = len(rows)
    y_positions = np.arange(n_rows)[::-1]
    score_values = np.asarray([float(row.get("score", np.nan)) for row in rows])
    finite_mask = np.isfinite(score_values)
    max_score = float(np.max(score_values[finite_mask])) if finite_mask.any() else 1.0
    max_score = max(max_score, 0.10)

    # Background pill so the card reads as a result block, not a plot.
    ax.set_facecolor(EEGDASH_SURFACE)

    for ypos, row, score in zip(y_positions, rows, score_values):
        regime = str(row.get("regime", "custom")).lower()
        color = _REGIME_COLORS.get(regime, _REGIME_COLORS["custom"])
        if not np.isfinite(score):
            # Placeholder for a leaderboard score that has not been
            # released yet; render as a dashed outline so the row stays
            # visible but communicates "unknown" rather than "zero".
            ax.barh(
                ypos,
                max_score,
                color="white",
                edgecolor=color,
                linewidth=1.2,
                linestyle="--",
                height=0.62,
                zorder=2,
            )
            score_text = "n/a"
        else:
            ax.barh(
                ypos,
                score,
                color=color,
                edgecolor=EEGDASH_INK,
                linewidth=0.6,
                height=0.62,
                zorder=3,
            )
            score_text = f"{score:+.3f}"

        team = str(row.get("team", regime))
        metric = str(row.get("metric", "r"))
        ax.text(
            -0.02 * max_score,
            ypos,
            team,
            ha="right",
            va="center",
            fontsize=9.0,
            color=EEGDASH_INK,
            fontweight="bold",
        )
        ax.text(
            max_score * 1.02,
            ypos,
            f"{score_text} ({metric})",
            ha="left",
            va="center",
            fontsize=8.6,
            family="monospace",
            color=EEGDASH_INK,
        )

    ax.set_yticks([])
    ax.set_xticks([0, max_score / 2, max_score])
    ax.set_xticklabels([f"{0.0:.2f}", f"{max_score / 2:.2f}", f"{max_score:.2f}"])
    ax.set_xlim(-0.65 * max_score, 1.40 * max_score)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel("score (higher is better)")
    ax.text(
        0.0,
        1.06,
        "Leaderboard-style result card",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )


def draw_challenge_2_figure(
    *,
    p_factor_distribution: np.ndarray,
    y_true_subj: np.ndarray,
    y_pred_subj: np.ndarray,
    leaderboard_rows: Sequence[Mapping[str, object]],
    subject_ids: Sequence[str] | None = None,
    plot_id: str = "tutorial_challenge_2",
) -> plt.Figure:
    """Render the three-panel EEG2025 Challenge 2 starter-kit figure.

    Parameters
    ----------
    p_factor_distribution : numpy.ndarray
        ``(n_train_subjects,)`` p-factor values across the training
        cohort. NaNs are dropped before plotting.
    y_true_subj : numpy.ndarray
        ``(n_held_out,)`` true p-factor values pooled across folds (or
        a single held-out fold) at the window level.
    y_pred_subj : numpy.ndarray
        ``(n_held_out,)`` predicted p-factor values, same length as
        ``y_true_subj``.
    leaderboard_rows : sequence of mapping
        Each mapping has keys ``team`` (display name), ``regime``
        (``chance`` / ``starter`` / ``target`` / ``custom``), ``score``
        (float or ``nan`` for placeholder), and ``metric`` (display
        suffix, defaults to ``r``).
    subject_ids : sequence of str, optional
        ``(n_held_out,)`` subject id per row. When provided, the
        scatter aggregates to one point per subject; otherwise every
        held-out window is plotted as a separate marker.
    plot_id : str
        Tutorial id forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure

    """
    train_p = np.asarray(p_factor_distribution, dtype=float).ravel()
    y_true_window = np.asarray(y_true_subj, dtype=float).ravel()
    y_pred_window = np.asarray(y_pred_subj, dtype=float).ravel()
    if y_true_window.shape != y_pred_window.shape:
        raise ValueError(
            "y_true_subj/y_pred_subj shape mismatch: "
            f"{y_true_window.shape} vs {y_pred_window.shape}"
        )
    if subject_ids is None:
        subj_ids_local = [f"sub-{i:03d}" for i in range(y_true_window.size)]
    else:
        if len(subject_ids) != y_true_window.size:
            raise ValueError(
                f"subject_ids length {len(subject_ids)} != "
                f"prediction length {y_true_window.size}"
            )
        subj_ids_local = list(subject_ids)

    yt_subj, yp_subj, unique_subjects = _aggregate_per_subject(
        y_true=y_true_window,
        y_pred=y_pred_window,
        subject_ids=subj_ids_local,
    )
    metrics = _compute_metrics(yt_subj, yp_subj)

    fig = plt.figure(figsize=(14.5, 5.2))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.05, 1.10, 1.20),
        wspace=0.42,
    )
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_board = fig.add_subplot(gs[0, 2])

    _draw_distribution_panel(ax_dist, p_factor_distribution=train_p)
    _draw_scatter_panel(
        ax_scatter,
        y_true_subj=yt_subj,
        y_pred_subj=yp_subj,
        metrics=metrics,
    )
    _draw_leaderboard_panel(ax_board, leaderboard_rows=leaderboard_rows)

    n_subjects = len(unique_subjects)
    train_n = int(train_p[~np.isnan(train_p)].size)
    subtitle = (
        f"dataset=EEG2025 R5 | n_subjects={n_subjects} | "
        f"r={metrics['pearson_r']:+.3f} | R^2={metrics['r2']:+.3f} | "
        f"MAE={metrics['mae']:.3f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=None,
        citation="HBN release R5 (Alexander et al. 2017) via NEMAR",
        extra=f"train cohort n={train_n}",
    )
    style_figure(
        fig,
        title="EEG2025 Challenge 2: predicting the p-factor from EEG",
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )

    # ``style_figure`` reserves the title/subtitle band; with three
    # panels carrying their own titles a hair more breathing room
    # around the leaderboard card reads cleaner.
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.07, right=0.97)
    fig._eegdash_challenge_2_metrics = {
        **metrics,
        "n_subjects": n_subjects,
        "n_train_subjects": train_n,
    }
    return fig


__all__ = ["draw_challenge_2_figure"]
