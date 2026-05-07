"""Drawing helpers for the ``plot_72`` p-factor regression diagnostic."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


def _aggregate_per_subject(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    subjects = np.asarray(list(subject_ids))
    unique = list(dict.fromkeys(subjects.tolist()))  # preserves first-seen order
    yt_subj = np.empty(len(unique), dtype=float)
    yp_subj = np.empty(len(unique), dtype=float)
    for idx, sid in enumerate(unique):
        mask = subjects == sid
        yt_subj[idx] = float(np.mean(y_true[mask]))
        yp_subj[idx] = float(np.mean(y_pred[mask]))
    return yt_subj, yp_subj, unique


def _draw_pred_vs_true_panel(
    ax,
    *,
    y_true_subj: np.ndarray,
    y_pred_subj: np.ndarray,
    metrics: dict,
) -> None:
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
        "Predicted vs true (per held-out subject)",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    metric_box = (
        f"r      = {metrics['pearson_r']:+.3f}\n"
        f"rho    = {metrics['spearman_rho']:+.3f}\n"
        f"R^2    = {metrics['r2']:+.3f}\n"
        f"MAE    = {metrics['mae']:.3f}"
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
    ax.legend(
        loc="lower right",
        fontsize=7.6,
        frameon=False,
    )


def _draw_residual_panel(
    ax,
    *,
    y_true_subj: np.ndarray,
    y_pred_subj: np.ndarray,
    subject_ids: Sequence[str],
) -> None:
    residuals = y_pred_subj - y_true_subj
    order = np.argsort(np.abs(residuals))
    residuals = residuals[order]
    labels = [str(subject_ids[i]) for i in order]

    positions = np.arange(len(residuals))
    colors = np.where(residuals >= 0.0, EEGDASH_ORANGE, EEGDASH_BLUE)

    ax.bar(
        positions,
        residuals,
        width=0.70,
        color=colors,
        edgecolor=EEGDASH_INK,
        linewidth=0.6,
        zorder=3,
    )
    ax.axhline(
        0.0,
        color=EEGDASH_MUTED,
        linestyle="-",
        linewidth=0.7,
        zorder=2,
    )

    ax.set_xticks(positions)
    # Compact tick labels: keep only the subject suffix to avoid crowding.
    short_labels = [s.replace("sub-", "") for s in labels]
    ax.set_xticklabels(
        short_labels,
        fontsize=7.6,
        rotation=0,
        color=EEGDASH_INK,
    )
    ax.set_xlabel("subject (sorted by |residual|)")
    ax.set_ylabel("predicted - true")
    ax.text(
        0.0,
        1.06,
        "Per-subject signed residual",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    # Two-entry legend so the orange/blue convention reads without a key
    # in the figure caption.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=EEGDASH_ORANGE, label="over-predict"),
        plt.Rectangle((0, 0), 1, 1, color=EEGDASH_BLUE, label="under-predict"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7.6,
        frameon=False,
        ncols=2,
        handlelength=1.2,
        columnspacing=0.9,
    )


def _draw_error_distribution_panel(
    ax,
    *,
    y_true_window: np.ndarray,
    y_pred_window: np.ndarray,
) -> None:
    errors = (y_pred_window - y_true_window).astype(float)
    mu = float(np.mean(errors))
    sigma = float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0

    # Histogram in density mode so the Gaussian overlay has a comparable
    # vertical scale; bin count via Freedman-Diaconis-like heuristic.
    n_bins = int(min(24, max(8, np.sqrt(errors.size))))
    counts, bins, patches = ax.hist(
        errors,
        bins=n_bins,
        density=True,
        color=EEGDASH_BLUE,
        edgecolor=EEGDASH_INK,
        linewidth=0.5,
        alpha=0.78,
        zorder=2,
    )

    # Overlay a Gaussian density evaluated on a smooth grid; std=0 is a
    # corner case (every error identical) where the overlay collapses.
    if sigma > 1e-9:
        grid = np.linspace(bins[0], bins[-1], 200)
        ax.plot(
            grid,
            norm.pdf(grid, loc=mu, scale=sigma),
            color=EEGDASH_BLUE_DARK,
            linewidth=1.4,
            zorder=4,
            label=f"N(mu={mu:+.2f}, sigma={sigma:.2f})",
        )

    # Annotate bias (mean) and zero-error reference. The mean line is the
    # one a reader is most likely to misread as zero, so it gets a label.
    ax.axvline(
        0.0,
        color=EEGDASH_MUTED,
        linestyle="--",
        linewidth=0.8,
        zorder=3,
    )
    ax.axvline(
        mu,
        color=EEGDASH_ORANGE,
        linestyle="-",
        linewidth=1.2,
        zorder=4,
        label=f"bias = {mu:+.3f}",
    )

    ax.set_xlabel("predicted - true (window-level)")
    ax.set_ylabel("density")
    ax.text(
        0.0,
        1.06,
        "Prediction-error distribution",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.legend(
        loc="upper right",
        fontsize=7.6,
        frameon=False,
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.size < 2:
        return {
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "r2": np.nan,
            "mae": np.nan,
        }

    pearson_r = float(pearsonr(y_true, y_pred).statistic)
    spearman_rho = float(spearmanr(y_true, y_pred).statistic)
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "r2": r2,
        "mae": mae,
    }


def draw_pfactor_figure(
    *,
    y_true_subj: np.ndarray,
    y_pred_subj: np.ndarray,
    subject_ids: Sequence[str],
    plot_id: str = "plot_72",
) -> plt.Figure:
    """Render the three-panel p-factor regression diagnostic.

    Parameters
    ----------
    y_true_subj : numpy.ndarray
        ``(n_held_out_windows,)`` true p-factor values pooled across every
        cross-subject fold (held-out side only).
    y_pred_subj : numpy.ndarray
        ``(n_held_out_windows,)`` predicted p-factor values, same length
        as ``y_true_subj``.
    subject_ids : sequence of str
        ``(n_held_out_windows,)`` subject id per row. The function
        aggregates to one point per subject for panels 1 and 2 and keeps
        the window-level distribution for panel 3.
    plot_id : str
        Tutorial id forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure

    """
    y_true_window = np.asarray(y_true_subj, dtype=float).ravel()
    y_pred_window = np.asarray(y_pred_subj, dtype=float).ravel()
    if y_true_window.shape != y_pred_window.shape:
        raise ValueError(
            f"y_true_subj/y_pred_subj shape mismatch: "
            f"{y_true_window.shape} vs {y_pred_window.shape}"
        )
    if len(subject_ids) != y_true_window.size:
        raise ValueError(
            f"subject_ids length {len(subject_ids)} != "
            f"prediction length {y_true_window.size}"
        )

    yt_subj, yp_subj, unique_subjects = _aggregate_per_subject(
        y_true=y_true_window,
        y_pred=y_pred_window,
        subject_ids=subject_ids,
    )
    metrics = _compute_metrics(yt_subj, yp_subj)
    n_features_hint = ""  # subtitle is filled by the tutorial caller via fig.text

    fig = plt.figure(figsize=(14.0, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.10, 1.10, 1.10),
        wspace=0.40,
    )
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_resid = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[0, 2])

    _draw_pred_vs_true_panel(
        ax_scatter,
        y_true_subj=yt_subj,
        y_pred_subj=yp_subj,
        metrics=metrics,
    )
    _draw_residual_panel(
        ax_resid,
        y_true_subj=yt_subj,
        y_pred_subj=yp_subj,
        subject_ids=unique_subjects,
    )
    _draw_error_distribution_panel(
        ax_dist,
        y_true_window=y_true_window,
        y_pred_window=y_pred_window,
    )

    n_subjects = len(unique_subjects)
    n_windows = int(y_true_window.size)
    subtitle = (
        f"EEG2025 Challenge 2 | n_subjects={n_subjects} | "
        f"n_windows={n_windows} | r={metrics['pearson_r']:+.3f} | "
        f"R^2={metrics['r2']:+.3f} | MAE={metrics['mae']:.3f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=None,
        citation="HBN release R5 (Alexander et al. 2017) via NEMAR",
        extra=n_features_hint or None,
    )
    style_figure(
        fig,
        title="Does the model predict p-factor on never-seen subjects?",
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )

    # ``style_figure`` reserves ``bottom=0.18`` and ``top=0.84`` for the
    # title/subtitle band; with three panels carrying their own titles
    # plus an x-tick row a hair more breathing room reads cleaner.
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.06, right=0.97)
    # Persist computed metrics on the figure so the tutorial can pull
    # them for its result-line print without recomputing.
    fig._eegdash_pfactor_metrics = {
        **metrics,
        "n_subjects": n_subjects,
        "n_windows": n_windows,
    }
    return fig


__all__ = ["draw_pfactor_figure"]
