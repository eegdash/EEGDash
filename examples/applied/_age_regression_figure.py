"""Drawing helpers for the ``project_age_regression`` diagnostic."""

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
    sex_or_fold: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str] | None]:
    subjects = np.asarray(list(subject_ids))
    unique = list(dict.fromkeys(subjects.tolist()))
    yt_subj = np.empty(len(unique), dtype=float)
    yp_subj = np.empty(len(unique), dtype=float)
    sex_subj: list[str] | None = None
    if sex_or_fold is not None:
        sex_arr = np.asarray(list(sex_or_fold))
        sex_subj = []
    for idx, sid in enumerate(unique):
        mask = subjects == sid
        yt_subj[idx] = float(np.mean(y_true[mask]))
        yp_subj[idx] = float(np.mean(y_pred[mask]))
        if sex_subj is not None:
            # Pick the first label seen for this subject.
            sex_subj.append(str(sex_arr[mask][0]))
    return yt_subj, yp_subj, unique, sex_subj


def _color_by_group(groups: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    unique = list(dict.fromkeys(groups))
    palette = [EEGDASH_BLUE, EEGDASH_ORANGE, EEGDASH_BLUE_DARK, EEGDASH_MUTED]
    mapping = {key: palette[i % len(palette)] for i, key in enumerate(unique)}
    return [mapping[g] for g in groups], mapping


def _draw_pred_vs_true_panel(
    ax,
    *,
    y_true_subj: np.ndarray,
    y_pred_subj: np.ndarray,
    metrics: dict,
    sex_or_fold_subj: list[str] | None,
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

    if sex_or_fold_subj is not None and len(sex_or_fold_subj) == len(y_true_subj):
        colors, mapping = _color_by_group(sex_or_fold_subj)
        for key, color in mapping.items():
            mask = np.array([g == key for g in sex_or_fold_subj])
            ax.scatter(
                y_true_subj[mask],
                y_pred_subj[mask],
                s=70,
                facecolor=color,
                edgecolor=EEGDASH_INK,
                linewidth=0.7,
                alpha=0.88,
                zorder=3,
                label=str(key),
            )
    else:
        ax.scatter(
            y_true_subj,
            y_pred_subj,
            s=70,
            facecolor=EEGDASH_BLUE,
            edgecolor=EEGDASH_INK,
            linewidth=0.7,
            alpha=0.88,
            zorder=3,
            label="held-out subject",
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("true age (years)")
    ax.set_ylabel("predicted age (years)")
    ax.text(
        0.0,
        1.06,
        "Predicted vs true age (per held-out subject)",
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
        f"MAE    = {metrics['mae']:.3f} yr"
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
    ax.axhline(0.0, color=EEGDASH_MUTED, linestyle="-", linewidth=0.7, zorder=2)

    ax.set_xticks(positions)
    # Keep the trailing chars of long HBN identifiers so each label stays unique.
    short_labels = []
    for s in labels:
        token = s.replace("sub-", "")
        if len(token) > 6:
            token = token[-6:]
        short_labels.append(token)
    ax.set_xticklabels(
        short_labels,
        fontsize=7.4,
        rotation=45,
        color=EEGDASH_INK,
        ha="right",
    )
    ax.set_xlabel("subject (sorted by |residual|)")
    ax.set_ylabel("predicted - true (years)")
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

    ax.axvline(0.0, color=EEGDASH_MUTED, linestyle="--", linewidth=0.8, zorder=3)
    ax.axvline(
        mu,
        color=EEGDASH_ORANGE,
        linestyle="-",
        linewidth=1.2,
        zorder=4,
        label=f"bias = {mu:+.3f} yr",
    )

    ax.set_xlabel("predicted - true (years)")
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
    ax.legend(loc="upper right", fontsize=7.6, frameon=False)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.size < 2:
        return {
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "r2": float("nan"),
            "mae": float("nan"),
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


def draw_age_regression_figure(
    *,
    y_true_subj: np.ndarray,
    y_pred_subj: np.ndarray,
    subject_ids: Sequence[str],
    sex_or_fold: Sequence[str] | None = None,
    plot_id: str = "project_age_regression",
) -> plt.Figure:
    """Render the three-panel age regression diagnostic.

    Parameters
    ----------
    y_true_subj : numpy.ndarray
        ``(n_held_out_windows,)`` true age values pooled across the
        held-out side of the subject-aware split.
    y_pred_subj : numpy.ndarray
        ``(n_held_out_windows,)`` predicted age values, same length as
        ``y_true_subj``.
    subject_ids : sequence of str
        ``(n_held_out_windows,)`` subject id per row. The function
        aggregates to one point per subject for panels 1 and 2 and keeps
        the window-level distribution for panel 3.
    sex_or_fold : sequence of str, optional
        Per-row categorical label (sex or fold). When supplied, panel 1
        colors points by category. When omitted, all points share the
        EEGDash blue.
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
    if sex_or_fold is not None and len(sex_or_fold) != y_true_window.size:
        raise ValueError(
            f"sex_or_fold length {len(sex_or_fold)} != "
            f"prediction length {y_true_window.size}"
        )

    yt_subj, yp_subj, unique_subjects, sex_subj = _aggregate_per_subject(
        y_true=y_true_window,
        y_pred=y_pred_window,
        subject_ids=subject_ids,
        sex_or_fold=sex_or_fold,
    )
    metrics = _compute_metrics(yt_subj, yp_subj)

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
        sex_or_fold_subj=sex_subj,
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
        f"HBN ds005505 | n_subjects={n_subjects} | n_windows={n_windows} | "
        f"r={metrics['pearson_r']:+.3f} | R^2={metrics['r2']:+.3f} | "
        f"MAE={metrics['mae']:.3f} yr"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id="ds005505",
        citation="HBN (Alexander et al. 2017) via NEMAR",
    )
    style_figure(
        fig,
        title="Does an EEG decoder predict age on held-out subjects?",
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )

    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.06, right=0.97)
    fig._eegdash_age_metrics = {
        **metrics,
        "n_subjects": n_subjects,
        "n_windows": n_windows,
    }
    return fig


__all__ = ["draw_age_regression_figure"]
