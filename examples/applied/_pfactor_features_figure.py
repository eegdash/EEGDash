"""Drawing helpers for the ``project_pfactor_features`` applied study."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_PURPLE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# Feature-family palette. Band-power features carry the EEGDash blue, the
# brand colour the rest of the gallery uses for spectral views; connectivity
# features take orange to flag cross-channel coupling; entropy features
# take purple, the EEGDash palette's complexity slot.
_FAMILY_COLORS = {
    "band_power": EEGDASH_BLUE,
    "connectivity": EEGDASH_ORANGE,
    "entropy": EEGDASH_PURPLE,
}
_FAMILY_LABELS = {
    "band_power": "band-power",
    "connectivity": "connectivity",
    "entropy": "entropy",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_feature(name: str) -> str:
    n = name.lower()
    if n.startswith(("conn_", "coh_", "plv_", "imcoh_")):
        return "connectivity"
    if n.startswith(("ent_",)) or "entropy" in n:
        return "entropy"
    return "band_power"


def _aggregate_per_subject(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    subjects = np.asarray(list(subject_ids))
    unique = list(dict.fromkeys(subjects.tolist()))  # first-seen order
    yt_subj = np.empty(len(unique), dtype=float)
    yp_subj = np.empty(len(unique), dtype=float)
    for idx, sid in enumerate(unique):
        mask = subjects == sid
        yt_subj[idx] = float(np.mean(y_true[mask]))
        yp_subj[idx] = float(np.mean(y_pred[mask]))
    return yt_subj, yp_subj, unique


def _style_axis(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(EEGDASH_GRID)
    ax.spines["bottom"].set_color(EEGDASH_GRID)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


def _draw_importance_panel(
    ax,
    *,
    feature_importances: np.ndarray,
    feature_names: Sequence[str],
    top_k: int,
) -> dict[str, int]:
    importances = np.asarray(feature_importances, dtype=float)
    abs_imp = np.abs(importances)
    order = np.argsort(abs_imp)[::-1][: int(top_k)]
    # Re-sort the selected slice ascending so the largest bar lands on top.
    order = order[np.argsort(abs_imp[order])]
    sel_imp = importances[order]
    sel_names = [str(feature_names[i]) for i in order]
    families = [_classify_feature(n) for n in sel_names]
    colors = [_FAMILY_COLORS.get(f, EEGDASH_MUTED) for f in families]

    y = np.arange(len(order))
    ax.barh(
        y,
        sel_imp,
        color=colors,
        edgecolor=EEGDASH_INK,
        linewidth=0.5,
        height=0.66,
        zorder=3,
    )
    # Inline value labels so the bars stay readable in print.
    span = float(np.max(np.abs(sel_imp))) if sel_imp.size else 1.0
    offset = 0.012 * max(span, 1e-9)
    for yi, vi in zip(y, sel_imp):
        ha = "left" if vi >= 0 else "right"
        ax.text(
            vi + (offset if vi >= 0 else -offset),
            yi,
            f"{vi:+.2f}",
            ha=ha,
            va="center",
            fontsize=7.6,
            family="monospace",
            color=EEGDASH_INK,
            zorder=4,
        )
    ax.axvline(0.0, color=EEGDASH_MUTED, linewidth=0.7, zorder=1.5)

    ax.set_yticks(y)
    ax.set_yticklabels(
        sel_names,
        fontsize=8.2,
        family="monospace",
        color=EEGDASH_INK,
    )
    ax.set_xlabel("importance (signed)", color=EEGDASH_INK, fontsize=9.0)
    ax.set_xlim(-span * 1.45, span * 1.45)
    ax.text(
        0.0,
        1.06,
        f"Top-{int(top_k)} feature importance",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.grid(True, axis="x", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)

    # Family legend, only the families actually present.
    present_families: list[str] = []
    for f in families:
        if f not in present_families:
            present_families.append(f)
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=_FAMILY_COLORS[f],
            label=_FAMILY_LABELS[f],
        )
        for f in present_families
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=7.6,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.9,
        ncols=len(handles),
    )

    return {
        "n_band_power": sum(f == "band_power" for f in families),
        "n_connectivity": sum(f == "connectivity" for f in families),
        "n_entropy": sum(f == "entropy" for f in families),
    }


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
    ax.set_xlabel("true p-factor", color=EEGDASH_INK, fontsize=9.0)
    ax.set_ylabel("predicted p-factor", color=EEGDASH_INK, fontsize=9.0)
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
    ax.grid(True, color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)


def _draw_error_distribution_panel(
    ax,
    *,
    y_true_window: np.ndarray,
    y_pred_window: np.ndarray,
) -> dict[str, float]:
    errors = (y_pred_window - y_true_window).astype(float)
    mu = float(np.mean(errors))
    sigma = float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0

    n_bins = int(min(28, max(10, np.sqrt(errors.size))))
    counts, bins, _ = ax.hist(
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
        grid = np.linspace(bins[0], bins[-1], 240)
        ax.plot(
            grid,
            norm.pdf(grid, loc=mu, scale=sigma),
            color=EEGDASH_BLUE_DARK,
            linewidth=1.4,
            zorder=4,
            label=f"N(mu={mu:+.2f}, sigma={sigma:.2f})",
        )
    ax.axvline(
        0.0,
        color=EEGDASH_MUTED,
        linestyle="--",
        linewidth=0.8,
        zorder=3,
        label="zero error",
    )
    ax.axvline(
        mu,
        color=EEGDASH_ORANGE,
        linestyle="-",
        linewidth=1.2,
        zorder=4,
        label=f"bias = {mu:+.3f}",
    )

    ax.set_xlabel("predicted - true (window-level)", color=EEGDASH_INK, fontsize=9.0)
    ax.set_ylabel("density", color=EEGDASH_INK, fontsize=9.0)
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
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)
    return {"bias": mu, "spread": sigma}


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.size < 2:
        return {"pearson_r": float("nan"), "r2": float("nan"), "mae": float("nan")}
    pearson_r = float(pearsonr(y_true, y_pred).statistic)
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"pearson_r": pearson_r, "r2": r2, "mae": mae}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def draw_pfactor_features_figure(
    *,
    feature_importances: np.ndarray,
    feature_names: Sequence[str],
    y_true_subj: np.ndarray,
    y_pred_subj: np.ndarray,
    subject_ids: Sequence[str] | None = None,
    top_k: int = 10,
    plot_id: str = "project_pfactor_features",
    citation: str = "HBN release R5 :cite:`alexander2017hbn` via NEMAR",
) -> plt.Figure:
    """Render the three-panel feature-based p-factor diagnostic.

    Parameters
    ----------
    feature_importances : numpy.ndarray
        ``(n_features,)`` per-feature importance scores. Either
        :attr:`sklearn.ensemble.RandomForestRegressor.feature_importances_`
        (non-negative) or coefficient magnitudes from
        :class:`sklearn.linear_model.Ridge`. Sign is preserved when
        present so the bars carry direction.
    feature_names : sequence of str
        ``(n_features,)`` column names matched to ``feature_importances``.
        Names control the family classification, see ``_classify_feature``.
    y_true_subj : numpy.ndarray
        ``(n_held_out_windows,)`` true p-factor values pooled across
        every cross-subject fold (held-out side only).
    y_pred_subj : numpy.ndarray
        ``(n_held_out_windows,)`` predicted p-factor values, same length
        as ``y_true_subj``.
    subject_ids : sequence of str, optional
        ``(n_held_out_windows,)`` subject id per row. When ``None`` the
        tutorial passes the pooled vectors as already aggregated; the
        helper then treats every row as its own subject so the scatter
        and the histogram coexist.
    top_k : int, optional
        Number of features in the importance panel (default 10).
    plot_id : str, optional
        Forwarded to :func:`add_provenance_footer`.
    citation : str, optional
        Short citation for the provenance footer.

    Returns
    -------
    matplotlib.figure.Figure

    """
    importances = np.asarray(feature_importances, dtype=float).ravel()
    names = list(feature_names)
    if importances.size != len(names):
        raise ValueError(
            f"feature_importances length {importances.size} != "
            f"feature_names length {len(names)}"
        )

    y_true_window = np.asarray(y_true_subj, dtype=float).ravel()
    y_pred_window = np.asarray(y_pred_subj, dtype=float).ravel()
    if y_true_window.shape != y_pred_window.shape:
        raise ValueError(
            f"y_true_subj/y_pred_subj shape mismatch: "
            f"{y_true_window.shape} vs {y_pred_window.shape}"
        )

    if subject_ids is None:
        # Treat every row as its own subject so panels 1+2 still render.
        subject_ids = [f"row{i:04d}" for i in range(y_true_window.size)]
    elif len(subject_ids) != y_true_window.size:
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

    fig = plt.figure(figsize=(15.4, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.40, 1.00, 1.10),
        wspace=0.48,
    )
    ax_imp = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[0, 2])

    family_counts = _draw_importance_panel(
        ax_imp,
        feature_importances=importances,
        feature_names=names,
        top_k=int(top_k),
    )
    _draw_pred_vs_true_panel(
        ax_scatter,
        y_true_subj=yt_subj,
        y_pred_subj=yp_subj,
        metrics=metrics,
    )
    bias_spread = _draw_error_distribution_panel(
        ax_dist,
        y_true_window=y_true_window,
        y_pred_window=y_pred_window,
    )

    n_subjects = len(unique_subjects)
    n_features = importances.size
    subtitle = (
        f"EEG2025r5 p-factor | n_subjects={n_subjects} | "
        f"n_features={n_features} | r={metrics['pearson_r']:+.3f} | "
        f"R^2={metrics['r2']:+.3f} | MAE={metrics['mae']:.3f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=None,
        citation=citation,
        extra=(
            f"top-{int(top_k)} families: band-power={family_counts['n_band_power']} "
            f"| connectivity={family_counts['n_connectivity']} "
            f"| entropy={family_counts['n_entropy']}"
        ),
    )
    style_figure(
        fig,
        title="Which features predict p-factor on never-seen subjects?",
        subtitle=subtitle,
        source=source,
        grid_axis="none",
    )
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.11, right=0.97)

    fig._eegdash_pfactor_features_metrics = {  # type: ignore[attr-defined]
        **metrics,
        "n_subjects": n_subjects,
        "n_features": n_features,
        "bias": bias_spread["bias"],
        "spread": bias_spread["spread"],
        "family_counts": family_counts,
    }
    return fig


__all__ = ["draw_pfactor_features_figure"]
