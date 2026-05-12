"""Drawing helpers for ``project_pfactor_deep`` (deep-learning case study)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
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


def _curve_band(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(curve, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected (n_seeds, n_epochs); got shape {arr.shape}")
    mean = arr.mean(axis=0)
    n = max(arr.shape[0], 1)
    sem = arr.std(axis=0, ddof=0) / np.sqrt(n)
    return mean, sem


def _draw_training_curves_panel(
    ax,
    *,
    epochs: Sequence[int],
    train_loss: np.ndarray,
    val_loss: np.ndarray,
    val_r: np.ndarray,
) -> None:
    ep = np.asarray(epochs, dtype=float)
    train_mu, train_se = _curve_band(train_loss)
    val_mu, val_se = _curve_band(val_loss)
    r_mu, r_se = _curve_band(val_r)

    ax.fill_between(
        ep,
        train_mu - train_se,
        train_mu + train_se,
        color=EEGDASH_BLUE,
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        ep,
        train_mu,
        color=EEGDASH_BLUE,
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        markerfacecolor=EEGDASH_BLUE,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="train MSE",
        zorder=3,
    )
    ax.fill_between(
        ep,
        val_mu - val_se,
        val_mu + val_se,
        color=EEGDASH_ORANGE,
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        ep,
        val_mu,
        color=EEGDASH_ORANGE,
        linewidth=2.0,
        marker="s",
        markersize=4.5,
        markerfacecolor=EEGDASH_ORANGE,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="val MSE",
        zorder=3,
    )

    ax.set_xlabel("training epoch")
    ax.set_ylabel("MSE loss")
    ax.set_xticks(ep.astype(int))
    ax.set_xlim(ep.min() - 0.4, ep.max() + 0.4)
    # MSE floor at zero; cap with a tiny gutter so the band reads cleanly.
    lo = float(min(train_mu.min() - train_se.max(), val_mu.min() - val_se.max()))
    hi = float(max(train_mu.max() + train_se.max(), val_mu.max() + val_se.max()))
    pad = 0.08 * max(hi - lo, 1e-3)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)

    # Twin axis carries Pearson r on the right so the curve panel doubles
    # as a learning-progress diagnostic without a second figure.
    ax_r = ax.twinx()
    ax_r.fill_between(
        ep,
        r_mu - r_se,
        r_mu + r_se,
        color=EEGDASH_BLUE_DARK,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    ax_r.plot(
        ep,
        r_mu,
        color=EEGDASH_BLUE_DARK,
        linewidth=1.8,
        marker="D",
        markersize=4.0,
        markerfacecolor=EEGDASH_BLUE_DARK,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="val r",
        zorder=3,
    )
    ax_r.axhline(0.0, color=EEGDASH_MUTED, linestyle=":", linewidth=0.7, zorder=1)
    ax_r.set_ylabel("validation Pearson r", color=EEGDASH_BLUE_DARK)
    ax_r.tick_params(axis="y", colors=EEGDASH_BLUE_DARK)
    r_lo = float(min(0.0, r_mu.min() - r_se.max()))
    r_hi = float(max(0.05, r_mu.max() + r_se.max()))
    r_pad = 0.10 * max(r_hi - r_lo, 1e-3)
    ax_r.set_ylim(r_lo - r_pad, r_hi + r_pad)
    ax_r.grid(False)

    ax.text(
        0.0,
        1.06,
        "Training curves (mean +/- SE across seeds)",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = ax_r.get_legend_handles_labels()
    ax.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="upper right",
        fontsize=7.6,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
        handletextpad=0.4,
    )


def _aggregate_per_subject(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if subject_ids is None or len(subject_ids) == 0:
        # No subject map -> treat each entry as its own subject.
        ids = [f"s{i:02d}" for i in range(y_true.size)]
        return y_true.copy(), y_pred.copy(), ids

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
    if y_true.size < 2:
        return {"pearson_r": float("nan"), "r2": float("nan"), "mae": float("nan")}
    return {
        "pearson_r": float(pearsonr(y_true, y_pred).statistic),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
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
        facecolor=EEGDASH_ORANGE,
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

    box = (
        f"r    = {metrics['pearson_r']:+.3f}\n"
        f"R^2  = {metrics['r2']:+.3f}\n"
        f"MAE  = {metrics['mae']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        box,
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


def _draw_saliency_panel(
    ax,
    *,
    saliency_map: np.ndarray,
    channel_names: Sequence[str],
    sfreq: float | None = None,
) -> None:
    sal = np.asarray(saliency_map, dtype=float)
    if sal.ndim != 2:
        raise ValueError(f"saliency_map must be 2D (channels, times); got {sal.shape}")
    n_chans, n_times = sal.shape

    # Normalise to [0, 1] so colour maps onto a comparable scale across runs.
    smin = float(sal.min())
    smax = float(sal.max())
    denom = max(smax - smin, 1e-12)
    sal_norm = (sal - smin) / denom

    # x-axis units: seconds when sfreq known, otherwise sample index.
    if sfreq is not None and sfreq > 0:
        extent = (0.0, n_times / float(sfreq), n_chans - 0.5, -0.5)
        x_label = "time within window (s)"
    else:
        extent = (0.0, float(n_times), n_chans - 0.5, -0.5)
        x_label = "sample within window"

    im = ax.imshow(
        sal_norm,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
        zorder=2,
    )
    ax.set_yticks(np.arange(n_chans))
    ax.set_yticklabels(list(channel_names), fontsize=8.0, color=EEGDASH_INK)
    ax.set_xlabel(x_label)
    ax.set_ylabel("channel")
    ax.text(
        0.0,
        1.06,
        "Saliency: where the model reads the signal",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("|grad| (normalised)", fontsize=8.4, color=EEGDASH_INK)
    cbar.ax.tick_params(labelsize=7.6, colors=EEGDASH_INK)

    # Mark the peak channel-time cell with a thin frame so the eye lands
    # on the strongest read without recomputing the argmax.
    peak_ch, peak_t = np.unravel_index(int(np.argmax(sal_norm)), sal_norm.shape)
    if sfreq is not None and sfreq > 0:
        peak_x = float(peak_t) / float(sfreq)
        x_width = 1.0 / float(sfreq)
    else:
        peak_x = float(peak_t)
        x_width = 1.0
    ax.add_patch(
        plt.Rectangle(
            (peak_x - 0.5 * x_width, peak_ch - 0.5),
            x_width,
            1.0,
            fill=False,
            edgecolor=EEGDASH_ORANGE,
            linewidth=1.2,
            zorder=4,
        )
    )


def draw_pfactor_deep_figure(
    *,
    train_curves: dict,
    y_true_subj: np.ndarray,
    y_pred_subj: np.ndarray,
    saliency_map: np.ndarray,
    channel_names: Sequence[str],
    subject_ids: Sequence[str] | None = None,
    sfreq: float | None = None,
    n_train_subjects: int | None = None,
    n_val_subjects: int | None = None,
    plot_id: str = "project_pfactor_deep",
) -> plt.Figure:
    """Render the three-panel deep-learning p-factor regression figure.

    Parameters
    ----------
    train_curves : dict
        Dict with keys ``"epochs"``, ``"train_loss"``, ``"val_loss"``,
        ``"val_r"``. Each value is shape ``(n_seeds, n_epochs)`` for the
        curves and a 1-D ``(n_epochs,)`` vector for ``epochs``.
    y_true_subj, y_pred_subj : numpy.ndarray
        Window-level (or already subject-aggregated) true and predicted
        p-factor scores on the held-out side.
    saliency_map : numpy.ndarray
        ``(n_channels, n_times)`` gradient-magnitude map averaged across
        high-confidence test windows.
    channel_names : sequence of str
        Channel labels for the saliency y-axis (length == n_channels).
    subject_ids : sequence of str, optional
        Per-row subject id; when given, panel 2 aggregates to one point
        per subject. When ``None`` the inputs are treated as one row per
        subject already.
    sfreq : float, optional
        Sampling rate of the saliency-map x-axis (seconds when given).
    n_train_subjects, n_val_subjects : int, optional
        Live runtime values for the subtitle.
    plot_id : str, default ``"project_pfactor_deep"``
        Plot id forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure

    """
    yt = np.asarray(y_true_subj, dtype=float).ravel()
    yp = np.asarray(y_pred_subj, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true_subj/y_pred_subj shape mismatch: {yt.shape} vs {yp.shape}"
        )

    yt_subj, yp_subj, unique_subjects = _aggregate_per_subject(
        y_true=yt,
        y_pred=yp,
        subject_ids=subject_ids,
    )
    metrics = _compute_metrics(yt_subj, yp_subj)

    epochs = np.asarray(train_curves["epochs"]).ravel()
    train_loss = np.asarray(train_curves["train_loss"], dtype=float)
    val_loss = np.asarray(train_curves["val_loss"], dtype=float)
    val_r = np.asarray(train_curves["val_r"], dtype=float)

    fig = plt.figure(figsize=(14.4, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.20, 1.00, 1.20),
        wspace=0.46,
    )
    ax_curves = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_saliency = fig.add_subplot(gs[0, 2])

    _draw_training_curves_panel(
        ax_curves,
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        val_r=val_r,
    )
    _draw_pred_vs_true_panel(
        ax_scatter,
        y_true_subj=yt_subj,
        y_pred_subj=yp_subj,
        metrics=metrics,
    )
    _draw_saliency_panel(
        ax_saliency,
        saliency_map=saliency_map,
        channel_names=channel_names,
        sfreq=sfreq,
    )

    n_epochs = int(epochs.size)
    best_val_r_per_seed = (
        val_r.reshape(-1, n_epochs).max(axis=1) if val_r.size else np.array([0.0])
    )
    best_val_r = (
        float(best_val_r_per_seed.mean()) if best_val_r_per_seed.size else float("nan")
    )
    n_train = int(n_train_subjects) if n_train_subjects is not None else -1
    n_val = int(n_val_subjects) if n_val_subjects is not None else len(unique_subjects)
    subtitle = (
        f"n_train_subjects={n_train} | n_val_subjects={n_val} | "
        f"n_epochs={n_epochs} | best_val_r={best_val_r:+.3f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        citation="HBN release :cite:`alexander2017hbn` via NEMAR | seeded across runs",
    )
    style_figure(
        fig,
        title="Can a Braindecode model predict p-factor on never-seen subjects?",
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )

    fig.subplots_adjust(top=0.74, bottom=0.18, left=0.06, right=0.97)
    fig._eegdash_pfactor_deep_metrics = {
        **metrics,
        "best_val_r": best_val_r,
        "n_subjects": len(unique_subjects),
        "n_epochs": n_epochs,
    }
    return fig


__all__ = ["draw_pfactor_deep_figure"]
