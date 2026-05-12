"""Drawing helpers for the ``plot_30`` eyes-open vs eyes-closed plate."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from eegdash.viz import (
    EEGDASH_AMBER,
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# ---------------------------------------------------------------------------
# Internal panel painters
# ---------------------------------------------------------------------------


def _style_axis(ax) -> None:
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(EEGDASH_GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


def _draw_psd_panel(
    ax,
    *,
    freqs: np.ndarray,
    psd_open: np.ndarray,
    psd_closed: np.ndarray,
    alpha_band: tuple[float, float],
    alpha_ratio: float,
    channel_label: str,
    n_open: int,
    n_closed: int,
) -> None:
    fmin = max(float(freqs[0]), 1.0)
    fmax = min(float(freqs[-1]), 40.0)
    band_lo, band_hi = alpha_band

    # Alpha band shading first, behind the curves.
    ax.axvspan(band_lo, band_hi, color=EEGDASH_AMBER, alpha=0.18, zorder=0)
    ax.plot(
        freqs,
        psd_open,
        color=EEGDASH_BLUE,
        linewidth=1.6,
        label=f"eyes open (n={n_open})",
        zorder=3,
    )
    ax.plot(
        freqs,
        psd_closed,
        color=EEGDASH_ORANGE,
        linewidth=1.8,
        label=f"eyes closed (n={n_closed})",
        zorder=4,
    )

    ax.set_yscale("log")
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("frequency (Hz)", color=EEGDASH_INK)
    ax.set_ylabel("PSD (uV$^2$/Hz)", color=EEGDASH_INK)
    ax.set_title(
        f"PSD at {channel_label}: eyes open vs eyes closed",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)

    # Band label tucked inside the amber shading at the top of the panel.
    ymin, ymax = ax.get_ylim()
    ax.text(
        (band_lo + band_hi) / 2,
        ymax * 0.85,
        f"alpha {int(band_lo)}-{int(band_hi)} Hz",
        color=EEGDASH_BLUE_DARK,
        fontsize=8.0,
        ha="center",
        va="top",
        alpha=0.95,
    )

    # Monospace ratio pill (closed / open). Pinned to the upper-right in
    # axes coordinates so it never collides with the legend in the
    # lower-left and is immune to the log y-scale.
    ax.text(
        0.985,
        0.965,
        f"closed / open alpha = {alpha_ratio:.2f}x",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        family="monospace",
        color=EEGDASH_INK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": EEGDASH_ORANGE,
            "linewidth": 0.6,
        },
    )


def _draw_topomap_panel(
    fig,
    ax,
    *,
    diff_log_power: np.ndarray,
    info: mne.Info,
    alpha_band: tuple[float, float],
) -> None:
    band_lo, band_hi = alpha_band
    vmax = float(np.max(np.abs(diff_log_power)))
    vmax = max(vmax, 1e-9)

    im, _ = mne.viz.plot_topomap(
        diff_log_power,
        info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=(-vmax, vmax),
        sensors=True,
        contours=4,
        outlines="head",
    )
    ax.set_title(
        f"alpha ({int(band_lo)}-{int(band_hi)} Hz) Δlog-power: closed - open",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.046,
        pad=0.04,
        shrink=0.78,
    )
    cbar.set_label("Δlog10 power (closed - open)", color=EEGDASH_INK, fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5, colors=EEGDASH_INK, length=0)
    cbar.outline.set_visible(False)


def _draw_accuracy_panel(
    ax,
    *,
    fold_subjects: Sequence[str],
    fold_accuracies: Sequence[float],
    chance_level: float,
) -> None:
    accs = np.asarray(list(fold_accuracies), dtype=float)
    labels = [f"held-out\nsub-{s[-4:]}" for s in fold_subjects]
    x = np.arange(len(accs))

    bars = ax.bar(
        x,
        accs,
        width=0.55,
        color=EEGDASH_INK,
        edgecolor=EEGDASH_BLUE_DARK,
        linewidth=0.6,
        zorder=3,
    )
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.014,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            family="monospace",
            color=EEGDASH_INK,
            zorder=4,
        )

    # Chance line; annotation pinned to the upper-right of the panel.
    ax.axhline(
        float(chance_level),
        color=EEGDASH_MUTED,
        linestyle="--",
        linewidth=0.9,
        zorder=1.5,
    )
    ax.text(
        0.99,
        0.97,
        f"chance = {float(chance_level):.2f}\n(majority class)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=EEGDASH_MUTED,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": EEGDASH_MUTED,
            "linewidth": 0.4,
            "alpha": 0.95,
        },
    )

    # Mean line with annotation tucked into the upper-left of the panel.
    mean = float(np.mean(accs))
    std = float(np.std(accs))
    ax.axhline(
        mean,
        color=EEGDASH_ORANGE,
        linewidth=1.2,
        linestyle="-",
        alpha=0.85,
        zorder=2,
    )
    ax.text(
        0.02,
        0.97,
        f"mean acc = {mean:.2f} +/- {std:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
        color=EEGDASH_ORANGE,
        zorder=4,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": EEGDASH_ORANGE,
            "linewidth": 0.4,
            "alpha": 0.95,
        },
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, color=EEGDASH_INK)
    ax.set_xlim(-0.6, len(accs) - 0.4)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test accuracy", color=EEGDASH_INK)
    ax.set_title(
        "Cross-subject decoding (leave-one-subject-out)",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)


def _draw_confusion_panel(
    ax,
    *,
    y_true_pooled: np.ndarray,
    y_pred_pooled: np.ndarray,
    class_names: tuple[str, str],
) -> None:
    y_true = np.asarray(y_true_pooled).astype(int)
    y_pred = np.asarray(y_pred_pooled).astype(int)

    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=list(class_names),
        cmap="Blues",
        colorbar=False,
        ax=ax,
        normalize="true",
        values_format=".2f",
    )

    # Tighten typography to match the rest of the plate. Sklearn writes
    # the cell text via Text artists; recolour and rescale them so the
    # readable values do not shrink against EEGDASH_INK headers.
    for txt in ax.texts:
        txt.set_fontsize(11)
        txt.set_fontfamily("monospace")
    ax.set_xlabel("predicted label", color=EEGDASH_INK, fontsize=9)
    ax.set_ylabel("true label", color=EEGDASH_INK, fontsize=9)
    ax.tick_params(
        axis="both", which="both", length=0, colors=EEGDASH_INK, labelsize=8.5
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "LOSO confusion matrix (pooled)",
        fontsize=10,
        color=EEGDASH_INK,
        loc="left",
        pad=6,
    )

    n_test = int(y_true.size)
    acc = float((y_true == y_pred).mean()) if n_test else 0.0
    ax.text(
        0.5,
        -0.18,
        f"n_test_windows={n_test}, accuracy={acc:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        family="monospace",
        color=EEGDASH_MUTED,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def draw_alpha_figure(
    *,
    freqs: np.ndarray,
    psd_open: np.ndarray,
    psd_closed: np.ndarray,
    alpha_topomap_data: np.ndarray,
    alpha_topomap_info: mne.Info,
    fold_subjects: Sequence[str],
    fold_accuracies: Sequence[float],
    alpha_ratio: float,
    chance_level: float,
    channel_label: str,
    n_open: int,
    n_closed: int,
    n_subjects: int,
    n_channels: int,
    sfreq: float,
    alpha_band: tuple[float, float],
    dataset: str,
    y_true_pooled: np.ndarray | None = None,
    y_pred_pooled: np.ndarray | None = None,
    class_names: tuple[str, str] = ("eyes open", "eyes closed"),
    plot_id: str = "plot_30",
):
    """Render the eyes-open vs eyes-closed plate.

    When ``y_true_pooled`` and ``y_pred_pooled`` are both provided the
    figure is laid out as a 2x2 plate (PSD, topomap, accuracy bars,
    confusion matrix). When either is ``None`` the helper falls back to
    the legacy 1x3 layout (PSD, topomap, accuracy bars).

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency axis (Hz) shared by both PSDs.
    psd_open, psd_closed : ndarray, shape (n_freqs,)
        Mean PSD at ``channel_label`` for each condition (uV^2 / Hz).
    alpha_topomap_data : ndarray, shape (n_channels,)
        Per-channel alpha-band log-power difference (closed - open).
    alpha_topomap_info : mne.Info
        :class:`mne.Info` describing the channels in ``alpha_topomap_data``,
        already carrying digitized positions.
    fold_subjects : sequence of str
        Subject ids held out in each fold of the LOSO split.
    fold_accuracies : sequence of float
        Per-fold test accuracy.
    alpha_ratio : float
        Mean ``closed / open`` alpha-power ratio at the anchor channel,
        printed in the PSD panel.
    chance_level : float
        Majority-class baseline drawn on the accuracy panel.
    channel_label : str
        Anchor channel used for the PSD (e.g., ``"E70 (Oz)"``).
    n_open, n_closed : int
        Window counts per condition.
    n_subjects : int
        Number of subjects feeding the LOSO split.
    n_channels : int
        Number of EEG channels carried into the topomap.
    sfreq : float
        Effective sampling rate (Hz).
    alpha_band : tuple of float
        Lower / upper alpha-band edges (Hz). Used for shading and labels.
    dataset : str
        OpenNeuro accession (e.g., ``"ds005514"``) for the provenance line.
    y_true_pooled, y_pred_pooled : ndarray or None
        Pooled true / predicted class indices across the full LOSO loop.
        When both are provided the 2x2 layout adds a confusion matrix.
    class_names : tuple of str
        Display labels for the two classes (in label-id order). Used as
        the axis tick labels of the confusion matrix.
    plot_id : str
        Tutorial id (default ``"plot_30"``).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure ready for ``plt.show()``.

    """
    show_confusion = y_true_pooled is not None and y_pred_pooled is not None

    if show_confusion:
        fig = plt.figure(figsize=(11.0, 7.8))
        gs = fig.add_gridspec(
            2,
            2,
            wspace=0.30,
            hspace=0.65,
            left=0.09,
            right=0.95,
            top=0.84,
            bottom=0.13,
            width_ratios=[1.30, 1.0],
            height_ratios=[1.0, 1.0],
        )
        ax_psd = fig.add_subplot(gs[0, 0])
        ax_topo = fig.add_subplot(gs[0, 1])
        ax_acc = fig.add_subplot(gs[1, 0])
        ax_cm = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(13.6, 5.2))
        gs = fig.add_gridspec(
            1,
            3,
            wspace=0.34,
            left=0.06,
            right=0.96,
            top=0.78,
            bottom=0.16,
            width_ratios=[1.25, 1.0, 1.0],
        )
        ax_psd = fig.add_subplot(gs[0, 0])
        ax_topo = fig.add_subplot(gs[0, 1])
        ax_acc = fig.add_subplot(gs[0, 2])
        ax_cm = None

    _draw_psd_panel(
        ax_psd,
        freqs=np.asarray(freqs),
        psd_open=np.asarray(psd_open),
        psd_closed=np.asarray(psd_closed),
        alpha_band=alpha_band,
        alpha_ratio=float(alpha_ratio),
        channel_label=channel_label,
        n_open=int(n_open),
        n_closed=int(n_closed),
    )

    _draw_topomap_panel(
        fig,
        ax_topo,
        diff_log_power=np.asarray(alpha_topomap_data),
        info=alpha_topomap_info,
        alpha_band=alpha_band,
    )

    _draw_accuracy_panel(
        ax_acc,
        fold_subjects=list(fold_subjects),
        fold_accuracies=list(fold_accuracies),
        chance_level=float(chance_level),
    )

    if show_confusion:
        _draw_confusion_panel(
            ax_cm,
            y_true_pooled=np.asarray(y_true_pooled),
            y_pred_pooled=np.asarray(y_pred_pooled),
            class_names=tuple(class_names),
        )

    mean_acc = float(np.mean(np.asarray(fold_accuracies)))
    band_lo, band_hi = alpha_band
    subtitle = (
        f"{dataset} | n_subjects={int(n_subjects)}, "
        f"n_open={int(n_open)}, n_closed={int(n_closed)}, "
        f"n_channels={int(n_channels)} | "
        f"sfreq {sfreq:.0f} Hz | "
        f"closed/open alpha = {alpha_ratio:.2f}x at {channel_label} | "
        f"LOSO acc {mean_acc:.2f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=dataset,
        citation="Alexander et al. 2017 (HBN)",
        extra=(
            f"task=RestingState; alpha {int(band_lo)}-{int(band_hi)} Hz; "
            "Berger 1929; Klimesch 2012"
        ),
    )
    style_figure(
        fig,
        title="Eyes closed releases posterior alpha: spectrum -> topography -> decoder -> errors",
        subtitle=subtitle,
        source=source,
        grid_axis="none",
    )
    return fig


__all__ = ["draw_alpha_figure"]
