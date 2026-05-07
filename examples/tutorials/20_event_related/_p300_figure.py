"""Drawing helpers for the ``plot_20`` visual-P300 oddball plate.

Sibling module to ``plot_20_visual_p300_oddball.py``. The leading
underscore tells sphinx-gallery to skip this file when building the
gallery, so the rendering plumbing stays out of the rendered tutorial;
the tutorial imports the public ``draw_p300_figure`` entry point.

The figure is a 2x2 plate that walks ERP -> topography -> decoder ->
error pattern:

1. *ERP at Pz* (top-left): target vs standard waveforms with shaded SE
   bands and the P300 search window highlighted.
2. *Topomap of the difference wave* (top-right; target - standard) at
   the peak latency, on the cleaned :class:`mne.Info`.
3. *Per-fold cross-subject accuracy* (bottom-left) of a logistic
   regression decoder trained on the centro-parietal P300
   mean-amplitude feature, with the chance level drawn in.
4. *Pooled LOSO confusion matrix* (bottom-right) rendered with
   :class:`sklearn.metrics.ConfusionMatrixDisplay`, normalised by row
   so the off-diagonals read as recall errors per true class.

The 4th panel is optional: if pooled ``y_true``/``y_pred`` are not
supplied, the helper falls back to the original 1x3 layout.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from eegdash.viz import (
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
    """Apply the EEGDash spine/tick treatment to a single axes."""
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(EEGDASH_GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


def _draw_erp_panel(
    ax,
    *,
    times_ms: np.ndarray,
    target_mean: np.ndarray,
    standard_mean: np.ndarray,
    target_se: np.ndarray,
    standard_se: np.ndarray,
    p300_window_ms: tuple[float, float],
    peak_time_ms: float,
    peak_amp_uv: float,
    channel_label: str,
    n_targets: int,
    n_standards: int,
) -> None:
    """ERP butterfly at the centro-parietal anchor channel."""
    p300_lo, p300_hi = p300_window_ms

    # Shaded P300 search window first so it sits behind the curves.
    ax.axvspan(
        p300_lo,
        p300_hi,
        color=EEGDASH_ORANGE,
        alpha=0.12,
        zorder=0,
    )
    # Vertical guides: stimulus onset and peak.
    ax.axvline(
        0.0, color=EEGDASH_INK, linewidth=0.8, linestyle="--", alpha=0.7, zorder=1
    )
    ax.axhline(0.0, color=EEGDASH_GRID, linewidth=0.6, alpha=0.7, zorder=1)
    ax.axvline(
        peak_time_ms,
        color=EEGDASH_ORANGE,
        linewidth=0.9,
        linestyle=":",
        alpha=0.85,
        zorder=2,
    )

    ax.fill_between(
        times_ms,
        standard_mean - standard_se,
        standard_mean + standard_se,
        color=EEGDASH_BLUE,
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    ax.fill_between(
        times_ms,
        target_mean - target_se,
        target_mean + target_se,
        color=EEGDASH_ORANGE,
        alpha=0.20,
        linewidth=0,
        zorder=3,
    )
    ax.plot(
        times_ms,
        standard_mean,
        color=EEGDASH_BLUE,
        linewidth=1.6,
        label=f"standard (n={n_standards})",
        zorder=4,
    )
    ax.plot(
        times_ms,
        target_mean,
        color=EEGDASH_ORANGE,
        linewidth=1.8,
        label=f"target (n={n_targets})",
        zorder=5,
    )

    # Window label inside the orange band, anchored just above center
    # so it does not collide with the peak pill in the lower-right.
    ax.text(
        (p300_lo + p300_hi) / 2,
        0.93,
        f"P300 search {int(p300_lo)}-{int(p300_hi)} ms",
        transform=ax.get_xaxis_transform(),
        color=EEGDASH_ORANGE,
        fontsize=7.8,
        ha="center",
        va="top",
        alpha=0.95,
    )

    # Monospace pill annotating the peak. Placed in the lower-right of
    # the panel where the post-peak ERP has settled, so it never collides
    # with the target curve or the legend.
    ax.text(
        0.985,
        0.04,
        f"peak {peak_amp_uv:+.2f} uV @ {int(round(peak_time_ms))} ms",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        family="monospace",
        color=EEGDASH_INK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": EEGDASH_ORANGE,
            "linewidth": 0.6,
        },
        zorder=6,
    )

    ax.set_xlim(times_ms[0], times_ms[-1])
    ax.set_xlabel("time relative to stimulus (ms)", color=EEGDASH_INK)
    ax.set_ylabel(f"amplitude at {channel_label} (uV)", color=EEGDASH_INK)
    ax.set_title(
        f"ERP at {channel_label}: target vs standard",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)


def _draw_topomap_panel(
    fig,
    ax,
    *,
    diff_at_peak: np.ndarray,
    info: mne.Info,
    peak_time_ms: float,
) -> None:
    """Scalp topomap of (target - standard) at the P300 peak."""
    vmax = float(np.max(np.abs(diff_at_peak)))
    vmax = max(vmax, 1e-9)

    im, _ = mne.viz.plot_topomap(
        diff_at_peak,
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
        f"target - standard at {int(round(peak_time_ms))} ms",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )

    # Side colorbar attached to the same axes via an inset.
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.046,
        pad=0.04,
        shrink=0.78,
    )
    cbar.set_label("uV (target - standard)", color=EEGDASH_INK, fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5, colors=EEGDASH_INK, length=0)
    cbar.outline.set_visible(False)


def _draw_accuracy_panel(
    ax,
    *,
    fold_subjects: Sequence[str],
    fold_accuracies: Sequence[float],
    chance_level: float,
) -> None:
    """Per-fold leave-one-subject-out accuracy bars."""
    accs = np.asarray(list(fold_accuracies), dtype=float)
    labels = [f"held-out\nsub-{s}" for s in fold_subjects]
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

    # Chance line. Annotation pinned to the upper-right of the panel.
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
        f"chance = {float(chance_level):.2f}",
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

    # Mean line with annotation tucked to the upper-left of the panel.
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
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("balanced accuracy", color=EEGDASH_INK)
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, str],
    balanced_acc: float,
) -> None:
    """Pooled LOSO confusion matrix rendered via sklearn."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_test = int(y_true.shape[0])

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

    # sklearn writes default axis labels and tick label colors that
    # collide with our identity. Re-style after the fact.
    ax.set_xlabel("predicted label", color=EEGDASH_INK, fontsize=9)
    ax.set_ylabel("true label", color=EEGDASH_INK, fontsize=9)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Keep cells square so the matrix reads as a matrix, not a table.
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "LOSO confusion matrix (pooled)",
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        loc="left",
        pad=6,
    )

    # Raw counts annotation pinned just under the matrix axes. xycoords
    # in axes fraction so it tracks the panel during savefig.
    ax.text(
        0.5,
        -0.30,
        f"n_test_windows={n_test}, balanced_acc={float(balanced_acc):.3f}",
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


def draw_p300_figure(
    *,
    times_ms: np.ndarray,
    target_mean: np.ndarray,
    standard_mean: np.ndarray,
    target_se: np.ndarray,
    standard_se: np.ndarray,
    channel_label: str,
    n_targets: int,
    n_standards: int,
    n_channels: int,
    sfreq: float,
    p300_window_ms: tuple[float, float],
    peak_time_ms: float,
    peak_amp_uv: float,
    diff_at_peak: np.ndarray,
    topomap_info: mne.Info,
    fold_subjects: Sequence[str],
    fold_accuracies: Sequence[float],
    chance_level: float,
    dataset: str,
    subject: str,
    plot_id: str = "plot_20",
    y_true_pooled: np.ndarray | None = None,
    y_pred_pooled: np.ndarray | None = None,
    class_names: tuple[str, str] = ("standard", "target"),
):
    """Render the P300 plate.

    The default layout is 2x2 (ERP, topomap, accuracy bars, confusion
    matrix). When pooled ``y_true``/``y_pred`` are not supplied, the
    helper falls back to the original 1x3 layout (ERP, topomap, bars).

    Parameters
    ----------
    times_ms : ndarray, shape (n_times,)
        Epoch time axis in milliseconds, with 0 at stimulus onset.
    target_mean, standard_mean : ndarray, shape (n_times,)
        Per-condition ERP at the anchor channel (microvolts).
    target_se, standard_se : ndarray, shape (n_times,)
        Standard error of the per-condition ERP at the anchor channel.
    channel_label : str
        Label of the anchor channel (e.g., ``"Pz"``).
    n_targets, n_standards : int
        Trial counts that fed the per-condition ERP.
    n_channels : int
        Number of EEG channels carried into the topomap.
    sfreq : float
        Effective sampling rate (Hz) used in the subtitle.
    p300_window_ms : tuple of float
        Lower/upper bound of the P300 search window in milliseconds.
    peak_time_ms : float
        Latency of the target-minus-standard peak inside the search window.
    peak_amp_uv : float
        Amplitude of the target-minus-standard peak at the anchor channel.
    diff_at_peak : ndarray, shape (n_channels,)
        Per-channel target-minus-standard amplitude at ``peak_time_ms``.
    topomap_info : mne.Info
        :class:`mne.Info` describing the channels used in ``diff_at_peak``.
    fold_subjects : sequence of str
        Subject ids held out in each fold (one per bar).
    fold_accuracies : sequence of float
        Test accuracy per fold (one per bar).
    chance_level : float
        Majority-class baseline drawn as a dashed reference.
    dataset, subject : str
        Provenance strings for the subtitle and source line.
    plot_id : str
        Tutorial id (``"plot_20"``).
    y_true_pooled, y_pred_pooled : ndarray, optional
        Pooled LOSO labels and predictions. If both are provided, the
        figure adds a 4th panel with a normalised confusion matrix.
    class_names : tuple of str
        Display labels for the two classes (default
        ``("standard", "target")``); ``0`` maps to the first label.

    Returns
    -------
    matplotlib.figure.Figure
        The 2x2 (or 1x3 fallback) figure ready for ``plt.show()``.

    """
    have_cm = y_true_pooled is not None and y_pred_pooled is not None

    if have_cm:
        fig = plt.figure(figsize=(11.2, 8.4))
        gs = fig.add_gridspec(
            2,
            2,
            wspace=0.40,
            hspace=0.55,
            left=0.10,
            right=0.94,
            top=0.81,
            bottom=0.11,
            width_ratios=[1.15, 1.0],
            height_ratios=[1.0, 1.0],
        )
        ax_erp = fig.add_subplot(gs[0, 0])
        ax_topo = fig.add_subplot(gs[0, 1])
        ax_acc = fig.add_subplot(gs[1, 0])
        ax_cm = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(13.6, 5.2))
        gs = fig.add_gridspec(
            1,
            3,
            wspace=0.30,
            left=0.06,
            right=0.97,
            top=0.78,
            bottom=0.16,
            width_ratios=[1.25, 1.0, 1.0],
        )
        ax_erp = fig.add_subplot(gs[0, 0])
        ax_topo = fig.add_subplot(gs[0, 1])
        ax_acc = fig.add_subplot(gs[0, 2])
        ax_cm = None

    _draw_erp_panel(
        ax_erp,
        times_ms=np.asarray(times_ms),
        target_mean=np.asarray(target_mean),
        standard_mean=np.asarray(standard_mean),
        target_se=np.asarray(target_se),
        standard_se=np.asarray(standard_se),
        p300_window_ms=p300_window_ms,
        peak_time_ms=float(peak_time_ms),
        peak_amp_uv=float(peak_amp_uv),
        channel_label=channel_label,
        n_targets=int(n_targets),
        n_standards=int(n_standards),
    )

    _draw_topomap_panel(
        fig,
        ax_topo,
        diff_at_peak=np.asarray(diff_at_peak),
        info=topomap_info,
        peak_time_ms=float(peak_time_ms),
    )

    _draw_accuracy_panel(
        ax_acc,
        fold_subjects=list(fold_subjects),
        fold_accuracies=list(fold_accuracies),
        chance_level=float(chance_level),
    )

    mean_acc = float(np.mean(np.asarray(fold_accuracies)))
    if have_cm:
        # Pooled balanced-accuracy on the concatenated LOSO predictions.
        # Computed locally so the panel annotation stays self-consistent
        # with the matrix it sits next to.
        from sklearn.metrics import balanced_accuracy_score as _bas

        pooled_bal_acc = float(
            _bas(np.asarray(y_true_pooled), np.asarray(y_pred_pooled))
        )
        _draw_confusion_panel(
            ax_cm,
            y_true=np.asarray(y_true_pooled),
            y_pred=np.asarray(y_pred_pooled),
            class_names=class_names,
            balanced_acc=pooled_bal_acc,
        )

    subtitle = (
        f"{dataset} | n_targets={int(n_targets)}, "
        f"n_standards={int(n_standards)}, n_channels={int(n_channels)} | "
        f"sfreq {sfreq:.0f} Hz | "
        f"peak {peak_amp_uv:+.2f} uV @ {int(round(peak_time_ms))} ms | "
        f"3-fold LOSO balanced acc {mean_acc:.2f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=dataset,
        citation="Pernet et al. 2019 (EEG-BIDS)",
        extra="task=visualoddball; P300 search 300-450 ms",
    )
    style_figure(
        fig,
        title="The visual P300: target vs standard, ERP -> topography -> decoder -> errors",
        subtitle=subtitle,
        source=source,
        grid_axis="none",
    )
    return fig


__all__ = ["draw_p300_figure"]
