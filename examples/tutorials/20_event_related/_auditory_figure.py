"""Drawing helpers for the ``plot_21`` "auditory vs. visual P300" figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

import mne

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_PURPLE,
    EEGDASH_SURFACE,
    style_figure,
)
# ---------------------------------------------------------------------------
# Constants for the two ERP windows (auditory paradigm).
# ---------------------------------------------------------------------------
#
# MMN: 150-250 ms (Naatanen et al. 2007, doi:10.1016/j.clinph.2007.04.026).
# P300 (P3a/P3b family): 250-400 ms (Polich 2007,
# doi:10.1016/j.clinph.2007.04.019; Squires et al. 1975).
# Auditory P300 typically peaks ~30 ms earlier than the visual P300 and is
# largest over frontal-central sites rather than the parietal sites of the
# visual P300.

MMN_WINDOW_MS = (150.0, 250.0)
P300_WINDOW_MS = (250.0, 400.0)


# ---------------------------------------------------------------------------
# Panel 1: ERP at Cz with SE bands and labelled task windows.
# ---------------------------------------------------------------------------


def _draw_erp_panel(
    ax,
    *,
    times_ms: np.ndarray,
    erp_dev: np.ndarray,
    erp_std: np.ndarray,
    se_dev: np.ndarray,
    se_std: np.ndarray,
    cz_label: str,
    peak_time_ms: float,
    peak_uv: float,
    n_deviants: int,
    n_standards: int,
) -> None:
    # MMN window in muted purple, P300 window in soft orange. Distinct hues
    # so the two task intervals are visually separable on the same panel.
    ax.axvspan(
        MMN_WINDOW_MS[0],
        MMN_WINDOW_MS[1],
        color=EEGDASH_PURPLE,
        alpha=0.10,
        zorder=0,
        linewidth=0,
    )
    ax.axvspan(
        P300_WINDOW_MS[0],
        P300_WINDOW_MS[1],
        color=EEGDASH_ORANGE,
        alpha=0.10,
        zorder=0,
        linewidth=0,
    )

    # SE bands give the reader a feel for trial-to-trial variability.
    ax.fill_between(
        times_ms,
        erp_std - se_std,
        erp_std + se_std,
        color=EEGDASH_BLUE,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax.fill_between(
        times_ms,
        erp_dev - se_dev,
        erp_dev + se_dev,
        color=EEGDASH_ORANGE,
        alpha=0.20,
        linewidth=0,
        zorder=1,
    )

    ax.plot(
        times_ms,
        erp_std,
        color=EEGDASH_BLUE,
        linewidth=1.6,
        label=f"standard (n={n_standards})",
        zorder=3,
    )
    ax.plot(
        times_ms,
        erp_dev,
        color=EEGDASH_ORANGE,
        linewidth=1.9,
        label=f"deviant (n={n_deviants})",
        zorder=4,
    )

    ax.axvline(0.0, color=EEGDASH_MUTED, linewidth=0.8, linestyle="--", zorder=2)
    ax.axhline(0.0, color=EEGDASH_GRID, linewidth=0.6, alpha=0.7, zorder=1)

    # Label each window in its own color, just above the data. Pad the
    # y-axis upper margin generously so the banners sit clear of the ERP
    # peak even on noisy single-subject runs.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.45 * (ymax - ymin))
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    ax.text(
        np.mean(MMN_WINDOW_MS),
        ymax - 0.04 * yspan,
        f"MMN\n{int(MMN_WINDOW_MS[0])}-{int(MMN_WINDOW_MS[1])} ms",
        ha="center",
        va="top",
        fontsize=8.4,
        color=EEGDASH_PURPLE,
        fontweight="bold",
    )
    ax.text(
        np.mean(P300_WINDOW_MS),
        ymax - 0.04 * yspan,
        f"P300\n{int(P300_WINDOW_MS[0])}-{int(P300_WINDOW_MS[1])} ms",
        ha="center",
        va="top",
        fontsize=8.4,
        color=EEGDASH_ORANGE,
        fontweight="bold",
    )

    # Annotate the live peak with a monospace label so it lines up with
    # code. The label sits in the lower-right of the panel, with a thin
    # leader line drawn from the peak marker. This keeps it clear of both
    # the upper window banners and the plotted ERP traces themselves.
    ax.plot(
        peak_time_ms,
        peak_uv,
        marker="o",
        markersize=6.0,
        markerfacecolor="white",
        markeredgecolor=EEGDASH_INK,
        markeredgewidth=1.3,
        zorder=5,
    )
    label_x = times_ms[-1] - 5.0
    label_y = ymin + 0.18 * yspan
    ax.annotate(
        f"peak: {peak_time_ms:+.0f} ms, {peak_uv:+.1f} uV",
        xy=(peak_time_ms, peak_uv),
        xytext=(label_x, label_y),
        ha="right",
        va="center",
        fontsize=8.4,
        family="monospace",
        color=EEGDASH_INK,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": EEGDASH_INK,
            "linewidth": 0.6,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": EEGDASH_INK,
            "linewidth": 0.6,
            "alpha": 0.55,
            "shrinkA": 0,
            "shrinkB": 4,
        },
        zorder=6,
    )

    ax.set_xlabel("time (ms)", color=EEGDASH_INK)
    ax.set_ylabel("amplitude (uV)", color=EEGDASH_INK)
    ax.set_title(
        f"ERP at {cz_label}: deviant vs. standard",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=8.5,
    )
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(EEGDASH_GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.20, linewidth=0.5)


# ---------------------------------------------------------------------------
# Panel 2: scalp topomap of the difference wave at the P300 peak.
# ---------------------------------------------------------------------------


def _draw_topomap_panel(
    ax,
    *,
    diff_uv_at_peak: np.ndarray,
    info: mne.Info,
    peak_time_ms: float,
    fig: plt.Figure,
) -> None:
    # Symmetric color scale around zero so positive (red) and negative
    # (blue) poles read at a glance.
    vlim = float(np.percentile(np.abs(diff_uv_at_peak), 98))
    vlim = max(vlim, 1e-3)
    im, _ = mne.viz.plot_topomap(
        diff_uv_at_peak,
        info,
        axes=ax,
        cmap="RdBu_r",
        vlim=(-vlim, vlim),
        contours=4,
        outlines="head",
        sensors=True,
        show=False,
    )
    ax.set_title(
        f"deviant - standard at {peak_time_ms:+.0f} ms",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("uV (diff)", color=EEGDASH_INK, fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5, colors=EEGDASH_INK)


# ---------------------------------------------------------------------------
# Panel 3: auditory vs. visual P300 comparison card.
# ---------------------------------------------------------------------------


def _draw_comparison_panel(
    ax,
    *,
    peak_time_ms: float,
    peak_channel: str,
    peak_uv: float,
    n_deviants: int,
    n_standards: int,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Header strip.
    ax.text(
        0.0,
        0.985,
        "auditory vs. visual P300",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.925,
        "same paradigm, different brain answer",
        ha="left",
        va="top",
        fontsize=8.7,
        style="italic",
        color=EEGDASH_MUTED,
        transform=ax.transAxes,
    )

    # One rounded card holds all rows. Three columns inside.
    card_left = 0.0
    card_w = 1.0
    card_top = 0.86
    card_h = 0.72
    ax.add_patch(
        FancyBboxPatch(
            (card_left, card_top - card_h),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.1,
            edgecolor=EEGDASH_GRID,
            facecolor=EEGDASH_SURFACE,
            transform=ax.transAxes,
            zorder=1,
        )
    )

    # Column x-anchors (in axes coordinates, inside the card).
    pad = 0.03
    col_label_x = card_left + pad
    # Right-edge anchors for each value column so commas / units / signs
    # align vertically inside their column. The auditory column ends near
    # 60% so each side has room for the longest expected string.
    col_aud_right = card_left + 0.61
    col_vis_right = card_left + card_w - pad
    # Header anchors (slightly inset from each column's right edge).
    col_aud_header = col_aud_right - 0.02
    col_vis_header = col_vis_right

    # Column headers.
    header_y = card_top - 0.06
    ax.text(
        col_label_x,
        header_y,
        "metric",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=EEGDASH_MUTED,
        transform=ax.transAxes,
    )
    ax.text(
        col_aud_header,
        header_y,
        "auditory (live)",
        ha="right",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=EEGDASH_ORANGE,
        transform=ax.transAxes,
    )
    ax.text(
        col_vis_header,
        header_y,
        "visual (plot_20)",
        ha="right",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=EEGDASH_BLUE_DARK,
        transform=ax.transAxes,
    )
    # Underline below header row.
    ax.plot(
        [card_left + pad, card_left + card_w - pad],
        [header_y - 0.045, header_y - 0.045],
        color=EEGDASH_GRID,
        linewidth=0.7,
        alpha=0.7,
        transform=ax.transAxes,
    )

    # Rows: (label, auditory-live, visual-reference). Strings are kept
    # short so two columns of monospace values fit inside the card without
    # bumping into each other at gallery render width.
    rows: list[tuple[str, str, str]] = [
        ("peak latency", f"{peak_time_ms:+.0f} ms", "~300-350 ms"),
        ("peak channel", peak_channel, "Pz"),
        ("peak amp.", f"{peak_uv:+.1f} uV", "~5-10 uV"),
        ("topography", "fronto-central", "centro-parietal"),
        ("class ratio", f"{n_deviants}:{n_standards}", "~1:4 oddball"),
        ("subcomponent", "MMN, P3a, P3b", "P3b dominant"),
    ]
    n_rows = len(rows)
    rows_top = header_y - 0.085
    rows_bot = card_top - card_h + 0.04
    row_h = (rows_top - rows_bot) / n_rows
    for i, (label, aud_val, vis_val) in enumerate(rows):
        y_center = rows_top - (i + 0.5) * row_h
        ax.text(
            col_label_x,
            y_center,
            label,
            ha="left",
            va="center",
            fontsize=8.4,
            color=EEGDASH_MUTED,
            transform=ax.transAxes,
        )
        ax.text(
            col_aud_right,
            y_center,
            aud_val,
            ha="right",
            va="center",
            fontsize=8.8,
            family="monospace",
            color=EEGDASH_INK,
            transform=ax.transAxes,
        )
        ax.text(
            col_vis_right,
            y_center,
            vis_val,
            ha="right",
            va="center",
            fontsize=8.8,
            family="monospace",
            color=EEGDASH_INK,
            transform=ax.transAxes,
        )
        # Thin separator below each row except the last.
        if i < n_rows - 1:
            sep_y = y_center - row_h / 2
            ax.plot(
                [card_left + pad, card_left + card_w - pad],
                [sep_y, sep_y],
                color=EEGDASH_GRID,
                linewidth=0.4,
                alpha=0.45,
                transform=ax.transAxes,
            )

    # Footer hint, sitting just below the card.
    ax.text(
        0.0,
        0.04,
        "visual reference values from Polich 2007 + plot_20.",
        ha="left",
        va="bottom",
        fontsize=7.7,
        style="italic",
        color=EEGDASH_MUTED,
        transform=ax.transAxes,
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def draw_auditory_figure(
    *,
    times_ms: np.ndarray,
    erp_deviant_cz: np.ndarray,
    erp_standard_cz: np.ndarray,
    se_deviant_cz: np.ndarray,
    se_standard_cz: np.ndarray,
    cz_label: str,
    peak_time_ms: float,
    peak_channel: str,
    peak_uv: float,
    diff_uv_at_peak: np.ndarray,
    info: mne.Info,
    n_deviants: int,
    n_standards: int,
    n_channels: int,
    sfreq: float,
    dataset: str,
    plot_id: str = "plot_21",
) -> plt.Figure:
    """Render the 1x3 auditory-vs-visual P300 figure.

    Parameters
    ----------
    times_ms : ndarray
        Epoch time vector in milliseconds. Length matches the ERP arrays.
    erp_deviant_cz, erp_standard_cz : ndarray
        Per-class evoked responses at the headline channel (microvolts).
    se_deviant_cz, se_standard_cz : ndarray
        Standard error bands at the same channel (microvolts).
    cz_label : str
        Channel name used in the panel title (typically ``"Cz"``).
    peak_time_ms : float
        Latency of the auditory P300 peak at the headline channel, in ms.
    peak_channel : str
        Scalp electrode where the difference wave is strongest at the peak.
    peak_uv : float
        Peak amplitude of the difference wave at the headline channel.
    diff_uv_at_peak : ndarray
        Per-EEG-channel difference wave (deviant - standard) sampled at
        the peak latency. Shape ``(n_eeg_channels,)``.
    info : mne.Info
        EEG-only :class:`mne.Info` with a montage attached, used by
        :func:`mne.viz.plot_topomap` to project sensors onto the head.
    n_deviants, n_standards : int
        Trial counts after epoch construction (printed in the legend and
        the comparison card).
    n_channels : int
        Number of EEG channels (drives the subtitle).
    sfreq : float
        Post-resample sampling rate in Hz.
    dataset : str
        OpenNeuro accession (``"ds003061"``) for the provenance footer.
    plot_id : str
        Tutorial id (``"plot_21"``) for the provenance footer.

    Returns
    -------
    matplotlib.figure.Figure
        A 1x3 figure ready for ``plt.show()``.

    """
    fig = plt.figure(figsize=(13.6, 5.6))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.45, 0.95, 1.10],
        wspace=0.33,
        left=0.060,
        right=0.985,
        top=0.78,
        bottom=0.16,
    )
    ax_erp = fig.add_subplot(gs[0, 0])
    ax_topo = fig.add_subplot(gs[0, 1])
    ax_card = fig.add_subplot(gs[0, 2])

    _draw_erp_panel(
        ax_erp,
        times_ms=times_ms,
        erp_dev=erp_deviant_cz,
        erp_std=erp_standard_cz,
        se_dev=se_deviant_cz,
        se_std=se_standard_cz,
        cz_label=cz_label,
        peak_time_ms=peak_time_ms,
        peak_uv=peak_uv,
        n_deviants=n_deviants,
        n_standards=n_standards,
    )

    _draw_topomap_panel(
        ax_topo,
        diff_uv_at_peak=diff_uv_at_peak,
        info=info,
        peak_time_ms=peak_time_ms,
        fig=fig,
    )

    _draw_comparison_panel(
        ax_card,
        peak_time_ms=peak_time_ms,
        peak_channel=peak_channel,
        peak_uv=peak_uv,
        n_deviants=n_deviants,
        n_standards=n_standards,
    )

    subtitle = (
        f"{dataset}  |  n_deviants={n_deviants}, n_standards={n_standards}, "
        f"n_channels={n_channels}  |  sfreq={int(sfreq)} Hz  |  "
        f"peak {peak_time_ms:+.0f} ms at {peak_channel}"
    )
    style_figure(
        fig,
        title="Auditory P300: same oddball plumbing, different brain answer",
        subtitle=subtitle,
        source=(
            f"EEGDash {plot_id} | OpenNeuro {dataset} (Delorme 2020) "
            "| Polich 2007 / Naatanen et al. 2007"
        ),
    )
    return fig


__all__ = ["draw_auditory_figure", "MMN_WINDOW_MS", "P300_WINDOW_MS"]
