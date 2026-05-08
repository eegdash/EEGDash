"""Drawing helpers for the ``plot_10`` "EEGPrep before/after" diagnostic."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

import mne

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_SURFACE,
    style_figure,
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _slice_data(
    raw: mne.io.BaseRaw, *, t_start: float, duration: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sfreq = raw.info["sfreq"]
    t_end = min(t_start + duration, raw.times[-1])
    sl = raw.copy().pick("eeg")
    s0 = int(round(t_start * sfreq))
    s1 = int(round(t_end * sfreq))
    data, times = sl[:, s0:s1]
    return data, times, sl.ch_names


def _shared_channels(
    names_before: Sequence[str], names_after: Sequence[str]
) -> list[str]:
    after_set = set(names_after)
    return [n for n in names_before if n in after_set]


def _draw_heatmap(
    ax,
    data_uv: np.ndarray,
    *,
    times: np.ndarray,
    vmax_uv: float,
    title: str,
    color: str,
    drop_mask: np.ndarray | None = None,
    n_yticks: int = 6,
) -> None:
    n_ch = data_uv.shape[0]
    extent = (times[0], times[-1], n_ch, 0)
    ax.imshow(
        data_uv,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-vmax_uv,
        vmax=vmax_uv,
        extent=extent,
    )
    ax.set_xlabel("time (s)", color=EEGDASH_INK)
    ax.set_ylabel("EEG channel index", color=EEGDASH_INK)
    ytick_pos = np.linspace(0, n_ch - 1, n_yticks).astype(int)
    ax.set_yticks(ytick_pos + 0.5)
    ax.set_yticklabels([str(p) for p in ytick_pos])
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(EEGDASH_GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_title(
        title, color=color, fontsize=10.5, fontweight="bold", loc="left", pad=6
    )
    if drop_mask is not None and drop_mask.any():
        for idx in np.where(drop_mask)[0]:
            ax.plot(
                times[0],
                idx + 0.5,
                marker="<",
                color=EEGDASH_ORANGE,
                markersize=6,
                clip_on=False,
                zorder=4,
            )


def _draw_psd_panel(
    ax,
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    *,
    fmax: float = 80.0,
    band: tuple[float, float] = (1.0, 40.0),
    line_freq: float = 50.0,
) -> None:
    psd_pre = (
        raw_before.copy()
        .pick("eeg")
        .compute_psd(fmax=fmax, n_fft=int(raw_before.info["sfreq"] * 4), verbose=False)
    )
    psd_post = (
        raw_after.copy()
        .pick("eeg")
        .compute_psd(fmax=fmax, n_fft=int(raw_after.info["sfreq"] * 4), verbose=False)
    )

    f_pre = psd_pre.freqs
    f_post = psd_post.freqs
    p_pre = np.mean(psd_pre.get_data(), axis=0)
    p_post = np.mean(psd_post.get_data(), axis=0)

    ax.axvspan(band[0], band[1], color=EEGDASH_ORANGE, alpha=0.13, zorder=0)
    ax.plot(f_pre, p_pre, color=EEGDASH_BLUE, linewidth=1.4, label="before", zorder=2)
    ax.plot(f_post, p_post, color=EEGDASH_INK, linewidth=1.4, label="after", zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(0, fmax)
    if line_freq <= fmax:
        ax.axvline(
            line_freq,
            color=EEGDASH_BLUE_DARK,
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
            zorder=1,
        )
        ymin, ymax = ax.get_ylim()
        ax.text(
            line_freq + 1.2,
            ymax * 0.8,
            f"{int(line_freq)} Hz line",
            color=EEGDASH_BLUE_DARK,
            fontsize=7.8,
            ha="left",
            va="top",
        )
    # Pass-band label inside the orange shading.
    ymin, ymax = ax.get_ylim()
    ax.text(
        (band[0] + band[1]) / 2,
        ymax * 0.5,
        f"{band[0]:.0f}-{band[1]:.0f} Hz target band",
        color=EEGDASH_ORANGE,
        fontsize=7.8,
        ha="center",
        va="center",
        alpha=0.95,
    )
    ax.set_xlabel("frequency (Hz)", color=EEGDASH_INK)
    ax.set_ylabel("PSD (V$^2$/Hz)", color=EEGDASH_INK)
    ax.set_title(
        f"PSD before vs after | {band[0]:.0f}-{band[1]:.0f} Hz target band",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(EEGDASH_GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)


def _draw_stage_bars(
    ax,
    stages: list[tuple[str, str, str]],
    *,
    title: str,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    n = len(stages)
    bar_h = 0.92 / max(n, 1)
    pad = bar_h * 0.18
    for i, (name, status, color) in enumerate(stages):
        y0 = 0.96 - (i + 1) * bar_h + pad / 2
        ax.add_patch(
            FancyBboxPatch(
                (0.02, y0),
                0.96,
                bar_h - pad,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                linewidth=1.1,
                edgecolor=color,
                facecolor=EEGDASH_SURFACE,
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (0.02, y0),
                0.012,
                bar_h - pad,
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )
        ax.text(
            0.06,
            y0 + (bar_h - pad) * 0.62,
            name,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=EEGDASH_INK,
            zorder=3,
        )
        ax.text(
            0.06,
            y0 + (bar_h - pad) * 0.22,
            status,
            ha="left",
            va="center",
            fontsize=8.5,
            family="monospace",
            color=EEGDASH_MUTED,
            zorder=3,
        )

    ax.text(
        0.0,
        1.0,
        title,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        transform=ax.transAxes,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def draw_eegprep_diagnostic(
    *,
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    sfreq: float,
    subject: str,
    dataset: str,
    plot_id: str = "plot_10",
    slice_seconds: float = 30.0,
    slice_start: float = 0.0,
):
    """Render the 2x2 EEGPrep before/after diagnostic figure.

    Parameters
    ----------
    raw_before, raw_after : mne.io.BaseRaw
        Recording before vs after :class:`braindecode.preprocessing.EEGPrep`.
        Both objects are read but never modified.
    sfreq : float
        Original (pre-EEGPrep) sampling rate, used in the subtitle.
    subject, dataset : str
        Used in the subtitle and provenance footer.
    plot_id : str
        Tutorial id (``"plot_10"``).
    slice_seconds : float
        Length of the heatmap slice in seconds.
    slice_start : float
        Where the slice starts inside the recording.

    Returns
    -------
    matplotlib.figure.Figure
        The 2x2 figure ready for ``plt.show()``.

    """
    # Pick EEG-only views and slice both to the same window. The "after"
    # object may have fewer channels than "before" (channels dropped by
    # bad-channel detection), so we render each side at its own height
    # and flag dropped channels with an orange marker on the "before" panel.
    eeg_before = raw_before.copy().pick("eeg")
    eeg_after = raw_after.copy().pick("eeg")

    data_b, times_b, names_b = _slice_data(
        eeg_before, t_start=slice_start, duration=slice_seconds
    )
    data_a, times_a, names_a = _slice_data(
        eeg_after, t_start=slice_start, duration=slice_seconds
    )

    # Convert to microvolts for display.
    data_b_uv = data_b * 1e6
    data_a_uv = data_a * 1e6

    # Symmetric color limits derived from the 99th percentile of |after|.
    # Bursts in the BEFORE panel then saturate red/blue at the same scale,
    # which is exactly the visual story the figure is asked to tell.
    vmax_uv = float(np.percentile(np.abs(data_a_uv), 99))
    vmax_uv = max(vmax_uv, 1e-9)

    drop_mask = np.array([name not in set(names_a) for name in names_b], dtype=bool)

    # Stage status lines built from live counts.
    n_before = len(names_b)
    n_after = len(names_a)
    n_dropped = int(drop_mask.sum())
    bad_annot = [a for a in raw_after.annotations if "BAD" in a["description"].upper()]
    n_bad = len(bad_annot)
    bad_seconds = float(sum(a["duration"] for a in bad_annot))
    pct_burst = 100.0 * bad_seconds / max(raw_after.times[-1], 1e-9)

    stages = [
        (
            "1. Resample",
            f"{int(sfreq)} Hz -> {int(raw_after.info['sfreq'])} Hz",
            EEGDASH_BLUE,
        ),
        (
            "2. High-pass (transition 0.25-0.75 Hz)",
            "drift below 0.25 Hz suppressed",
            EEGDASH_BLUE,
        ),
        (
            "3. Bad-channel removal (corr + HF)",
            f"{n_before} -> {n_after} channels  ({n_dropped} dropped)",
            EEGDASH_ORANGE if n_dropped else EEGDASH_BLUE,
        ),
        (
            "4. ASR burst removal (cutoff = 10)",
            f"{n_bad} bad-window markers, {pct_burst:.1f}% of duration",
            EEGDASH_ORANGE if n_bad else EEGDASH_BLUE,
        ),
        (
            "5. Common average reference",
            "applied across surviving EEG channels",
            EEGDASH_INK,
        ),
    ]

    # Layout: 2x2 grid. Headroom on top for style_figure title/subtitle and
    # a generous bottom band for the source line.
    fig = plt.figure(figsize=(11.4, 8.4))
    gs = fig.add_gridspec(
        2,
        2,
        wspace=0.22,
        hspace=0.42,
        left=0.085,
        right=0.965,
        top=0.78,
        bottom=0.13,
    )
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    _draw_heatmap(
        ax_tl,
        data_b_uv,
        times=times_b,
        vmax_uv=vmax_uv,
        title="before EEGPrep",
        color=EEGDASH_BLUE,
        drop_mask=drop_mask,
    )
    _draw_heatmap(
        ax_tr,
        data_a_uv,
        times=times_a,
        vmax_uv=vmax_uv,
        title="after EEGPrep",
        color=EEGDASH_INK,
    )

    # Shape labels in the upper-right of each heatmap (white pill on top
    # of the data so the panel stays self-contained).
    ax_tl.text(
        0.985,
        0.96,
        f"({n_before} ch, {data_b_uv.shape[1]} samples)",
        transform=ax_tl.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        family="monospace",
        color=EEGDASH_INK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": EEGDASH_BLUE,
            "linewidth": 0.6,
        },
    )
    ax_tr.text(
        0.985,
        0.96,
        f"({n_after} ch, {data_a_uv.shape[1]} samples)",
        transform=ax_tr.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        family="monospace",
        color=EEGDASH_INK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": EEGDASH_INK,
            "linewidth": 0.6,
        },
    )

    _draw_psd_panel(ax_bl, raw_before, raw_after, fmax=80.0, band=(1.0, 40.0))

    _draw_stage_bars(
        ax_br,
        stages,
        title="EEGPrep stages (live status)",
    )

    subtitle = (
        f"sub-{subject}  |  {dataset}  |  "
        f"{n_before} -> {n_after} channels, "
        f"{n_dropped} dropped, "
        f"{pct_burst:.1f}% of duration flagged as burst  |  "
        f"sfreq {int(sfreq)} -> {int(raw_after.info['sfreq'])} Hz"
    )
    style_figure(
        fig,
        title="EEGPrep before vs after: one recording, four views",
        subtitle=subtitle,
        source=(
            f"EEGDash {plot_id} | OpenNeuro {dataset} :cite:`wakeman2015` "
            f"| ASR per Mullen et al. 2015"
        ),
    )
    return fig


__all__ = ["draw_eegprep_diagnostic"]
