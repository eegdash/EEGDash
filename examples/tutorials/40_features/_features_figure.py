"""Drawing helpers for the ``plot_40`` first-features plate.

Sibling module to ``plot_40_first_features.py``. The leading underscore
tells sphinx-gallery to skip this file when building the gallery (see
``docs/source/conf.py`` ``ignore_pattern``); the rendered tutorial only
imports the public ``draw_features_figure`` entry point.

The figure is a 1x3 plate that reads the same way the feature matrix
does in plot_42:

1. *Band x channel feature-matrix heatmap* (left) -- mean log power
   across windows for theta, alpha, beta, gamma. ``RdBu_r`` divergent
   colormap symmetric around the per-band mean so increases above and
   suppressions below the mean read as red and blue. The alpha row
   carries a thin EEGDASH_ORANGE tick to flag the band Berger first
   reported.
2. *Per-band histogram strip* (middle) -- four small inset histograms
   stacked vertically with one EEGDASH palette colour per band. Each
   inset shows the log-power distribution across all (window, channel)
   pairs for that band; the stripe lets the reader read the variance
   structure that plot_42 hands to sklearn.
3. *Top-K alpha discriminative channels* (right) -- horizontal bar
   chart of per-channel ``closed - open`` log-power difference (or
   per-channel variance when condition labels are absent). Bars are
   EEGDASH_BLUE; the channel that peaks first sits at the top.

Geometry, colours, and annotations follow the same conventions as
``_alpha_figure.py`` and ``_pipeline_diagram.py`` so the gallery reads
as one consistent set of figures.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_PALETTE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _style_axis(ax) -> None:
    """Apply the EEGDash spine/tick treatment to a single axes."""
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(EEGDASH_GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


def _draw_heatmap_panel(
    fig,
    ax,
    *,
    feature_matrix: np.ndarray,
    band_names: Sequence[str],
    channel_names: Sequence[str],
    highlight_band: str = "alpha",
) -> None:
    """Band x channel heatmap of mean log power across windows.

    Each row is z-scored across channels so the band-internal
    parieto-occipital structure reads at the same intensity in every
    band; the absolute log-power scale (which spans orders of magnitude
    because alpha can be 100x the broadband floor) is reported in the
    panel below as a histogram strip. The ``RdBu_r`` colormap is
    symmetric around zero and the alpha row carries a thin orange tick
    on the left margin to flag the Berger band.
    """
    matrix = np.asarray(feature_matrix, dtype=float)
    row_mean = matrix.mean(axis=1, keepdims=True)
    row_std = matrix.std(axis=1, keepdims=True)
    row_std = np.where(row_std < 1e-9, 1.0, row_std)
    z_matrix = (matrix - row_mean) / row_std
    span = float(np.max(np.abs(z_matrix)))
    span = max(span, 1e-9)

    im = ax.imshow(
        z_matrix,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-span,
        vmax=span,
        interpolation="nearest",
    )

    n_bands = len(band_names)
    n_channels = len(channel_names)

    # X ticks: every channel when n_channels <= 16, else every 4th.
    x_step = 1 if n_channels <= 16 else 4
    x_ticks = list(range(0, n_channels, x_step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [channel_names[i] for i in x_ticks],
        rotation=0,
        fontsize=8.0,
        family="monospace",
        color=EEGDASH_INK,
    )
    ax.set_yticks(range(n_bands))
    ax.set_yticklabels(
        list(band_names),
        fontsize=9.0,
        family="monospace",
        color=EEGDASH_INK,
    )
    ax.set_xlabel("channel", color=EEGDASH_INK, fontsize=9.0)
    ax.set_ylabel("band", color=EEGDASH_INK, fontsize=9.0)
    ax.set_title(
        "Feature matrix: mean log power per (band, channel)",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )

    # Highlight the alpha row with a thin orange tick on the left.
    if highlight_band in band_names:
        row = list(band_names).index(highlight_band)
        ax.plot(
            [-0.6, -0.6],
            [row - 0.42, row + 0.42],
            color=EEGDASH_ORANGE,
            linewidth=2.6,
            solid_capstyle="butt",
            clip_on=False,
            zorder=5,
        )

    # Spines off; ticks invisible (heatmap reads cleaner).
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.038,
        pad=0.025,
        shrink=0.88,
    )
    cbar.set_label(
        "z-score per band (across channels)",
        color=EEGDASH_INK,
        fontsize=8.5,
    )
    cbar.ax.tick_params(labelsize=7.5, colors=EEGDASH_INK, length=0)
    cbar.outline.set_visible(False)


def _draw_histogram_panel(
    fig,
    gridspec_slot,
    *,
    log_power_per_band: dict[str, np.ndarray],
    band_names: Sequence[str],
) -> None:
    """Stacked per-band histograms of (window, channel) log-power values.

    Splits the GridSpec slot into one axes per band, stacked vertically.
    Each sub-axes uses a different EEGDash palette colour so the
    distribution shape per band is visible at a glance; the band label
    sits inside each sub-axes in monospace.
    """
    n_bands = len(band_names)
    inner = gridspec_slot.subgridspec(n_bands, 1, hspace=0.55)

    # Shared x range so the four insets are comparable.
    all_values = np.concatenate(
        [np.asarray(log_power_per_band[b]).ravel() for b in band_names]
    )
    finite = all_values[np.isfinite(all_values)]
    x_lo = float(np.percentile(finite, 1.0))
    x_hi = float(np.percentile(finite, 99.0))
    if x_hi - x_lo < 1e-6:
        pad = max(abs(x_lo), 1.0) * 0.1
        x_lo, x_hi = x_lo - pad, x_hi + pad

    palette = list(EEGDASH_PALETTE)
    first_ax = None
    for k, band in enumerate(band_names):
        ax = fig.add_subplot(inner[k, 0])
        if first_ax is None:
            first_ax = ax
        values = np.asarray(log_power_per_band[band]).ravel()
        values = values[np.isfinite(values)]
        color = palette[k % len(palette)]
        ax.hist(
            values,
            bins=22,
            range=(x_lo, x_hi),
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.92,
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7.5, colors=EEGDASH_INK, length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(EEGDASH_GRID)
        ax.spines["bottom"].set_linewidth(0.6)

        # Band label inside the panel, top-right in axes coords so the
        # rotation does not collide with bars.
        ax.text(
            0.985,
            0.92,
            band,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            family="monospace",
            color=EEGDASH_INK,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.5,
                "alpha": 0.95,
            },
        )
        if k < n_bands - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(
                "log10 power",
                color=EEGDASH_INK,
                fontsize=8.5,
            )

    if first_ax is not None:
        first_ax.set_title(
            "Per-band log-power distribution",
            color=EEGDASH_INK,
            fontsize=10.5,
            fontweight="bold",
            loc="left",
            pad=6,
        )


def _draw_topk_panel(
    ax,
    *,
    channel_names: Sequence[str],
    discriminative_score: np.ndarray,
    score_label: str,
    top_k: int = 8,
) -> None:
    """Horizontal bar chart of the top-K discriminative channels for alpha.

    ``discriminative_score`` is the per-channel score (closed - open log
    power if condition labels exist, else per-channel variance). The
    largest absolute scores sit at the top; the bars are EEGDASH_BLUE.
    """
    score = np.asarray(discriminative_score, dtype=float)
    order = np.argsort(np.abs(score))[::-1][: int(top_k)]
    # Re-sort the selected slice in ascending value so the largest bar
    # ends up at the top of the panel.
    order = order[np.argsort(score[order])]
    labels = [channel_names[i] for i in order]
    values = score[order]

    y = np.arange(len(order))
    ax.barh(
        y,
        values,
        color=EEGDASH_BLUE,
        edgecolor=EEGDASH_BLUE_DARK,
        linewidth=0.5,
        height=0.62,
        zorder=3,
    )
    for yi, vi in zip(y, values):
        ha = "left" if vi >= 0 else "right"
        offset = 0.012 * max(np.max(np.abs(values)), 1e-9)
        ax.text(
            vi + (offset if vi >= 0 else -offset),
            yi,
            f"{vi:+.2f}",
            ha=ha,
            va="center",
            fontsize=7.8,
            family="monospace",
            color=EEGDASH_INK,
            zorder=4,
        )

    ax.axvline(0.0, color=EEGDASH_MUTED, linewidth=0.8, zorder=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=8.5,
        family="monospace",
        color=EEGDASH_INK,
    )
    ax.set_xlabel(score_label, color=EEGDASH_INK, fontsize=9.0)
    ax.set_title(
        f"Top-{int(top_k)} alpha-discriminative channels",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    # Pad the x-axis so the inline value labels do not get clipped.
    span = float(np.max(np.abs(values))) if values.size else 1.0
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.grid(True, axis="x", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def draw_features_figure(
    *,
    feature_matrix: np.ndarray,
    band_names: Sequence[str],
    channel_names: Sequence[str],
    log_power_per_band: dict[str, np.ndarray],
    discriminative_score: np.ndarray,
    score_label: str,
    n_windows: int,
    sfreq: float,
    dataset: str = "ds005514",
    citation: str = "Alexander et al. 2017 (HBN, mock)",
    top_k: int = 8,
    plot_id: str = "plot_40",
):
    """Render the band x channel x distribution x discriminator plate.

    Parameters
    ----------
    feature_matrix : ndarray, shape (n_bands, n_channels)
        Mean log power across windows per (band, channel).
    band_names : sequence of str
        Frequency-band labels in row order. The label ``"alpha"`` (when
        present) is highlighted with an orange margin tick.
    channel_names : sequence of str
        Channel labels in column order. Tick labels show every channel
        when ``len(channel_names) <= 16`` and every fourth channel
        otherwise.
    log_power_per_band : dict[str, ndarray]
        Per-band 2-D arrays of shape ``(n_windows, n_channels)`` holding
        log10 band power. Used by the histogram strip; one inset axes
        per band.
    discriminative_score : ndarray, shape (n_channels,)
        Per-channel alpha score for the right panel: ``closed - open``
        log-power difference when condition labels are available, else
        per-channel variance across windows.
    score_label : str
        x-axis label of the right panel (e.g.
        ``"alpha (closed - open) log10 power"``).
    n_windows : int
        Total number of windows in the feature table; printed in the
        figure subtitle.
    sfreq : float
        Effective sampling rate (Hz); printed in the subtitle.
    dataset : str, optional
        OpenNeuro accession for the provenance footer (default
        ``"ds005514"``).
    citation : str, optional
        Short citation for the provenance footer.
    top_k : int, optional
        Number of channels in the right panel (default 8).
    plot_id : str, optional
        Tutorial id for the provenance footer (default ``"plot_40"``).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure ready for ``plt.show()``.

    """
    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.shape != (len(band_names), len(channel_names)):
        raise ValueError(
            f"feature_matrix shape {matrix.shape} does not match "
            f"(n_bands={len(band_names)}, n_channels={len(channel_names)})"
        )
    for band in band_names:
        if band not in log_power_per_band:
            raise ValueError(f"log_power_per_band missing entry for band {band!r}")

    fig = plt.figure(figsize=(13.6, 5.0))
    gs = fig.add_gridspec(
        1,
        3,
        wspace=0.46,
        left=0.06,
        right=0.97,
        top=0.80,
        bottom=0.16,
        width_ratios=[1.55, 1.0, 1.0],
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_topk = fig.add_subplot(gs[0, 2])

    _draw_heatmap_panel(
        fig,
        ax_heat,
        feature_matrix=matrix,
        band_names=list(band_names),
        channel_names=list(channel_names),
        highlight_band="alpha",
    )

    _draw_histogram_panel(
        fig,
        gs[0, 1],
        log_power_per_band={b: np.asarray(log_power_per_band[b]) for b in band_names},
        band_names=list(band_names),
    )

    _draw_topk_panel(
        ax_topk,
        channel_names=list(channel_names),
        discriminative_score=np.asarray(discriminative_score),
        score_label=score_label,
        top_k=int(top_k),
    )

    n_bands = len(band_names)
    n_channels = len(channel_names)
    subtitle = (
        f"{dataset} | n_windows={int(n_windows)} | n_bands={int(n_bands)} | "
        f"n_channels={int(n_channels)} | sfreq {sfreq:.0f} Hz"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=dataset,
        citation=citation,
        extra=(
            "bands theta/alpha/beta/gamma; Berger 1929; Klimesch 2012; "
            "Welch psd_array_welch"
        ),
    )
    style_figure(
        fig,
        title=(
            "From windows to a feature matrix: heatmap -> distribution -> discriminator"
        ),
        subtitle=subtitle,
        source=source,
        grid_axis="none",
    )
    return fig


__all__ = ["draw_features_figure"]
