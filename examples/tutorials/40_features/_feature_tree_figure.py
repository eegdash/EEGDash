"""Drawing helpers for the ``plot_41`` feature-tree plate.

Sibling module to ``plot_41_feature_trees.py``. The leading underscore
tells sphinx-gallery to skip this file when building the gallery (see
``docs/source/conf.py`` ``ignore_pattern``); the rendered tutorial only
imports the public ``draw_feature_tree_figure`` entry point.

The figure is a three-panel plate that matches the polish standard set by
``_pipeline_diagram.py`` (plot_02) and ``_alpha_figure.py`` (plot_30):

1. *Feature tree* (left). A directed graph drawn with FancyBboxPatch
   nodes and FancyArrowPatch edges. Root ``raw window`` feeds the
   ``Welch PSD`` intermediate node; four leaves hang off the PSD
   (``band power``, ``spectral entropy``, ``peak frequency``,
   ``1/f slope``). Each leaf carries a tiny inset thumbnail showing
   the relevant slice of the PSD.
2. *Per-feature distributions* (centre). Four mini KDEs, one per
   derived feature, coloured from ``EEGDASH_PALETTE``. The reader sees
   the live spread of each feature across all windows of the recording
   and confirms that the four scalars are not redundant.
3. *Feature-feature Pearson correlation heatmap* (right). 4x4 with
   value annotations in monospace, ``RdBu_r`` divergent symmetric. Tells
   the reader, in one panel, which derived markers move together.

Helvetica is missing the U+2192 right-arrow glyph, so every prose label
uses ``->`` instead of an actual arrow character.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_PALETTE,
    EEGDASH_SURFACE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# ---------------------------------------------------------------------------
# Internal panel painters
# ---------------------------------------------------------------------------


def _style_axis(ax) -> None:
    """Apply the EEGDash spine and tick treatment to a single axes."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(EEGDASH_GRID)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=2.5, colors=EEGDASH_INK)


def _draw_node(
    ax,
    *,
    cx: float,
    cy: float,
    w: float,
    h: float,
    color: str,
    title: str,
    sub: str | None = None,
    title_size: float = 10.5,
    sub_size: float = 8.4,
    title_anchor: str = "center",
) -> tuple[float, float, float, float]:
    """Draw a rounded rectangle node centred on ``(cx, cy)`` and label it.

    Returns the bounding box ``(x0, y0, w, h)`` so callers can wire edges
    to the node's borders without recomputing the geometry.

    ``title_anchor`` selects between ``"center"`` (one-line caption) and
    ``"top"`` (title pinned to the top edge so an inset thumbnail can
    occupy the bottom of the box).
    """
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.022",
            linewidth=1.6,
            edgecolor=color,
            facecolor=EEGDASH_SURFACE,
            zorder=2,
        )
    )
    if sub is None:
        if title_anchor == "top":
            ax.text(
                cx,
                cy + h * 0.42,
                title,
                ha="center",
                va="top",
                fontsize=title_size,
                fontweight="bold",
                color=EEGDASH_INK,
                linespacing=1.0,
                zorder=3,
            )
        else:
            ax.text(
                cx,
                cy,
                title,
                ha="center",
                va="center",
                fontsize=title_size,
                fontweight="bold",
                color=EEGDASH_INK,
                zorder=3,
            )
    else:
        ax.text(
            cx,
            cy + h * 0.20,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=EEGDASH_INK,
            zorder=3,
        )
        ax.text(
            cx,
            cy - h * 0.22,
            sub,
            ha="center",
            va="center",
            fontsize=sub_size,
            color=color,
            family="monospace",
            zorder=3,
        )
    return x0, y0, w, h


def _connect(ax, x_from, y_from, x_to, y_to, *, color=EEGDASH_GRID) -> None:
    """Draw a straight FancyArrowPatch between two anchor points."""
    ax.add_patch(
        FancyArrowPatch(
            (x_from, y_from),
            (x_to, y_to),
            arrowstyle="-|>,head_length=7,head_width=5",
            linewidth=1.4,
            color=color,
            zorder=1,
        )
    )


def _add_leaf_thumb(fig, ax, x0, y0, w, h, painter) -> None:
    """Carve an inner axes inside a leaf node and call ``painter(ax)``.

    The inset is positioned in figure coordinates so it survives any later
    layout adjustments. Spines are hidden so the thumbnail reads as
    ornament, not as a chart with its own axes.
    """
    # Trim the inset to the lower ~45 % of the node; the title row
    # (which can be one or two lines) sits in the upper half.
    pad = 0.012
    ix0 = x0 + pad
    iy0 = y0 + pad
    iw = w - 2 * pad
    ih = h * 0.45
    fig_w_in, fig_h_in = fig.get_size_inches()
    px0, py0 = ax.transData.transform((ix0, iy0))
    px1, py1 = ax.transData.transform((ix0 + iw, iy0 + ih))
    rect = (
        px0 / (fig_w_in * fig.dpi),
        py0 / (fig_h_in * fig.dpi),
        (px1 - px0) / (fig_w_in * fig.dpi),
        (py1 - py0) / (fig_h_in * fig.dpi),
    )
    inset = fig.add_axes(rect, frame_on=False)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)
    painter(inset)


def _paint_psd_band(ax, freqs, psd_mean, color):
    """Plot the mean PSD with a shaded 8-12 Hz alpha band."""
    ax.plot(freqs, psd_mean, color=color, linewidth=1.1)
    ax.fill_between(
        freqs, psd_mean, psd_mean.min(), color=color, alpha=0.25, linewidth=0
    )
    ax.axvspan(8.0, 12.0, color=color, alpha=0.20, zorder=0)
    ax.set_xlim(freqs.min(), freqs.max())


def _paint_psd_norm(ax, freqs, psd_mean, color):
    """Plot the *normalised* PSD (a probability density)."""
    pdf = psd_mean / (psd_mean.sum() + 1e-30)
    ax.plot(freqs, pdf, color=color, linewidth=1.1)
    ax.fill_between(freqs, pdf, 0.0, color=color, alpha=0.25, linewidth=0)
    ax.set_xlim(freqs.min(), freqs.max())


def _paint_psd_peak(ax, freqs, psd_mean, color):
    """Plot the PSD with a vertical marker at the peak frequency."""
    ax.plot(freqs, psd_mean, color=color, linewidth=1.1)
    peak_f = freqs[int(np.argmax(psd_mean))]
    ax.axvline(peak_f, color=color, linewidth=1.6, linestyle="--", alpha=0.9)
    ax.set_xlim(freqs.min(), freqs.max())


def _paint_psd_loglog(ax, freqs, psd_mean, color):
    """Plot log-log PSD with a fitted line, showing the 1/f slope."""
    f_pos = freqs > 0
    log_f = np.log10(freqs[f_pos])
    log_p = np.log10(psd_mean[f_pos] + 1e-30)
    slope, intercept = np.polyfit(log_f, log_p, 1)
    fit = slope * log_f + intercept
    ax.plot(log_f, log_p, color=color, linewidth=1.0, alpha=0.7)
    ax.plot(log_f, fit, color=color, linewidth=1.6, linestyle="--")
    ax.set_xlim(log_f.min(), log_f.max())


def _draw_tree_panel(
    fig,
    ax,
    *,
    psd_array: np.ndarray,
    freqs: np.ndarray,
    feature_names: Sequence[str],
    palette: Sequence[str],
) -> None:
    """Paint Panel 1 (the dependency tree) on ``ax``.

    Geometry uses axes coordinates 0..1 so the panel reads identically at
    any DPI. The four leaf painters are wired by name so the label, the
    thumbnail painter, and the palette colour stay zipped together.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.set_axis_off()

    psd_mean = psd_array.mean(axis=(0, 1))

    # Root and intermediate nodes. Y values give breathing room for the
    # panel-title strip at y in [0.93, 1.0] and the leaf row at y in
    # [0.05, 0.33].
    root_w, root_h = 0.42, 0.12
    root_cx, root_cy = 0.5, 0.83
    _draw_node(
        ax,
        cx=root_cx,
        cy=root_cy,
        w=root_w,
        h=root_h,
        color=EEGDASH_BLUE,
        title="raw window",
        sub=f"({psd_array.shape[1]} ch, time)",
    )

    psd_w, psd_h = 0.44, 0.13
    psd_cx, psd_cy = 0.5, 0.58
    _draw_node(
        ax,
        cx=psd_cx,
        cy=psd_cy,
        w=psd_w,
        h=psd_h,
        color=EEGDASH_INK,
        title="Welch PSD",
        sub=f"({psd_array.shape[1]} ch, {freqs.size} freqs)",
    )

    # Edge: root -> PSD
    _connect(
        ax,
        root_cx,
        root_cy - root_h / 2,
        psd_cx,
        psd_cy + psd_h / 2,
        color=EEGDASH_BLUE_DARK,
    )

    # Four leaf nodes. Width 0.235 keeps "spectral entropy" inside the
    # box; vertical span 0.30 leaves room for the inset thumbnail.
    leaf_painters = [
        _paint_psd_band,
        _paint_psd_norm,
        _paint_psd_peak,
        _paint_psd_loglog,
    ]
    leaf_w, leaf_h = 0.235, 0.30
    leaf_cy = 0.18
    leaf_cxs = [0.135, 0.378, 0.622, 0.865]

    for cx, name, color, painter in zip(
        leaf_cxs, feature_names, palette[: len(feature_names)], leaf_painters
    ):
        # Two-word leaf names get a soft line break so the title fits
        # inside the leaf box at any reasonable figure width.
        leaf_title = name.replace(" ", "\n", 1) if " " in name else name
        x0, y0, w, h = _draw_node(
            ax,
            cx=cx,
            cy=leaf_cy,
            w=leaf_w,
            h=leaf_h,
            color=color,
            title=leaf_title,
            sub=None,
            title_size=9.0,
            title_anchor="top",
        )
        # Leaf title sits at the top of the box; thumbnail fills the bottom.
        # Paint a small thumbnail inside the lower 50 % of the node.
        _add_leaf_thumb(
            fig,
            ax,
            x0,
            y0,
            w,
            h,
            painter=lambda inset, c=color, p=painter: p(inset, freqs, psd_mean, c),
        )
        # Edge: PSD -> leaf
        _connect(
            ax,
            psd_cx + (cx - psd_cx) * 0.35,
            psd_cy - psd_h / 2,
            cx,
            leaf_cy + leaf_h / 2,
            color=color,
        )

    # Panel title sits inside the axes so it never collides with the
    # figure-level suptitle / subtitle banner.
    ax.text(
        0.5,
        0.965,
        "Welch PSD is the parent of every classical marker",
        ha="center",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=EEGDASH_INK,
    )


def _draw_distribution_panel(
    fig,
    ax,
    *,
    derived_features: np.ndarray,
    feature_names: Sequence[str],
    palette: Sequence[str],
) -> None:
    """Paint Panel 2 (per-feature distributions).

    ``derived_features`` is shape ``(n_windows, n_features)``. Each feature
    is drawn as a Gaussian-KDE curve over its own y-row; a small dot
    annotation marks the mean. Coloured from ``palette`` so the row colour
    in Panel 2 matches the leaf colour in Panel 1.
    """
    ax.set_axis_off()

    n_feat = derived_features.shape[1]
    inner = ax.get_subplotspec().subgridspec(n_feat, 1, hspace=0.55)
    fig_axes = []
    for i in range(n_feat):
        sub_ax = fig.add_subplot(inner[i, 0])
        _style_axis(sub_ax)
        sub_ax.set_yticks([])
        fig_axes.append(sub_ax)
        x = derived_features[:, i]
        x = x[np.isfinite(x)]
        x_range = float(x.max() - x.min()) if x.size else 0.0
        if x.size < 2 or x_range == 0.0:
            # Fallback for degenerate columns: draw a thin spike at the mean.
            sub_ax.axvline(
                float(x.mean()) if x.size else 0.0,
                color=palette[i % len(palette)],
                linewidth=2.0,
            )
            sub_ax.set_xlim(
                x.min() - 1.0 if x.size else -1.0,
                x.max() + 1.0 if x.size else 1.0,
            )
        else:
            # Pad the x-range by 8 % each side so KDE tails do not clip.
            pad = 0.08 * x_range
            grid = np.linspace(x.min() - pad, x.max() + pad, 200)
            # Silverman bandwidth, with a floor proportional to the
            # column's range so values at picovolt scale still render.
            bw = 1.06 * x.std(ddof=1) * x.size ** (-1 / 5)
            bw = max(bw, 0.05 * x_range)
            kde = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(
                axis=1
            ) / (x.size * bw * np.sqrt(2 * np.pi))
            sub_ax.plot(grid, kde, color=palette[i % len(palette)], linewidth=1.4)
            sub_ax.fill_between(
                grid, kde, 0.0, color=palette[i % len(palette)], alpha=0.30, linewidth=0
            )
            mean = float(x.mean())
            sub_ax.axvline(
                mean, color=palette[i % len(palette)], linewidth=1.0, linestyle=":"
            )
            sub_ax.set_xlim(grid.min(), grid.max())
        sub_ax.text(
            0.0,
            1.04,
            feature_names[i],
            transform=sub_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.4,
            color=palette[i % len(palette)],
            fontweight="bold",
        )
        # n and mean as a monospace tag in the lower-right of each row,
        # away from any matplotlib scientific-notation offset text.
        if x.size:
            sub_ax.text(
                0.985,
                0.92,
                f"n={x.size}  mean={x.mean():.3g}",
                transform=sub_ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.8,
                family="monospace",
                color=EEGDASH_MUTED,
            )
        # Style the matplotlib offset-text (e.g. "1e-11") that floats up
        # next to the right tick label at picovolt scale; we keep it on so
        # readers know the band-power axis is V^2/Hz at 1e-12.
        offset = sub_ax.xaxis.get_offset_text()
        offset.set_color(EEGDASH_MUTED)
        offset.set_fontsize(7.5)

    fig_axes[0].set_title(
        "Distribution of each derived feature",
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        pad=14,
        loc="center",
    )
    fig_axes[-1].set_xlabel("feature value", fontsize=9, color=EEGDASH_MUTED)


def _draw_correlation_panel(
    fig,
    ax,
    *,
    derived_features: np.ndarray,
    feature_names: Sequence[str],
) -> None:
    """Paint Panel 3 (4x4 Pearson correlation heatmap)."""
    # Per-column standardise, then compute Pearson correlation.
    finite = np.all(np.isfinite(derived_features), axis=1)
    X = derived_features[finite]
    # Replace zero-variance columns with a tiny jitter to avoid NaN corr.
    stds = X.std(axis=0)
    safe = np.where(stds > 0, stds, 1.0)
    X = (X - X.mean(axis=0)) / safe
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    im = ax.imshow(
        corr,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=30, ha="right", fontsize=8.6)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8.6)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n = corr.shape[0]
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            txt_color = "white" if abs(v) > 0.55 else EEGDASH_INK
            ax.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=8.6,
                family="monospace",
                color=txt_color,
            )

    ax.set_title(
        "Pairwise Pearson correlation",
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        pad=10,
    )

    # Slim colourbar pinned to the right of the heatmap.
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, shrink=0.85)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=8, colors=EEGDASH_INK)
    cbar.outline.set_visible(False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def draw_feature_tree_figure(
    *,
    psd_array: np.ndarray,
    freqs: np.ndarray,
    derived_features: np.ndarray,
    feature_names: Sequence[str],
    plot_id: str = "plot_41",
    sfreq: float | None = None,
    n_windows: int | None = None,
    citation: str = "Demanuele et al. 2007 / Donoghue et al. 2020",
):
    """Render the three-panel feature-tree plate and return the Figure.

    Parameters
    ----------
    psd_array : ndarray
        Per-window Welch PSD with shape ``(n_windows, n_channels, n_freqs)``.
        Used by Panel 1 to draw the per-leaf thumbnails from the recording's
        mean PSD; the absolute scale does not matter, only the shape.
    freqs : ndarray
        Frequency vector matching ``psd_array.shape[-1]``.
    derived_features : ndarray
        Shape ``(n_windows, n_features)``; one column per name in
        ``feature_names``. The four standard columns are
        band power, spectral entropy, peak frequency, 1/f slope, but any
        four-feature matrix renders correctly as long as the columns are
        ordered to match ``feature_names``.
    feature_names : sequence of str
        Display labels for the four leaves and the four KDE rows. Must
        have length 4 to match the panel-1 geometry.
    plot_id : str, optional
        Used in the provenance footer.
    sfreq : float, optional
        Sampling rate, surfaced in the subtitle.
    n_windows : int, optional
        Window count, surfaced in the subtitle. Falls back to
        ``derived_features.shape[0]``.
    citation : str, optional
        Short citation line for the provenance footer.

    Returns
    -------
    matplotlib.figure.Figure

    """
    if len(feature_names) != 4:
        raise ValueError(
            f"draw_feature_tree_figure expects 4 feature_names; got {len(feature_names)}."
        )
    if derived_features.shape[1] != 4:
        raise ValueError(
            "draw_feature_tree_figure expects derived_features with 4 columns; "
            f"got shape {derived_features.shape}."
        )
    n_windows = (
        int(n_windows) if n_windows is not None else int(derived_features.shape[0])
    )

    fig = plt.figure(figsize=(13.8, 5.2))
    gs = GridSpec(
        1,
        3,
        width_ratios=[1.30, 1.00, 0.90],
        wspace=0.32,
        left=0.045,
        right=0.985,
        bottom=0.16,
        top=0.85,
    )
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_corr = fig.add_subplot(gs[0, 2])

    palette = list(EEGDASH_PALETTE[:4])
    if len(palette) < 4:
        # Pad with the orange accent if the project palette is ever shortened.
        palette += [EEGDASH_ORANGE] * (4 - len(palette))

    _draw_tree_panel(
        fig,
        ax_tree,
        psd_array=psd_array,
        freqs=freqs,
        feature_names=feature_names,
        palette=palette,
    )
    _draw_distribution_panel(
        fig,
        ax_dist,
        derived_features=derived_features,
        feature_names=feature_names,
        palette=palette,
    )
    _draw_correlation_panel(
        fig,
        ax_corr,
        derived_features=derived_features,
        feature_names=feature_names,
    )

    sfreq_label = f"{int(sfreq)} Hz" if sfreq else "PSD inputs"
    subtitle = f"n_windows={n_windows} | n_features=4 | sfreq={sfreq_label}"
    style_figure(
        fig,
        title="Feature trees: one Welch PSD, four derived markers",
        subtitle=subtitle,
        source=add_provenance_footer(
            None,
            plot_id=plot_id,
            openneuro_id=None,
            citation=citation,
        ),
    )
    return fig


__all__ = ["draw_feature_tree_figure"]
