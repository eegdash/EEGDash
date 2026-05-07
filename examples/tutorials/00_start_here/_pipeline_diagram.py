"""Drawing helpers for the ``plot_02`` "Pipeline at a glance" figure.

Sibling module to ``plot_02_dataset_to_dataloader.py``. The leading
underscore tells sphinx-gallery to skip this file when building the
gallery, so the helpers stay out of the rendered tutorial; the tutorial
imports the public ``draw_pipeline`` entry point.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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


# Geometry budget for one stage box (axes coords, 0..1):
#   - title band:    y0 + h - 0.07  (12 pt bold)
#   - subtitle band: y0 + h - 0.16  (9 pt, in box colour)
#   - inner figure:  y0 + 0.22 .. y0 + h - 0.22  (the imshow / bars)
#   - shape band:    y0 + 0.05  (monospace shape label, below inner)
#   - foot1 band:    y0 - 0.06   (contract, monospace, muted)
#   - foot2 band:    y0 - 0.13


def _draw_stage(ax, x0, y0, w, h, *, color, title, sub, foot1, foot2):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.022",
            linewidth=1.7,
            edgecolor=color,
            facecolor=EEGDASH_SURFACE,
            zorder=1,
        )
    )
    ax.text(
        x0 + w / 2,
        y0 + h - 0.07,
        title,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=EEGDASH_INK,
    )
    ax.text(
        x0 + w / 2,
        y0 + h - 0.16,
        sub,
        ha="center",
        va="top",
        fontsize=9.0,
        color=color,
    )
    ax.plot(
        [x0 + 0.012, x0 + 0.012],
        [y0 - 0.005, y0 - 0.16],
        color=color,
        linewidth=0.9,
        zorder=3,
    )
    ax.text(
        x0 + 0.022,
        y0 - 0.06,
        foot1,
        ha="left",
        va="center",
        fontsize=8.6,
        family="monospace",
        color=EEGDASH_MUTED,
    )
    ax.text(
        x0 + 0.022,
        y0 - 0.13,
        foot2,
        ha="left",
        va="center",
        fontsize=8.6,
        family="monospace",
        color=EEGDASH_MUTED,
    )


def _add_inner_axes(fig, ax, x0, y0, w, h, *, top_clear, bot_clear):
    inner_x = x0 + 0.018
    inner_w = w - 0.036
    inner_y = y0 + bot_clear
    inner_h = h - top_clear - bot_clear
    fig_x0, fig_y0 = ax.transData.transform((inner_x, inner_y))
    fig_x1, fig_y1 = ax.transData.transform((inner_x + inner_w, inner_y + inner_h))
    fig_size = fig.get_size_inches() * fig.dpi
    rect = (
        fig_x0 / fig_size[0],
        fig_y0 / fig_size[1],
        (fig_x1 - fig_x0) / fig_size[0],
        (fig_y1 - fig_y0) / fig_size[1],
    )
    return fig.add_axes(rect, frame_on=False)


def _draw_records_inner(inset, raw_data, sfreq, color, color_dark):
    n_ch_show = 3
    n_t = raw_data.shape[1]
    step = max(1, n_t // 600)
    n_ch = raw_data.shape[0]
    picks = np.linspace(0, n_ch - 1, n_ch_show, dtype=int)
    for k, ci in enumerate(picks):
        trace = raw_data[ci, ::step]
        t = np.arange(trace.size) * step / sfreq
        amp = np.abs(trace).max() or 1.0
        center = n_ch_show - 1 - k
        normalised = trace / amp
        inset.plot(t, normalised + 2.4 * center, color=color_dark, linewidth=0.6)
        inset.fill_between(
            t,
            normalised + 2.4 * center,
            2.4 * center,
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_ylim(-1.4, 2.4 * (n_ch_show - 1) + 1.4)
    for spine in inset.spines.values():
        spine.set_visible(False)


def _draw_windows_inner(inset, window_xy, color):
    inset.imshow(
        window_xy,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-np.abs(window_xy).max(),
        vmax=np.abs(window_xy).max(),
    )
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(0.9)


def _draw_batches_layered(ax, x0, y0, w, h, *, color, top_clear, bot_clear):
    inner_top = y0 + h - top_clear
    inner_bot = y0 + bot_clear
    offset = 0.012
    box_w = w - 0.10
    box_h = inner_top - inner_bot
    stack_total_w = box_w + 2 * offset
    bx = x0 + (w - stack_total_w) / 2
    by = inner_bot
    for i in [2, 1]:
        ax.add_patch(
            Rectangle(
                (bx + i * offset, by - i * offset),
                box_w,
                box_h,
                facecolor=EEGDASH_SURFACE,
                edgecolor=color,
                linewidth=0.9,
                alpha=0.85,
                zorder=2,
            )
        )
    return bx, by, box_w, box_h


def _draw_arrow(ax, x_from, x_to, y, *, label_top, label_bot):
    ax.add_patch(
        FancyArrowPatch(
            (x_from, y),
            (x_to, y),
            arrowstyle="-|>,head_length=9,head_width=7",
            linewidth=2.0,
            color=EEGDASH_GRID,
            zorder=4,
        )
    )
    xc = (x_from + x_to) / 2
    ax.text(
        xc,
        y + 0.075,
        label_top,
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=EEGDASH_INK,
        fontweight="bold",
    )
    ax.text(
        xc,
        y + 0.020,
        label_bot,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=EEGDASH_MUTED,
    )


def draw_pipeline(
    *,
    record_signal: np.ndarray,
    window_xy: np.ndarray,
    batch_xy: np.ndarray,
    n_records: int,
    n_channels: int,
    sfreq: float,
    window_size_samples: int,
    batch_size: int,
    n_windows: int,
    subject: str,
    n_channels_full: int | None = None,
):
    """Render the three-stage pipeline diagram and return the Figure.

    All inputs are live values from the caller (the tutorial passes
    ``raw_pp.get_data(picks='eeg')``, ``windows[0][0]``, ``X_batch[0]``
    and the scalars). The figure is laid out in axes coordinates so the
    geometry is independent of figure DPI.
    """
    n_batches_est = max(1, n_windows // batch_size)
    if n_channels_full is None:
        n_channels_full = n_channels

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.set_axis_off()

    stage_geom: list[dict[str, Any]] = [
        {
            "x": 0.015,
            "color": EEGDASH_BLUE,
            "color_dark": EEGDASH_BLUE_DARK,
            "title": "EEGDashDataset",
            "sub": f"{n_records} record - BIDS meta",
            "shape_label": (
                f"({n_channels_full} ch, {int(record_signal.shape[1]):,} samples)"
            ),
            "foot1": "__len__   = n_records",
            "foot2": "__getitem__ -> Raw, ...",
        },
        {
            "x": 0.395,
            "color": EEGDASH_INK,
            "color_dark": EEGDASH_INK,
            "title": "WindowsDataset",
            "sub": f"{n_windows:,} windows",
            "shape_label": f"({n_channels} ch, {window_size_samples} samples)",
            "foot1": "__len__   = n_windows",
            "foot2": "__getitem__ -> (X, y, idx)",
        },
        {
            "x": 0.775,
            "color": EEGDASH_ORANGE,
            "color_dark": EEGDASH_ORANGE,
            "title": "DataLoader",
            "sub": f"~{n_batches_est:,} batches of {batch_size}",
            "shape_label": f"({batch_size}, {n_channels}, {window_size_samples})",
            "foot1": "iter()    -> batches",
            "foot2": "stacked across n=batch_size",
        },
    ]
    stage_w, stage_h, stage_y = 0.21, 0.66, 0.22

    for stage in stage_geom:
        _draw_stage(
            ax,
            stage["x"],
            stage_y,
            stage_w,
            stage_h,
            color=stage["color"],
            title=stage["title"],
            sub=stage["sub"],
            foot1=stage["foot1"],
            foot2=stage["foot2"],
        )
        ax.text(
            stage["x"] + stage_w / 2,
            stage_y + 0.05,
            stage["shape_label"],
            ha="center",
            va="center",
            fontsize=8.4,
            family="monospace",
            color=EEGDASH_INK,
            zorder=15,
        )

    inset_records = _add_inner_axes(
        fig,
        ax,
        stage_geom[0]["x"],
        stage_y,
        stage_w,
        stage_h,
        top_clear=0.22,
        bot_clear=0.22,
    )
    _draw_records_inner(
        inset_records, record_signal, sfreq, EEGDASH_BLUE, EEGDASH_BLUE_DARK
    )

    inset_windows = _add_inner_axes(
        fig,
        ax,
        stage_geom[1]["x"],
        stage_y,
        stage_w,
        stage_h,
        top_clear=0.22,
        bot_clear=0.22,
    )
    _draw_windows_inner(inset_windows, window_xy, EEGDASH_INK)

    bx_b, by_b, bw_b, bh_b = _draw_batches_layered(
        ax,
        stage_geom[2]["x"],
        stage_y,
        stage_w,
        stage_h,
        color=EEGDASH_ORANGE,
        top_clear=0.22,
        bot_clear=0.22,
    )
    fig_size_inches = fig.get_size_inches() * fig.dpi
    fx0, fy0 = ax.transData.transform((bx_b + 0.005, by_b + 0.005))
    fx1, fy1 = ax.transData.transform((bx_b + bw_b - 0.005, by_b + bh_b - 0.005))
    inset_batch = fig.add_axes(
        (
            fx0 / fig_size_inches[0],
            fy0 / fig_size_inches[1],
            (fx1 - fx0) / fig_size_inches[0],
            (fy1 - fy0) / fig_size_inches[1],
        ),
        frame_on=False,
    )
    inset_batch.imshow(
        batch_xy,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-np.abs(batch_xy).max(),
        vmax=np.abs(batch_xy).max(),
    )
    inset_batch.set_xticks([])
    inset_batch.set_yticks([])
    for spine in inset_batch.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(EEGDASH_ORANGE)
        spine.set_linewidth(0.9)
    ax.text(
        bx_b + bw_b - 0.008,
        by_b + bh_b - 0.012,
        f"n={batch_size}",
        ha="right",
        va="top",
        fontsize=8.0,
        family="monospace",
        color=EEGDASH_ORANGE,
        zorder=30,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": EEGDASH_ORANGE,
            "linewidth": 0.6,
        },
    )

    arrow_y = stage_y + stage_h * 0.55
    _draw_arrow(
        ax,
        stage_geom[0]["x"] + stage_w + 0.020,
        stage_geom[1]["x"] - 0.020,
        arrow_y,
        label_top="preprocess",
        label_bot="+ cut windows",
    )
    _draw_arrow(
        ax,
        stage_geom[1]["x"] + stage_w + 0.020,
        stage_geom[2]["x"] - 0.020,
        arrow_y,
        label_top="batch",
        label_bot="+ shuffle",
    )

    style_figure(
        fig,
        title="Dataset vs DataLoader: the pipeline",
        subtitle=(
            f"sub-{subject} - {n_records} record - {n_channels} ch @ "
            f"{int(sfreq)} Hz\n{n_windows:,} windows of "
            f"{window_size_samples} samples -> "
            f"{n_batches_est:,} batches of {batch_size}"
        ),
        source="EEGDash plot_02 | OpenNeuro ds002718 (Wakeman & Henson 2015)",
    )
    return fig
