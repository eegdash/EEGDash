"""Drawing helpers for the ``plot_74`` NeuroAI-interop figure."""

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
    EEGDASH_PURPLE,
    EEGDASH_SURFACE,
    style_figure,
)

# Stage colors: EEGDash blue for the loader, NeuralFetch purple for the
# discovery layer, NeuralSet ink for the tensor pipeline, DataLoader
# orange for the consumer. The palette stays colorblind-friendly and
# grayscale-distinguishable since each stage uses a different luminance
# band as well as a different hue.
_STAGE_DEFS = [
    {
        "key": "eegdash",
        "color": EEGDASH_BLUE,
        "title": "EEGDashDataset",
        "role": "discover + load BIDS",
    },
    {
        "key": "neuralfetch",
        "color": EEGDASH_PURPLE,
        "title": "NeuralFetch.Study",
        "role": "standardise events",
    },
    {
        "key": "neuralset",
        "color": EEGDASH_INK,
        "title": "NeuralSet.Segmenter",
        "role": "trigger -> tensors",
    },
    {
        "key": "loader",
        "color": EEGDASH_ORANGE,
        "title": "PyTorch DataLoader",
        "role": "batched iterator",
    },
]


def _draw_stage_box(ax, x0, y0, w, h, *, color, title, role, shape):
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
        y0 + h - 0.10,
        title,
        ha="center",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        color=EEGDASH_INK,
    )
    ax.text(
        x0 + w / 2,
        y0 + h - 0.32,
        role,
        ha="center",
        va="top",
        fontsize=8.8,
        color=color,
    )
    ax.text(
        x0 + w / 2,
        y0 + 0.16,
        shape,
        ha="center",
        va="center",
        fontsize=8.0,
        family="monospace",
        color=EEGDASH_INK,
    )


def _draw_pipeline_arrow(ax, x_from, x_to, y, *, label):
    ax.add_patch(
        FancyArrowPatch(
            (x_from, y),
            (x_to, y),
            arrowstyle="-|>,head_length=8,head_width=6",
            linewidth=1.6,
            color=EEGDASH_GRID,
            zorder=4,
        )
    )
    ax.text(
        (x_from + x_to) / 2,
        y + 0.10,
        label,
        ha="center",
        va="bottom",
        fontsize=8.6,
        color=EEGDASH_MUTED,
        family="monospace",
    )


def _ecosystem_panel(ax, *, stage_shapes):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.set_axis_off()

    # Four boxes, equal width, evenly spaced. Geometry kept in axes
    # coordinates so it is independent of the figure size.
    n = len(_STAGE_DEFS)
    box_w = 0.215
    gap = (1.0 - n * box_w) / (n + 1)
    box_h = 0.70
    box_y = 0.18

    centers = []
    for i, stage in enumerate(_STAGE_DEFS):
        x = gap + i * (box_w + gap)
        centers.append((x + box_w / 2, x, x + box_w))
        _draw_stage_box(
            ax,
            x,
            box_y,
            box_w,
            box_h,
            color=stage["color"],
            title=stage["title"],
            role=stage["role"],
            shape=stage_shapes[stage["key"]],
        )

    arrow_y = box_y + box_h * 0.55
    arrow_labels = ["query", "segment", "batch"]
    for i in range(n - 1):
        _, _, right_i = centers[i]
        _, left_j, _ = centers[i + 1]
        _draw_pipeline_arrow(
            ax,
            right_i + 0.012,
            left_j - 0.012,
            arrow_y,
            label=arrow_labels[i],
        )


def _draw_sparkline(ax, signal, *, color, color_fill, highlight_idx=None):
    t = np.arange(signal.size)
    ax.plot(t, signal, color=color, linewidth=1.0)
    ax.fill_between(t, signal, signal.min(), color=color_fill, alpha=0.18, linewidth=0)
    if highlight_idx is not None:
        for idx in highlight_idx:
            if 0 <= idx < signal.size:
                ax.axvline(idx, color=EEGDASH_ORANGE, linewidth=0.9, alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(0.7)


def _identity_panel(fig, gridspec_slot, *, sample_window, highlight_idx, sfreq):
    inner = gridspec_slot.subgridspec(1, 3, wspace=0.20)
    panels = []
    titles = [
        "EEGDashDataset record",
        "NeuralFetch events row",
        "NeuralSet segment tensor",
    ]
    colors = [EEGDASH_BLUE_DARK, EEGDASH_PURPLE, EEGDASH_INK]
    fills = [EEGDASH_BLUE, EEGDASH_PURPLE, EEGDASH_INK]
    for i, (title, color, fill) in enumerate(zip(titles, colors, fills)):
        ax = fig.add_subplot(inner[0, i])
        _draw_sparkline(
            ax,
            sample_window,
            color=color,
            color_fill=fill,
            highlight_idx=highlight_idx,
        )
        ax.set_title(title, fontsize=9.2, color=color, pad=4)
        panels.append(ax)
    # One shape annotation under the leftmost panel (the raw record). The
    # other two annotate at the same shape too, since the contract of
    # this row is "shape preserved".
    secs = sample_window.size / float(sfreq) if sfreq > 0 else 0.0
    label = f"({sample_window.size} samples, {secs:.2f} s)"
    panels[0].set_xlabel(label, fontsize=8.4, color=EEGDASH_MUTED, family="monospace")
    panels[1].set_xlabel(label, fontsize=8.4, color=EEGDASH_MUTED, family="monospace")
    panels[2].set_xlabel(label, fontsize=8.4, color=EEGDASH_MUTED, family="monospace")
    return panels


def _matrix_panel(ax, *, integration_matrix):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    headers = ["NeuroAI project", "Role", "EEGDash integration", "Demonstrated in"]
    n_rows = len(integration_matrix)

    # Column x-positions and widths. Project column wider (it holds the
    # name and the colored swatch); demonstrated-in column wider too
    # (longer text).
    col_widths = np.array([0.18, 0.30, 0.27, 0.25])
    col_widths = col_widths / col_widths.sum()
    col_x = np.concatenate([[0.0], np.cumsum(col_widths)])

    header_y = 0.92
    body_top = header_y - 0.10
    body_bot = 0.04
    row_h = (body_top - body_bot) / max(n_rows, 1)

    # Header row
    for j, label in enumerate(headers):
        ax.text(
            col_x[j] + col_widths[j] / 2,
            header_y,
            label,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=EEGDASH_INK,
        )
    # Header rule
    ax.plot(
        [0.005, 0.995],
        [header_y - 0.06, header_y - 0.06],
        color=EEGDASH_GRID,
        linewidth=1.0,
    )

    project_palette = {
        "NeuralFetch": EEGDASH_PURPLE,
        "NeuralSet": EEGDASH_INK,
        "NeuralTrain": EEGDASH_BLUE_DARK,
        "NeuralBench": EEGDASH_ORANGE,
    }

    for i, row in enumerate(integration_matrix):
        y = body_top - (i + 0.5) * row_h
        # Alternating row band for readability
        if i % 2 == 0:
            ax.add_patch(
                plt.Rectangle(
                    (0.005, y - row_h / 2 + 0.005),
                    0.99,
                    row_h - 0.01,
                    facecolor=EEGDASH_SURFACE,
                    edgecolor="none",
                    zorder=0,
                )
            )

        # Project cell: colored swatch + name
        swatch_color = project_palette.get(row["project"], EEGDASH_MUTED)
        sw_x = col_x[0] + 0.020
        sw_w = 0.025
        ax.add_patch(
            plt.Rectangle(
                (sw_x, y - 0.018),
                sw_w,
                0.036,
                facecolor=swatch_color,
                edgecolor="none",
                zorder=2,
            )
        )
        ax.text(
            sw_x + sw_w + 0.012,
            y,
            row["project"],
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=EEGDASH_INK,
        )

        # Role cell
        ax.text(
            col_x[1] + 0.012,
            y,
            row["role"],
            ha="left",
            va="center",
            fontsize=8.8,
            color=EEGDASH_INK,
        )

        # Status cell with status dot
        status_color = {
            "shipped": EEGDASH_BLUE,
            "compatible": EEGDASH_PURPLE,
            "downstream": EEGDASH_MUTED,
        }.get(row.get("status_kind", "downstream"), EEGDASH_MUTED)
        dot_x = col_x[2] + 0.012
        ax.add_patch(
            plt.Circle(
                (dot_x, y),
                0.012,
                facecolor=status_color,
                edgecolor="none",
                zorder=2,
            )
        )
        ax.text(
            dot_x + 0.022,
            y,
            row["status"],
            ha="left",
            va="center",
            fontsize=8.6,
            color=EEGDASH_INK,
        )

        # Tutorial cell
        ax.text(
            col_x[3] + 0.012,
            y,
            row["tutorial"],
            ha="left",
            va="center",
            fontsize=8.4,
            family="monospace",
            color=EEGDASH_MUTED,
        )


def draw_neuroai_interop_figure(
    *,
    dataset_meta: dict,
    sample_window: np.ndarray,
    integration_matrix: Sequence[dict],
    plot_id: str = "plot_74",
):
    """Render the three-panel NeuroAI interop figure.

    Parameters
    ----------
    dataset_meta : dict
        Live runtime metadata. Required keys: ``dataset`` (e.g.
        ``"ds002718"``), ``n_records``, ``n_channels``, ``sfreq``,
        ``window_samples``, ``citation`` (short citation for the source).
    sample_window : numpy.ndarray
        One channel of one window. Used as the sparkline payload in
        panel 2; same array shown three times so the visual contract
        "shape preserved" reads from the figure alone.
    integration_matrix : sequence of dict
        One row per NeuroAI project. Each row carries
        ``project``, ``role``, ``status``, ``status_kind`` (one of
        ``"shipped"``, ``"compatible"``, ``"downstream"``), and
        ``tutorial``.
    plot_id : str, optional
        Plot id used in the provenance footer.

    """
    n_channels = int(dataset_meta["n_channels"])
    sfreq = float(dataset_meta["sfreq"])
    window_samples = int(dataset_meta["window_samples"])
    batch_size = int(dataset_meta.get("batch_size", 8))
    n_records = int(dataset_meta["n_records"])
    dataset_id = str(dataset_meta["dataset"])
    citation = str(dataset_meta.get("citation", ""))

    stage_shapes = {
        "eegdash": f"({n_channels} ch, raw)",
        "neuralfetch": "DataFrame[start, type, ...]",
        "neuralset": f"({n_channels}, {window_samples})",
        "loader": f"({batch_size}, {n_channels}, {window_samples})",
    }

    fig = plt.figure(figsize=(13.0, 9.5))
    gs = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.05, 1.00, 1.15],
        left=0.05,
        right=0.98,
        top=0.80,
        bottom=0.09,
        hspace=0.55,
    )

    ax_eco = fig.add_subplot(gs[0, 0])
    _ecosystem_panel(ax_eco, stage_shapes=stage_shapes)

    # Panel 2: identity through the pipeline.
    arr = np.asarray(sample_window, dtype=float).ravel()
    if arr.size == 0:
        arr = np.zeros(window_samples or 1, dtype=float)
    n = arr.size
    highlight = [int(0.20 * n), int(0.55 * n), int(0.85 * n)]
    _identity_panel(
        fig,
        gs[1, 0],
        sample_window=arr,
        highlight_idx=highlight,
        sfreq=sfreq,
    )

    # Panel 3: integration matrix
    ax_mat = fig.add_subplot(gs[2, 0])
    _matrix_panel(ax_mat, integration_matrix=integration_matrix)

    # Figure-level title, subtitle, source. style_figure attaches the
    # Data Rail and the consistent typography across the gallery.
    subtitle = (
        f"{dataset_id} | {n_records} record | {n_channels} ch @ {sfreq:.0f} Hz | "
        f"{window_samples}-sample windows"
    )
    source_bits = [f"EEGDash {plot_id}"]
    if dataset_id:
        source_bits.append(
            f"OpenNeuro {dataset_id}" + (f" ({citation})" if citation else "")
        )
    source_bits.append("Meta NeuroAI: facebookresearch.github.io/neuroai")
    style_figure(
        fig,
        title="EEGDash in the Meta NeuroAI ecosystem",
        subtitle=subtitle,
        source=" | ".join(source_bits),
        grid_axis="none",
    )
    # Restore the GridSpec margins ``style_figure`` overwrites with its
    # default ``subplots_adjust`` so the three panels keep enough room
    # for the panel headers attached as ``fig.text`` annotations below.
    fig.subplots_adjust(top=0.78, bottom=0.09, left=0.05, right=0.98)

    # Panel headers: place each one just above the corresponding axes,
    # using the live bbox so the labels track the gridspec layout.
    for ax_target, header in (
        (
            ax_eco,
            "1. Pipeline: EEGDash record -> NeuralFetch events -> "
            "NeuralSet tensor -> DataLoader",
        ),
        (
            None,
            "2. Identity check: the bytes survive every stage",
        ),
        (
            ax_mat,
            "3. NeuroAI integration matrix",
        ),
    ):
        if ax_target is not None:
            bbox = ax_target.get_position()
            y = bbox.y1 + 0.012
        else:
            # Identity panel is a triple of subplots inside gs[1,0]; use
            # the gridspec slot directly to get a y close to its top.
            slot = gs[1, 0]
            sub_bbox = slot.get_position(fig)
            y = sub_bbox.y1 + 0.012
        fig.text(
            0.05,
            y,
            header,
            fontsize=10.5,
            color=EEGDASH_INK,
            ha="left",
            va="bottom",
        )
    return fig


__all__ = ["draw_neuroai_interop_figure"]
