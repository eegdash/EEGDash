"""Drawing helpers for the ``plot_13`` "round-trip ledger" figure.

Sibling module to ``plot_13_save_and_reuse_prepared_data.py``. The leading
underscore tells sphinx-gallery to skip this file when building the gallery,
so the helper code stays out of the rendered tutorial; the tutorial imports
the public ``draw_ledger_figure`` entry point.

The figure has three panels:

1. Grouped horizontal bars: write-time vs read-time per format, with the
   on-disk size (MB) annotated to the right of each row.
2. Residual heatmap: ``reloaded - original`` for one window, hard-clipped to
   plus-or-minus ``1e-7`` in float32 units. Title carries ``np.allclose`` and
   ``max|delta|``.
3. Provenance text card: package versions, seed, and git short-SHA. Renders
   with monospace font so the cache stays reusable next month.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_SURFACE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# ---------------------------------------------------------------------------
# Panel 1: write-time vs read-time bars
# ---------------------------------------------------------------------------


def _draw_cost_bars(ax, format_records: Sequence[Mapping]) -> None:
    """Render grouped horizontal bars for write-time and read-time per format."""
    n = len(format_records)
    y = np.arange(n)
    bar_h = 0.36
    write_s = np.array([r["write_s"] for r in format_records], dtype=float)
    read_s = np.array([r["read_s"] for r in format_records], dtype=float)
    sizes_mb = np.array([r["size_mb"] for r in format_records], dtype=float)
    names = [r["name"] for r in format_records]

    ax.barh(
        y - bar_h / 2,
        write_s,
        height=bar_h,
        color=EEGDASH_BLUE,
        edgecolor="white",
        linewidth=0.7,
        label="write",
        zorder=3,
    )
    ax.barh(
        y + bar_h / 2,
        read_s,
        height=bar_h,
        color=EEGDASH_ORANGE,
        edgecolor="white",
        linewidth=0.7,
        label="read",
        zorder=3,
    )

    x_max = max(float(write_s.max()), float(read_s.max()), 1e-3)
    pad = x_max * 0.02
    # Reserve the rightmost band for the MB annotation so it never collides
    # with the bar-end ms labels, even when read_s is much smaller than write_s.
    mb_x = x_max * 1.42
    for i, (w, r, mb) in enumerate(zip(write_s, read_s, sizes_mb)):
        ax.text(
            w + pad,
            i - bar_h / 2,
            f"{w * 1000:.1f} ms",
            va="center",
            ha="left",
            fontsize=8.4,
            color=EEGDASH_INK,
            family="monospace",
        )
        ax.text(
            r + pad,
            i + bar_h / 2,
            f"{r * 1000:.1f} ms",
            va="center",
            ha="left",
            fontsize=8.4,
            color=EEGDASH_INK,
            family="monospace",
        )
        ax.text(
            mb_x,
            i,
            f"{mb:>6.3f} MB",
            va="center",
            ha="left",
            fontsize=8.6,
            color=EEGDASH_MUTED,
            family="monospace",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=EEGDASH_INK)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max * 1.75)
    ax.set_xlabel("seconds", fontsize=9, color=EEGDASH_INK)
    ax.set_title("write vs read cost", fontsize=10.5, color=EEGDASH_INK, loc="left")
    # Place the legend above the axes so it never collides with bar labels.
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        frameon=False,
        fontsize=8.6,
        handlelength=1.4,
        labelcolor=EEGDASH_INK,
        ncol=2,
    )
    ax.grid(True, axis="x", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


# ---------------------------------------------------------------------------
# Panel 2: residual heatmap
# ---------------------------------------------------------------------------


def _draw_residual_heatmap(ax, residual_array: np.ndarray) -> None:
    """Render the (n_channels, n_times) residual heatmap clipped to plus-or-minus 1e-7."""
    arr = np.asarray(residual_array, dtype=float)
    clip = 1.0e-7
    arr_clipped = np.clip(arr, -clip, clip)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    allclose = bool(np.allclose(arr, 0.0, atol=1.0e-7))

    im = ax.imshow(
        arr_clipped,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-clip,
        vmax=clip,
    )
    ax.set_xlabel("sample index", fontsize=9, color=EEGDASH_INK)
    ax.set_ylabel("channel", fontsize=9, color=EEGDASH_INK)
    n_ch = arr.shape[0]
    ax.set_yticks(np.arange(n_ch))
    ax.set_yticklabels([f"ch{i}" for i in range(n_ch)], fontsize=8)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)
    ax.grid(False)

    parity_line = (
        f"np.allclose=True, max|delta|={max_abs:.2e}"
        if allclose
        else (f"np.allclose=False, max|delta|={max_abs:.2e}")
    )
    ax.set_title(
        "residual: reloaded - original (FIF)\n" + parity_line,
        fontsize=10.0,
        color=EEGDASH_INK,
        loc="left",
        pad=8.0,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.040, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.set_ticks([-clip, -clip / 2, 0.0, clip / 2, clip])
    cbar.set_ticklabels(
        ["-1e-7", "-5e-8", "0", "+5e-8", "+1e-7"],
        fontsize=7.5,
        color=EEGDASH_MUTED,
    )
    cbar.set_label(
        "delta, clipped",
        fontsize=8,
        color=EEGDASH_MUTED,
        labelpad=2.0,
    )
    cbar.ax.yaxis.get_offset_text().set_visible(False)


# ---------------------------------------------------------------------------
# Panel 3: provenance card
# ---------------------------------------------------------------------------


def _format_provenance(provenance: Mapping[str, str]) -> str:
    """Pretty-print the provenance dict as monospace ``key: value`` rows."""
    width = max(len(k) for k in provenance) if provenance else 0
    rows = []
    for k, v in provenance.items():
        rows.append(f"{k.ljust(width)}  {v}")
    return "\n".join(rows)


def _draw_provenance_card(ax, provenance: Mapping[str, str]) -> None:
    """Render a monospace text block listing the cache provenance."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        plt.Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            transform=ax.transAxes,
            facecolor=EEGDASH_SURFACE,
            edgecolor=EEGDASH_BLUE,
            linewidth=1.2,
            zorder=1,
        )
    )
    ax.text(
        0.04,
        0.93,
        "cache provenance",
        ha="left",
        va="top",
        fontsize=10.0,
        color=EEGDASH_INK,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.78,
        "what a future reader needs to rerun this exactly",
        ha="left",
        va="top",
        fontsize=8.4,
        color=EEGDASH_MUTED,
        style="italic",
        transform=ax.transAxes,
    )
    # Render provenance rows in two columns so seven keys never overflow.
    rows = list(provenance.items())
    half = (len(rows) + 1) // 2
    col1, col2 = rows[:half], rows[half:]
    width1 = max((len(k) for k, _ in col1), default=0)
    width2 = max((len(k) for k, _ in col2), default=0)
    text1 = "\n".join(f"{k.ljust(width1)}  {v}" for k, v in col1)
    text2 = "\n".join(f"{k.ljust(width2)}  {v}" for k, v in col2)
    ax.text(
        0.04,
        0.62,
        text1,
        ha="left",
        va="top",
        fontsize=9.0,
        family="monospace",
        color=EEGDASH_INK,
        transform=ax.transAxes,
    )
    if text2:
        ax.text(
            0.40,
            0.62,
            text2,
            ha="left",
            va="top",
            fontsize=9.0,
            family="monospace",
            color=EEGDASH_INK,
            transform=ax.transAxes,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def draw_ledger_figure(
    *,
    format_records: Sequence[Mapping],
    residual_array: np.ndarray,
    provenance_dict: Mapping[str, str],
    n_windows: int,
    n_channels: int,
    window_samples: int,
    plot_id: str = "plot_13",
):
    """Render the three-panel round-trip ledger figure.

    Parameters
    ----------
    format_records : sequence of mapping
        Each entry has keys ``name``, ``write_s``, ``read_s``, ``size_mb``.
        The order is preserved on the y-axis.
    residual_array : numpy.ndarray
        A ``(n_channels, n_times)`` array of ``reloaded - original`` for one
        window, in the same float32 units as the recorded signal.
    provenance_dict : mapping of str -> str
        Key-value pairs rendered as a monospace text card (Panel 3). Caller
        supplies whatever keys it wants; typical entries are
        ``eegdash``, ``braindecode``, ``numpy``, ``seed``, ``git``.
    n_windows, n_channels, window_samples : int
        Live values from the runtime, surfaced in the figure subtitle.
    plot_id : str, optional
        Tutorial id, forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure. The caller is responsible for ``plt.show()``.

    """
    total_disk_mb = float(sum(r["size_mb"] for r in format_records))

    fig = plt.figure(figsize=(11.0, 6.6))
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.05, 0.75],
        wspace=0.40,
        hspace=0.60,
        left=0.16,
        right=0.965,
        top=0.80,
        bottom=0.12,
    )
    ax_bars = fig.add_subplot(gs[0, 0])
    ax_resid = fig.add_subplot(gs[0, 1])
    ax_prov = fig.add_subplot(gs[1, :])

    _draw_cost_bars(ax_bars, format_records)
    _draw_residual_heatmap(ax_resid, residual_array)
    _draw_provenance_card(ax_prov, provenance_dict)

    subtitle = (
        f"{n_windows} windows | {n_channels} channels | "
        f"{window_samples} samples/window | "
        f"{total_disk_mb:.3f} MB on disk"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        citation="synthetic recording",
        extra="seed=42",
    )
    style_figure(
        fig,
        title="Round-trip ledger: cost and parity of cached prepared data",
        subtitle=subtitle,
        source=source,
        grid_axis="x",
    )
    return fig


__all__ = ["draw_ledger_figure"]
