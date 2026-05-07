"""Drawing helpers for the ``tutorial_challenge_1`` starter-kit figure.

Sibling module to ``tutorial_challenge_1.py``. The leading underscore tells
sphinx-gallery to skip this file when building the gallery (see
``docs/source/conf.py`` ``ignore_pattern``); the rendered tutorial only
imports the public :func:`draw_challenge_1_figure` entry point.

The figure is a 1x3 plate that reads as the canonical "where do I start
with Challenge 1?" view of the EEG2025 Foundation Challenge:

1. *Trial-structure schematic* (left). Nodes arranged on a timeline:
   pre-stimulus baseline -> contrast ramp onset (stim cue) -> response
   window -> feedback. The decoder window (+0.5 s .. +2.5 s after the
   stimulus anchor) sits as a translucent band over a synthetic EEG
   sparkline so the reader sees both the trial structure and what the
   model actually consumes.
2. *One CCD window heatmap* (centre). A single ``(n_channels, n_samples)``
   trial rendered with a diverging ``RdBu_r`` colormap. The shape
   ``(129, 200)`` is annotated inline so the tensor contract is visible
   at the same scale as the data.
3. *Baseline decoder accuracy* (right). One bar per cross-validation
   fold (or subject), a ``chance_line(ax, level=0.5)`` reference, and a
   mean +/- std band so the headline number ships next to its noise
   floor. The mean is annotated above the band.

Geometry and colour conventions follow ``_challenge_basics_figure.py`` and
``_cross_task_figure.py`` so the gallery reads as one consistent set of
figures.
"""

from __future__ import annotations

from typing import Mapping, Sequence

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
    chance_line,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer, shape_label


# ---------------------------------------------------------------------------
# Panel 1: trial-structure schematic
# ---------------------------------------------------------------------------


def _draw_node(ax, x_center, y_center, width, height, *, color, title, sub):
    """One rounded node on the trial-timeline schematic."""
    ax.add_patch(
        FancyBboxPatch(
            (x_center - width / 2, y_center - height / 2),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.020",
            linewidth=1.4,
            edgecolor=color,
            facecolor=EEGDASH_SURFACE,
            zorder=3,
        )
    )
    ax.text(
        x_center,
        y_center + 0.022,
        title,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=EEGDASH_INK,
        zorder=4,
    )
    ax.text(
        x_center,
        y_center - 0.034,
        sub,
        ha="center",
        va="center",
        fontsize=7.8,
        color=color,
        zorder=4,
    )


def _draw_paradigm_panel(ax, paradigm_schematic_data: Mapping) -> None:
    """Render the CCD trial timeline + decoder-window highlight + EEG trace.

    ``paradigm_schematic_data`` carries:
      - ``trace`` : ``(n_samples,)`` synthetic EEG sparkline.
      - ``sfreq`` : sampling frequency of the trace (Hz).
      - ``shift_after_stim`` : decoder window start, seconds after stim.
      - ``window_len`` : decoder window length, seconds.
      - ``stim_time`` : seconds at which the stimulus cue lands.
      - ``response_time`` : seconds at which the synthetic response lands.
    """
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    # Title band
    ax.text(
        0.0,
        0.99,
        "CCD trial -> decoder window",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.93,
        "flickering discs; one disc ramps; subject presses left/right",
        ha="left",
        va="top",
        fontsize=8.1,
        color=EEGDASH_MUTED,
        style="italic",
        transform=ax.transAxes,
    )

    # Time axis line (a thin horizontal rule across the panel).
    y_axis = 0.62
    ax.plot([0.04, 0.96], [y_axis, y_axis], color=EEGDASH_GRID, linewidth=0.7, zorder=1)

    # Four trial-structure nodes.
    nodes = [
        (0.13, "baseline", "pre-stim flicker"),
        (0.36, "stimulus", "contrast ramp"),
        (0.62, "response", "key press"),
        (0.86, "feedback", "smile / frown"),
    ]
    colors = [EEGDASH_MUTED, EEGDASH_BLUE_DARK, EEGDASH_ORANGE, EEGDASH_MUTED]
    for (x, title, sub), color in zip(nodes, colors):
        _draw_node(ax, x, y_axis, 0.18, 0.16, color=color, title=title, sub=sub)

    # Arrows between nodes (left to right).
    for i in range(len(nodes) - 1):
        x0 = nodes[i][0] + 0.09
        x1 = nodes[i + 1][0] - 0.09
        ax.add_patch(
            FancyArrowPatch(
                (x0, y_axis),
                (x1, y_axis),
                arrowstyle="->,head_length=6,head_width=4",
                linewidth=1.0,
                color=EEGDASH_GRID,
                zorder=2,
            )
        )

    # Decoder window highlight: a translucent blue band spanning from
    # stim+shift to stim+shift+window_len, projected onto the axis line.
    stim_x = nodes[1][0]
    win_start_x = stim_x + 0.06  # represents +0.5 s after stim
    win_end_x = stim_x + 0.34  # represents +2.5 s after stim
    band_y0 = 0.30
    band_y1 = 0.45
    ax.add_patch(
        Rectangle(
            (win_start_x, band_y0),
            win_end_x - win_start_x,
            band_y1 - band_y0,
            facecolor=EEGDASH_BLUE,
            edgecolor=EEGDASH_BLUE_DARK,
            linewidth=0.9,
            alpha=0.22,
            zorder=2,
        )
    )
    ax.text(
        (win_start_x + win_end_x) / 2,
        (band_y0 + band_y1) / 2,
        "decoder window  +0.5 s .. +2.5 s",
        ha="center",
        va="center",
        fontsize=8.0,
        family="monospace",
        color=EEGDASH_BLUE_DARK,
        zorder=4,
    )

    # Synthetic EEG sparkline below the band so the reader sees what the
    # model actually consumes for one trial.
    trace = np.asarray(paradigm_schematic_data.get("trace"), dtype=float)
    if trace.size:
        # Map trace x to span the full panel width [0.04, 0.96]; y stays
        # in a thin band under the decoder-window highlight.
        n = trace.size
        x_trace = np.linspace(0.04, 0.96, n)
        amp = float(np.abs(trace).max() or 1.0)
        # Center the sparkline around y=0.16, span ~0.14 vertically.
        y_trace = 0.16 + 0.10 * (trace / amp)
        ax.plot(x_trace, y_trace, color=EEGDASH_INK, linewidth=0.7, zorder=2)
        ax.text(
            0.04,
            0.02,
            "synthetic EEG sparkline (1 channel, one CCD trial)",
            ha="left",
            va="bottom",
            fontsize=7.6,
            color=EEGDASH_MUTED,
            style="italic",
            transform=ax.transAxes,
        )


# ---------------------------------------------------------------------------
# Panel 2: one CCD window heatmap
# ---------------------------------------------------------------------------


def _draw_window_panel(ax, sample_window: np.ndarray, *, sfreq: float) -> None:
    """Render one ``(n_channels, n_samples)`` CCD window as a heatmap."""
    win = np.asarray(sample_window, dtype=float)
    if win.ndim != 2:
        raise ValueError("sample_window must be 2-D (n_channels, n_samples)")
    n_chans, n_samples = win.shape
    vmax = float(np.abs(win).max() or 1.0)

    im = ax.imshow(
        win,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xlabel("time (samples)", color=EEGDASH_INK, fontsize=9)
    ax.set_ylabel("EEG channels", color=EEGDASH_INK, fontsize=9)
    ax.set_title(
        "one CCD trial window",
        fontsize=10.5,
        color=EEGDASH_INK,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    # Light spine treatment.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(EEGDASH_GRID)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK, labelsize=8)

    # Inline shape annotation, white-pill background so it reads on the
    # diverging colormap.
    shape_label(
        ax,
        x=n_samples * 0.02,
        y=n_chans * 0.06,
        shape_tuple=f"({n_chans} ch, {n_samples} samples)  sfreq={sfreq:.0f} Hz",
        color=EEGDASH_INK,
        fontsize=8.4,
        ha="left",
        va="center",
        background=EEGDASH_BLUE,
    )

    cbar = ax.figure.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.046,
        pad=0.04,
        shrink=0.78,
    )
    cbar.set_label("uV (z-scored)", color=EEGDASH_INK, fontsize=8.4)
    cbar.ax.tick_params(labelsize=7.4, colors=EEGDASH_INK, length=0)
    cbar.outline.set_visible(False)


# ---------------------------------------------------------------------------
# Panel 3: per-fold baseline accuracy
# ---------------------------------------------------------------------------


def _draw_accuracy_panel(
    ax,
    fold_accuracies: Sequence[float],
    *,
    chance_level: float,
) -> None:
    """One bar per fold (or subject), with mean +/- std band and chance line."""
    accs = np.asarray(list(fold_accuracies), dtype=float)
    n = accs.size
    if n == 0:
        ax.text(0.5, 0.5, "no folds", ha="center", va="center", color=EEGDASH_MUTED)
        ax.set_axis_off()
        return

    x = np.arange(n)
    mean = float(accs.mean())
    std = float(accs.std()) if n > 1 else 0.0

    # Shaded mean +/- std band so the noise floor is visible.
    if std > 0.0:
        ax.axhspan(
            mean - std,
            mean + std,
            color=EEGDASH_ORANGE,
            alpha=0.14,
            zorder=1,
            label=f"mean +/- std = {mean:.2f} +/- {std:.2f}",
        )
    # Solid mean line.
    ax.axhline(
        mean,
        color=EEGDASH_ORANGE,
        linewidth=1.2,
        zorder=2,
        label=f"mean = {mean:.2f}",
    )

    # Per-fold bars.
    bars = ax.bar(
        x,
        accs,
        width=0.62,
        color=EEGDASH_BLUE,
        edgecolor=EEGDASH_INK,
        linewidth=0.5,
        zorder=3,
        label="per fold",
    )
    for bar, value in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            family="monospace",
            color=EEGDASH_INK,
            zorder=4,
        )

    # Compute a sane y-limit before adding the chance line so the level
    # always lands within view.
    y_max = max(0.95, float(accs.max()) + 0.20)
    y_min = max(0.0, min(0.30, float(min(accs.min(), chance_level)) - 0.05))
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.7, n - 0.3)

    # Use the EEGDash chance-line primitive; it draws the dashed reference
    # line and registers a legend handle so the label stays inside the
    # panel even when the axes sit close to the figure right edge.
    chance_line(ax, level=float(chance_level))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"fold {i + 1}" for i in range(n)], fontsize=8.2, color=EEGDASH_INK
    )
    ax.set_ylabel("test accuracy", color=EEGDASH_INK, fontsize=9)
    ax.set_title(
        "baseline decoder per fold",
        fontsize=10.5,
        color=EEGDASH_INK,
        fontweight="bold",
        loc="left",
        pad=8,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(EEGDASH_GRID)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.legend(
        loc="upper right",
        fontsize=7.4,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def draw_challenge_1_figure(
    *,
    paradigm_schematic_data: Mapping,
    sample_window: np.ndarray,
    fold_accuracies: Sequence[float],
    dataset: str = "EEG2025 R5 mini",
    plot_id: str = "tutorial_challenge_1",
    chance_level: float = 0.5,
    n_subjects: int | None = None,
    task: str = "contrastChangeDetection",
    sfreq: float = 100.0,
):
    """Render the three-panel Challenge 1 starter-kit figure.

    Parameters
    ----------
    paradigm_schematic_data : mapping
        Inputs for the trial-structure schematic. Must carry the keys
        ``trace`` (1-D array, the EEG sparkline), ``sfreq``,
        ``shift_after_stim``, ``window_len``, ``stim_time``,
        ``response_time``.
    sample_window : numpy.ndarray
        ``(n_channels, n_samples)`` array; one CCD trial window after the
        stimulus anchor.
    fold_accuracies : sequence of float
        Test accuracy of the baseline decoder per cross-validation fold
        (or per held-out subject). The mean +/- std band is drawn from
        this list.
    dataset : str, default ``"EEG2025 R5 mini"``
        Free-form dataset descriptor for the subtitle.
    plot_id : str, default ``"tutorial_challenge_1"``
        Tutorial id forwarded to :func:`add_provenance_footer`.
    chance_level : float, default ``0.5``
        Chance/baseline level for the accuracy panel reference line.
    n_subjects : int or None, optional
        Subject count for the subtitle (when known from the live data).
    task : str, default ``"contrastChangeDetection"``
        Task name for the subtitle.
    sfreq : float, default ``100.0``
        Sampling frequency for the window-panel annotation.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure. The caller is responsible for ``plt.show()``.

    """
    fig = plt.figure(figsize=(15.6, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.30, 1.05, 1.20),
        wspace=0.40,
        left=0.05,
        right=0.93,
        top=0.78,
        bottom=0.18,
    )
    ax_paradigm = fig.add_subplot(gs[0, 0])
    ax_window = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[0, 2])

    _draw_paradigm_panel(ax_paradigm, paradigm_schematic_data)
    _draw_window_panel(ax_window, sample_window, sfreq=sfreq)
    _draw_accuracy_panel(ax_acc, fold_accuracies, chance_level=chance_level)

    accs = np.asarray(list(fold_accuracies), dtype=float)
    mean_acc = float(accs.mean()) if accs.size else 0.0
    n_chans, n_samples = (
        np.asarray(sample_window).shape
        if np.asarray(sample_window).ndim == 2
        else (0, 0)
    )

    subj_part = f"n_subjects={n_subjects}" if n_subjects is not None else None
    subtitle_bits = [
        f"dataset={dataset}",
        f"task={task}",
        subj_part,
        f"sample window=({n_chans}, {n_samples})",
        f"baseline_acc={mean_acc:.2f}",
    ]
    subtitle = " | ".join(b for b in subtitle_bits if b)

    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        citation="HBN R5 mini, Alexander et al. 2017",
        extra="EEG2025 Challenge 1 starter kit, downsampled 100 Hz",
    )
    style_figure(
        fig,
        title="Challenge 1 starter kit: load CCD, build a window, train a baseline",
        subtitle=subtitle,
        source=source,
        grid_axis="none",
    )
    return fig


__all__ = ["draw_challenge_1_figure"]
