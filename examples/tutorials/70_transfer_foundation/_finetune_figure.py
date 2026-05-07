"""Drawing helpers for the ``plot_73`` fine-tune comparison figure.

Sibling module to ``plot_73_finetune_pretrained_model.py``. The leading
underscore tells sphinx-gallery to skip this file when building the
gallery, so the rendering plumbing stays out of the rendered tutorial;
the tutorial imports the public :func:`draw_finetune_figure` entry
point.

The figure has three panels arranged left-to-right:

1. Validation curves: per-epoch validation accuracy across seeds for
   from-scratch, linear-probe, and full-finetune. Mean is solid, +/-1
   std is shaded. The chance line sits underneath as a muted dashed
   reference.
2. Final-accuracy bars: the same three regimes plotted on the same
   data axis as panel 1. Each bar is annotated with its absolute
   accuracy and the gain over from-scratch.
3. Parameter cost vs accuracy scatter: trainable parameter count on a
   log x-axis vs final accuracy. The cheap-but-good zone (upper-left)
   is annotated; from-scratch and full-finetune cluster at high
   parameter counts, linear-probe sits to the left.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_SURFACE,
    chance_line,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# Regime order is fixed so colors and annotations stay consistent across
# the three panels. The mapping mirrors the tutorial's narrative:
# from-scratch is the no-transfer baseline (dark blue), linear-probe is
# the cheap fine-tune (mid blue), full-finetune is the expensive one
# (orange). Choosing dark-blue / mid-blue / orange keeps the figure
# colorblind-readable and grayscale-distinguishable.
_REGIME_ORDER = ("from-scratch", "linear-probe", "full-finetune")
_REGIME_COLORS = {
    "from-scratch": EEGDASH_BLUE_DARK,
    "linear-probe": EEGDASH_BLUE,
    "full-finetune": EEGDASH_ORANGE,
}
_REGIME_MARKERS = {
    "from-scratch": "o",
    "linear-probe": "s",
    "full-finetune": "D",
}


def _curve_stats(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-epoch mean and std for a ``(n_seeds, n_epochs)`` array."""
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected (n_seeds, n_epochs); got shape {arr.shape}")
    return arr.mean(axis=0), arr.std(axis=0, ddof=0)


def _draw_curves_panel(
    ax,
    *,
    epochs: Sequence[int],
    curves: dict[str, np.ndarray],
    chance_level: float,
) -> None:
    """Render the per-epoch validation-accuracy curves with std bands."""
    epochs = np.asarray(epochs)
    # Set xlim before drawing the chance line so its right-edge text
    # annotation lands past the last training-epoch tick instead of
    # crowding the rightmost data point.
    ax.set_xlim(epochs.min() - 0.5, epochs.max() + 0.5)
    chance_line(ax, level=float(chance_level), label="chance")
    for name in _REGIME_ORDER:
        if name not in curves:
            continue
        mean, std = _curve_stats(curves[name])
        color = _REGIME_COLORS[name]
        ax.fill_between(
            epochs,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            epochs,
            mean,
            color=color,
            linewidth=2.0,
            marker=_REGIME_MARKERS[name],
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=name,
            zorder=3,
        )

    ax.set_xlabel("training epoch")
    ax.set_ylabel("validation accuracy")
    # Y range matches panel 2 so the eye reads both panels on one scale.
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks(epochs)
    ax.text(
        0.0,
        1.06,
        "Validation accuracy across epochs",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.legend(
        loc="lower right",
        fontsize=7.6,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
        handletextpad=0.4,
    )


def _draw_bars_panel(
    ax,
    *,
    final_accuracies: dict[str, float],
    chance_level: float,
) -> None:
    """Render the final-accuracy bars on the same data axis as panel 1."""
    names = [n for n in _REGIME_ORDER if n in final_accuracies]
    accs = np.asarray([float(final_accuracies[n]) for n in names])
    positions = np.arange(len(names))
    base = float(final_accuracies.get("from-scratch", float("nan")))

    # Set xlim before drawing the chance line so its right-edge annotation
    # lands cleanly past the last bar without overlapping it.
    ax.set_xlim(positions.min() - 0.6, positions.max() + 0.6)
    chance_line(ax, level=float(chance_level), label="chance")

    bars = ax.bar(
        positions,
        accs,
        width=0.62,
        color=[_REGIME_COLORS[n] for n in names],
        edgecolor=EEGDASH_INK,
        linewidth=0.8,
        zorder=3,
    )
    for bar, name, acc in zip(bars, names, accs):
        gain = acc - base if not np.isnan(base) else float("nan")
        # Absolute accuracy stays inside the bar so it remains legible
        # whether the bar is short or tall.
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(acc - 0.04, 0.96),
            f"{acc:.2f}",
            ha="center",
            va="top",
            fontsize=9.0,
            fontweight="bold",
            color="white" if acc >= 0.18 else EEGDASH_INK,
        )
        # Gain caption sits just above the bar.
        if name == "from-scratch":
            gain_text = "baseline"
        elif np.isnan(gain):
            gain_text = ""
        else:
            sign = "+" if gain >= 0 else ""
            gain_text = f"{sign}{gain:.2f} vs scratch"
        if gain_text:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                acc + 0.02,
                gain_text,
                ha="center",
                va="bottom",
                fontsize=8.4,
                color=EEGDASH_INK,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=9.0, color=EEGDASH_INK)
    # Top headroom keeps the gain captions from clipping into the panel
    # title; the headroom is consistent across the bar and curve panels.
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("final validation accuracy")
    ax.text(
        0.0,
        1.06,
        "Final accuracy by regime",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.legend(loc="lower right", fontsize=7.6, frameon=False)


def _draw_scatter_panel(
    ax,
    *,
    trainable_params: dict[str, int],
    final_accuracies: dict[str, float],
) -> None:
    """Render the trainable-parameter vs final-accuracy scatter."""
    names = [
        n for n in _REGIME_ORDER if n in trainable_params and n in final_accuracies
    ]
    if not names:
        return
    xs = np.asarray([max(int(trainable_params[n]), 1) for n in names])
    ys = np.asarray([float(final_accuracies[n]) for n in names])

    # Compute axis bounds before drawing the cheap-but-good band so the
    # band fills the visible area cleanly. The x bound is in log space
    # with a 30 percent gutter on each side so labels do not clip.
    log_xs = np.log10(xs)
    log_pad = 0.4
    xmin = 10 ** (log_xs.min() - log_pad)
    xmax = 10 ** (log_xs.max() + log_pad)
    ymin, ymax = 0.0, 1.02
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xscale("log")

    # The cheap-but-good zone is "few trainable params AND high
    # accuracy". Visualise it as a soft surface-colored band in the
    # upper-left quadrant; the median trainable count splits the panel.
    median_x = 10 ** (log_xs.min() + 0.5 * (log_xs.max() - log_xs.min()))
    midline_y = 0.5
    ax.add_patch(
        FancyBboxPatch(
            (xmin, midline_y),
            median_x - xmin,
            ymax - midline_y,
            boxstyle="round,pad=0.001,rounding_size=0.002",
            facecolor=EEGDASH_SURFACE,
            edgecolor=EEGDASH_MUTED,
            linewidth=0.8,
            linestyle="--",
            alpha=0.8,
            zorder=1,
        )
    )
    # Place the zone label inside the band, tucked against the upper
    # boundary so it never collides with the linear-probe data label.
    ax.text(
        np.sqrt(xmin * median_x),
        midline_y + 0.04,
        "cheap-but-good zone",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=EEGDASH_MUTED,
        fontstyle="italic",
        zorder=2,
    )

    # Per-regime label offset chosen so labels do not overlap the band
    # label or each other. linear-probe labels above its marker (clear
    # of the band), full-finetune labels below-right of its marker, and
    # from-scratch labels above its marker.
    label_offsets = {
        "linear-probe": (0, 12, "center", "bottom"),
        "from-scratch": (-12, -2, "right", "center"),
        "full-finetune": (-12, 2, "right", "center"),
    }
    # Scatter the regime points last so their markers sit on top of the
    # band annotation.
    for name, x, y in zip(names, xs, ys):
        ax.scatter(
            x,
            y,
            s=160,
            facecolor=_REGIME_COLORS[name],
            edgecolor=EEGDASH_INK,
            linewidth=1.0,
            marker=_REGIME_MARKERS[name],
            zorder=4,
            label=name,
        )
        dx, dy, ha, va = label_offsets.get(name, (8, 8, "left", "bottom"))
        ax.annotate(
            f"{name}\n{int(x):,} params, {y:.2f}",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            color=EEGDASH_INK,
            ha=ha,
            va=va,
        )

    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("final validation accuracy")
    ax.text(
        0.0,
        1.06,
        "Parameter cost vs accuracy",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )


def draw_finetune_figure(
    *,
    epochs: Sequence[int],
    scratch_curve: np.ndarray,
    probe_curve: np.ndarray,
    finetune_curve: np.ndarray,
    final_accuracies: dict[str, float],
    trainable_params: dict[str, int],
    chance_level: float,
    pretrain_task: str,
    target_task: str,
    n_train_subjects: int,
    n_test_subjects: int,
    plot_id: str = "plot_73",
) -> plt.Figure:
    """Render the three-panel fine-tune comparison figure.

    Parameters
    ----------
    epochs : sequence of int
        Epoch indices on the x axis of panel 1 (e.g. ``[1, 2, 3, 4]``).
    scratch_curve, probe_curve, finetune_curve : numpy.ndarray
        ``(n_seeds, n_epochs)`` validation-accuracy curves for the three
        regimes. Mean +/- std across seeds is rendered as a shaded band.
    final_accuracies : dict[str, float]
        Final validation accuracy per regime. Keys must include
        ``"from-scratch"``, ``"linear-probe"``, and ``"full-finetune"``.
    trainable_params : dict[str, int]
        Trainable parameter count per regime, same key set.
    chance_level : float
        Majority-vote chance accuracy on the held-out subjects.
    pretrain_task, target_task : str
        Task names for the figure subtitle (live runtime values).
    n_train_subjects, n_test_subjects : int
        Subject counts on either side of the cross-subject split.
    plot_id : str, default ``"plot_73"``
        Tutorial id forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure

    """
    curves = {
        "from-scratch": np.asarray(scratch_curve, dtype=float),
        "linear-probe": np.asarray(probe_curve, dtype=float),
        "full-finetune": np.asarray(finetune_curve, dtype=float),
    }

    fig = plt.figure(figsize=(14.0, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.20, 0.95, 1.10),
        wspace=0.42,
    )
    ax_curves = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[0, 2])

    _draw_curves_panel(
        ax_curves,
        epochs=epochs,
        curves=curves,
        chance_level=chance_level,
    )
    _draw_bars_panel(
        ax_bars,
        final_accuracies=final_accuracies,
        chance_level=chance_level,
    )
    _draw_scatter_panel(
        ax_scatter,
        trainable_params=trainable_params,
        final_accuracies=final_accuracies,
    )

    # Subtitle threads the live runtime values: which task pretrained the
    # encoder, which task it was adapted to, the train/test subject
    # counts, and which regime came out on top.
    best_regime = max(final_accuracies, key=final_accuracies.get)
    subtitle = (
        f"pretrain={pretrain_task} | target={target_task} | "
        f"n_train_subjects={n_train_subjects} | "
        f"n_test_subjects={n_test_subjects} | "
        f"best_regime={best_regime}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        citation="synthetic source + target | seeded across runs",
    )
    style_figure(
        fig,
        title="Which fine-tune regime wins on a small EEG target task?",
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )

    fig.subplots_adjust(top=0.74, bottom=0.18, left=0.07, right=0.97)
    return fig


__all__ = ["draw_finetune_figure"]
