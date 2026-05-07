"""Drawing helpers for the ``plot_53`` learning-curves figure.

Sibling module to ``plot_53_learning_curves.py``. The leading underscore
tells sphinx-gallery to skip this file when building the gallery (see
``docs/source/conf.py`` ``ignore_pattern``); the rendered tutorial only
imports the public ``draw_learning_curves_figure`` entry point.

The figure is a 1x2 plate that reads like a textbook learning curve:

1. *Training-size x accuracy curve* (left) -- balanced accuracy on the
   training fold (dashed blue) and on the held-out validation fold
   (solid orange) as the training set grows. ``+/- 1 std`` bands across
   ``n_perms`` repeats sit behind each curve. A horizontal chance line
   anchors the y-axis. Two callouts mark the saturation point where val
   accuracy reaches 90% of its plateau, and the plateau itself.
2. *Train-minus-val gap* (right) -- the same x-axis, with
   ``train_acc - val_acc`` on the y-axis. A widening gap is overfitting;
   a stable gap is a healthy decoder. The bias-variance trade-off lives
   here.

Geometry, colours, and annotation conventions follow the rest of the
gallery (see ``_alpha_figure.py`` and ``_p300_figure.py``) so the plate
sits visually next to its 50-track siblings.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    chance_line,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# ---------------------------------------------------------------------------
# Internal panel painters
# ---------------------------------------------------------------------------


def _style_axis(ax) -> None:
    """Apply the EEGDash spine/tick treatment to a single axes."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(EEGDASH_GRID)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


def _draw_curve_panel(
    ax,
    *,
    train_sizes: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    val_mean: np.ndarray,
    val_std: np.ndarray,
    plateau_point: tuple[int, float],
    saturation_point: tuple[int, float] | None,
    chance_level: float,
    n_perms: int,
) -> None:
    """Train + val accuracy versus training-set size (log x)."""
    xs = np.asarray(train_sizes, dtype=float)
    # Train curve: dashed blue.
    ax.plot(
        xs,
        train_mean,
        color=EEGDASH_BLUE,
        linestyle="--",
        linewidth=1.6,
        marker="o",
        markersize=4.0,
        label=f"train (n_perms={n_perms})",
        zorder=4,
    )
    ax.fill_between(
        xs,
        train_mean - train_std,
        train_mean + train_std,
        color=EEGDASH_BLUE,
        alpha=0.14,
        linewidth=0,
        zorder=2,
    )
    # Val curve: solid orange.
    ax.plot(
        xs,
        val_mean,
        color=EEGDASH_ORANGE,
        linestyle="-",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        label=f"validation (n_perms={n_perms})",
        zorder=5,
    )
    ax.fill_between(
        xs,
        val_mean - val_std,
        val_mean + val_std,
        color=EEGDASH_ORANGE,
        alpha=0.18,
        linewidth=0,
        zorder=3,
    )
    chance_line(ax, level=float(chance_level), label="chance")

    # Plateau callout: thin arrow + monospace pill above the curve.
    plateau_n, plateau_acc = plateau_point
    ax.annotate(
        f"plateau\nval={plateau_acc:.2f} @ n={int(plateau_n)}",
        xy=(float(plateau_n), float(plateau_acc)),
        xytext=(0.55, 0.86),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=8.4,
        family="monospace",
        color=EEGDASH_INK,
        arrowprops={
            "arrowstyle": "->",
            "mutation_scale": 8,
            "color": EEGDASH_INK,
            "linewidth": 0.7,
            "shrinkA": 2,
            "shrinkB": 6,
        },
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": EEGDASH_ORANGE,
            "linewidth": 0.6,
        },
        zorder=8,
    )

    # 90% saturation callout (when distinct from the plateau).
    if saturation_point is not None and saturation_point[0] != plateau_point[0]:
        sat_n, sat_acc = saturation_point
        ax.annotate(
            f"90% of plateau\n@ n={int(sat_n)}",
            xy=(float(sat_n), float(sat_acc)),
            xytext=(0.04, 0.18),
            textcoords="axes fraction",
            ha="left",
            va="center",
            fontsize=8.2,
            family="monospace",
            color=EEGDASH_INK,
            arrowprops={
                "arrowstyle": "->",
                "mutation_scale": 8,
                "color": EEGDASH_MUTED,
                "linewidth": 0.7,
                "shrinkA": 2,
                "shrinkB": 6,
            },
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": EEGDASH_MUTED,
                "linewidth": 0.6,
            },
            zorder=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("training-set size (windows, log scale)", color=EEGDASH_INK)
    ax.set_ylabel("balanced accuracy", color=EEGDASH_INK)
    ax.set_xlim(xs.min() * 0.85, xs.max() * 1.18)
    y_low = max(0.0, min(float(np.min(val_mean - val_std)), float(chance_level)) - 0.04)
    y_high = min(1.02, float(np.max(train_mean + train_std)) + 0.06)
    ax.set_ylim(y_low, y_high)
    ax.set_title(
        "Accuracy vs training-set size",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=False, fontsize=8.4
    )


def _draw_gap_panel(
    ax,
    *,
    train_sizes: np.ndarray,
    train_mean: np.ndarray,
    val_mean: np.ndarray,
    train_std: np.ndarray,
    val_std: np.ndarray,
) -> None:
    """``train_acc - val_acc`` versus training-set size (log x)."""
    xs = np.asarray(train_sizes, dtype=float)
    gap_mean = train_mean - val_mean
    # Std of a difference with independent permutations: sqrt(var_a + var_b).
    gap_std = np.sqrt(train_std**2 + val_std**2)

    ax.axhline(
        0.0,
        color=EEGDASH_MUTED,
        linestyle="--",
        linewidth=0.9,
        zorder=1.5,
    )
    ax.plot(
        xs,
        gap_mean,
        color=EEGDASH_BLUE_DARK,
        linewidth=1.8,
        marker="o",
        markersize=4.5,
        label="train - val (mean)",
        zorder=4,
    )
    ax.fill_between(
        xs,
        gap_mean - gap_std,
        gap_mean + gap_std,
        color=EEGDASH_BLUE_DARK,
        alpha=0.14,
        linewidth=0,
        zorder=2,
    )

    # Annotate first and last points with their values so the trend reads
    # without scanning the y-axis.
    for idx in (0, len(xs) - 1):
        ax.text(
            float(xs[idx]),
            float(gap_mean[idx]) + 0.013,
            f"{gap_mean[idx]:+.2f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            family="monospace",
            color=EEGDASH_INK,
            zorder=6,
        )

    # Verdict pill: "shrinks" / "stable" / "widens" based on the slope from
    # the smallest training size to the largest.
    delta = float(gap_mean[-1] - gap_mean[0])
    if delta < -0.02:
        verdict = "gap shrinks: overfitting recedes"
        accent = EEGDASH_BLUE
    elif delta > 0.02:
        verdict = "gap widens: more data is making it worse"
        accent = EEGDASH_ORANGE
    else:
        verdict = "gap stable: bias-limited"
        accent = EEGDASH_INK
    ax.text(
        0.97,
        0.97,
        verdict,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        family="monospace",
        color=accent,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": accent,
            "linewidth": 0.6,
        },
        zorder=6,
    )

    ax.set_xscale("log")
    ax.set_xlabel("training-set size (windows, log scale)", color=EEGDASH_INK)
    ax.set_ylabel("train - val (balanced accuracy)", color=EEGDASH_INK)
    ax.set_xlim(xs.min() * 0.85, xs.max() * 1.18)
    y_low = min(-0.05, float(np.min(gap_mean - gap_std)) - 0.02)
    y_high = max(0.05, float(np.max(gap_mean + gap_std)) + 0.04)
    ax.set_ylim(y_low, y_high)
    ax.set_title(
        "Bias-variance gap: train minus val",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def draw_learning_curves_figure(
    *,
    train_sizes: Iterable[int],
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    plateau_point: tuple[int, float],
    saturation_point: tuple[int, float] | None = None,
    chance_level: float = 0.5,
    n_total_windows: int,
    n_subjects: int,
    n_features: int,
    plot_id: str = "plot_53",
):
    """Render the learning-curves plate and return the Figure.

    Parameters
    ----------
    train_sizes : iterable of int
        Absolute training-set sizes (windows) sampled along the curve.
        Drawn on a log x-axis.
    train_scores, val_scores : ndarray, shape (n_sizes, n_perms)
        Per-size balanced accuracy across ``n_perms`` repeats. ``learning_curve``
        from sklearn returns the same layout.
    plateau_point : tuple of (int, float)
        ``(n_train, balanced_accuracy)`` at the plateau marker. Annotated
        as the upper-right callout on the left panel.
    saturation_point : tuple of (int, float), optional
        ``(n_train, balanced_accuracy)`` at the size where val accuracy
        first reaches 90% of its plateau. When omitted (or equal to the
        plateau), the second annotation is suppressed.
    chance_level : float, default ``0.5``
        Drawn as a horizontal dashed line on the left panel.
    n_total_windows : int
        Total window count in the cohort (for the subtitle).
    n_subjects : int
        Subject count in the cohort (for the subtitle).
    n_features : int
        Feature dimension (for the subtitle).
    plot_id : str, default ``"plot_53"``
        Tutorial id used in the provenance footer string.

    Returns
    -------
    matplotlib.figure.Figure

    """
    train_sizes_arr = np.asarray(list(train_sizes), dtype=int)
    train_scores = np.asarray(train_scores)
    val_scores = np.asarray(val_scores)
    if train_scores.shape != val_scores.shape:
        raise ValueError(
            "train_scores and val_scores must share shape; got "
            f"{train_scores.shape} vs {val_scores.shape}."
        )
    if train_scores.shape[0] != train_sizes_arr.size:
        raise ValueError(
            f"len(train_sizes)={train_sizes_arr.size} does not match "
            f"train_scores.shape[0]={train_scores.shape[0]}."
        )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    n_perms = int(train_scores.shape[1])

    fig = plt.figure(figsize=(11.6, 4.6))
    gs = fig.add_gridspec(
        1,
        2,
        wspace=0.32,
        left=0.07,
        right=0.93,
        top=0.78,
        bottom=0.20,
        width_ratios=[1.0, 1.0],
    )
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_gap = fig.add_subplot(gs[0, 1])

    _draw_curve_panel(
        ax_curve,
        train_sizes=train_sizes_arr,
        train_mean=train_mean,
        train_std=train_std,
        val_mean=val_mean,
        val_std=val_std,
        plateau_point=plateau_point,
        saturation_point=saturation_point,
        chance_level=float(chance_level),
        n_perms=n_perms,
    )
    _draw_gap_panel(
        ax_gap,
        train_sizes=train_sizes_arr,
        train_mean=train_mean,
        val_mean=val_mean,
        train_std=train_std,
        val_std=val_std,
    )

    plateau_n, plateau_acc = plateau_point
    subtitle = (
        f"n_total_windows={int(n_total_windows):,} | "
        f"n_subjects={int(n_subjects)} | "
        f"n_features={int(n_features)} | "
        f"train sizes {int(train_sizes_arr.min())}..{int(train_sizes_arr.max())} | "
        f"plateau {plateau_acc:.2f} at n_train={int(plateau_n)}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=None,
        citation="synthetic split manifest",
        extra=(
            "scoring=balanced_accuracy; "
            "Hoffmann et al. 2014 (sample size in BCI); "
            "Schirrmeister et al. 2017 (Braindecode)"
        ),
    )
    style_figure(
        fig,
        title="Where does decoding accuracy plateau as the training set grows?",
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )
    return fig


__all__ = ["draw_learning_curves_figure"]
