# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""EEGDash data-visualization identity helpers.

The visual identity centres on the **Data Rail** — a thin EEGDash-blue
horizontal line near the top of every figure, with a short EEGDash-orange
pulse segment at its left edge. Tutorials call :func:`use_eegdash_style`
once at the top, then :func:`style_figure` per figure to get the rail,
title/subtitle/source band, and consistent spines/grids/ticks across
every axes.
"""

from __future__ import annotations

from itertools import cycle

EEGDASH_BLUE = "#006CA3"
EEGDASH_BLUE_DARK = "#004A76"
EEGDASH_ORANGE = "#F7941D"
EEGDASH_SKY = "#4F8CFF"
EEGDASH_MINT = "#22D3EE"
EEGDASH_PURPLE = "#7C3AED"
EEGDASH_AMBER = "#EAB308"
EEGDASH_CORAL = "#D65F5F"
EEGDASH_INK = "#102A43"
EEGDASH_MUTED = "#64748B"
EEGDASH_GRID = "#7A8CA0"
EEGDASH_SURFACE = "#F7FBFE"

EEGDASH_PALETTE = [
    EEGDASH_BLUE,
    EEGDASH_ORANGE,
    EEGDASH_SKY,
    EEGDASH_MINT,
    EEGDASH_PURPLE,
    EEGDASH_AMBER,
    EEGDASH_CORAL,
]

_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "axes.grid": True,
    "grid.color": EEGDASH_GRID,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    "text.color": EEGDASH_INK,
    "axes.labelcolor": EEGDASH_INK,
    "xtick.color": EEGDASH_INK,
    "ytick.color": EEGDASH_INK,
    "axes.titlecolor": EEGDASH_INK,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
}


def get_eegdash_palette(n: int) -> list[str]:
    """Return ``n`` colors from the EEGDash data-viz palette."""
    if n <= len(EEGDASH_PALETTE):
        return EEGDASH_PALETTE[:n]
    return [color for _, color in zip(range(n), cycle(EEGDASH_PALETTE))]


def use_eegdash_style() -> None:
    """Configure matplotlib (and seaborn if installed) for EEGDash plots."""
    try:
        import seaborn as sns

        sns.set_theme(style="white", palette=EEGDASH_PALETTE, font="sans-serif", rc=_RC)
    except Exception:
        import matplotlib as mpl
        from cycler import cycler

        mpl.rcParams.update(_RC)
        mpl.rcParams["axes.prop_cycle"] = cycler(color=EEGDASH_PALETTE)


def _draw_data_rail(fig) -> None:
    """Append the Data Rail rectangles to ``fig`` (idempotent)."""
    if getattr(fig, "_eegdash_data_rail", False):
        return

    import matplotlib.pyplot as plt

    fig.patches.append(
        plt.Rectangle(
            (0.08, 0.945),
            0.84,
            0.006,
            transform=fig.transFigure,
            clip_on=False,
            facecolor=EEGDASH_BLUE,
            edgecolor="none",
        )
    )
    fig.patches.append(
        plt.Rectangle(
            (0.08, 0.938),
            0.045,
            0.006,
            transform=fig.transFigure,
            clip_on=False,
            facecolor=EEGDASH_ORANGE,
            edgecolor="none",
        )
    )
    fig._eegdash_data_rail = True


def _style_axes(ax, *, grid_axis: str = "y") -> None:
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(EEGDASH_GRID)
    ax.spines["bottom"].set_linewidth(0.8)

    if grid_axis == "none":
        ax.grid(False)
    elif grid_axis == "both":
        ax.grid(True, axis="both", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    elif grid_axis == "x":
        ax.grid(True, axis="x", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
        ax.grid(False, axis="y")
    else:
        ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
        ax.grid(False, axis="x")

    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)
    ax.set_title("")


def style_figure(
    fig,
    *,
    title: str,
    subtitle: str = "",
    source: str = "",
    data_rail: bool = True,
    grid_axis: str = "y",
) -> None:
    """Apply the EEGDash identity to every axes in ``fig``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    title : str
        Short, 1-line figure title (top-left).
    subtitle : str
        Dataset/task/split context line.
    source : str
        Provenance footer (italic, bottom-left).
    data_rail : bool, default ``True``
        Attach the Data Rail.
    grid_axis : {"y", "x", "both", "none"}
        Grid orientation.

    """
    if fig is None:
        return

    for ax in list(getattr(fig, "axes", []) or []):
        _style_axes(ax, grid_axis=grid_axis)
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.12, right=0.95)

    if data_rail:
        _draw_data_rail(fig)

    if title:
        fig.text(
            0.08,
            0.93,
            title,
            fontsize=14,
            fontweight="bold",
            color=EEGDASH_INK,
            ha="left",
            va="top",
        )
    if subtitle:
        fig.text(
            0.08, 0.88, subtitle, fontsize=10, color=EEGDASH_MUTED, ha="left", va="top"
        )
    if source:
        fig.text(
            0.08,
            0.018,
            source,
            fontsize=8,
            color=EEGDASH_MUTED,
            ha="left",
            va="bottom",
            style="italic",
        )


def chance_line(ax, level: float, *, label: str = "chance"):
    """Add a horizontal dashed reference line at ``level``."""
    if ax is None:
        return None

    line = ax.axhline(
        level,
        color=EEGDASH_MUTED,
        linestyle="--",
        linewidth=0.8,
        label=label,
        zorder=1.5,
    )
    try:
        _, xmax = ax.get_xlim()
    except Exception:
        xmax = 1.0
    ax.annotate(
        f"{label} = {level:.2f}",
        xy=(xmax, level),
        xycoords="data",
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        color=EEGDASH_MUTED,
    )
    return line


__all__ = [
    "EEGDASH_BLUE",
    "EEGDASH_BLUE_DARK",
    "EEGDASH_ORANGE",
    "EEGDASH_SKY",
    "EEGDASH_MINT",
    "EEGDASH_PURPLE",
    "EEGDASH_AMBER",
    "EEGDASH_CORAL",
    "EEGDASH_INK",
    "EEGDASH_MUTED",
    "EEGDASH_GRID",
    "EEGDASH_SURFACE",
    "EEGDASH_PALETTE",
    "get_eegdash_palette",
    "use_eegdash_style",
    "style_figure",
    "chance_line",
]
