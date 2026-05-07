# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""EEGDash data-visualization identity helpers.

This module defines the EEGDash visual identity for documentation plots,
tutorials, dataset/task cards, and benchmark outputs. The core motif is the
**Data Rail**: a thin EEGDash-blue horizontal line near the top of the figure
with a short EEGDash-orange pulse segment at its left edge.

The helpers here are intentionally stand-alone -- they only require
matplotlib at runtime. Seaborn-aware defaults are loaded if seaborn is
installed; otherwise the helpers fall back to plain matplotlib rcParams so
tutorials can run without seaborn as a dependency.
"""

from __future__ import annotations

from html import escape
from itertools import cycle
from typing import Iterable

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

EEGDASH_STATUS_COLORS = {
    "ready": EEGDASH_MINT,
    "warning": EEGDASH_ORANGE,
    "risk": EEGDASH_CORAL,
    "unknown": EEGDASH_MUTED,
    "selected": EEGDASH_BLUE,
    "baseline": EEGDASH_AMBER,
}

EEGDASH_VIZ_CSS = f"""\
:root {{
  --eegdash-viz-blue: {EEGDASH_BLUE};
  --eegdash-viz-blue-dark: {EEGDASH_BLUE_DARK};
  --eegdash-viz-orange: {EEGDASH_ORANGE};
  --eegdash-viz-sky: {EEGDASH_SKY};
  --eegdash-viz-mint: {EEGDASH_MINT};
  --eegdash-viz-ink: {EEGDASH_INK};
  --eegdash-viz-muted: {EEGDASH_MUTED};
  --eegdash-viz-grid: {EEGDASH_GRID};
  --eegdash-viz-surface: {EEGDASH_SURFACE};
}}

.eegdash-viz-card {{
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(0, 108, 163, 0.18);
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff 0%, var(--eegdash-viz-surface) 100%);
  box-shadow: 0 10px 24px rgba(16, 42, 67, 0.08);
  padding: 1rem 1rem 0.9rem;
  color: var(--eegdash-viz-ink);
}}

.eegdash-viz-card::before {{
  content: "";
  position: absolute;
  inset: 0 0 auto 0;
  height: 4px;
  background:
    linear-gradient(90deg,
      var(--eegdash-viz-orange) 0 36px,
      transparent 36px 44px,
      var(--eegdash-viz-blue) 44px 100%);
}}

.eegdash-viz-kicker {{
  margin: 0 0 0.28rem;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--eegdash-viz-blue);
}}

.eegdash-viz-title {{
  margin: 0;
  font-size: 1.18rem;
  line-height: 1.2;
  font-weight: 760;
  color: var(--eegdash-viz-ink);
}}

.eegdash-viz-body {{
  margin: 0.7rem 0 0;
  color: var(--eegdash-viz-muted);
  line-height: 1.45;
}}

.eegdash-viz-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.8rem;
}}

.eegdash-viz-pill {{
  display: inline-flex;
  align-items: center;
  border-radius: 4px;
  border: 1px solid rgba(0, 108, 163, 0.2);
  background: rgba(0, 108, 163, 0.08);
  color: var(--eegdash-viz-blue-dark);
  padding: 0.15rem 0.45rem;
  font-size: 0.72rem;
  font-weight: 650;
}}
"""


def get_eegdash_palette(n: int) -> list[str]:
    """Return ``n`` colors from the EEGDash data-viz palette."""
    if n <= len(EEGDASH_PALETTE):
        return EEGDASH_PALETTE[:n]
    return [color for _, color in zip(range(n), cycle(EEGDASH_PALETTE))]


# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------

# Helvetica/Arial fallback chain. Matplotlib resolves the first installed
# family, so providing several alternatives keeps the contract honoured even
# on systems that lack the platform-default sans-serif fonts.
_EEGDASH_FONT_FAMILY = ["Helvetica", "Arial", "DejaVu Sans"]


def _matplotlib_rc_dict() -> dict:
    """Return the rcParams dictionary for the EEGDash identity."""
    return {
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
        "font.sans-serif": _EEGDASH_FONT_FAMILY,
    }


def set_eegdash_matplotlib_defaults() -> None:
    """Configure Matplotlib (and Seaborn if installed) for EEGDash plots.

    If seaborn is available, this calls :func:`seaborn.set_theme` so seaborn
    aware tutorials inherit the theme; otherwise it falls back to plain
    matplotlib ``rcParams`` updates so the helpers do not require seaborn.
    """
    rc = _matplotlib_rc_dict()
    try:  # pragma: no cover - exercised by tutorials with seaborn installed
        import seaborn as sns

        sns.set_theme(
            style="white",
            palette=EEGDASH_PALETTE,
            font="sans-serif",
            rc=rc,
        )
    except Exception:  # pragma: no cover - exercised when seaborn missing
        import matplotlib as mpl

        mpl.rcParams.update(rc)
        # Categorical color cycle aligned with the EEGDash palette.
        from cycler import cycler

        mpl.rcParams["axes.prop_cycle"] = cycler(color=EEGDASH_PALETTE)


def use_eegdash_style() -> None:
    """One-call setup for tutorial code.

    Configures matplotlib rcParams, the EEGDash categorical palette, and the
    Data Rail defaults. Tolerates the absence of seaborn -- tutorials only
    need matplotlib to use this helper.
    """
    set_eegdash_matplotlib_defaults()


# ---------------------------------------------------------------------------
# Per-axes styling
# ---------------------------------------------------------------------------


def _draw_data_rail(fig) -> None:
    """Append the EEGDash Data Rail rectangles to ``fig``.

    Idempotent: if a Data Rail has already been attached to this figure (we
    flag it via ``fig._eegdash_data_rail``), the call is a no-op.
    """
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


def apply_eegdash_matplotlib_style(
    ax,
    *,
    title: str = "",
    subtitle: str = "",
    source: str = "",
    data_rail: bool = True,
    grid_axis: str = "y",
) -> None:
    """Apply the EEGDash Data Rail identity to a single Matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    title, subtitle, source : str
        Optional figure-level annotations placed in the upper-left and
        lower-left corners of the figure (not the axes).
    data_rail : bool, default ``True``
        Attach the Data Rail rectangles to the figure once.
    grid_axis : {"y", "x", "both", "none"}
        Grid orientation.

    """
    fig = ax.get_figure()

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

    if data_rail:
        _draw_data_rail(fig)

    if title:
        fig.text(
            0.08,
            0.922,
            title,
            fontsize=14,
            fontweight="bold",
            color=EEGDASH_INK,
            ha="left",
            va="top",
        )

    if subtitle:
        fig.text(
            0.08,
            0.89,
            subtitle,
            fontsize=10,
            color=EEGDASH_MUTED,
            ha="left",
            va="top",
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

    ax.set_title("")


# ---------------------------------------------------------------------------
# Figure-level helpers
# ---------------------------------------------------------------------------


def style_figure(
    fig,
    *,
    title: str,
    subtitle: str = "",
    source: str = "",
    data_rail: bool = True,
    grid_axis: str = "y",
) -> None:
    """Apply the EEGDash identity to *every* axes in ``fig``.

    Intended for tutorial code: instead of calling
    :func:`apply_eegdash_matplotlib_style` once per axes, pass the figure and
    a single title/subtitle/source block. The Data Rail and the title/subtitle
    /source footer are attached to the figure (not duplicated per axes), and
    each child axes gets the spine/grid/tick treatment.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    title : str
        Short, 1-line figure title (top-left).
    subtitle : str
        Dataset/task/split context line (e.g., ``"ds002718 | n_subjects=18"``).
    source : str
        Provenance footer (e.g., ``"EEGDash plot_10 | doi:10.18112/openneuro.ds002718"``).
    data_rail : bool, default ``True``
        Attach the Data Rail.
    grid_axis : {"y", "x", "both", "none"}
        Grid orientation forwarded to the per-axes call.

    """
    if fig is None:
        return

    children = list(getattr(fig, "axes", []) or [])
    if not children:
        # Even axes-less figures should keep the Data Rail and footer when
        # explicitly requested, so we still annotate the figure below.
        children = []

    # First pass: spines + grid + ticks for every axes.
    for ax in children:
        apply_eegdash_matplotlib_style(
            ax,
            data_rail=False,
            grid_axis=grid_axis,
        )

    # Second pass: figure-level annotations (Data Rail + title/subtitle/source)
    # attached to the figure exactly once.
    if data_rail:
        _draw_data_rail(fig)

    if title:
        fig.text(
            0.08,
            0.922,
            title,
            fontsize=14,
            fontweight="bold",
            color=EEGDASH_INK,
            ha="left",
            va="top",
        )

    if subtitle:
        fig.text(
            0.08,
            0.89,
            subtitle,
            fontsize=10,
            color=EEGDASH_MUTED,
            ha="left",
            va="top",
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


def chance_line(ax, level: float, *, label: str = "chance") -> None:
    """Add a horizontal dashed reference line at ``level`` (in [0, 1]).

    Used to mark the chance level (or any baseline) on accuracy/score plots.
    The line is drawn with the EEGDash muted slate color and a dashed style;
    a small right-side text marker annotates the level so the line remains
    interpretable when rendered in grayscale or by colorblind readers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.
    level : float
        Chance/baseline level on the axes' y-units (typically 0..1).
    label : str, default ``"chance"``
        Legend label and annotation text.

    """
    if ax is None:
        return

    line = ax.axhline(
        level,
        color=EEGDASH_MUTED,
        linestyle="--",
        linewidth=0.8,
        label=label,
        zorder=1.5,
    )

    # Annotate the level to the right of the axes so the chance line is
    # readable without referring to the legend.
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


# ---------------------------------------------------------------------------
# Plotly + HTML helpers (preserved for the docs site)
# ---------------------------------------------------------------------------


def apply_eegdash_plotly_layout(
    fig,
    *,
    title: str = "",
    subtitle: str = "",
    source: str = "",
    width: int | None = None,
    height: int | None = None,
):
    """Apply the EEGDash Data Rail identity to a Plotly figure."""
    layout_updates = {
        "template": "plotly_white",
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Source Sans 3, Arial, sans-serif", "color": EEGDASH_INK},
        "margin": {"l": 72, "r": 36, "t": 118, "b": 70},
        "colorway": EEGDASH_PALETTE,
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "linecolor": EEGDASH_GRID,
            "tickcolor": EEGDASH_GRID,
        },
        "yaxis": {
            "gridcolor": "rgba(122, 140, 160, 0.28)",
            "zeroline": False,
            "linecolor": EEGDASH_GRID,
            "tickcolor": EEGDASH_GRID,
        },
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    }
    if width is not None:
        layout_updates["width"] = width
    if height is not None:
        layout_updates["height"] = height

    fig.update_layout(**layout_updates)

    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=1.13,
        y1=1.13,
        line={"color": EEGDASH_BLUE, "width": 4},
    )
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        x1=0.06,
        y0=1.105,
        y1=1.105,
        line={"color": EEGDASH_ORANGE, "width": 4},
    )

    annotations = []
    if title:
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.09,
                "showarrow": False,
                "text": f"<b>{escape(title)}</b>",
                "align": "left",
                "xanchor": "left",
                "font": {"size": 20, "color": EEGDASH_INK},
            }
        )
    if subtitle:
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.04,
                "showarrow": False,
                "text": escape(subtitle),
                "align": "left",
                "xanchor": "left",
                "font": {"size": 13, "color": EEGDASH_MUTED},
            }
        )
    if source:
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.16,
                "showarrow": False,
                "text": escape(source),
                "align": "left",
                "xanchor": "left",
                "font": {"size": 10, "color": EEGDASH_MUTED},
            }
        )
    if annotations:
        fig.update_layout(annotations=list(fig.layout.annotations or []) + annotations)

    return fig


def eegdash_viz_card_html(
    *,
    title: str,
    kicker: str,
    body: str,
    meta: Iterable[str] | None = None,
) -> str:
    """Return a compact HTML card using the EEGDash Data Rail identity."""
    meta_html = ""
    if meta:
        pills = "".join(
            f'<span class="eegdash-viz-pill">{escape(str(item))}</span>'
            for item in meta
        )
        meta_html = f'<div class="eegdash-viz-meta">{pills}</div>'

    return (
        '<div class="eegdash-viz-card">'
        f'<p class="eegdash-viz-kicker">{escape(kicker)}</p>'
        f'<h3 class="eegdash-viz-title">{escape(title)}</h3>'
        f'<p class="eegdash-viz-body">{escape(body)}</p>'
        f"{meta_html}"
        "</div>"
    )


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
    "EEGDASH_STATUS_COLORS",
    "EEGDASH_VIZ_CSS",
    "get_eegdash_palette",
    "set_eegdash_matplotlib_defaults",
    "use_eegdash_style",
    "apply_eegdash_matplotlib_style",
    "style_figure",
    "chance_line",
    "apply_eegdash_plotly_layout",
    "eegdash_viz_card_html",
]
