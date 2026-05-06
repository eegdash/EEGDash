"""EEGDash data-visualization identity helpers.

This module defines the first reusable EEGDash visual identity element for
documentation plots and cards: the Data Rail. It mirrors MOABB's disciplined
plot accent system while using EEGDash's own blue/orange brand colors.
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


def set_eegdash_matplotlib_defaults() -> None:
    """Configure Matplotlib/Seaborn defaults for EEGDash documentation plots."""
    import seaborn as sns

    sns.set_theme(
        style="white",
        palette=EEGDASH_PALETTE,
        font="sans-serif",
        rc={
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
            "legend.fontsize": 11,
        },
    )


def apply_eegdash_matplotlib_style(
    ax,
    *,
    title: str = "",
    subtitle: str = "",
    source: str = "",
    data_rail: bool = True,
    grid_axis: str = "y",
) -> None:
    """Apply the EEGDash Data Rail identity to a Matplotlib axes."""
    import matplotlib.pyplot as plt

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

    if title:
        fig.text(
            0.08,
            0.922,
            title,
            fontsize=18,
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
            fontsize=12,
            color=EEGDASH_MUTED,
            ha="left",
            va="top",
        )

    if source:
        fig.text(
            0.08,
            0.018,
            source,
            fontsize=9,
            color=EEGDASH_MUTED,
            ha="left",
            va="bottom",
            style="italic",
        )

    ax.set_title("")


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
