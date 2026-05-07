# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""EEGDash visualization helpers (palette, Data Rail, figure-level styling).

Public surface:

- :func:`use_eegdash_style` -- one-call rcParams + palette setup for tutorials.
- :func:`style_figure` -- apply the EEGDash identity to every axes in a figure
  and attach the title/subtitle/source block + Data Rail.
- :func:`chance_line` -- add a labelled chance-level reference line.
- :func:`apply_eegdash_matplotlib_style` -- per-axes styling primitive.
- :func:`apply_eegdash_plotly_layout` -- Plotly equivalent of the rail.
- :func:`eegdash_viz_card_html` -- compact HTML card for docs pages.
- Palette constants ``EEGDASH_BLUE``, ``EEGDASH_ORANGE``, ..., ``EEGDASH_PALETTE``.
"""

from __future__ import annotations

from .identity import (
    EEGDASH_AMBER,
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_CORAL,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MINT,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_PALETTE,
    EEGDASH_PURPLE,
    EEGDASH_SKY,
    EEGDASH_STATUS_COLORS,
    EEGDASH_SURFACE,
    EEGDASH_VIZ_CSS,
    apply_eegdash_matplotlib_style,
    apply_eegdash_plotly_layout,
    chance_line,
    eegdash_viz_card_html,
    get_eegdash_palette,
    set_eegdash_matplotlib_defaults,
    style_figure,
    use_eegdash_style,
)

__all__ = [
    "EEGDASH_AMBER",
    "EEGDASH_BLUE",
    "EEGDASH_BLUE_DARK",
    "EEGDASH_CORAL",
    "EEGDASH_GRID",
    "EEGDASH_INK",
    "EEGDASH_MINT",
    "EEGDASH_MUTED",
    "EEGDASH_ORANGE",
    "EEGDASH_PALETTE",
    "EEGDASH_PURPLE",
    "EEGDASH_SKY",
    "EEGDASH_STATUS_COLORS",
    "EEGDASH_SURFACE",
    "EEGDASH_VIZ_CSS",
    "apply_eegdash_matplotlib_style",
    "apply_eegdash_plotly_layout",
    "chance_line",
    "eegdash_viz_card_html",
    "get_eegdash_palette",
    "set_eegdash_matplotlib_defaults",
    "style_figure",
    "use_eegdash_style",
]
