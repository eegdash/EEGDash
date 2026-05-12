"""Cross-tutorial figure primitives for the EEGDash gallery.

Private submodule (leading underscore) — not part of the public API. Used by
the gallery tutorials in ``examples/tutorials/`` to avoid copy-pasting the
same provenance-footer / shape-label / before-after-axes / subject-color
glue across every plot.

The four exports here cover the patterns that already proved their weight in
``plot_02_dataset_to_dataloader``; bespoke per-tutorial geometry stays in
sibling ``_<topic>_figure.py`` modules next to each ``plot_*.py``.

Why a private package submodule rather than ``examples/tutorials/_shared/``:
sphinx-gallery executes a tutorial with the *script's* directory on
``sys.path``, so a sibling import works but a top-level shared dir would need
``sys.path`` cruft. ``eegdash`` is always on ``sys.path``; importing from
``eegdash.viz._tutorial_panels`` is zero-friction.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from .identity import (
    EEGDASH_INK,
    EEGDASH_MUTED,
    get_eegdash_palette,
)


def add_provenance_footer(
    fig,
    *,
    plot_id: str,
    openneuro_id: str | None = None,
    citation: str | None = None,
    extra: str | None = None,
) -> str:
    """Format and attach the standard tutorial provenance footer.

    The output string is also returned so callers that already drive
    :func:`~eegdash.viz.style_figure` can pass it as ``source=...`` instead
    of attaching a separate ``fig.text(...)``. Call shape:

    >>> source = add_provenance_footer(  # doctest: +SKIP
    ...     fig,
    ...     plot_id="plot_10",
    ...     openneuro_id="ds002718",
    ...     citation="Wakeman & Henson 2015",
    ... )

    Parameters
    ----------
    fig : matplotlib.figure.Figure | None
        When ``None``, only the formatted string is returned (useful for
        ``style_figure(source=add_provenance_footer(None, ...))``).
    plot_id : str
        Short tutorial id, e.g. ``"plot_10"``.
    openneuro_id : str, optional
        OpenNeuro accession (``"ds002718"``). Pass ``None`` for synthetic data.
    citation : str, optional
        Short citation (e.g. ``"Wakeman & Henson 2015"``).
    extra : str, optional
        Trailing free-form text (e.g. doi or run-time note).

    Returns
    -------
    str
        The formatted footer string. When ``fig`` is not ``None``, the same
        string is also placed at ``(0.08, 0.018)`` of the figure in muted
        italic, matching :func:`~eegdash.viz.style_figure`'s conventions.

    """
    parts = [f"EEGDash {plot_id}"]
    if openneuro_id:
        if citation:
            parts.append(f"OpenNeuro {openneuro_id} ({citation})")
        else:
            parts.append(f"OpenNeuro {openneuro_id}")
    elif citation:
        parts.append(citation)
    if extra:
        parts.append(extra)
    text = " | ".join(parts)
    if fig is not None:
        fig.text(
            0.08,
            0.018,
            text,
            fontsize=8,
            color=EEGDASH_MUTED,
            ha="left",
            va="bottom",
            style="italic",
        )
    return text


def shape_label(
    ax,
    x: float,
    y: float,
    shape_tuple: Sequence[int] | str,
    *,
    color: str = EEGDASH_INK,
    fontsize: float = 8.4,
    ha: str = "center",
    va: str = "center",
    background: str | None = None,
) -> None:
    """Place a monospace tensor-shape annotation on an axes.

    Used by every tutorial that wants to print ``(70 ch, 200 samples)`` or
    ``(8, 70, 200)`` inline on or near a panel. Rendered in monospace so
    the parentheses and commas line up with code blocks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (uses its data coordinates by default).
    x, y : float
        Position in ``ax``'s data coordinates.
    shape_tuple : sequence of int | str
        ``(70, 200)`` becomes ``"(70, 200)"``. A pre-formatted string is
        passed through unchanged so callers can attach units
        (``"(70 ch, 200 samples)"``).
    color : str, optional
        Text color (defaults to ``EEGDASH_INK``).
    fontsize : float, optional
    ha, va : {"left", "center", "right"} / {"top", "center", "bottom"}
    background : str | None, optional
        When set, render with a white-pill background outlined in the given
        color. Useful when the label sits on top of a heatmap.

    """
    if isinstance(shape_tuple, str):
        text = shape_tuple
    else:
        text = "(" + ", ".join(str(int(v)) for v in shape_tuple) + ")"
    bbox = None
    if background is not None:
        bbox = {
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": background,
            "linewidth": 0.6,
        }
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        family="monospace",
        color=color,
        bbox=bbox,
    )


def before_after_axes(
    fig,
    gridspec_slot,
    *,
    label_before: str = "before",
    label_after: str = "after",
    label_color: str = EEGDASH_MUTED,
) -> tuple:
    """Carve a single GridSpec slot into a paired (before, after) axes pair.

    Splits the slot horizontally with a thin gutter and labels each half in
    the upper-left corner. Used by plot_10's PSD/heatmap before-vs-after
    panel and plot_13's residual heatmap.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    gridspec_slot : matplotlib.gridspec.SubplotSpec
        Result of ``fig.add_gridspec(...)[r, c]``.
    label_before, label_after : str, optional
        Labels rendered as muted monospace tags inside each axes.
    label_color : str, optional
        Color for the corner labels.

    Returns
    -------
    (ax_before, ax_after) : tuple of matplotlib.axes.Axes

    """
    inner = gridspec_slot.subgridspec(1, 2, wspace=0.08)
    ax_before = fig.add_subplot(inner[0, 0])
    ax_after = fig.add_subplot(inner[0, 1])
    for ax, label in [(ax_before, label_before), (ax_after, label_after)]:
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            family="monospace",
            color=label_color,
        )
    return ax_before, ax_after


def subject_color_map(
    subjects: Iterable[str | int] | pd.Series,
) -> dict[str, str]:
    """Stable subject -> color mapping shared across tutorials.

    Two tutorials need the *same* subject to render in the *same* color
    (plot_11's leakage Sankey and plot_12's PCA scatter, when they discuss
    the same recording). This helper picks a deterministic palette entry per
    subject id so a multi-figure narrative stays visually coherent.

    Parameters
    ----------
    subjects : iterable of str or int, or pandas.Series
        Subject ids in any order. Duplicates are ignored.

    Returns
    -------
    dict[str, str]
        ``{subject_id_as_str: hex_color}``. Iteration order matches the
        first-seen order of ``subjects``.

    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for s in subjects:
        key = str(s)
        if key not in seen_set:
            seen_set.add(key)
            seen.append(key)
    palette = get_eegdash_palette(max(len(seen), 1))
    return {s: palette[i % len(palette)] for i, s in enumerate(seen)}


__all__ = [
    "add_provenance_footer",
    "shape_label",
    "before_after_axes",
    "subject_color_map",
]
