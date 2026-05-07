"""Drawing helpers for the clinical-dataset-summary applied case study.

Sibling module to ``project_clinical_dataset_summary.py``. The leading
underscore tells sphinx-gallery to skip this file when building the gallery,
so the helper code stays out of the rendered tutorial; the tutorial imports
the public ``draw_clinical_summary_figure`` entry point.

Three panels:

1. Subjects-per-condition horizontal bar chart, color-coded by diagnostic
   group from the participants.tsv ``Group`` field.
2. Per-condition age histogram drawn with the EEGDash palette so the
   distributions are easy to read against each other.
3. A four-cell metadata card with the headline numbers a project plan
   needs at hand: n_subjects, n_recordings, mean recording duration,
   n_channels.
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
    EEGDASH_PALETTE,
    EEGDASH_PURPLE,
    EEGDASH_SKY,
    EEGDASH_SURFACE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# ---------------------------------------------------------------------------
# Group palette (Panel 1 + Panel 2)
# ---------------------------------------------------------------------------

# Stable group -> color mapping. Keys are the long-form labels we expect
# the tutorial to pass in (the BIDS ``Group`` codes A / F / C are remapped
# to readable strings before the figure is drawn).
GROUP_COLOR = {
    "Alzheimer's disease": EEGDASH_BLUE,
    "Frontotemporal dementia": EEGDASH_ORANGE,
    "Healthy control": EEGDASH_PURPLE,
    "Other": EEGDASH_SKY,
}


def _color_for(group: str) -> str:
    return GROUP_COLOR.get(group, EEGDASH_PALETTE[0])


# ---------------------------------------------------------------------------
# Panel 1: subjects per condition
# ---------------------------------------------------------------------------


def _draw_condition_bars(ax, condition_counts: Mapping[str, int]) -> None:
    """Render horizontal bars of subjects-per-condition, color-coded."""
    items = sorted(condition_counts.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    counts = np.array([v for _, v in items], dtype=float)
    colors = [_color_for(n) for n in names]

    y = np.arange(len(names))
    ax.barh(
        y,
        counts,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    x_max = max(float(counts.max()), 1.0)
    pad = x_max * 0.02
    for i, c in enumerate(counts):
        ax.text(
            c + pad,
            i,
            f"{int(c)}",
            va="center",
            ha="left",
            fontsize=9.0,
            color=EEGDASH_INK,
            family="monospace",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.0, color=EEGDASH_INK)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max * 1.18)
    ax.set_xlabel("subjects", fontsize=9, color=EEGDASH_INK)
    ax.set_title(
        "subjects per condition",
        fontsize=10.5,
        color=EEGDASH_INK,
        loc="left",
        pad=18.0,
    )
    ax.grid(True, axis="x", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


# ---------------------------------------------------------------------------
# Panel 2: age distribution per condition
# ---------------------------------------------------------------------------


def _draw_age_histogram(
    ax,
    ages_by_condition: Mapping[str, Sequence[float]],
    bins: int = 12,
) -> None:
    """Stacked age histogram, one color band per condition."""
    conditions = list(ages_by_condition.keys())
    if not conditions:
        ax.text(0.5, 0.5, "no age data", ha="center", va="center", color=EEGDASH_MUTED)
        ax.set_axis_off()
        return

    all_ages = np.concatenate(
        [np.asarray(ages_by_condition[c], dtype=float) for c in conditions]
    )
    all_ages = all_ages[np.isfinite(all_ages)]
    if all_ages.size == 0:
        ax.text(0.5, 0.5, "no age data", ha="center", va="center", color=EEGDASH_MUTED)
        ax.set_axis_off()
        return

    edges = np.linspace(np.floor(all_ages.min()), np.ceil(all_ages.max()), bins + 1)
    bottom = np.zeros(bins, dtype=float)
    for cond in conditions:
        ages = np.asarray(ages_by_condition[cond], dtype=float)
        ages = ages[np.isfinite(ages)]
        if ages.size == 0:
            continue
        counts, _ = np.histogram(ages, bins=edges)
        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            bottom=bottom,
            color=_color_for(cond),
            edgecolor="white",
            linewidth=0.6,
            align="edge",
            label=cond,
            zorder=3,
        )
        bottom = bottom + counts

    ax.set_xlabel("age (years)", fontsize=9, color=EEGDASH_INK)
    ax.set_ylabel("subjects", fontsize=9, color=EEGDASH_INK)
    ax.set_title(
        "age distribution per condition",
        fontsize=10.5,
        color=EEGDASH_INK,
        loc="left",
        pad=22.0,
    )
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=8.2,
        handlelength=1.0,
        handleheight=0.8,
        labelcolor=EEGDASH_INK,
        ncol=1,
        columnspacing=0.9,
    )
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


# ---------------------------------------------------------------------------
# Panel 3: metadata card with four numbers
# ---------------------------------------------------------------------------


def _format_duration_seconds(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds <= 0:
        return "n/a"
    minutes = seconds / 60.0
    if minutes >= 10:
        return f"{minutes:.0f} min"
    return f"{minutes:.1f} min"


def _draw_metadata_card(ax, summary_metrics: Mapping[str, float]) -> None:
    """Render a 2x2 grid of headline numbers inside a soft-bordered panel."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        plt.Rectangle(
            (0.012, 0.012),
            0.976,
            0.976,
            transform=ax.transAxes,
            facecolor=EEGDASH_SURFACE,
            edgecolor=EEGDASH_BLUE,
            linewidth=1.1,
            zorder=1,
        )
    )
    ax.text(
        0.04,
        0.93,
        "dataset at a glance",
        ha="left",
        va="top",
        fontsize=10.0,
        color=EEGDASH_INK,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.86,
        "four numbers a project plan needs at hand before training a model",
        ha="left",
        va="top",
        fontsize=8.4,
        color=EEGDASH_MUTED,
        style="italic",
        transform=ax.transAxes,
    )

    # Four-cell layout: top-left, top-right, bottom-left, bottom-right.
    n_subjects = int(summary_metrics.get("n_subjects", 0))
    n_recordings = int(summary_metrics.get("n_recordings", 0))
    mean_dur_s = float(summary_metrics.get("mean_duration_seconds", 0.0))
    n_channels = summary_metrics.get("n_channels", "n/a")
    n_channels_str = (
        f"{int(n_channels)}"
        if isinstance(n_channels, (int, float)) and np.isfinite(float(n_channels))
        else str(n_channels)
    )

    # Two-row x two-column layout. Each cell has the small caption above
    # the big number; labels are placed well clear of the value glyph height.
    cells = [
        (0.06, 0.70, "n_subjects", f"{n_subjects}"),
        (0.55, 0.70, "n_recordings", f"{n_recordings}"),
        (0.06, 0.30, "mean recording", _format_duration_seconds(mean_dur_s)),
        (0.55, 0.30, "n_channels", n_channels_str),
    ]
    for x, y, label, value in cells:
        ax.text(
            x,
            y,
            label,
            ha="left",
            va="bottom",
            fontsize=9.0,
            color=EEGDASH_MUTED,
            family="monospace",
            transform=ax.transAxes,
        )
        ax.text(
            x,
            y - 0.04,
            value,
            ha="left",
            va="top",
            fontsize=22.0,
            fontweight="bold",
            color=EEGDASH_INK,
            family="monospace",
            transform=ax.transAxes,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def draw_clinical_summary_figure(
    *,
    condition_counts: Mapping[str, int],
    ages_by_condition: Mapping[str, Sequence[float]],
    summary_metrics: Mapping[str, float],
    dataset: str,
    plot_id: str = "project_clinical_dataset_summary",
):
    """Render the three-panel clinical-dataset-summary figure.

    Parameters
    ----------
    condition_counts : mapping of str -> int
        Subjects per condition. Keys are readable group labels
        (``"Alzheimer's disease"``, ``"Healthy control"``, ...). Values are
        non-negative integers.
    ages_by_condition : mapping of str -> sequence of float
        Per-condition lists of subject ages. Keys must overlap with
        ``condition_counts`` for the colors to line up between Panel 1 and
        Panel 2.
    summary_metrics : mapping
        The four headline numbers rendered in Panel 3:
        ``n_subjects``, ``n_recordings``, ``mean_duration_seconds``,
        ``n_channels``.
    dataset : str
        Dataset accession (e.g., ``"ds004504"``). Used in the subtitle and
        the provenance footer.
    plot_id : str, optional
        Tutorial id, forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure. The caller is responsible for ``plt.show()``.

    """
    n_subjects = int(summary_metrics.get("n_subjects", 0))
    n_recordings = int(summary_metrics.get("n_recordings", 0))

    fig = plt.figure(figsize=(11.4, 7.4))
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[1.0, 1.05],
        height_ratios=[1.0, 0.85],
        wspace=0.55,
        hspace=0.65,
        left=0.16,
        right=0.965,
        top=0.78,
        bottom=0.10,
    )
    ax_cond = fig.add_subplot(gs[0, 0])
    ax_age = fig.add_subplot(gs[0, 1])
    ax_meta = fig.add_subplot(gs[1, :])

    _draw_condition_bars(ax_cond, condition_counts)
    _draw_age_histogram(ax_age, ages_by_condition)
    _draw_metadata_card(ax_meta, summary_metrics)

    subtitle = f"{dataset} | n_subjects={n_subjects} | n_recordings={n_recordings}"

    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=dataset,
        citation="Miltiadous et al. 2023",
        extra="metadata-only catalog query (no signal download)",
    )
    style_figure(
        fig,
        title="Clinical EEG dataset survey: cohort, ages, and headline metadata",
        subtitle=subtitle,
        source=source,
        grid_axis="x",
    )
    return fig


__all__ = ["draw_clinical_summary_figure"]
