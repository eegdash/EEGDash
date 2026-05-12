"""Drawing helpers for the ``plot_70`` "Challenge dataset basics" figure."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_PURPLE,
    EEGDASH_SKY,
    EEGDASH_SURFACE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# ---------------------------------------------------------------------------
# Task-family palette (Panel 1 + Panel 2)
# ---------------------------------------------------------------------------

# HBN protocol groups documented in Alexander et al. 2017 sec. 4.4
# (https://doi.org/10.1038/sdata.2017.181). Task names match the BIDS task
# entity exposed by the eegdash metadata catalog.
TASK_FAMILY = {
    "RestingState": "resting",
    "DespicableMe": "passive video",
    "DiaryOfAWimpyKid": "passive video",
    "FunwithFractals": "passive video",
    "ThePresent": "passive video",
    "MovieDM": "passive video",
    "contrastChangeDetection": "active cognitive",
    "surroundSupp": "active cognitive",
    "symbolSearch": "active cognitive",
    "seqLearning6target": "sequence learning",
    "seqLearning8target": "sequence learning",
}

FAMILY_COLOR = {
    "resting": EEGDASH_BLUE,
    "passive video": EEGDASH_SKY,
    "active cognitive": EEGDASH_ORANGE,
    "sequence learning": EEGDASH_PURPLE,
    "other": EEGDASH_MUTED,
}


def _family_for(task: str) -> str:
    return TASK_FAMILY.get(task, "other")


# ---------------------------------------------------------------------------
# Panel 1: records per task
# ---------------------------------------------------------------------------


def _draw_task_bars(ax, task_counts: Mapping[str, int]) -> None:
    items = sorted(task_counts.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    counts = np.array([v for _, v in items], dtype=float)
    families = [_family_for(n) for n in names]
    colors = [FAMILY_COLOR[f] for f in families]

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
            fontsize=8.6,
            color=EEGDASH_INK,
            family="monospace",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.6, color=EEGDASH_INK)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max * 1.18)
    ax.set_xlabel("records", fontsize=9, color=EEGDASH_INK)
    ax.set_title(
        "records per task",
        fontsize=10.5,
        color=EEGDASH_INK,
        loc="left",
        pad=22.0,
    )

    # Family legend: ordered list of families that appear in this release.
    seen: list[str] = []
    for f in families:
        if f not in seen:
            seen.append(f)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f], linewidth=0) for f in seen
    ]
    ax.legend(
        handles,
        seen,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        frameon=False,
        fontsize=8.0,
        handlelength=1.0,
        handleheight=0.8,
        labelcolor=EEGDASH_INK,
        ncol=min(4, len(seen)),
        columnspacing=0.9,
    )

    ax.grid(True, axis="x", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


# ---------------------------------------------------------------------------
# Panel 2: age x task availability
# ---------------------------------------------------------------------------


def _draw_age_task_stack(
    ax, age_task_matrix: pd.DataFrame, task_order: Sequence[str]
) -> None:
    age_bins = list(age_task_matrix.index)
    # Reorder columns so a family's segments sit contiguous in the stacks.
    family_order = ["resting", "passive video", "active cognitive", "sequence learning"]
    ordered_tasks: list[str] = []
    for fam in family_order:
        ordered_tasks.extend(
            [
                t
                for t in task_order
                if _family_for(t) == fam and t in age_task_matrix.columns
            ]
        )
    # Append any leftover columns the caller passed but the family map missed.
    for t in age_task_matrix.columns:
        if t not in ordered_tasks:
            ordered_tasks.append(t)

    y = np.arange(len(age_bins))
    left = np.zeros(len(age_bins), dtype=float)
    legend_entries: list[tuple[str, str]] = []

    for task in ordered_tasks:
        seg = age_task_matrix[task].to_numpy(dtype=float)
        if not np.any(seg > 0):
            continue
        family = _family_for(task)
        color = FAMILY_COLOR[family]
        ax.barh(
            y,
            seg,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        left = left + seg
        if family not in {f for f, _ in legend_entries}:
            legend_entries.append((family, color))

    # Row totals as monospace annotations on the right.
    totals = age_task_matrix.sum(axis=1).to_numpy(dtype=float)
    x_max = float(totals.max()) if len(totals) else 1.0
    pad = x_max * 0.02
    for i, total in enumerate(totals):
        ax.text(
            total + pad,
            i,
            f"{int(total)}",
            va="center",
            ha="left",
            fontsize=8.4,
            color=EEGDASH_INK,
            family="monospace",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(age_bins, fontsize=8.6, color=EEGDASH_INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(x_max * 1.18, 1.0))
    ax.set_xlabel("records", fontsize=9, color=EEGDASH_INK)
    ax.set_title(
        "age bin x task availability",
        fontsize=10.5,
        color=EEGDASH_INK,
        loc="left",
        pad=22.0,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, linewidth=0) for _, c in legend_entries
    ]
    labels = [f for f, _ in legend_entries]
    ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        frameon=False,
        fontsize=8.0,
        handlelength=1.0,
        handleheight=0.8,
        labelcolor=EEGDASH_INK,
        ncol=min(4, len(labels)),
        columnspacing=0.9,
    )

    ax.grid(True, axis="x", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


# ---------------------------------------------------------------------------
# Panel 3: metadata card for one record
# ---------------------------------------------------------------------------


def _format_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        if pd.isna(value):
            return "NaN"
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        head = ", ".join(_format_value(v) for v in list(value)[:3])
        tail = " ..." if len(value) > 3 else ""
        return f"[{head}{tail}]  (n={len(value)})"
    if isinstance(value, dict):
        keys = list(value.keys())[:4]
        body = ", ".join(str(k) for k in keys)
        tail = " ..." if len(value) > 4 else ""
        return f"{{{body}{tail}}}  (n={len(value)})"
    text = str(value)
    if len(text) > 36:
        text = text[:33] + "..."
    return text


def _draw_metadata_card(ax, sample_metadata_row: Mapping) -> None:
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
        0.03,
        0.95,
        "one record from ds.records",
        ha="left",
        va="top",
        fontsize=10.0,
        color=EEGDASH_INK,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.03,
        0.88,
        "every BDF file in the challenge bucket has a row like this in the metadata catalog",
        ha="left",
        va="top",
        fontsize=8.2,
        color=EEGDASH_MUTED,
        style="italic",
        transform=ax.transAxes,
    )

    # Pick keys in a curated order so the card stays readable. The catalog
    # exposes around 20 keys; a dozen carry information a tutorial reader
    # cares about.
    preferred_order = [
        "data_name",
        "dataset",
        "subject",
        "task",
        "session",
        "run",
        "sampling_frequency",
        "nchans",
        "ntimes",
        "datatype",
        "suffix",
        "extension",
        "recording_modality",
        "bidspath",
        "digested_at",
    ]
    rows: list[tuple[str, str]] = []
    for key in preferred_order:
        if key in sample_metadata_row:
            rows.append((key, _format_value(sample_metadata_row[key])))
    # Two-column layout so the card stays readable without overflowing.
    rows = rows[:14]
    half = (len(rows) + 1) // 2
    col_left = rows[:half]
    col_right = rows[half:]

    width_left = max((len(k) for k, _ in col_left), default=0)
    width_right = max((len(k) for k, _ in col_right), default=0)
    text_left = "\n".join(f"{k.ljust(width_left)}  {v}" for k, v in col_left)
    text_right = "\n".join(f"{k.ljust(width_right)}  {v}" for k, v in col_right)

    ax.text(
        0.03,
        0.78,
        text_left,
        ha="left",
        va="top",
        fontsize=8.6,
        family="monospace",
        color=EEGDASH_INK,
        transform=ax.transAxes,
    )
    if text_right:
        ax.text(
            0.52,
            0.78,
            text_right,
            ha="left",
            va="top",
            fontsize=8.6,
            family="monospace",
            color=EEGDASH_INK,
            transform=ax.transAxes,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def draw_challenge_basics_figure(
    *,
    task_counts: Mapping[str, int],
    age_task_matrix: pd.DataFrame,
    sample_metadata_row: Mapping,
    plot_id: str = "plot_70",
):
    """Render the three-panel "challenge dataset basics" figure.

    Parameters
    ----------
    task_counts : mapping of str -> int
        Records per task name, computed from ``ds.description['task']``.
        Names are rendered as-is on the y-axis of Panel 1.
    age_task_matrix : pandas.DataFrame
        Non-negative-integer matrix; one row per age-bin label, one column per
        task name. Values are the per-record-count of recordings in that bin
        for that task. Used as the data source for Panel 2.
    sample_metadata_row : mapping
        One row of ``ds.records`` (or the equivalent dict). Rendered as a
        monospace metadata card in Panel 3.
    plot_id : str, optional
        Tutorial id, forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure. The caller is responsible for ``plt.show()``.

    """
    n_records = int(sum(task_counts.values()))
    n_tasks = len(task_counts)
    n_age_groups = int(len(age_task_matrix.index))
    # Subjects can be inferred from the matrix only if rows are subjects;
    # we instead source it from the metadata row when possible, otherwise
    # the caller supplies it via the subtitle override (kept here for
    # determinism without an extra parameter).
    n_subjects = sample_metadata_row.get("__n_subjects", None)

    fig = plt.figure(figsize=(11.4, 7.4))
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[1.05, 1.0],
        height_ratios=[1.0, 0.85],
        wspace=0.55,
        hspace=0.65,
        left=0.16,
        right=0.965,
        top=0.78,
        bottom=0.10,
    )
    ax_tasks = fig.add_subplot(gs[0, 0])
    ax_age = fig.add_subplot(gs[0, 1])
    ax_meta = fig.add_subplot(gs[1, :])

    _draw_task_bars(ax_tasks, task_counts)
    _draw_age_task_stack(
        ax_age,
        age_task_matrix,
        task_order=sorted(task_counts.keys(), key=lambda k: -task_counts[k]),
    )
    _draw_metadata_card(ax_meta, sample_metadata_row)

    subtitle_bits = [
        f"n_subjects={n_subjects}" if n_subjects is not None else None,
        f"n_recordings={n_records}",
        f"n_tasks={n_tasks}",
        f"n_age_groups={n_age_groups}",
    ]
    subtitle = " | ".join(b for b in subtitle_bits if b)

    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        citation="HBN R5 mini, Alexander et al. 2017",
        extra="EEG2025 challenge bucket, downsampled 100 Hz",
    )
    style_figure(
        fig,
        title="EEGChallengeDataset at a glance: tasks, ages, and one catalog row",
        subtitle=subtitle,
        source=source,
        grid_axis="x",
    )
    return fig


__all__ = ["draw_challenge_basics_figure"]
