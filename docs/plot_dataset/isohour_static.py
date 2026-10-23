"""Static, journal-ready version of the iso-hour landscape (matplotlib).

Four panels, built from the cleaned table of :mod:`plot_dataset.isohour`:

A. Sampling landscape: subjects against hours per subject, log-log, with
   iso-hour diagonals, one marker and colour per recording modality, and
   well-known datasets labelled on the points.
B. Where the hours are: cumulative share of the catalogue's hours (and
   subjects) held by the largest datasets.
C. Hours per subject by study type (LLM-tagger ``tags.type``), sharing the
   x axis of A.
D. Share of datasets, subjects and hours per recording modality.

Usage::

    cd docs
    python -m plot_dataset.isohour_static --out paper_figures/isohour
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

try:
    from .isohour import (
        ISO_HOURS,
        LANDMARKS,
        MODALITY_STYLE,
        build_isohour_table,
        correct_durations_from_records,
        fetch_catalogue,
        summarise_isohour,
    )
except ImportError:  # pragma: no cover - direct-script execution
    from isohour import (  # type: ignore
        ISO_HOURS,
        LANDMARKS,
        MODALITY_STYLE,
        build_isohour_table,
        correct_durations_from_records,
        fetch_catalogue,
        summarise_isohour,
    )

X_LIM = (1 / 60, 400.0)  # one minute to 400 h per subject
Y_LIM = (0.75, 4500.0)
INK = "#222222"
MUTED = "#6b7280"
ISO_COLOR = "#cbd5e1"

#: Label offsets (points) for the landmarks of panel A; default (4, 2).
_LABEL_OFFSETS: dict[str, tuple[float, float, str]] = {
    "nm000181": (5, -1, "left"),
    "nm000225": (-5, 3, "right"),
    "nm000289": (5, 0, "left"),
    "on005873": (-4, 4, "right"),
    "on004395": (5, -1, "left"),
    "nm000185": (5, -2, "left"),
    "on005178": (0, -6, "center"),
    "nm000110": (5, -1, "left"),
    "nm000358": (5, -2, "left"),
    "nm000232": (5, 2, "left"),
    "ds004212": (5, -3, "left"),
    "nm000229": (5, -1, "left"),
    "nm000180": (-5, 0, "right"),
    "nm000179": (-5, 1, "right"),
    "on004362": (5, -2, "left"),
    "nm000139": (-5, -2, "right"),
    "on004789": (5, 1, "left"),
    "on005515": (5, 2, "left"),
    "on006317": (-5, 0, "right"),
    "nm000238": (5, 0, "left"),
}

_TYPE_MIN_DATASETS = 20


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )


def _hour_fmt(h: float) -> str:
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 10:
        return f"{h:.1f} h"
    return f"{h:,.0f} h"


def _log_hours_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ticks = [1 / 60, 10 / 60, 1, 10, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1 min", "10 min", "1 h", "10 h", "100 h"])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def _panel_letter(ax: plt.Axes, letter: str, title: str, *, dx: float = -0.11) -> None:
    ax.text(
        dx,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
    )
    ax.set_title(title, loc="left", pad=4)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _iso_lines(ax: plt.Axes) -> None:
    xs = np.logspace(np.log10(X_LIM[0]), np.log10(X_LIM[1]), 200)
    for hours in ISO_HOURS:
        ax.plot(
            xs,
            hours / xs,
            color=ISO_COLOR,
            lw=4.5,
            alpha=0.55,
            zorder=0,
            solid_capstyle="butt",
        )
    ax.figure.canvas.draw()  # transforms are needed for the label angle
    for hours in ISO_HOURS:
        # Label each line near the bottom edge, or on the right edge when the
        # line leaves the plot through it.
        y = 1.45
        x = hours / y
        if x > X_LIM[1] / 1.7:
            x = X_LIM[1] / 1.7
            y = hours / x
        p1 = ax.transData.transform((x, hours / x))
        p2 = ax.transData.transform((x * 1.5, hours / (x * 1.5)))
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        ax.text(
            x,
            y,
            f"{hours:,} h",
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=6,
            color=MUTED,
            zorder=1,
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"),
        )


def panel_landscape(ax: plt.Axes, kept: pd.DataFrame, stats: dict) -> None:
    ax.set_yscale("log")
    _log_hours_axis(ax)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1,000"])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    _iso_lines(ax)

    # Region where many subjects are each sampled for many hours.
    ax.fill_between([10, X_LIM[1]], 100, Y_LIM[1], color="#f1f5f9", zorder=-1, lw=0)
    n_dense = len(stats["dense"])
    ax.text(
        X_LIM[1] / 1.15,
        Y_LIM[1] / 1.25,
        f"≥100 subjects × ≥10 h\n{n_dense} datasets",
        ha="right",
        va="top",
        fontsize=6,
        color=MUTED,
        linespacing=1.2,
    )

    handles = []
    order = ["eeg", "ieeg", "meg", "fnirs", "emg"]
    for mod in order:
        sub = kept[kept["modality"] == mod]
        if sub.empty:
            continue
        st = MODALITY_STYLE[mod]
        ax.scatter(
            sub["hours_per_subject"],
            sub["n_subjects"],
            s=7,
            marker=st["marker"],
            c=st["color"],
            alpha=0.7,
            lw=0,
            zorder=2 + order.index(mod) * 0.1,
        )
        handles.append(
            Line2D(
                [],
                [],
                ls="",
                marker=st["marker"],
                color=st["color"],
                markersize=3.5,
                label=f"{st['label']} ({len(sub)})",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        handletextpad=0.1,
        borderaxespad=0.3,
        labelspacing=0.25,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )

    for ds, label in LANDMARKS.items():
        row = kept[kept["dataset_id"] == ds]
        if row.empty:
            continue
        r = row.iloc[0]
        dx, dy, ha = _LABEL_OFFSETS.get(ds, (4, 2, "left"))
        ax.scatter(
            [r["hours_per_subject"]],
            [r["n_subjects"]],
            s=10,
            facecolors="none",
            edgecolors=INK,
            lw=0.5,
            zorder=4,
            marker=MODALITY_STYLE.get(r["modality"], MODALITY_STYLE["eeg"])["marker"],
        )
        ax.annotate(
            label,
            (r["hours_per_subject"], r["n_subjects"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=5.6,
            color=INK,
            zorder=5,
            path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")],
        )

    ax.set_xlabel("Recorded hours per subject")
    ax.set_ylabel("Subjects per dataset")


def panel_concentration(ax: plt.Axes, kept: pd.DataFrame, stats: dict) -> None:
    n = len(kept)
    ranks = np.arange(1, n + 1)
    hours = np.sort(kept["hours_total"].to_numpy())[::-1]
    subj = np.sort(kept["n_subjects"].to_numpy())[::-1]
    cum_h = 100 * np.cumsum(hours) / hours.sum()
    cum_s = 100 * np.cumsum(subj) / subj.sum()
    ax.plot(ranks, 100 * ranks / n, color=MUTED, lw=0.7, ls="--", zorder=1)
    ax.plot(ranks, cum_s, color="#56B4E9", lw=1.3, zorder=2)
    ax.plot(ranks, cum_h, color=INK, lw=1.3, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1, n)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([1, 10, 100, n])
    ax.set_xticklabels(["1", "10", "100", f"{n}"])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

    k50 = stats["n_datasets_half_hours"]
    top10 = 100 * stats["share_hours_top10"]
    ax.plot([k50], [cum_h[k50 - 1]], "o", ms=2.5, color=INK, zorder=4)
    ax.plot([10], [top10], "o", ms=2.5, color=INK, zorder=4)
    ax.annotate(
        f"{k50} datasets:\n{cum_h[k50 - 1]:.0f}% of hours",
        (k50, cum_h[k50 - 1]),
        xytext=(5, -2),
        textcoords="offset points",
        fontsize=5.8,
        va="top",
        linespacing=1.15,
    )
    ax.annotate(
        f"10 datasets: {top10:.0f}%",
        (10, top10),
        xytext=(5, -3),
        textcoords="offset points",
        fontsize=5.8,
        va="top",
    )
    i_h, i_s = min(39, n - 1), min(219, n - 1)
    ax.text(
        i_h + 1,
        cum_h[i_h] + 5,
        "hours",
        ha="center",
        va="bottom",
        fontsize=6,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        i_s + 1,
        cum_s[i_s] - 6,
        "subjects",
        ha="left",
        va="top",
        fontsize=6,
        color="#3a8fc4",
        fontweight="bold",
    )
    ax.text(
        n / 1.4,
        100 * (n / 2.2) / n - 9,
        "equal shares",
        ha="right",
        va="top",
        fontsize=5.8,
        color=MUTED,
    )
    ax.set_xlabel("Largest datasets (rank)")
    ax.set_ylabel("Cumulative share (%)")


def panel_types(ax: plt.Axes, kept: pd.DataFrame) -> None:
    exploded = kept.explode("types").dropna(subset=["types"])
    counts = exploded["types"].value_counts()
    types = counts[counts >= _TYPE_MIN_DATASETS].index
    med = (
        exploded[exploded["types"].isin(types)]
        .groupby("types")["hours_per_subject"]
        .median()
    )
    types = med.sort_values().index.tolist()
    rng = np.random.default_rng(0)
    for i, t in enumerate(types):
        sub = exploded[exploded["types"] == t]
        y = i + rng.uniform(-0.28, 0.28, len(sub))
        colors = [
            MODALITY_STYLE.get(m, MODALITY_STYLE["eeg"])["color"]
            for m in sub["modality"]
        ]
        ax.scatter(
            sub["hours_per_subject"], y, s=3, c=colors, alpha=0.55, lw=0, zorder=2
        )
        ax.plot(
            [med[t], med[t]],
            [i - 0.38, i + 0.38],
            color=INK,
            lw=1.4,
            zorder=3,
            solid_capstyle="butt",
        )
        ax.text(
            1.01,
            i,
            _hour_fmt(med[t]),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=5.8,
            color=INK,
        )
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{t} ({counts[t]})" for t in types])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(types) - 0.4)
    _log_hours_axis(ax)
    ax.set_xlabel("Recorded hours per subject")


def panel_modalities(ax: plt.Axes, kept: pd.DataFrame) -> None:
    order = ["eeg", "ieeg", "meg", "fnirs", "emg"]
    measures = {
        "Hours": kept.groupby("modality")["hours_total"].sum(),
        "Subjects": kept.groupby("modality")["n_subjects"].sum(),
        "Datasets": kept.groupby("modality").size(),
    }
    for row, (label, series) in enumerate(measures.items()):
        share = 100 * series.reindex(order).fillna(0) / series.sum()
        left = 0.0
        for mod in order:
            w = share[mod]
            st = MODALITY_STYLE[mod]
            ax.barh(
                row,
                w,
                left=left,
                height=0.62,
                color=st["color"],
                lw=0.4,
                edgecolor="white",
            )
            if w >= 7:
                text = f"{st['label']}\n{w:.0f}%" if w >= 12 else f"{w:.0f}%"
                ax.text(
                    left + w / 2,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color="white",
                    fontweight="bold",
                    linespacing=1.1,
                )
            left += w
    ax.set_yticks(range(len(measures)))
    ax.set_yticklabels(list(measures))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Share of the catalogue (%)")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_isohour_figure(table: pd.DataFrame, out_stem: str | Path) -> dict:
    """Draw the four-panel figure and write ``.pdf``, ``.png`` and ``.svg``."""
    _style()
    kept = table[table["status"] == "kept"].copy()
    stats = summarise_isohour(table)

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.75, 1],
        height_ratios=[2.25, 1.25],
        left=0.14,
        right=0.985,
        top=0.95,
        bottom=0.075,
        wspace=0.42,
        hspace=0.32,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    right = gs[:, 1].subgridspec(2, 1, height_ratios=[1.45, 1], hspace=0.62)
    ax_b = fig.add_subplot(right[0])
    ax_d = fig.add_subplot(right[1])

    panel_landscape(ax_a, kept, stats)
    _panel_letter(ax_a, "A", "Sampling landscape", dx=-0.1)
    panel_concentration(ax_b, kept, stats)
    _panel_letter(ax_b, "B", "Where the hours are", dx=-0.3)
    panel_types(ax_c, kept)
    _panel_letter(ax_c, "C", "Hours per subject by study type", dx=-0.1)
    panel_modalities(ax_d, kept)
    _panel_letter(ax_d, "D", "Share by modality", dx=-0.3)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)
    return stats


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="paper_figures/isohour")
    parser.add_argument(
        "--table-csv",
        default=None,
        help="also write the cleaned table (with exclusion reasons)",
    )
    parser.add_argument("--no-record-correction", action="store_true")
    args = parser.parse_args()

    table = build_isohour_table(fetch_catalogue())
    if not args.no_record_correction:
        table = correct_durations_from_records(table)
    stats = make_isohour_figure(table, args.out)
    if args.table_csv:
        table.to_csv(args.table_csv, index=False)
    print({k: v for k, v in stats.items() if k != "dense"})


if __name__ == "__main__":  # pragma: no cover
    main()
