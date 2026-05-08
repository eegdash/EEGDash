"""Drawing helpers for the ``project_p300_transfer`` applied case study."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_GRID,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# ---------------------------------------------------------------------------
# Internal panel painters
# ---------------------------------------------------------------------------


def _style_axis(ax) -> None:
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(EEGDASH_GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=0, colors=EEGDASH_INK)


def _draw_accuracy_panel(
    ax,
    *,
    accuracies: dict,
    chance_level: float,
) -> None:
    naive = float(accuracies.get("naive", 0.0))
    mmd = float(accuracies.get("mmd", 0.0))
    oracle = float(accuracies.get("oracle", 0.0))
    gain = mmd - naive

    labels = ["naive transfer", "AS-MMD aligned", "oracle (target)"]
    values = np.asarray([naive, mmd, oracle], dtype=float)
    colours = [EEGDASH_BLUE, EEGDASH_ORANGE, EEGDASH_MUTED]
    x = np.arange(values.size)

    bars = ax.bar(
        x,
        values,
        width=0.55,
        color=colours,
        edgecolor=EEGDASH_INK,
        linewidth=0.6,
        zorder=3,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            family="monospace",
            color=EEGDASH_INK,
            zorder=4,
        )

    # Gain bracket spanning naive -> AS-MMD bars.
    x_lo = bars[0].get_x() + bars[0].get_width() / 2
    x_hi = bars[1].get_x() + bars[1].get_width() / 2
    y_top = max(values[0], values[1]) + 0.085
    ax.annotate(
        "",
        xy=(x_hi, y_top - 0.012),
        xytext=(x_lo, y_top - 0.012),
        arrowprops={
            "arrowstyle": "-",
            "color": EEGDASH_INK,
            "linewidth": 0.7,
        },
    )
    ax.text(
        (x_lo + x_hi) / 2,
        y_top + 0.005,
        f"AS-MMD gain = {gain:+.02f}",
        ha="center",
        va="bottom",
        fontsize=8.4,
        family="monospace",
        color=EEGDASH_INK if gain >= 0 else EEGDASH_ORANGE,
        fontweight="bold",
        zorder=4,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": EEGDASH_INK if gain >= 0 else EEGDASH_ORANGE,
            "linewidth": 0.5,
            "alpha": 0.95,
        },
    )

    # Chance line.
    ax.axhline(
        float(chance_level),
        color=EEGDASH_MUTED,
        linestyle="--",
        linewidth=0.9,
        zorder=1.5,
    )
    # Chance label tucked into the bottom-left corner where the panel
    # has the most empty real estate; the oracle bar dominates the
    # right-hand side and the gain bracket sits up top.
    ax.text(
        0.02,
        0.04,
        f"chance = {float(chance_level):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=EEGDASH_MUTED,
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": EEGDASH_MUTED,
            "linewidth": 0.4,
            "alpha": 0.95,
        },
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.4, color=EEGDASH_INK)
    ax.set_xlim(-0.6, values.size - 0.4)
    # Headroom big enough for the gain pill above the bar tops.
    ax.set_ylim(0.0, max(1.15, float(values.max()) + 0.32))
    ax.set_ylabel("target-set accuracy", color=EEGDASH_INK)
    ax.set_title(
        "Target accuracy: naive vs AS-MMD vs oracle",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)


def _project_2d(features: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    feats = np.asarray(features, dtype=float)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    if feats.shape[1] < 2:
        feats = np.column_stack([feats, np.zeros_like(feats)])
    feats_std = StandardScaler(with_mean=True, with_std=True).fit_transform(feats)
    pca = PCA(n_components=2, random_state=0).fit(feats_std)
    coords = pca.transform(feats_std)
    var = tuple(float(v) for v in pca.explained_variance_ratio_)
    return coords, var


def _draw_embedding_panel(
    fig,
    gridspec_slot,
    *,
    embeddings: dict,
) -> None:
    inner = gridspec_slot.subgridspec(1, 2, wspace=0.10)
    ax_before = fig.add_subplot(inner[0, 0])
    ax_after = fig.add_subplot(inner[0, 1])

    src_before = np.asarray(embeddings["source_before"], dtype=float)
    tgt_before = np.asarray(embeddings["target_before"], dtype=float)
    src_after = np.asarray(embeddings["source_after"], dtype=float)
    tgt_after = np.asarray(embeddings["target_after"], dtype=float)

    # Joint PCA per panel keeps both clouds on the same axes for that
    # panel. PCA fit on the stacked data so source/target share basis.
    feats_before = np.vstack([src_before, tgt_before])
    feats_after = np.vstack([src_after, tgt_after])
    coords_before, var_b = _project_2d(feats_before)
    coords_after, var_a = _project_2d(feats_after)

    n_src_b = src_before.shape[0]
    n_src_a = src_after.shape[0]
    src_b_xy, tgt_b_xy = coords_before[:n_src_b], coords_before[n_src_b:]
    src_a_xy, tgt_a_xy = coords_after[:n_src_a], coords_after[n_src_a:]

    # Lock both panels to one set of axis limits for fair visual compare.
    all_xy = np.vstack([coords_before, coords_after])
    pad = 0.4
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    panels = (
        (ax_before, src_b_xy, tgt_b_xy, var_b, "before AS-MMD"),
        (ax_after, src_a_xy, tgt_a_xy, var_a, "after AS-MMD"),
    )
    for ax, src_xy, tgt_xy, var, label in panels:
        ax.scatter(
            src_xy[:, 0],
            src_xy[:, 1],
            s=22,
            facecolor=EEGDASH_BLUE,
            edgecolor=EEGDASH_INK,
            linewidth=0.4,
            alpha=0.78,
            label="source",
        )
        ax.scatter(
            tgt_xy[:, 0],
            tgt_xy[:, 1],
            s=22,
            facecolor=EEGDASH_ORANGE,
            edgecolor=EEGDASH_INK,
            linewidth=0.4,
            alpha=0.78,
            label="target",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f"PC1 ({var[0] * 100:.0f} % var)", color=EEGDASH_INK)
        if ax is ax_before:
            ax.set_ylabel(f"PC2 ({var[1] * 100:.0f} % var)", color=EEGDASH_INK)
        else:
            ax.set_yticklabels([])
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            family="monospace",
            color=EEGDASH_MUTED,
        )
        ax.grid(True, axis="both", color=EEGDASH_GRID, alpha=0.20, linewidth=0.5)
        _style_axis(ax)

    ax_before.legend(
        loc="lower right",
        fontsize=7.6,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
    )
    ax_before.text(
        0.0,
        1.06,
        "Feature space (PC1, PC2 of penultimate activations)",
        transform=ax_before.transAxes,
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _draw_erp_panel(
    ax,
    *,
    erp: dict,
    channel_label: str,
) -> None:
    times_ms = np.asarray(erp["times_ms"], dtype=float)
    src_curve = np.asarray(
        erp.get("source_after", erp.get("source_before")), dtype=float
    )
    tgt_curve = np.asarray(
        erp.get("target_after", erp.get("target_before")), dtype=float
    )
    src_se = np.asarray(erp.get("source_se", np.zeros_like(src_curve)), dtype=float)
    tgt_se = np.asarray(erp.get("target_se", np.zeros_like(tgt_curve)), dtype=float)

    # Vertical guides: stimulus onset and a faint zero line.
    ax.axvline(
        0.0, color=EEGDASH_INK, linewidth=0.8, linestyle="--", alpha=0.7, zorder=1
    )
    ax.axhline(0.0, color=EEGDASH_GRID, linewidth=0.6, alpha=0.7, zorder=1)

    # SE bands behind the curves so the lines stay readable on top.
    ax.fill_between(
        times_ms,
        src_curve - src_se,
        src_curve + src_se,
        color=EEGDASH_BLUE,
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    ax.fill_between(
        times_ms,
        tgt_curve - tgt_se,
        tgt_curve + tgt_se,
        color=EEGDASH_ORANGE,
        alpha=0.20,
        linewidth=0,
        zorder=2,
    )

    ax.plot(
        times_ms,
        src_curve,
        color=EEGDASH_BLUE_DARK,
        linewidth=1.8,
        label="source",
        zorder=4,
    )
    ax.plot(
        times_ms,
        tgt_curve,
        color=EEGDASH_ORANGE,
        linewidth=1.8,
        label="target",
        zorder=4,
    )

    # Highlight the canonical P3 search window so the reader can verify
    # both cohorts carry a centro-parietal positivity in the 300-450 ms
    # band; the alignment claim hinges on that band surviving.
    p300_lo, p300_hi = 300.0, 450.0
    ax.axvspan(
        p300_lo,
        p300_hi,
        color=EEGDASH_ORANGE,
        alpha=0.10,
        zorder=0,
    )
    # Anchored inside the orange band, low enough to clear the legend
    # in the upper-left and the curve peaks above ~3 uV.
    ax.text(
        (p300_lo + p300_hi) / 2,
        0.06,
        f"P3 search {int(p300_lo)}-{int(p300_hi)} ms",
        transform=ax.get_xaxis_transform(),
        color=EEGDASH_ORANGE,
        fontsize=7.6,
        ha="center",
        va="bottom",
        alpha=0.95,
    )

    ax.set_xlim(times_ms[0], times_ms[-1])
    ax.set_xlabel("time relative to stimulus (ms)", color=EEGDASH_INK)
    ax.set_ylabel(f"target - standard at {channel_label} (uV)", color=EEGDASH_INK)
    ax.set_title(
        f"ERP overlay at {channel_label}: source vs target",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.legend(
        loc="upper left",
        fontsize=7.8,
        frameon=False,
    )
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def draw_p300_transfer_figure(
    *,
    accuracies_dict: dict,
    embeddings_dict: dict,
    erp_dict: dict,
    chance_level: float = 0.5,
    channel_label: str = "Pz",
    source_id: str = "source",
    target_id: str = "target",
    plot_id: str = "project_p300_transfer",
):
    """Render the three-panel AS-MMD P3 transfer figure.

    Parameters
    ----------
    accuracies_dict : dict
        Keys ``naive``, ``mmd``, ``oracle`` with target-set accuracy
        floats. ``naive`` is the source-trained encoder evaluated on the
        target with no alignment; ``mmd`` is the same encoder with
        AS-MMD applied; ``oracle`` is a target-trained ceiling.
    embeddings_dict : dict
        Keys ``source_before``, ``target_before``, ``source_after``,
        ``target_after``. Each value is an ``(n, d)`` array of
        penultimate-layer activations on a fixed pool of windows.
        ``before`` = naive encoder; ``after`` = AS-MMD encoder.
    erp_dict : dict
        Keys ``times_ms``, ``source_before``, ``source_after``,
        ``target_before``, ``target_after``. Optional ``source_se`` and
        ``target_se`` for shaded standard-error bands on the post-
        alignment curves.
    chance_level : float, default ``0.5``
        Majority-class chance baseline drawn on the bar panel.
    channel_label : str, default ``"Pz"``
        Anchor channel used for the ERP overlay panel.
    source_id, target_id : str
        Dataset accessions used in the live subtitle.
    plot_id : str, default ``"project_p300_transfer"``
        Tutorial id forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure ready for ``plt.show()``.

    """
    fig = plt.figure(figsize=(15.4, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.10, 1.55, 1.30),
        wspace=0.42,
        left=0.06,
        right=0.97,
        top=0.78,
        bottom=0.20,
    )
    ax_bars = fig.add_subplot(gs[0, 0])
    ax_erp = fig.add_subplot(gs[0, 2])

    _draw_accuracy_panel(
        ax_bars,
        accuracies=accuracies_dict,
        chance_level=chance_level,
    )
    _draw_embedding_panel(
        fig,
        gs[0, 1],
        embeddings=embeddings_dict,
    )
    _draw_erp_panel(
        ax_erp,
        erp=erp_dict,
        channel_label=channel_label,
    )

    naive = float(accuracies_dict.get("naive", 0.0))
    mmd = float(accuracies_dict.get("mmd", 0.0))
    oracle = float(accuracies_dict.get("oracle", 0.0))
    subtitle = (
        f"source={source_id} | target={target_id} | "
        f"naive={naive:.02f} | mmd={mmd:.02f} | oracle={oracle:.02f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=source_id,
        citation="Polich 2007 (P3a/P3b); NEMAR :cite:`delorme2022nemar`",
        extra="encoder=ShallowFBCSPNet; AS-MMD on logit-space features",
    )
    style_figure(
        fig,
        title=("Does AS-MMD pull two oddball cohorts into a shared P3 subspace?"),
        subtitle=subtitle,
        source=source,
        grid_axis="none",
    )
    return fig


__all__ = ["draw_p300_transfer_figure"]
