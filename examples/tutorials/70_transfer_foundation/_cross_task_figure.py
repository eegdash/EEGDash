"""Drawing helpers for the ``plot_71`` cross-task transfer figure."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eegdash.viz import (
    EEGDASH_BLUE,
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


def _draw_transfer_matrix(
    ax,
    *,
    transfer_matrix: np.ndarray,
    source_tasks: Sequence[str],
    target_tasks: Sequence[str],
) -> None:
    matrix = np.asarray(transfer_matrix, dtype=float)
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    vmax = max(vmax, 0.02)

    im = ax.imshow(
        matrix,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    # Cell text. Pick black or white based on the underlying tile so the
    # numbers stay readable on the saturated ends of the colormap.
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            tone = "white" if abs(value) > 0.55 * vmax else EEGDASH_INK
            ax.text(
                c,
                r,
                f"{value:+.02f}",
                ha="center",
                va="center",
                fontsize=8.5,
                family="monospace",
                color=tone,
            )

    ax.set_xticks(np.arange(len(target_tasks)))
    ax.set_xticklabels(list(target_tasks), rotation=20, ha="right", fontsize=8.4)
    ax.set_yticks(np.arange(len(source_tasks)))
    ax.set_yticklabels(list(source_tasks), fontsize=8.4)
    ax.set_xlabel("downstream target task", color=EEGDASH_INK)
    ax.set_ylabel("pretrained source task", color=EEGDASH_INK)
    ax.set_title(
        "Δacc vs from-scratch (blue helps, orange hurts)",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.grid(False)

    cbar = ax.figure.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.046,
        pad=0.04,
        shrink=0.78,
    )
    cbar.set_label("Δaccuracy", color=EEGDASH_INK, fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5, colors=EEGDASH_INK, length=0)
    cbar.outline.set_visible(False)


def _draw_bar_panel(
    ax,
    *,
    target_tasks: Sequence[str],
    scratch_acc: Sequence[float],
    finetune_acc: Sequence[float],
    chance_level: float,
) -> None:
    scratch = np.asarray(scratch_acc, dtype=float)
    finetune = np.asarray(finetune_acc, dtype=float)
    n = scratch.size
    width = 0.36
    x = np.arange(n)

    bars_scratch = ax.bar(
        x - width / 2,
        scratch,
        width=width,
        color=EEGDASH_BLUE,
        edgecolor=EEGDASH_INK,
        linewidth=0.6,
        label="from-scratch",
        zorder=3,
    )
    bars_finetune = ax.bar(
        x + width / 2,
        finetune,
        width=width,
        color=EEGDASH_ORANGE,
        edgecolor=EEGDASH_INK,
        linewidth=0.6,
        label="pretrained -> finetuned",
        zorder=3,
    )

    for bs, bf, sc, ft in zip(bars_scratch, bars_finetune, scratch, finetune):
        # Per-bar value labels in monospace.
        for bar, value in ((bs, sc), (bf, ft)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
                family="monospace",
                color=EEGDASH_INK,
                zorder=4,
            )
        # Δacc bracket above the bar pair.
        gain = ft - sc
        x_centre = (bs.get_x() + bs.get_width() + bf.get_x()) / 2
        y_top = max(bs.get_height(), bf.get_height()) + 0.085
        ax.text(
            x_centre,
            y_top,
            f"Δ={gain:+.02f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
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

    # Chance line drawn inline so the right-side label does not collide
    # with the embedding panel sitting next to this one in the gridspec.
    ax.axhline(
        float(chance_level),
        color=EEGDASH_MUTED,
        linestyle="--",
        linewidth=0.9,
        zorder=1.5,
        label=f"chance = {float(chance_level):.2f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(list(target_tasks), rotation=20, ha="right", fontsize=8.4)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0.0, max(1.0, float(max(scratch.max(), finetune.max())) + 0.30))
    ax.set_ylabel("test accuracy", color=EEGDASH_INK)
    ax.set_title(
        "From-scratch vs finetuned (Δacc per target)",
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.legend(
        loc="upper right",
        fontsize=7.8,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
        ncols=1,
    )
    ax.grid(True, axis="y", color=EEGDASH_GRID, alpha=0.28, linewidth=0.6)
    _style_axis(ax)


def _project_2d(features: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    feats = np.asarray(features, dtype=float)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    if feats.shape[1] < 2:
        # Pad to two dims so PCA is well-defined; rare edge case.
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
    embeddings_scratch: np.ndarray,
    embeddings_finetuned: np.ndarray,
    classes_target: np.ndarray,
    class_names: Sequence[str],
) -> None:
    inner = gridspec_slot.subgridspec(1, 2, wspace=0.10)
    ax_scratch = fig.add_subplot(inner[0, 0])
    ax_finetune = fig.add_subplot(inner[0, 1])

    coords_scratch, var_s = _project_2d(np.asarray(embeddings_scratch))
    coords_finetune, var_f = _project_2d(np.asarray(embeddings_finetuned))
    classes = np.asarray(classes_target).astype(int)
    class_colors = {0: EEGDASH_BLUE, 1: EEGDASH_ORANGE}

    # Lock both panels to one set of axis limits so the reader's eye can
    # compare cluster spread without rescaling.
    all_xy = np.vstack([coords_scratch, coords_finetune])
    pad = 0.4
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    for ax, coords, var, label in (
        (ax_scratch, coords_scratch, var_s, "from-scratch"),
        (ax_finetune, coords_finetune, var_f, "pretrained -> finetuned"),
    ):
        for cls_idx, cls_label in enumerate(class_names):
            mask = classes == cls_idx
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=24,
                facecolor=class_colors.get(cls_idx, EEGDASH_INK),
                edgecolor=EEGDASH_INK,
                linewidth=0.4,
                alpha=0.78,
                label=cls_label,
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f"PC1 ({var[0] * 100:.0f} % var)", color=EEGDASH_INK)
        if ax is ax_scratch:
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

    # Single legend on the left axes is enough; both panels share classes.
    ax_scratch.legend(
        loc="lower right",
        fontsize=7.6,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
    )
    # Shared title spans the joint subplot pair via axes-fraction text on
    # ax_scratch; loc='left' on a half-width axes truncates the string.
    ax_scratch.text(
        0.0,
        1.06,
        "Encoder feature space (PC1, PC2 of penultimate activations)",
        transform=ax_scratch.transAxes,
        color=EEGDASH_INK,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def draw_cross_task_figure(
    *,
    transfer_matrix: np.ndarray,
    source_task: str,
    target_tasks: Sequence[str],
    scratch_acc: Sequence[float],
    finetune_acc: Sequence[float],
    embeddings_scratch: np.ndarray,
    embeddings_finetuned: np.ndarray,
    classes_target: np.ndarray,
    chance_level: float = 0.5,
    class_names: Sequence[str] = ("class 0", "class 1"),
    source_tasks_full: Sequence[str] | None = None,
    plot_id: str = "plot_71",
):
    """Render the three-panel cross-task transfer figure.

    Parameters
    ----------
    transfer_matrix : numpy.ndarray
        ``(n_source, n_target)`` accuracy-delta matrix vs the
        from-scratch baseline.
    source_task : str
        Anchor source task highlighted in the subtitle (e.g.
        ``"RestingState"``).
    target_tasks : sequence of str
        Downstream target task names; one per column of
        ``transfer_matrix`` and one bar pair in the centre panel.
    scratch_acc, finetune_acc : sequence of float
        Per-target test accuracy of the from-scratch baseline and the
        pretrained-then-finetuned encoder. Same length as
        ``target_tasks``.
    embeddings_scratch, embeddings_finetuned : numpy.ndarray
        ``(n_target_windows, d)`` penultimate-layer activations of each
        encoder on the same target-task windows.
    classes_target : numpy.ndarray
        ``(n_target_windows,)`` integer labels matching the embeddings.
    chance_level : float, default ``0.5``
        Majority-class baseline drawn on the bar panel.
    class_names : sequence of str, default ``("class 0", "class 1")``
        Display labels for the two classes (in label-id order).
    source_tasks_full : sequence of str or None
        Row labels for the transfer matrix. Defaults to a single-row
        matrix with ``source_task`` if ``transfer_matrix`` has one row.
    plot_id : str, default ``"plot_71"``
        Tutorial id forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure ready for ``plt.show()``.

    """
    matrix = np.asarray(transfer_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if source_tasks_full is None:
        source_tasks_full = (
            [source_task]
            if matrix.shape[0] == 1
            else [f"src_{i}" for i in range(matrix.shape[0])]
        )

    fig = plt.figure(figsize=(15.4, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.30, 1.10, 1.55),
        wspace=0.42,
        left=0.06,
        right=0.97,
        top=0.78,
        bottom=0.20,
    )
    ax_matrix = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[0, 1])

    _draw_transfer_matrix(
        ax_matrix,
        transfer_matrix=matrix,
        source_tasks=source_tasks_full,
        target_tasks=target_tasks,
    )
    _draw_bar_panel(
        ax_bars,
        target_tasks=target_tasks,
        scratch_acc=scratch_acc,
        finetune_acc=finetune_acc,
        chance_level=chance_level,
    )
    _draw_embedding_panel(
        fig,
        gs[0, 2],
        embeddings_scratch=embeddings_scratch,
        embeddings_finetuned=embeddings_finetuned,
        classes_target=classes_target,
        class_names=class_names,
    )

    # Live subtitle: pretrain task | n target tasks | mean gain | best target.
    gains = np.asarray(finetune_acc, dtype=float) - np.asarray(scratch_acc, dtype=float)
    mean_gain = float(gains.mean()) if gains.size else 0.0
    if gains.size:
        best_idx = int(np.argmax(gains))
        best_target = list(target_tasks)[best_idx]
        best_gain = float(gains[best_idx])
    else:
        best_target = "n/a"
        best_gain = 0.0

    subtitle = (
        f"pretrain={source_task} | n_target_tasks={len(list(target_tasks))} | "
        f"mean Δacc={mean_gain:+.02f} | best target={best_target} ({best_gain:+.02f})"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id="ds005514",
        citation="Alexander et al. 2017 (HBN); EEG2025 Challenge mini",
        extra="encoder=ShallowFBCSPNet; pretrain on passive, finetune on active",
    )
    style_figure(
        fig,
        title=("Does pretraining on RestingState transfer to active EEG2025 tasks?"),
        subtitle=subtitle,
        source=source,
        grid_axis="none",
    )
    return fig


__all__ = ["draw_cross_task_figure"]
