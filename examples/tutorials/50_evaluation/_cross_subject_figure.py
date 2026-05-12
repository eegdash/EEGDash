"""Drawing helpers for the ``plot_51`` cross-subject evaluation figure."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    chance_line,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


def _draw_transfer_matrix(
    ax,
    *,
    transfer_matrix: np.ndarray,
    subject_ids: Sequence[str],
) -> None:
    matrix = np.asarray(transfer_matrix, dtype=float)
    n = matrix.shape[0]
    masked = np.ma.array(matrix, mask=np.eye(n, dtype=bool))

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white")
    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=0.3,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )

    # Per-cell value annotations in monospace; pick a contrasting color
    # so the number stays readable against the deep-blue cells.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = float(matrix[i, j])
            color = "white" if value >= 0.72 else EEGDASH_INK
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.6,
                family="monospace",
                color=color,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_ids = [str(s).replace("sub-", "") for s in subject_ids]
    ax.set_xticklabels(short_ids, fontsize=7.6, family="monospace")
    ax.set_yticklabels(short_ids, fontsize=7.6, family="monospace")
    ax.set_xlabel("test subject (held out)")
    ax.set_ylabel("source subject (in training fold)")
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)
    ax.grid(False)
    ax.text(
        0.0,
        1.06,
        "Train-test transfer (balanced accuracy)",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
    cbar.set_label("balanced accuracy", fontsize=8.4, color=EEGDASH_INK)
    cbar.ax.tick_params(labelsize=7.6, colors=EEGDASH_INK)
    cbar.outline.set_edgecolor(EEGDASH_MUTED)
    cbar.outline.set_linewidth(0.6)


def _draw_loso_bars(
    ax,
    *,
    fold_accuracies: Sequence[float],
    held_out_subjects: Sequence[str],
    chance_level: float,
) -> None:
    accuracies = np.asarray(fold_accuracies, dtype=float)
    order = np.argsort(accuracies)
    sorted_acc = accuracies[order]
    sorted_subjects = [str(held_out_subjects[i]).replace("sub-", "") for i in order]

    n_folds = sorted_acc.size
    positions = np.arange(n_folds)
    mean_acc = float(sorted_acc.mean())
    std_acc = float(sorted_acc.std(ddof=0))

    ax.axhspan(
        max(0.0, mean_acc - std_acc),
        min(1.0, mean_acc + std_acc),
        color=EEGDASH_BLUE,
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )
    ax.axhline(
        mean_acc,
        color=EEGDASH_BLUE,
        linestyle=":",
        linewidth=1.0,
        zorder=2,
        label=f"mean = {mean_acc:.2f}",
    )

    bars = ax.bar(
        positions,
        sorted_acc,
        width=0.62,
        color=EEGDASH_ORANGE,
        edgecolor=EEGDASH_INK,
        linewidth=0.8,
        zorder=3,
    )
    for bar, acc in zip(bars, sorted_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.018,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=EEGDASH_INK,
            fontweight="bold",
        )

    chance_line(ax, level=float(chance_level), label="chance")

    ax.set_xticks(positions)
    ax.set_xticklabels(sorted_subjects, fontsize=8.0, family="monospace")
    ax.set_xlabel("held-out subject")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.text(
        0.0,
        1.06,
        f"LOSO per-subject: {mean_acc:.2f} +/- {std_acc:.2f}",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.legend(
        loc="lower left",
        fontsize=7.6,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
    )


def _draw_pooled_confusion(
    ax,
    *,
    y_true_pooled: np.ndarray,
    y_pred_pooled: np.ndarray,
    class_names: Sequence[str],
) -> None:
    y_true = np.asarray(y_true_pooled).ravel()
    y_pred = np.asarray(y_pred_pooled).ravel()

    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=list(class_names),
        cmap="Blues",
        colorbar=False,
        ax=ax,
        normalize="true",
        values_format=".2f",
    )

    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.text(
        0.0,
        1.06,
        "Pooled LOSO confusion matrix",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    n_test_windows = int(y_true.size)
    pooled_balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    ax.text(
        0.5,
        -0.20,
        f"n_test_windows={n_test_windows}, balanced_acc={pooled_balanced_acc:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        family="monospace",
        color=EEGDASH_INK,
    )

    ax.grid(False)


def draw_cross_subject_figure(
    *,
    transfer_matrix: np.ndarray,
    subject_ids: Sequence[str],
    fold_accuracies: Sequence[float],
    y_true_pooled: np.ndarray,
    y_pred_pooled: np.ndarray,
    class_names: Sequence[str],
    held_out_subjects: Sequence[str] | None = None,
    chance_level: float = 0.5,
    plot_id: str = "plot_51",
) -> plt.Figure:
    """Render the three-panel cross-subject evaluation figure.

    Parameters
    ----------
    transfer_matrix : numpy.ndarray
        ``(n_subjects, n_subjects)`` square matrix of balanced accuracy.
        Rows index the source subject (in the training fold); columns
        index the held-out test subject. The diagonal is ignored
        (within-subject scores are not the cross-subject question).
    subject_ids : sequence of str
        Subject ids in the same order as the rows / columns of
        ``transfer_matrix``. ``"sub-XX"`` prefixes are stripped from
        the tick labels for compactness.
    fold_accuracies : sequence of float
        Per-fold balanced accuracy across the LOSO loop. Bars are
        sorted ascending so the panel reads worst -> best.
    y_true_pooled, y_pred_pooled : numpy.ndarray
        ``(n_test_total,)`` ground-truth labels and held-out predictions
        pooled across every LOSO fold. ``ConfusionMatrixDisplay`` runs
        its own ``normalize="true"`` reduction.
    class_names : sequence of str
        Display labels for the classes (e.g. ``("class 0", "class 1")``).
    held_out_subjects : sequence of str, optional
        Subject id held out on each LOSO fold (one per
        ``fold_accuracies`` entry). Falls back to ``subject_ids`` when
        the LOSO loop covers every subject in the cohort.
    chance_level : float, default 0.5
        Reference chance accuracy from
        ``majority_baseline`` (or 0.5 for a balanced
        binary task). Drawn as a dashed horizontal line on Panel 2.
    plot_id : str, default ``"plot_51"``
        Tutorial id forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure

    """
    transfer_matrix = np.asarray(transfer_matrix, dtype=float)
    if (
        transfer_matrix.ndim != 2
        or transfer_matrix.shape[0] != transfer_matrix.shape[1]
    ):
        raise ValueError(
            "transfer_matrix must be a square 2-D array; got shape "
            f"{transfer_matrix.shape}."
        )
    if transfer_matrix.shape[0] != len(subject_ids):
        raise ValueError(
            f"len(subject_ids)={len(subject_ids)} does not match "
            f"transfer_matrix side ({transfer_matrix.shape[0]})."
        )
    held_out = (
        list(held_out_subjects) if held_out_subjects is not None else list(subject_ids)
    )
    if len(held_out) != len(fold_accuracies):
        raise ValueError(
            f"len(held_out_subjects)={len(held_out)} must equal "
            f"len(fold_accuracies)={len(fold_accuracies)}."
        )

    fig = plt.figure(figsize=(14.4, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.20, 1.05, 0.95),
        wspace=0.50,
    )
    ax_transfer = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_cm = fig.add_subplot(gs[0, 2])

    _draw_transfer_matrix(
        ax_transfer,
        transfer_matrix=transfer_matrix,
        subject_ids=subject_ids,
    )
    _draw_loso_bars(
        ax_bars,
        fold_accuracies=fold_accuracies,
        held_out_subjects=held_out,
        chance_level=chance_level,
    )
    _draw_pooled_confusion(
        ax_cm,
        y_true_pooled=y_true_pooled,
        y_pred_pooled=y_pred_pooled,
        class_names=class_names,
    )

    n_subjects = len(subject_ids)
    n_folds = len(fold_accuracies)
    accuracies = np.asarray(fold_accuracies, dtype=float)
    mean_acc = float(accuracies.mean())
    std_acc = float(accuracies.std(ddof=0))

    subtitle = (
        f"n_subjects={n_subjects} | n_folds={n_folds} | "
        f"mean_loso_acc={mean_acc:.2f} +/- {std_acc:.2f} | "
        f"chance_balanced_acc={chance_level:.2f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        citation="synthetic cross-subject cohort",
    )

    style_figure(
        fig,
        title="How well does an EEG decoder generalize to a never-seen subject?",
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )

    # ``style_figure`` resets the bottom margin to 0.18; the three-panel
    # layout needs more breathing room for the tick labels, panel titles,
    # and the confusion-matrix annotation strip.
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.06, right=0.98)
    return fig


__all__ = ["draw_cross_subject_figure"]
