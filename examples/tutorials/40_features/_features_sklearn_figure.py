"""Drawing helpers for the ``plot_42`` features-to-sklearn figure."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    chance_line,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# Marker shapes cycle for up to ~7 cross-subject folds; four is typical
# for plot_42 (one per held-out subject in a 4-subject mock cohort).
_FOLD_MARKERS = ("o", "s", "D", "^", "v", "P", "X")


def _draw_pca_panel(
    ax,
    *,
    X_features: np.ndarray,
    y_classes: np.ndarray,
    fold_assignment: np.ndarray,
    class_names: Sequence[str],
) -> None:
    scaler = StandardScaler().fit(X_features)
    X_std = scaler.transform(X_features)
    pca = PCA(n_components=2, random_state=0).fit(X_std)
    components = pca.transform(X_std)
    var_explained = pca.explained_variance_ratio_

    # Fit a logistic regression in PCA space so the decision contour lives
    # in the same coordinates as the scattered points; the actual model
    # the tutorial reports lives in the original feature space.
    lr_pca = LogisticRegression(max_iter=400).fit(components, y_classes)

    # ``DecisionBoundaryDisplay`` is sklearn's canonical helper for this
    # exact job. The display draws first; the fold-shaped scatter overlay
    # is layered on top so the markers stay readable.
    DecisionBoundaryDisplay.from_estimator(
        lr_pca,
        components,
        ax=ax,
        alpha=0.3,
        response_method="predict",
        grid_resolution=200,
        cmap="coolwarm",
        eps=0.5,
    )

    class_colors = {0: EEGDASH_BLUE, 1: EEGDASH_ORANGE}
    folds = sorted(np.unique(fold_assignment).tolist())
    for fold_idx, fold_id in enumerate(folds):
        marker = _FOLD_MARKERS[fold_idx % len(_FOLD_MARKERS)]
        for cls_idx, cls_label in enumerate(class_names):
            mask = (fold_assignment == fold_id) & (y_classes == cls_idx)
            if not mask.any():
                continue
            ax.scatter(
                components[mask, 0],
                components[mask, 1],
                marker=marker,
                s=34,
                facecolor=class_colors[cls_idx],
                edgecolor=EEGDASH_INK,
                linewidth=0.5,
                alpha=0.85,
                label=f"{cls_label} | fold {fold_idx + 1}",
            )

    ax.set_xlabel(f"PC1 ({var_explained[0] * 100:.0f} % var)")
    ax.set_ylabel(f"PC2 ({var_explained[1] * 100:.0f} % var)")
    ax.text(
        0.0,
        1.06,
        "PCA of feature matrix",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    # Compact legend with one entry per class and one per fold-marker.
    class_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=class_colors[i],
            markeredgecolor=EEGDASH_INK,
            markersize=7,
            label=class_names[i],
        )
        for i in range(len(class_names))
    ]
    fold_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=_FOLD_MARKERS[i % len(_FOLD_MARKERS)],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=EEGDASH_INK,
            markersize=7,
            label=f"fold {i + 1}",
        )
        for i in range(len(folds))
    ]
    ax.legend(
        handles=class_handles + fold_handles,
        loc="lower right",
        fontsize=7.4,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
        ncols=1,
        handletextpad=0.4,
        columnspacing=0.9,
    )


def _draw_accuracy_panel(
    ax,
    *,
    fold_accuracies: Sequence[float],
    chance_level: float,
    held_out_subjects: Sequence[str],
) -> None:
    accuracies = np.asarray(fold_accuracies, dtype=float)
    n_folds = accuracies.size
    positions = np.arange(n_folds)
    mean_acc = float(accuracies.mean())
    std_acc = float(accuracies.std(ddof=0))

    # Mean +/- std band first so the bars sit above it.
    ax.axhspan(
        mean_acc - std_acc,
        mean_acc + std_acc,
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
        accuracies,
        width=0.62,
        color=EEGDASH_ORANGE,
        edgecolor=EEGDASH_INK,
        linewidth=0.8,
        zorder=3,
    )
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.018,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color=EEGDASH_INK,
            fontweight="bold",
        )

    chance_line(ax, level=float(chance_level), label="chance")

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"fold {i + 1}\n{sid}" for i, sid in enumerate(held_out_subjects)],
        fontsize=8.4,
        color=EEGDASH_INK,
    )
    ax.set_ylim(0.0, max(1.0, accuracies.max() + 0.18))
    ax.set_ylabel("accuracy")
    ax.text(
        0.0,
        1.06,
        f"LOSO CV: {mean_acc:.2f} +/- {std_acc:.2f}",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.legend(loc="upper right", fontsize=7.6, frameon=False)


def _draw_confusion_panel(
    ax,
    *,
    y_true_pooled: np.ndarray,
    y_pred_pooled: np.ndarray,
    class_names: Sequence[str],
) -> None:
    y_true = np.asarray(y_true_pooled).ravel()
    y_pred = np.asarray(y_pred_pooled).ravel()

    # ``ConfusionMatrixDisplay`` does the imshow + cell-text plumbing
    # itself; ``normalize='true'`` gives a row-normalized matrix and
    # ``values_format='.2f'`` keeps the cell labels short.
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
        "Pooled LOSO: confusion matrix",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    # Normalized cells hide absolute counts; the small monospace
    # annotation below the matrix carries the raw counts so the reader
    # can still tell whether 0.92 came from 22/24 or 220/240.
    n_test_windows = int(y_true.size)
    n_correct = int((y_true == y_pred).sum())
    ax.text(
        0.5,
        -0.18,
        f"n_test_windows={n_test_windows}, n_correct={n_correct}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        family="monospace",
        color=EEGDASH_INK,
    )

    # Hide grid lines on the heatmap so the cell numbers stay crisp.
    ax.grid(False)


def draw_features_sklearn_figure(
    *,
    X_features: np.ndarray,
    y_classes: np.ndarray,
    fold_assignment: np.ndarray,
    fold_accuracies: Sequence[float],
    y_true_pooled: np.ndarray,
    y_pred_pooled: np.ndarray,
    class_names: Sequence[str],
    subjects: Sequence[str],
    plot_id: str = "plot_42",
) -> plt.Figure:
    """Render the three-panel features-to-sklearn diagnostic figure.

    Parameters
    ----------
    X_features : numpy.ndarray
        ``(n_windows, n_features)`` band-power + variance feature matrix.
    y_classes : numpy.ndarray
        ``(n_windows,)`` integer class labels (0 = eyes-open,
        1 = eyes-closed in the plot_42 mock).
    fold_assignment : numpy.ndarray
        ``(n_windows,)`` integer fold id per window (held-out fold for
        that window in a leave-one-subject-out scheme).
    fold_accuracies : sequence of float
        Per-fold held-out accuracy across the cross-subject folds.
    y_true_pooled, y_pred_pooled : numpy.ndarray
        ``(n_test_total,)`` ground-truth labels and held-out predictions
        pooled across every cross-subject fold. Used to render the
        confusion matrix; ``ConfusionMatrixDisplay.from_predictions``
        computes the matrix internally.
    class_names : sequence of str
        Display labels for the two classes (e.g. ``("eyes-open",
        "eyes-closed")``).
    subjects : sequence of str
        Held-out subject id on each fold (one per ``fold_accuracies``
        bar). The full cohort feeds the subtitle.
    plot_id : str
        Tutorial id forwarded to :func:`add_provenance_footer`.

    Returns
    -------
    matplotlib.figure.Figure

    """
    fig = plt.figure(figsize=(14.6, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.25, 1.0, 1.10),
        wspace=0.62,
    )
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_cm = fig.add_subplot(gs[0, 2])

    accuracies = np.asarray(fold_accuracies, dtype=float)
    n_folds = int(accuracies.size)
    held_out_subjects = list(subjects)

    # Chance level on a balanced cross-subject contrast lands at 1/n_classes
    # by construction. The plot_42 mock keeps classes balanced in every
    # fold, so the held-out test mode-frequency lands at 0.5; using a
    # constant here keeps the chance reference identical across folds and
    # matches what ``majority_baseline`` reports on this contrast.
    chance_value = 1.0 / max(len(class_names), 2)

    _draw_pca_panel(
        ax_pca,
        X_features=X_features,
        y_classes=y_classes,
        fold_assignment=fold_assignment,
        class_names=class_names,
    )
    _draw_accuracy_panel(
        ax_acc,
        fold_accuracies=accuracies,
        chance_level=chance_value,
        held_out_subjects=held_out_subjects,
    )
    _draw_confusion_panel(
        ax_cm,
        y_true_pooled=y_true_pooled,
        y_pred_pooled=y_pred_pooled,
        class_names=class_names,
    )

    n_test_per_fold = max(1, len(held_out_subjects) // max(1, n_folds))
    n_total_subjects = len(set(held_out_subjects))
    n_train_per_fold = max(1, n_total_subjects - n_test_per_fold)
    n_features = X_features.shape[1]
    class_pair = " vs ".join(class_names)
    subtitle = (
        f"{class_pair} | per-fold n_train_subjects="
        f"{n_train_per_fold} | n_test_subjects={n_test_per_fold} | "
        f"n_features={n_features} | n_folds={n_folds}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id="ds005514",
        citation="HBN, mock for plot_42",
    )
    style_figure(
        fig,
        title=(
            f"Does StandardScaler -> LogisticRegression on EEGDash features "
            f"separate {class_pair}?"
        ),
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )

    # ``style_figure`` resets the bottom margin to ``0.18``; the
    # multi-panel layout needs slightly more breathing room for the
    # held-out-subject x-tick labels, panel titles, and the confusion
    # matrix's raw-count annotation.
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.06, right=0.97)
    return fig


__all__ = ["draw_features_sklearn_figure"]
