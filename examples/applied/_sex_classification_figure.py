"""Drawing helpers for the ``project_sex_classification`` figure."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    balanced_accuracy_score,
    roc_auc_score,
)

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# Marker shapes cycle for up to ~7 cross-subject folds; three is typical.
_FOLD_MARKERS = ("o", "s", "D", "^", "v", "P", "X")


def _draw_pca_panel(
    ax,
    *,
    X_pca: np.ndarray,
    y_classes: np.ndarray,
    fold_assignment: np.ndarray,
    estimator,
    class_names: Sequence[str],
) -> None:
    # ``DecisionBoundaryDisplay`` is sklearn's canonical helper for this
    # exact job. The display draws first; the fold-shaped scatter overlay
    # is layered on top so the markers stay readable.
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X_pca,
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
                X_pca[mask, 0],
                X_pca[mask, 1],
                marker=marker,
                s=34,
                facecolor=class_colors[cls_idx],
                edgecolor=EEGDASH_INK,
                linewidth=0.5,
                alpha=0.85,
                label=f"{cls_label} | fold {fold_idx + 1}",
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.text(
        0.0,
        1.06,
        "PCA of band-power features",
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


def _draw_roc_panel(
    ax,
    *,
    y_true_pooled: np.ndarray,
    y_score_pooled: np.ndarray,
    fold_assignment: np.ndarray,
    fold_aucs: Sequence[float],
) -> None:
    fold_aucs_arr = np.asarray(fold_aucs, dtype=float)

    # Per-fold curves first so they sit underneath the pooled curve. The
    # AUC value appears in the legend automatically; we still pass the
    # display the per-fold positive scores so the curve geometry tracks
    # the held-out predictions. ``curve_kwargs`` is the sklearn 1.7+ way
    # to forward matplotlib keyword arguments to the curve (the old
    # ``**kwargs`` path is deprecated).
    folds = sorted(np.unique(fold_assignment).tolist())
    fold_curve_kw = {
        "color": EEGDASH_MUTED,
        "alpha": 0.75,
        "linewidth": 1.2,
    }
    for i, fold_id in enumerate(folds):
        mask = fold_assignment == fold_id
        if not mask.any():
            continue
        RocCurveDisplay.from_predictions(
            y_true=y_true_pooled[mask],
            y_score=y_score_pooled[mask],
            ax=ax,
            name=f"fold {i + 1}",
            curve_kwargs=fold_curve_kw,
            plot_chance_level=False,
        )

    # Pooled curve in dark ink on top.
    RocCurveDisplay.from_predictions(
        y_true=y_true_pooled,
        y_score=y_score_pooled,
        ax=ax,
        name="pooled",
        curve_kwargs={"color": EEGDASH_INK, "linewidth": 2.2},
        plot_chance_level=True,
        chance_level_kw={
            "linestyle": ":",
            "color": EEGDASH_MUTED,
            "label": "chance",
        },
    )

    mean_auc = float(np.mean(fold_aucs_arr))
    std_auc = float(np.std(fold_aucs_arr, ddof=0))
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.text(
        0.0,
        1.06,
        f"Cross-subject ROC: AUC = {mean_auc:.2f} +/- {std_auc:.2f}",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(
        loc="lower right",
        fontsize=7.4,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
        handlelength=2.0,
    )


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
    # ``values_format='.2f'`` keeps the cell labels short. ``Blues`` is
    # colorblind-safe; red/green diverging maps are the anti-pattern.
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
        "Pooled cross-subject confusion",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    # Normalized cells hide absolute counts. The monospace annotation
    # below the matrix carries the raw window count and the balanced
    # accuracy so the reader can still tell whether 0.55 came from 11/20
    # or 550/1000, and so the headline metric sits next to the matrix.
    n_test_windows = int(y_true.size)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    ax.text(
        0.5,
        -0.20,
        f"n_test_windows={n_test_windows}, balanced_acc={bal_acc:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        family="monospace",
        color=EEGDASH_INK,
    )

    # Hide grid lines on the heatmap so the cell numbers stay crisp.
    ax.grid(False)


def draw_sex_classification_figure(
    *,
    X_pca: np.ndarray,
    y_classes: np.ndarray,
    fold_assignment: np.ndarray,
    estimator: LogisticRegression,
    fold_aucs: Sequence[float],
    y_true_pooled: np.ndarray,
    y_pred_pooled: np.ndarray,
    y_score_pooled: np.ndarray,
    class_names: Sequence[str],
    plot_id: str = "project_sex_classification",
    n_train_subjects: int | None = None,
    n_test_subjects: int | None = None,
    n_features: int | None = None,
) -> plt.Figure:
    """Render the three-panel sex-classification diagnostic figure.

    Parameters
    ----------
    X_pca : numpy.ndarray
        ``(n_windows, 2)`` PCA-projected feature matrix used to draw the
        scatter and the decision-boundary contour.
    y_classes : numpy.ndarray
        ``(n_windows,)`` integer class labels (0 = first ``class_names``
        entry, 1 = second).
    fold_assignment : numpy.ndarray
        ``(n_windows,)`` integer fold id per window.
    estimator : sklearn estimator
        A fitted estimator that predicts on the 2-D PCA coordinates;
        :class:`sklearn.inspection.DecisionBoundaryDisplay` calls
        ``estimator.predict`` to draw the contour.
    fold_aucs : sequence of float
        Held-out ROC AUC per cross-subject fold.
    y_true_pooled, y_pred_pooled, y_score_pooled : numpy.ndarray
        ``(n_test_total,)`` ground-truth labels, hard predictions, and
        positive-class scores pooled across every cross-subject fold.
    class_names : sequence of str
        Display labels for the two classes (e.g. ``("F", "M")``).
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
        width_ratios=(1.25, 1.10, 1.10),
        wspace=0.55,
    )
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_roc = fig.add_subplot(gs[0, 1])
    ax_cm = fig.add_subplot(gs[0, 2])

    _draw_pca_panel(
        ax_pca,
        X_pca=X_pca,
        y_classes=y_classes,
        fold_assignment=fold_assignment,
        estimator=estimator,
        class_names=class_names,
    )
    _draw_roc_panel(
        ax_roc,
        y_true_pooled=y_true_pooled,
        y_score_pooled=y_score_pooled,
        fold_assignment=fold_assignment,
        fold_aucs=fold_aucs,
    )
    _draw_confusion_panel(
        ax_cm,
        y_true_pooled=y_true_pooled,
        y_pred_pooled=y_pred_pooled,
        class_names=class_names,
    )

    # Subtitle carries every live runtime value so the reader can compare
    # this figure against the budget the spec yaml pins. Subject counts
    # and the original feature count are not inferable from ``X_pca``
    # alone, so the caller passes them explicitly; the helper falls back
    # to placeholders only if a value is missing.
    n_folds = int(np.unique(fold_assignment).size)
    train_str = "?" if n_train_subjects is None else str(int(n_train_subjects))
    test_str = "?" if n_test_subjects is None else str(int(n_test_subjects))
    feat_str = "?" if n_features is None else str(int(n_features))
    pooled_auc = float(roc_auc_score(y_true_pooled, y_score_pooled))
    bal_acc = float(balanced_accuracy_score(y_true_pooled, y_pred_pooled))
    class_pair = " vs ".join(class_names)
    subtitle = (
        f"ds005505 RestingState | {class_pair} | n_train_subjects="
        f"{train_str} | n_test_subjects={test_str} | n_features={feat_str} "
        f"| n_folds={n_folds} | AUC={pooled_auc:.2f} "
        f"| balanced_acc={bal_acc:.2f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id="ds005505",
        citation="HBN; Alexander et al. 2017",
    )
    style_figure(
        fig,
        title=(
            f"Does resting-state EEG predict the BIDS sex label "
            f"({class_pair}) above chance?"
        ),
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )

    # ``style_figure`` resets the bottom margin to ``0.18``; the
    # multi-panel layout needs slightly more breathing room for the
    # confusion matrix's raw-count annotation and the ROC legend.
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.06, right=0.97)
    return fig


__all__ = ["draw_sex_classification_figure"]
