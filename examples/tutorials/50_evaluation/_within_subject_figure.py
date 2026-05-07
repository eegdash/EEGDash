"""Drawing helpers for the ``plot_50`` within-subject diagnostic figure.

Sibling module to ``plot_50_within_subject_evaluation.py``. The leading
underscore makes sphinx-gallery skip this file, so the rendering plumbing
stays out of the rendered tutorial; the tutorial imports the public
:func:`draw_within_subject_figure` entry point.

The figure has three panels arranged left-to-right:

1. Per-subject within-CV accuracy bars. One bar per subject, height = mean
   within-subject 5-fold accuracy, error bar = +/- std. Bars colored
   uniformly with :data:`~eegdash.viz.EEGDASH_BLUE`. A dashed
   :func:`~eegdash.viz.chance_line` reference at ``level=0.5`` and a solid
   :data:`~eegdash.viz.EEGDASH_ORANGE` reference line for the cross-subject
   pooled accuracy.
2. Variance comparison between within-subject CV and cross-subject CV,
   rendered as side-by-side boxplots.
3. Pooled :class:`sklearn.metrics.ConfusionMatrixDisplay` over all
   within-subject test predictions, ``Blues`` colormap, ``normalize="true"``.

Mirrors the visual identity of ``_leakage_figure.py`` (plot_11): same blue
+ orange anchor, same shared :func:`~eegdash.viz.style_figure` driver, same
provenance footer.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    chance_line,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


def _draw_per_subject_panel(
    ax,
    *,
    per_subject_accuracies: np.ndarray,
    cross_subject_accuracy: float,
    subjects: Sequence[str],
) -> None:
    """Per-subject within-CV accuracy bars with chance + cross-subject refs.

    ``per_subject_accuracies`` is a ``(n_subjects, n_folds)`` matrix; bar
    height is the per-subject mean and the error bar is the per-subject
    standard deviation across folds.
    """
    accuracies = np.asarray(per_subject_accuracies, dtype=float)
    if accuracies.ndim != 2:
        raise ValueError("per_subject_accuracies must be 2-D (n_subjects, n_folds)")
    n_subjects = accuracies.shape[0]
    means = accuracies.mean(axis=1)
    stds = accuracies.std(axis=1, ddof=0)
    positions = np.arange(n_subjects)

    bars = ax.bar(
        positions,
        means,
        yerr=stds,
        width=0.68,
        color=EEGDASH_BLUE,
        edgecolor=EEGDASH_INK,
        linewidth=0.7,
        ecolor=EEGDASH_INK,
        capsize=3.0,
        zorder=3,
    )

    # Bar-top accuracy labels for read-at-a-glance comparison.
    for bar, mean_value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_value + 0.025,
            f"{mean_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=EEGDASH_INK,
            fontweight="bold",
        )

    # Cross-subject pooled accuracy as a solid orange reference line. Two
    # very different questions ("can a model calibrate to subject X?" vs
    # "does it transfer to a new person?") share the same y-axis, so a
    # second annotated reference is the cleanest way to compare them.
    cs_handle = ax.axhline(
        float(cross_subject_accuracy),
        color=EEGDASH_ORANGE,
        linestyle="-",
        linewidth=1.4,
        zorder=2,
        label=f"cross-subject = {cross_subject_accuracy:.2f}",
    )

    chance_line(ax, level=0.5, label="chance")

    ax.set_xticks(positions)
    # Strip the redundant ``sub-`` prefix; the x-axis label already says
    # ``subject``. Rotate so 12+ ids fit without overlapping.
    short_ids = [str(sid).replace("sub-", "") for sid in subjects]
    ax.set_xticklabels(
        short_ids,
        rotation=45,
        ha="right",
        fontsize=7.8,
        color=EEGDASH_INK,
    )
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("subject")
    ax.text(
        0.0,
        1.06,
        "Per-subject within-CV accuracy",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )
    ax.legend(
        handles=[cs_handle],
        loc="lower left",
        fontsize=7.6,
        frameon=True,
        framealpha=0.92,
        edgecolor=EEGDASH_MUTED,
    )


def _draw_variance_panel(
    ax,
    *,
    within_distribution: np.ndarray,
    cross_distribution: np.ndarray,
) -> None:
    """Side-by-side boxplots: within-subject CV vs cross-subject CV.

    The point of the panel: within-subject variance is typically tighter
    than cross-subject variance, because inter-subject differences (skull
    geometry, baseline alpha amplitude, electrode placement) dominate the
    cross-subject error budget but are absorbed inside the per-subject
    loop.
    """
    within = np.asarray(within_distribution, dtype=float).ravel()
    cross = np.asarray(cross_distribution, dtype=float).ravel()
    data = [within, cross]
    positions = [1, 2]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": EEGDASH_INK, "linewidth": 1.4},
        whiskerprops={"color": EEGDASH_INK, "linewidth": 0.8},
        capprops={"color": EEGDASH_INK, "linewidth": 0.8},
        flierprops={
            "marker": "o",
            "markersize": 3.0,
            "markerfacecolor": EEGDASH_MUTED,
            "markeredgecolor": EEGDASH_INK,
            "alpha": 0.65,
        },
    )
    box_colors = (EEGDASH_BLUE, EEGDASH_ORANGE)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(EEGDASH_INK)
        patch.set_linewidth(0.8)

    # Strip plot overlay so the reader sees the actual fold/subject points.
    rng = np.random.default_rng(0)
    for pos, sample, color in zip(positions, data, box_colors):
        if sample.size == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=sample.size)
        ax.scatter(
            np.full_like(sample, pos) + jitter,
            sample,
            s=18,
            facecolor=color,
            edgecolor=EEGDASH_INK,
            linewidth=0.4,
            alpha=0.75,
            zorder=4,
        )

    chance_line(ax, level=0.5, label="chance")

    within_std = float(within.std(ddof=0)) if within.size else 0.0
    cross_std = float(cross.std(ddof=0)) if cross.size else 0.0
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"within-subject\n(std = {within_std:.3f})",
            f"cross-subject\n(std = {cross_std:.3f})",
        ],
        fontsize=8.4,
        color=EEGDASH_INK,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.text(
        0.0,
        1.06,
        "Variance: within-subject vs cross-subject",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )


def _draw_confusion_panel(
    ax,
    *,
    y_true_pooled: np.ndarray,
    y_pred_pooled: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """Pooled within-subject confusion matrix in the Blues colormap."""
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
        "Within-subject CV confusion matrix (pooled)",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=EEGDASH_INK,
        ha="left",
        va="bottom",
    )

    n_test_windows = int(y_true.size)
    n_correct = int((y_true == y_pred).sum())
    ax.text(
        0.5,
        -0.20,
        f"n_pooled_test_windows={n_test_windows}, n_correct={n_correct}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        family="monospace",
        color=EEGDASH_INK,
    )
    ax.grid(False)


def draw_within_subject_figure(
    *,
    per_subject_accuracies: np.ndarray,
    cross_subject_accuracy: float,
    y_true_pooled: np.ndarray,
    y_pred_pooled: np.ndarray,
    class_names: Sequence[str],
    subjects: Sequence[str],
    cross_fold_accuracies: Sequence[float] | None = None,
    n_windows_per_subject: int | None = None,
    plot_id: str = "plot_50",
) -> plt.Figure:
    """Render the three-panel within-subject diagnostic figure.

    Parameters
    ----------
    per_subject_accuracies : numpy.ndarray
        ``(n_subjects, n_folds)`` matrix of per-subject within-CV fold
        accuracies. Bar height = row mean; error bar = row std.
    cross_subject_accuracy : float
        Pooled accuracy under a leave-one-subject-out (cross-subject)
        loop on the same data, drawn as a horizontal orange reference on
        panel 1 and as the right-hand boxplot input on panel 2.
    y_true_pooled, y_pred_pooled : numpy.ndarray
        Concatenated within-subject test predictions across every fold,
        used to render the pooled confusion matrix.
    class_names : sequence of str
        Display labels for the two classes.
    subjects : sequence of str
        Subject ids in the same order as the rows of
        ``per_subject_accuracies``.
    cross_fold_accuracies : sequence of float, optional
        Distribution of cross-subject fold accuracies (one per held-out
        subject). When ``None``, falls back to a one-point distribution
        at ``cross_subject_accuracy`` so panel 2 still renders.
    n_windows_per_subject : int, optional
        Forwarded into the subtitle when provided.
    plot_id : str, default ``"plot_50"``
        Tutorial id used in the provenance footer string.

    Returns
    -------
    matplotlib.figure.Figure

    """
    accuracies = np.asarray(per_subject_accuracies, dtype=float)
    if accuracies.ndim != 2:
        raise ValueError(
            "per_subject_accuracies must be 2-D (n_subjects, n_folds); "
            f"got shape {accuracies.shape}."
        )
    if accuracies.shape[0] != len(subjects):
        raise ValueError(
            f"len(subjects)={len(subjects)} does not match the row count "
            f"of per_subject_accuracies ({accuracies.shape[0]})."
        )

    n_subjects, n_folds = accuracies.shape
    # Per-subject mean keeps the comparison apples-to-apples: each
    # within-subject point is one subject's CV-mean, each cross-subject
    # point is one held-out-subject's accuracy. Both have ``n_subjects``
    # entries so the boxplot widths are not driven by sample count.
    within_distribution = np.nanmean(accuracies, axis=1)
    if cross_fold_accuracies is None:
        cross_distribution = np.asarray([float(cross_subject_accuracy)])
    else:
        cross_distribution = np.asarray(cross_fold_accuracies, dtype=float)

    fig = plt.figure(figsize=(14.0, 5.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1.30, 0.95, 1.05),
        wspace=0.55,
    )
    ax_bars = fig.add_subplot(gs[0, 0])
    ax_var = fig.add_subplot(gs[0, 1])
    ax_cm = fig.add_subplot(gs[0, 2])

    _draw_per_subject_panel(
        ax_bars,
        per_subject_accuracies=accuracies,
        cross_subject_accuracy=float(cross_subject_accuracy),
        subjects=subjects,
    )
    _draw_variance_panel(
        ax_var,
        within_distribution=within_distribution,
        cross_distribution=cross_distribution,
    )
    _draw_confusion_panel(
        ax_cm,
        y_true_pooled=y_true_pooled,
        y_pred_pooled=y_pred_pooled,
        class_names=class_names,
    )

    within_mean = (
        float(within_distribution.mean()) if within_distribution.size else float("nan")
    )
    within_std = (
        float(within_distribution.std(ddof=0))
        if within_distribution.size
        else float("nan")
    )
    n_windows_part = (
        f" | n_windows_per_subject={n_windows_per_subject}"
        if n_windows_per_subject is not None
        else ""
    )
    subtitle = (
        f"n_subjects={n_subjects} | n_folds={n_folds}{n_windows_part} | "
        f"within_acc = {within_mean:.2f} +/- {within_std:.2f}"
    )
    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        openneuro_id=None,
        citation="synthetic per-subject manifold; palette: Okabe-Ito-aligned",
    )

    style_figure(
        fig,
        title="Within-subject CV: how stable is per-individual decoding?",
        subtitle=subtitle,
        source=source,
        grid_axis="y",
    )
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.06, right=0.97)
    return fig


__all__ = ["draw_within_subject_figure"]
