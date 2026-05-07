"""Drawing helpers for the ``plot_11`` leakage-vs-safe-split figure.

Sibling module to ``plot_11_leakage_safe_split.py``. The leading underscore
tells sphinx-gallery to skip this file when building the gallery, so the
helpers stay out of the rendered tutorial; the tutorial imports the public
``draw_leakage_figure`` entry point.

The figure is a 2 x 3 grid: row 1 = naive random split, row 2 = cross-subject
(leakage-safe) split. Each row carries the same three panels:

* col 1 -- subject x fold matrix; cell encodes the per-fold per-subject status.
  ``0`` = subject is fully on the train side of this fold (blue),
  ``1`` = subject is fully on the test side (orange),
  ``2`` = subject is split across train and test within this fold (striped).
* col 2 -- horizontal Sankey-lite for fold 0; segmented by subject so colors
  appear once on each side under a clean split, and on both sides under a
  leaky split.
* col 3 -- big-number callouts: subjects that leak across train/test in fold
  0, and the test share of windows in that fold.

Same data underneath; only the split strategy differs across rows.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_SURFACE,
    style_figure,
)
from eegdash.viz._tutorial_panels import subject_color_map

# Column titles (rendered above row 1 only).
_COL_TITLES = (
    "Per-fold subject status",
    "Window flow within fold 0",
    "Split health (fold 0)",
)
_ROW_LABELS = (
    ("Naive random split", "shuffle windows, ignore subject"),
    ("Cross-subject split", "GroupKFold by subject id"),
)
# Fold-0 train / test pill colors. Both come from EEGDash blue + orange, which
# match the Okabe-Ito-derived palette and stay distinguishable in grayscale.
_TRAIN_COLOR = EEGDASH_BLUE
_TEST_COLOR = EEGDASH_ORANGE
_MIXED_BG = EEGDASH_BLUE
_MIXED_FG = EEGDASH_ORANGE


def _draw_status_matrix(
    ax,
    assignment: np.ndarray,
    subjects: Iterable[str],
) -> None:
    """Cell color encodes subject status PER fold.

    ``assignment[i, j]`` carries one of ``{0, 1, 2}``:

    * ``0`` -- subject ``i`` is fully on the train side of fold ``j``.
    * ``1`` -- subject ``i`` is fully on the test side of fold ``j``.
    * ``2`` -- subject ``i`` is split across train and test within fold ``j``
      (the leakage failure mode).
    """
    subjects = list(subjects)
    n_subj, n_folds = assignment.shape
    cell_w = 0.92 / max(n_folds, 1)
    cell_h = 0.92 / max(n_subj, 1)
    x0, y0 = 0.06, 0.05
    for i in range(n_subj):
        for j in range(n_folds):
            value = int(assignment[i, j])
            x = x0 + j * cell_w
            y = y0 + (n_subj - 1 - i) * cell_h
            w = cell_w * 0.94
            h = cell_h * 0.86
            if value == 2:
                # Striped cell: blue background, orange diagonal hatching.
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        w,
                        h,
                        facecolor=_MIXED_BG,
                        edgecolor="white",
                        linewidth=0.8,
                        alpha=0.85,
                    )
                )
                # Overlay an orange diagonal stripe to mark "subject splits in this fold".
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        w,
                        h,
                        facecolor=_MIXED_FG,
                        edgecolor="white",
                        linewidth=0.0,
                        alpha=0.55,
                        hatch="///",
                    )
                )
            else:
                color = _TEST_COLOR if value == 1 else _TRAIN_COLOR
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        w,
                        h,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.8,
                        alpha=0.92,
                    )
                )
    # x ticks: fold labels.
    for j in range(n_folds):
        ax.text(
            x0 + (j + 0.47) * cell_w,
            y0 - 0.045,
            f"f{j}",
            ha="center",
            va="top",
            fontsize=8.0,
            family="monospace",
            color=EEGDASH_MUTED,
        )
    # y ticks: subject ids.
    for i, sub in enumerate(subjects):
        ax.text(
            x0 - 0.012,
            y0 + (n_subj - 1 - i + 0.43) * cell_h,
            str(sub),
            ha="right",
            va="center",
            fontsize=8.0,
            family="monospace",
            color=EEGDASH_MUTED,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def _draw_flow_bars(
    ax,
    train_counts_per_subject: np.ndarray,
    test_counts_per_subject: np.ndarray,
    subjects: Iterable[str],
    color_map: dict,
) -> None:
    """Two horizontal stacked bars (train above, test below) for ONE fold.

    Each bar is segmented by subject. Naive: every subject color appears in
    both bars (leakage). Cross-subject: each color appears in exactly one
    bar.
    """
    subjects = list(subjects)
    total = max(train_counts_per_subject.sum(), test_counts_per_subject.sum(), 1.0)
    bar_h = 0.20
    bar_x0 = 0.13
    bar_w = 0.74
    y_train = 0.58
    y_test = 0.26

    def _draw_bar(y, counts, label):
        cursor = bar_x0
        for i, sub in enumerate(subjects):
            count = float(counts[i])
            if count <= 0:
                continue
            seg_w = (count / total) * bar_w
            ax.add_patch(
                Rectangle(
                    (cursor, y),
                    seg_w,
                    bar_h,
                    facecolor=color_map.get(str(sub), EEGDASH_MUTED),
                    edgecolor="white",
                    linewidth=0.7,
                )
            )
            cursor += seg_w
        # Bar outline + total.
        ax.add_patch(
            Rectangle(
                (bar_x0, y),
                cursor - bar_x0,
                bar_h,
                facecolor="none",
                edgecolor=EEGDASH_MUTED,
                linewidth=0.6,
            )
        )
        ax.text(
            bar_x0 - 0.012,
            y + bar_h / 2,
            label,
            ha="right",
            va="center",
            fontsize=9.5,
            color=EEGDASH_INK,
            fontweight="bold",
        )
        ax.text(
            cursor + 0.012,
            y + bar_h / 2,
            f"{int(counts.sum()):,} win",
            ha="left",
            va="center",
            fontsize=8.4,
            family="monospace",
            color=EEGDASH_MUTED,
        )

    _draw_bar(y_train, train_counts_per_subject, "train")
    _draw_bar(y_test, test_counts_per_subject, "test")

    n_train_subj = int((train_counts_per_subject > 0).sum())
    n_test_subj = int((test_counts_per_subject > 0).sum())
    overlap = int(
        ((train_counts_per_subject > 0) & (test_counts_per_subject > 0)).sum()
    )
    overlap_color = EEGDASH_INK if overlap == 0 else EEGDASH_ORANGE
    ax.text(
        0.5,
        0.07,
        f"distinct subjects: train={n_train_subj}, test={n_test_subj}, both={overlap}",
        ha="center",
        va="center",
        fontsize=8.6,
        family="monospace",
        color=overlap_color,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def _draw_callouts(
    ax,
    *,
    overlap_count: int,
    n_subjects: int,
    test_share: float,
    target_test_share: float,
    is_safe: bool,
) -> None:
    """Big-number callouts: subject overlap and test-window share."""
    pill_y_top = 0.62
    pill_y_bot = 0.18
    pill_x = 0.06
    pill_w = 0.88
    pill_h = 0.30

    def _pill(y, big_text, label, accent):
        ax.add_patch(
            FancyBboxPatch(
                (pill_x, y),
                pill_w,
                pill_h,
                boxstyle="round,pad=0.005,rounding_size=0.024",
                facecolor=EEGDASH_SURFACE,
                edgecolor=accent,
                linewidth=1.4,
            )
        )
        ax.text(
            pill_x + pill_w / 2,
            y + pill_h * 0.66,
            big_text,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color=accent,
        )
        ax.text(
            pill_x + pill_w / 2,
            y + pill_h * 0.22,
            label,
            ha="center",
            va="center",
            fontsize=8.6,
            color=EEGDASH_MUTED,
        )

    overlap_accent = EEGDASH_BLUE if overlap_count == 0 else EEGDASH_ORANGE
    _pill(
        pill_y_top,
        f"{overlap_count} / {n_subjects}",
        "subjects in train AND test",
        overlap_accent,
    )
    balance_label = f"test window share (target {target_test_share:.2f})"
    _pill(
        pill_y_bot,
        f"{test_share:.2f}",
        balance_label,
        EEGDASH_INK,
    )

    verdict = "leakage-safe" if is_safe else "LEAKS"
    verdict_color = EEGDASH_BLUE if is_safe else EEGDASH_ORANGE
    ax.text(
        0.5,
        0.06,
        verdict,
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=verdict_color,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def _row_label(ax, label: str, sublabel: str) -> None:
    """Place a left-margin label naming the row's split strategy."""
    ax.text(
        -0.16,
        0.78,
        label,
        ha="right",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=EEGDASH_INK,
        transform=ax.transAxes,
    )
    ax.text(
        -0.16,
        0.62,
        sublabel,
        ha="right",
        va="center",
        fontsize=8.4,
        color=EEGDASH_MUTED,
        transform=ax.transAxes,
    )


def _legend(fig) -> None:
    """Single shared legend for the matrix color encoding."""
    swatches = [
        (_TRAIN_COLOR, "train fold", None, 0.085),
        (_TEST_COLOR, "test fold", None, 0.085),
        (_MIXED_BG, "subject split across train+test (leak)", "///", 0.235),
    ]
    x = 0.34
    y = 0.815
    for facecolor, label, hatch, advance in swatches:
        rect = Rectangle(
            (x, y),
            0.016,
            0.020,
            transform=fig.transFigure,
            facecolor=facecolor,
            edgecolor=EEGDASH_INK if hatch else "none",
            linewidth=0.5,
            clip_on=False,
        )
        if hatch:
            rect.set_hatch(hatch)
        fig.patches.append(rect)
        fig.text(
            x + 0.020,
            y + 0.010,
            label,
            fontsize=9,
            color=EEGDASH_INK,
            ha="left",
            va="center",
        )
        x += advance


def _per_fold_window_counts(
    naive_assignment_full: np.ndarray | None,
    safe_assignment: np.ndarray,
    n_subjects: int,
    n_folds: int,
    n_windows_per_subject: int,
    fold: int,
):
    """Compute per-subject train/test window counts for fold ``fold``.

    For the cross-subject row we treat every window of a subject as test
    exactly when the safe assignment puts that subject in the test fold.
    For the naive row we simulate the random shuffle: each window has the
    same per-fold test probability ``1 / n_folds``, so we draw counts from a
    Binomial that respects the realised marginal of the matrix.
    """
    rng = np.random.default_rng(0)
    target_test_frac = 1.0 / max(n_folds, 1)
    train_counts = np.zeros(n_subjects)
    test_counts = np.zeros(n_subjects)
    if safe_assignment.ndim != 2:
        raise ValueError("safe_assignment must be 2-D")
    if naive_assignment_full is not None and naive_assignment_full.ndim == 2:
        # Naive: per (subject, fold) status of {0,1,2}. The Sankey for fold k
        # looks at windows within that fold, so a "split" cell contributes to
        # both bars proportional to the realised in-fold split.
        for i in range(n_subjects):
            v = int(naive_assignment_full[i, fold])
            if v == 2:
                # Ensure at least 1 train and 1 test window per subject so the
                # "every subject leaks" message stays unambiguous on the bars.
                test_w = int(rng.binomial(n_windows_per_subject, target_test_frac))
                test_w = max(1, min(n_windows_per_subject - 1, test_w))
                test_counts[i] = test_w
                train_counts[i] = n_windows_per_subject - test_w
            elif v == 1:
                test_counts[i] = n_windows_per_subject
            else:
                train_counts[i] = n_windows_per_subject
    return train_counts, test_counts


def draw_leakage_figure(
    *,
    naive_assignment: np.ndarray,
    safe_assignment: np.ndarray,
    subjects: Iterable[str],
    n_windows_per_subject: int,
    plot_id: str = "plot_11",
):
    """Render the 2 x 3 leakage-vs-safe-split figure and return the Figure.

    Parameters
    ----------
    naive_assignment, safe_assignment : numpy.ndarray
        Per-fold per-subject status matrices of shape ``(n_subjects, n_folds)``.
        Values: ``0`` = subject is fully on the train side of that fold,
        ``1`` = fully on the test side, ``2`` = split across train and test
        within that fold (leakage). Cross-subject splits never produce a
        ``2``; naive random window splits produce a ``2`` in essentially
        every cell.
    subjects : iterable of str
        Subject ids in the same order as the rows of the assignment arrays.
    n_windows_per_subject : int
        Used to size the Sankey-lite flow bars and the test-share callout.
    plot_id : str, default ``"plot_11"``
        Tutorial id used in the provenance footer string.

    """
    naive_assignment = np.asarray(naive_assignment, dtype=int)
    safe_assignment = np.asarray(safe_assignment, dtype=int)
    subjects = [str(s) for s in subjects]
    if naive_assignment.shape != safe_assignment.shape:
        raise ValueError(
            "naive_assignment and safe_assignment must share shape; got "
            f"{naive_assignment.shape} vs {safe_assignment.shape}."
        )
    if naive_assignment.shape[0] != len(subjects):
        raise ValueError(
            f"len(subjects)={len(subjects)} does not match the row count of "
            f"the assignment arrays ({naive_assignment.shape[0]})."
        )

    n_subjects, n_folds = naive_assignment.shape
    n_windows_total = n_subjects * n_windows_per_subject
    color_map = subject_color_map(subjects)
    target_test_share = 1.0 / max(n_folds, 1)
    fold = 0

    # Compute fold-0 window counts per subject for both rows.
    naive_train, naive_test = _per_fold_window_counts(
        naive_assignment,
        safe_assignment,
        n_subjects,
        n_folds,
        n_windows_per_subject,
        fold,
    )
    safe_train = np.zeros(n_subjects)
    safe_test = np.zeros(n_subjects)
    for i in range(n_subjects):
        v = int(safe_assignment[i, fold])
        if v == 1:
            safe_test[i] = n_windows_per_subject
        else:
            safe_train[i] = n_windows_per_subject

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.0, 7.6),
        gridspec_kw={
            "width_ratios": [1.05, 1.45, 0.95],
            "wspace": 0.22,
            "hspace": 0.40,
        },
    )

    # Row 1 -- naive.
    _draw_status_matrix(axes[0, 0], naive_assignment, subjects)
    _draw_flow_bars(axes[0, 1], naive_train, naive_test, subjects, color_map)
    naive_overlap = int(((naive_train > 0) & (naive_test > 0)).sum())
    naive_test_share = float(naive_test.sum() / max(n_windows_total, 1))
    _draw_callouts(
        axes[0, 2],
        overlap_count=naive_overlap,
        n_subjects=n_subjects,
        test_share=naive_test_share,
        target_test_share=target_test_share,
        is_safe=naive_overlap == 0,
    )
    _row_label(axes[0, 0], _ROW_LABELS[0][0], _ROW_LABELS[0][1])

    # Row 2 -- safe.
    _draw_status_matrix(axes[1, 0], safe_assignment, subjects)
    _draw_flow_bars(axes[1, 1], safe_train, safe_test, subjects, color_map)
    safe_overlap = int(((safe_train > 0) & (safe_test > 0)).sum())
    safe_test_share = float(safe_test.sum() / max(n_windows_total, 1))
    _draw_callouts(
        axes[1, 2],
        overlap_count=safe_overlap,
        n_subjects=n_subjects,
        test_share=safe_test_share,
        target_test_share=target_test_share,
        is_safe=safe_overlap == 0,
    )
    _row_label(axes[1, 0], _ROW_LABELS[1][0], _ROW_LABELS[1][1])

    # Column titles on the top row only.
    for col_index, title in enumerate(_COL_TITLES):
        axes[0, col_index].text(
            0.5,
            1.05,
            title,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=EEGDASH_INK,
            transform=axes[0, col_index].transAxes,
        )

    _legend(fig)

    style_figure(
        fig,
        title="How do I split EEG data without subject leakage?",
        subtitle=(
            f"n_subjects={n_subjects}, n_folds={n_folds}, "
            f"n_windows_total={n_windows_total:,}. Naive shuffle puts every "
            "subject on both sides of every fold; GroupKFold keeps each "
            "subject in one test fold."
        ),
        source=f"EEGDash {plot_id} | synthetic split assignments | palette: Okabe-Ito-aligned",
    )
    # Layout band reservations:
    #   [0.92, 1.00] -- title strip (style_figure)
    #   [0.84, 0.92] -- subtitle (style_figure: y=0.88)
    #   [0.79, 0.84] -- shared swatch legend (this module)
    #   [0.06, 0.78] -- 2x3 axes grid
    #   [0.00, 0.05] -- provenance footer (style_figure)
    fig.subplots_adjust(top=0.76, bottom=0.07, left=0.18, right=0.97)
    return fig
