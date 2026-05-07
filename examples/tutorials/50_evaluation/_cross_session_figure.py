"""Drawing helpers for the ``plot_52`` cross-session drift figure.

Sibling module to ``plot_52_cross_session_evaluation.py``. The leading
underscore tells sphinx-gallery to skip this file when building the gallery,
so the helpers stay out of the rendered tutorial; the tutorial imports the
public ``draw_cross_session_figure`` entry point.

The figure is a 1 x 3 grid that quantifies *within-subject between-session*
drift on a fixed cohort:

* col 1 -- session x session transfer matrix (mean across subjects). Rows
  index the *train* session, columns index the *test* session. Cell shading
  encodes accuracy on a sequential ``Blues`` ramp; the diagonal carries the
  highest values (within-session ceiling) and the off-diagonal carries the
  lower values (cross-session generalisation under calibration drift).
* col 2 -- paired bars per subject. ``EEGDASH_BLUE`` for within-session,
  ``EEGDASH_ORANGE`` for between-session. The visible gap is the subject's
  drift magnitude; the chance line ties both bars to the majority baseline.
* col 3 -- distribution of ``(within - between)`` across subjects. The right
  tail picks out subjects whose decoder collapses across days; the left tail
  marks subjects whose calibration transfers cleanly.

Same data underneath all three panels; only the lens differs.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_BLUE_DARK,
    EEGDASH_INK,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    chance_line,
    style_figure,
)
from eegdash.viz._tutorial_panels import add_provenance_footer


# ----------------------------------------------------------------------------
# Panel 1 -- session x session transfer matrix
# ----------------------------------------------------------------------------


def _draw_transfer_matrix(
    ax,
    matrix: np.ndarray,
    *,
    chance: float,
) -> ScalarMappable:
    """Render the per-cell accuracy heatmap.

    ``matrix[r, c]`` is the mean test accuracy when training on session ``r``
    and testing on session ``c`` (averaged across subjects). The diagonal is
    the within-session ceiling; the off-diagonal is the cross-session score.
    """
    n = int(matrix.shape[0])
    # Anchor the colormap at chance so the diagonal stands out and the
    # off-diagonal cells stay readable when accuracy hovers near 0.5.
    vmin = max(0.0, float(chance) - 0.05)
    vmax = 1.0
    cmap = plt.get_cmap("Blues")
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")

    # Cell-level annotations. Light text on dark cells, dark text on light
    # cells, picked from a luminance threshold against the colormap so the
    # numbers stay legible in print and grayscale.
    luminance_threshold = 0.62
    for r in range(n):
        for c in range(n):
            value = float(matrix[r, c])
            shade = norm(value)
            text_color = "white" if shade >= luminance_threshold else EEGDASH_INK
            label = f"{value:.2f}"
            if r == c:
                # Pin the diagonal as the within-session ceiling visually.
                label = f"{value:.2f}*"
            ax.text(
                c,
                r,
                label,
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold" if r == c else "normal",
                color=text_color,
            )

    session_labels = [f"ses-{i + 1:02d}" for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(session_labels, fontsize=9)
    ax.set_yticklabels(session_labels, fontsize=9)
    ax.set_xlabel("test session", fontsize=9.5, color=EEGDASH_INK)
    ax.set_ylabel("train session", fontsize=9.5, color=EEGDASH_INK)
    ax.tick_params(axis="both", which="both", length=0)
    # Light cell separators read as a grid without competing with the data.
    for x in np.arange(-0.5, n, 1):
        ax.axhline(x, color="white", linewidth=1.2)
        ax.axvline(x, color="white", linewidth=1.2)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Diagonal-vs-off-diagonal callout. The asterisk in the diagonal cells
    # points back to this single legend, so the message reads even without
    # the colorbar. Anchored above the panel title in axes coords so the
    # x-axis label below stays clear.
    diag_mean = float(np.mean(np.diag(matrix)))
    off_mask = ~np.eye(n, dtype=bool)
    off_mean = float(np.mean(matrix[off_mask])) if off_mask.any() else float("nan")
    ax.text(
        0.5,
        -0.22,
        f"diag* = {diag_mean:.2f}   off-diag = {off_mean:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
        family="monospace",
        color=EEGDASH_MUTED,
    )
    return im


# ----------------------------------------------------------------------------
# Panel 2 -- paired within vs between bars per subject
# ----------------------------------------------------------------------------


def _draw_paired_bars(
    ax,
    *,
    subject_ids: Sequence[str],
    within: np.ndarray,
    between: np.ndarray,
    chance: float,
) -> None:
    """Paired bars per subject: blue = within, orange = between."""
    n = len(subject_ids)
    x = np.arange(n)
    bar_w = 0.36
    bars_within = ax.bar(
        x - bar_w / 2,
        within,
        width=bar_w,
        color=EEGDASH_BLUE,
        edgecolor="white",
        linewidth=0.6,
        label="within-session",
        zorder=2.5,
    )
    bars_between = ax.bar(
        x + bar_w / 2,
        between,
        width=bar_w,
        color=EEGDASH_ORANGE,
        edgecolor="white",
        linewidth=0.6,
        label="between-session",
        zorder=2.5,
    )

    # Drift connector: thin grey line from each within-bar top to its paired
    # between-bar top. Reads as the per-subject drop without legend lookup.
    for i in range(n):
        ax.plot(
            [x[i] - bar_w / 2, x[i] + bar_w / 2],
            [float(within[i]), float(between[i])],
            color=EEGDASH_MUTED,
            linewidth=0.8,
            alpha=0.7,
            zorder=2.2,
        )

    ax.set_xticks(x)
    # Short subject labels on the x-axis. Helvetica is missing U+2192 so we
    # keep this monospace and ASCII (the polish memo flags this).
    short_labels = [str(s).replace("sub-", "") for s in subject_ids]
    ax.set_xticklabels(short_labels, fontsize=9, family="monospace")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy", fontsize=9.5, color=EEGDASH_INK)
    ax.set_xlabel("subject", fontsize=9.5, color=EEGDASH_INK)
    ax.tick_params(axis="both", which="both", length=0)
    # Mean reference lines per condition. Aggregate values are the headline;
    # marking them keeps the eye on the gap rather than on the per-subject
    # noise. Labels live in the LEFT margin so the right edge (where the
    # chance annotation sits) stays uncluttered. Set xlim BEFORE calling
    # chance_line so its right-edge annotation lands on the new limit.
    within_mean = float(np.mean(within))
    between_mean = float(np.mean(between))
    ax.set_xlim(-2.0, n - 0.5)

    # Chance reference. ``chance_line`` handles the dashed line plus the
    # small right-side marker so the level reads in grayscale.
    chance_line(ax, level=float(chance), label="chance")
    ax.axhline(
        within_mean,
        color=EEGDASH_BLUE_DARK,
        linestyle=":",
        linewidth=0.9,
        alpha=0.85,
        zorder=1.8,
    )
    ax.axhline(
        between_mean,
        color=EEGDASH_ORANGE,
        linestyle=":",
        linewidth=0.9,
        alpha=0.85,
        zorder=1.8,
    )
    ax.text(
        -1.95,
        within_mean,
        f"within\n{within_mean:.2f}",
        ha="left",
        va="center",
        fontsize=8.0,
        color=EEGDASH_BLUE_DARK,
        family="monospace",
        fontweight="bold",
    )
    ax.text(
        -1.95,
        between_mean,
        f"between\n{between_mean:.2f}",
        ha="left",
        va="center",
        fontsize=8.0,
        color=EEGDASH_ORANGE,
        family="monospace",
        fontweight="bold",
    )

    legend = ax.legend(
        handles=[bars_within, bars_between],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncols=2,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(EEGDASH_INK)


# ----------------------------------------------------------------------------
# Panel 3 -- drift-magnitude distribution
# ----------------------------------------------------------------------------


def _draw_drift_histogram(
    ax,
    *,
    drift: np.ndarray,
    subject_ids: Sequence[str],
) -> None:
    """Histogram of ``(within - between)`` across subjects.

    Bars are tinted on a diverging scheme: positive drift (typical) in
    EEGDash orange because that is the calibration-drift story; negative
    drift (between > within, rare) in EEGDash blue because the decoder
    transferred *better* to a held-out session than within-session.
    """
    n = len(drift)
    if n == 0:
        ax.set_axis_off()
        return

    span = max(0.20, float(np.max(np.abs(drift))) + 0.05)
    n_bins = max(5, min(9, n // 2 + 4))
    edges = np.linspace(-span, span, n_bins + 1)
    counts, _ = np.histogram(drift, bins=edges)

    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0]) * 0.92
    for c, h in zip(centers, counts):
        if h <= 0:
            continue
        color = EEGDASH_ORANGE if c >= 0 else EEGDASH_BLUE
        ax.bar(
            float(c),
            int(h),
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=2.5,
        )

    # Per-subject rug marks under the histogram so individual subjects stay
    # readable even when bins lump them together. Skip the per-tick text
    # labels (they collide on dense bins); the bars + the mean line carry
    # the message and panel 2 already names every subject.
    rug_y = -0.45
    for value in drift:
        ax.plot(
            [float(value)],
            [rug_y],
            marker="|",
            markersize=10,
            color=EEGDASH_MUTED,
            zorder=2.0,
        )

    # Zero-line marks the no-drift contract. The mean drift (vertical dashed
    # line) is the headline number echoed in the subtitle.
    mean_drift = float(np.mean(drift))
    ax.axvline(0.0, color=EEGDASH_INK, linewidth=0.9, alpha=0.7, zorder=1.6)
    ax.axvline(
        mean_drift,
        color=EEGDASH_ORANGE if mean_drift >= 0 else EEGDASH_BLUE,
        linestyle="--",
        linewidth=1.1,
        zorder=2.4,
    )
    y_top = float(max(counts.max() if counts.size else 1, 1)) + 1.4
    # Mean-drift label sits ABOVE the dashed line, in the reserved band
    # between the bars and the panel title. Tail callouts live below the
    # mean-drift label so the three never overlap.
    ax.text(
        mean_drift,
        y_top - 0.10,
        f"mean drift = {mean_drift:+.2f}",
        ha="center",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=EEGDASH_ORANGE if mean_drift >= 0 else EEGDASH_BLUE,
        family="monospace",
    )

    ax.set_xlim(-span, span)
    ax.set_ylim(-1.6, y_top)
    ax.set_xlabel(
        "drift = within - between (per subject)", fontsize=9.5, color=EEGDASH_INK
    )
    ax.set_ylabel("subjects", fontsize=9.5, color=EEGDASH_INK)
    ax.tick_params(axis="both", which="both", length=0)
    # Y ticks on integer counts only; the rug strip below the axis is
    # decorative and should not pull the y-axis down.
    y_max_int = int(np.ceil(max(counts.max() if counts.size else 1, 1)))
    ax.set_yticks(range(0, max(y_max_int, 1) + 1))
    # Annotate left + right tails inside the data area, anchored on bottom
    # corners so they cannot collide with the mean-drift label.
    ax.text(
        -span + 0.01,
        y_top * 0.42,
        "left tail:\nclean transfer",
        ha="left",
        va="center",
        fontsize=8.0,
        color=EEGDASH_BLUE_DARK,
    )
    ax.text(
        span - 0.01,
        y_top * 0.42,
        "right tail:\nsevere drift",
        ha="right",
        va="center",
        fontsize=8.0,
        color=EEGDASH_ORANGE,
    )


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def draw_cross_session_figure(
    *,
    session_matrix: np.ndarray,
    subject_ids: Sequence[str],
    within_session_acc: np.ndarray,
    between_session_acc: np.ndarray,
    chance: float = 0.5,
    plot_id: str = "plot_52",
):
    """Render the 1 x 3 cross-session drift figure and return the Figure.

    Parameters
    ----------
    session_matrix : numpy.ndarray
        Square ``(n_sessions, n_sessions)`` matrix of accuracies; ``[r, c]``
        is the mean accuracy when training on session ``r`` and testing on
        session ``c``, averaged across subjects.
    subject_ids : sequence of str
        Per-subject identifiers in the order of ``within_session_acc`` and
        ``between_session_acc``.
    within_session_acc : numpy.ndarray
        Per-subject within-session accuracy (one value per subject).
    between_session_acc : numpy.ndarray
        Per-subject between-session accuracy (one value per subject).
    chance : float, default ``0.5``
        Majority-class chance level on the test labels. Used by the matrix
        colormap anchor and by the panel-2 chance line.
    plot_id : str, default ``"plot_52"``
        Tutorial id used in the provenance footer.

    """
    session_matrix = np.asarray(session_matrix, dtype=float)
    if session_matrix.ndim != 2 or session_matrix.shape[0] != session_matrix.shape[1]:
        raise ValueError(
            "session_matrix must be a square 2-D array; got shape "
            f"{session_matrix.shape}."
        )
    within = np.asarray(within_session_acc, dtype=float)
    between = np.asarray(between_session_acc, dtype=float)
    subject_ids = [str(s) for s in subject_ids]
    if within.shape != between.shape:
        raise ValueError(
            "within_session_acc and between_session_acc must share shape; "
            f"got {within.shape} vs {between.shape}."
        )
    if within.shape[0] != len(subject_ids):
        raise ValueError(
            f"len(subject_ids)={len(subject_ids)} does not match the length "
            f"of the per-subject accuracy arrays ({within.shape[0]})."
        )

    n_subjects = len(subject_ids)
    n_sessions = int(session_matrix.shape[0])
    drift_per_subject = within - between
    mean_within = float(np.mean(within))
    mean_between = float(np.mean(between))
    mean_drift = float(np.mean(drift_per_subject))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.6, 5.0),
        gridspec_kw={
            "width_ratios": [0.95, 1.55, 1.20],
            "wspace": 0.34,
        },
    )

    # Panel 1 -- transfer matrix (mean across subjects).
    im = _draw_transfer_matrix(axes[0], session_matrix, chance=float(chance))
    cbar = fig.colorbar(
        im,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        shrink=0.82,
    )
    cbar.set_label("accuracy", fontsize=8.6, color=EEGDASH_INK)
    cbar.ax.tick_params(labelsize=8, length=0)
    cbar.outline.set_visible(False)

    # Panel 2 -- paired within vs between bars per subject.
    _draw_paired_bars(
        axes[1],
        subject_ids=subject_ids,
        within=within,
        between=between,
        chance=float(chance),
    )

    # Panel 3 -- drift-magnitude histogram across subjects.
    _draw_drift_histogram(
        axes[2],
        drift=drift_per_subject,
        subject_ids=subject_ids,
    )

    # Column titles above each axes. style_figure leaves the per-axes title
    # empty so we attach these as figure-coordinate text via the axes.
    panel_titles = (
        "Transfer matrix (mean across subjects)",
        "Within vs between, per subject",
        "Drift = within - between",
    )
    for ax, panel_title in zip(axes, panel_titles):
        ax.text(
            0.5,
            1.06,
            panel_title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=EEGDASH_INK,
        )

    source = add_provenance_footer(
        None,
        plot_id=plot_id,
        citation="synthetic 8-subject x 3-session cohort",
        extra="palette: Okabe-Ito-aligned (EEGDash blue/orange); cmap: Blues",
    )

    style_figure(
        fig,
        title="How much does a within-session decoder drift across sessions?",
        subtitle=(
            f"n_subjects={n_subjects} | n_sessions={n_sessions} | "
            f"mean within = {mean_within:.2f} | "
            f"mean between = {mean_between:.2f} | "
            f"drift = {mean_drift:+.2f}"
        ),
        source=source,
        grid_axis="y",
    )
    # Layout band reservations:
    #   [0.92, 1.00] -- title strip (style_figure)
    #   [0.85, 0.92] -- subtitle (style_figure: y=0.88)
    #   [0.78, 0.85] -- panel-title row (this module, ax.text y=1.06)
    #   [0.28, 0.78] -- 1x3 axes grid
    #   [0.10, 0.22] -- legend strip under panel 2
    #   [0.00, 0.06] -- provenance footer (style_figure)
    fig.subplots_adjust(top=0.74, bottom=0.28, left=0.10, right=0.97)
    return fig


__all__ = ["draw_cross_session_figure"]
