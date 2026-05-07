"""Unit tests for the EEGDash visual identity helpers."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (headless backend for CI)

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from eegdash.viz import (  # noqa: E402
    EEGDASH_BLUE,
    EEGDASH_MUTED,
    EEGDASH_ORANGE,
    EEGDASH_PALETTE,
    chance_line,
    style_figure,
    use_eegdash_style,
)


@pytest.fixture(autouse=True)
def _isolate_rc():
    """Reset rcParams between tests so use_eegdash_style is observable."""
    with plt.rc_context():
        yield


def test_use_eegdash_style_sets_rcparams_when_seaborn_missing(monkeypatch):
    """use_eegdash_style must work even if seaborn import fails."""
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def _no_seaborn(name, *args, **kwargs):
        if name == "seaborn":
            raise ImportError("simulated missing seaborn")
        return real_import(name, *args, **kwargs)

    # Drop any cached seaborn module and stub the import.
    monkeypatch.delitem(sys.modules, "seaborn", raising=False)
    with patch("builtins.__import__", side_effect=_no_seaborn):
        # Re-import the module under the patched importer to trigger the
        # fallback branch.
        module = importlib.import_module("eegdash.viz.identity")
        importlib.reload(module)
        module.use_eegdash_style()

    # Re-load cleanly so subsequent tests see a normal module state.
    importlib.reload(importlib.import_module("eegdash.viz.identity"))

    # The seaborn-free path updates rcParams directly.
    rc = matplotlib.rcParams
    assert rc["axes.spines.top"] is False
    assert rc["axes.spines.right"] is False
    # Font fallback chain must include Helvetica/Arial (the contract).
    sans = rc["font.sans-serif"]
    assert "Helvetica" in sans and "Arial" in sans


def test_use_eegdash_style_sets_palette_when_seaborn_missing(monkeypatch):
    """When seaborn is missing the prop_cycle must use the EEGDash palette."""
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def _no_seaborn(name, *args, **kwargs):
        if name == "seaborn":
            raise ImportError("simulated missing seaborn")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "seaborn", raising=False)
    with patch("builtins.__import__", side_effect=_no_seaborn):
        module = importlib.import_module("eegdash.viz.identity")
        importlib.reload(module)
        module.use_eegdash_style()

    importlib.reload(importlib.import_module("eegdash.viz.identity"))

    cycle_colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    # First two entries of the cycle are EEGDash blue + orange.
    assert cycle_colors[0].lower() == EEGDASH_BLUE.lower()
    assert cycle_colors[1].lower() == EEGDASH_ORANGE.lower()


def test_style_figure_attaches_data_rail_and_source():
    """style_figure must attach the Data Rail rectangles and source text."""
    use_eegdash_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0], color=EEGDASH_PALETTE[0])

    # No data rail before styling.
    assert not getattr(fig, "_eegdash_data_rail", False)
    n_patches_before = len(fig.patches)
    n_texts_before = len(fig.texts)

    style_figure(
        fig,
        title="Title",
        subtitle="ds002718 | n_subjects=18",
        source="EEGDash plot_test",
    )

    # Data Rail flag set + 2 rectangles added.
    assert getattr(fig, "_eegdash_data_rail", False) is True
    assert len(fig.patches) >= n_patches_before + 2

    # 3 figure-level texts added (title, subtitle, source).
    assert len(fig.texts) == n_texts_before + 3

    # The two Data Rail rectangles use EEGDash blue and orange.
    rail_colors = []
    for p in fig.patches:
        try:
            rail_colors.append(matplotlib.colors.to_hex(p.get_facecolor()).lower())
        except Exception:
            continue
    assert EEGDASH_BLUE.lower() in rail_colors
    assert EEGDASH_ORANGE.lower() in rail_colors

    # Source string must appear somewhere in the figure-level texts.
    rendered = [t.get_text() for t in fig.texts]
    assert "EEGDash plot_test" in rendered

    plt.close(fig)


def test_style_figure_idempotent_data_rail():
    """Calling style_figure twice must not duplicate the rail rectangles."""
    use_eegdash_style()
    fig, _ = plt.subplots()
    style_figure(fig, title="A", subtitle="b", source="c")
    rail_count = sum(
        1
        for p in fig.patches
        if matplotlib.colors.to_hex(p.get_facecolor()).lower()
        in (EEGDASH_BLUE.lower(), EEGDASH_ORANGE.lower())
    )
    style_figure(fig, title="A", subtitle="b", source="c")
    rail_count_after = sum(
        1
        for p in fig.patches
        if matplotlib.colors.to_hex(p.get_facecolor()).lower()
        in (EEGDASH_BLUE.lower(), EEGDASH_ORANGE.lower())
    )
    assert rail_count == 2
    assert rail_count_after == 2
    plt.close(fig)


def test_chance_line_adds_axhline_with_expected_style():
    """chance_line must add a dashed muted-grey axhline + an annotation."""
    use_eegdash_style()
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.plot([0, 5, 10], [0.4, 0.55, 0.6], color=EEGDASH_PALETTE[0])

    n_lines_before = len(ax.lines)
    chance_line(ax, level=0.25, label="chance")

    # One new line was added.
    assert len(ax.lines) == n_lines_before + 1
    line = ax.lines[-1]

    # Color and linestyle match the contract.
    assert matplotlib.colors.to_hex(line.get_color()).lower() == EEGDASH_MUTED.lower()
    assert line.get_linestyle() in ("--", "dashed")
    assert line.get_label() == "chance"

    # The y data is the chance level on both endpoints.
    ydata = line.get_ydata()
    assert pytest.approx(ydata[0]) == 0.25
    assert pytest.approx(ydata[-1]) == 0.25

    # An annotation was added near the right edge of the axes.
    annotations = [
        c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)
    ]
    assert len(annotations) >= 1
    assert any("chance" in a.get_text() for a in annotations)

    plt.close(fig)


def test_chance_line_handles_missing_xlim():
    """chance_line should not raise if the axes has no data yet."""
    use_eegdash_style()
    fig, ax = plt.subplots()
    chance_line(ax, level=0.5)
    assert any(
        line.get_linestyle() in ("--", "dashed")
        and matplotlib.colors.to_hex(line.get_color()).lower() == EEGDASH_MUTED.lower()
        for line in ax.lines
    )
    plt.close(fig)


def test_palette_constants_match_design_contract():
    """The palette constants must match the data-viz-design.md contract."""
    assert EEGDASH_PALETTE[0] == "#006CA3"
    assert EEGDASH_PALETTE[1] == "#F7941D"
    assert "#4F8CFF" in EEGDASH_PALETTE
    assert "#22D3EE" in EEGDASH_PALETTE


def test_shim_reexport_path_works():
    """The docs/plot_dataset/identity.py shim must continue to export the API."""
    # The shim lives outside any package, so we add docs/ to sys.path and import.
    import os
    import sys as _sys

    docs_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    )
    if docs_path not in _sys.path:
        _sys.path.insert(0, docs_path)
    try:
        from plot_dataset.identity import (
            EEGDASH_BLUE as ShimBlue,
        )
        from plot_dataset.identity import (
            chance_line as shim_chance_line,
        )
        from plot_dataset.identity import (
            style_figure as shim_style_figure,
        )
        from plot_dataset.identity import (
            use_eegdash_style as shim_use_eegdash_style,
        )
    finally:
        # Don't pollute sys.path for other tests.
        if docs_path in _sys.path:
            _sys.path.remove(docs_path)

    # Re-import the live identity module so we are comparing against the
    # currently-loaded version (earlier tests may have reloaded it to exercise
    # the seaborn-fallback branch).
    import eegdash.viz.identity as live_identity

    assert ShimBlue == EEGDASH_BLUE
    assert shim_use_eegdash_style is live_identity.use_eegdash_style
    assert shim_style_figure is live_identity.style_figure
    assert shim_chance_line is live_identity.chance_line
