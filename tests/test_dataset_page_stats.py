"""Tests for the dataset_page "Dataset Statistics" section.

Regression coverage for the case where a dataset carries no usable
subject, age, channel-count or sampling-frequency metadata. The section
must collapse entirely rather than leave a bare heading above an empty
chart grid.

These tests only call pure formatting helpers, so the suite stays
offline-safe.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging as stdlib_logging
import sys
import types
from pathlib import Path

import pytest


def _ensure_sphinx_logging_shim():
    """Make ``sections`` importable without sphinx installed.

    ``sections.py`` touches sphinx exactly once, ``LOGGER =
    logging.getLogger(__name__)`` against ``sphinx.util.logging``, whose
    ``getLogger`` returns a stdlib-compatible adapter. The functions under
    test never log, so standing in a stdlib logger keeps them unchanged
    while letting the module import on the test matrix, which installs the
    package without doc dependencies. Skipping instead left this file
    unexecuted in CI.
    """
    # Check sys.modules first. Once the shim below is installed, find_spec on it
    # raises because a hand-built module has no __spec__ of its own.
    if "sphinx" in sys.modules or importlib.util.find_spec("sphinx") is not None:
        return

    def _stub(name):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return module

    sphinx = _stub("sphinx")
    sphinx_util = _stub("sphinx.util")
    sphinx_util_logging = _stub("sphinx.util.logging")
    sphinx_util_logging.getLogger = stdlib_logging.getLogger
    sphinx_util.logging = sphinx_util_logging
    sphinx.util = sphinx_util
    sys.modules["sphinx"] = sphinx
    sys.modules["sphinx.util"] = sphinx_util
    sys.modules["sphinx.util.logging"] = sphinx_util_logging


@pytest.fixture
def section_module():
    """Load ``docs/source/_extensions/dataset_page/sections.py``.

    The dataset_page package lives under ``docs/`` (not on the Python
    path by default), so we put its parent on ``sys.path`` and import it
    as a normal package, matching ``test_dataset_page_electrodes.py``.
    """
    _ensure_sphinx_logging_shim()
    repo_root = Path(__file__).resolve().parent.parent
    pkg_root = repo_root / "docs" / "source" / "_extensions"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from dataset_page import sections  # type: ignore[import-not-found]

    return sections


def test_stats_section_empty_when_no_metadata_at_all(section_module):
    """No demographics and no recording aggregates means no section."""
    assert section_module._format_recording_stats_section({}) == ""


def test_stats_section_empty_when_channel_counts_have_no_values(section_module):
    """A non-empty ``nchans_counts`` whose entries carry no ``val`` sets
    the ``has_nchans`` flag but renders nothing, so the section must
    still collapse instead of emitting a bare heading.
    """
    context = {
        "nchans_counts": [{"val": None, "count": 12}],
    }

    assert section_module._format_recording_stats_section(context) == ""


def test_stats_section_empty_when_sampling_freqs_have_no_values(section_module):
    """Same collapse for the sampling-frequency chart."""
    context = {
        "sfreq_counts": [{"val": None, "count": 4}],
    }

    assert section_module._format_recording_stats_section(context) == ""


def test_stats_section_empty_when_every_chart_renders_nothing(section_module):
    """Several unusable inputs together still produce no section, and in
    particular no ``eegdash-ed-cohort-grid`` wrapper left behind.
    """
    context = {
        "demographics": {"ages": [], "sex_distribution": {}},
        "nchans_counts": [{"val": None, "count": 3}],
        "sfreq_counts": [{"val": None, "count": 3}],
        "total_duration_s": "not-a-number",
    }

    rst = section_module._format_recording_stats_section(context)

    assert rst == ""
    assert "Dataset Statistics" not in rst
    assert "eegdash-ed-cohort-grid" not in rst


def test_stats_section_still_renders_when_channel_counts_are_usable(section_module):
    """Guard against over-correcting: real data must still render."""
    context = {
        "nchans_counts": [{"val": 64, "count": 10}, {"val": 128, "count": 2}],
    }

    rst = section_module._format_recording_stats_section(context)

    assert "Dataset Statistics" in rst
    assert "Channel counts" in rst
    assert "eegdash-ed-cohort-grid" in rst


def test_stats_section_renders_duration_only_dataset(section_module):
    """A dataset with only a total duration is still worth a section."""
    context = {"total_duration_s": 7200.0}

    rst = section_module._format_recording_stats_section(context)

    assert "Dataset Statistics" in rst
    assert "Total recording duration" in rst
