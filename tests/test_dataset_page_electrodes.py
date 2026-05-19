"""Tests for the dataset_page electrode-layout section.

Locks in arch #5: the section reads montage data through
:meth:`DatasetSnapshot.montage`, not from a build-time
``electrode-layouts.json`` file. The retired script
``docs/build_electrode_layouts.py`` is gone and the consumer must
render the same RST output it did before — same heading, same
``label`` / ``n_channels`` / ``montage_id`` keys driving the iframe.

These tests stub :meth:`DatasetSnapshot.build` rather than the
network so the suite stays offline-safe.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# The dataset_page package transitively imports sphinx (it's a Sphinx
# extension). Skip the whole module when sphinx isn't installed —
# this happens on the test-only CI matrix that doesn't pull doc deps.
pytest.importorskip("sphinx", reason="dataset_page extension requires sphinx")

from eegdash.dataset.snapshot import DatasetSnapshot


@pytest.fixture
def section_module():
    """Load ``docs/source/_extensions/dataset_page/sections.py``.

    The dataset_page package lives under ``docs/`` (not on the
    Python path by default), so we load it via ``importlib`` rather
    than mutating ``sys.path``. The directive's heavy imports
    (sphinx, JSON-LD writers, etc.) are not needed for these tests —
    we only touch the electrode section helpers.
    """
    repo_root = Path(__file__).resolve().parent.parent
    pkg_root = repo_root / "docs" / "source" / "_extensions"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    # Standard package import now that the parent is on sys.path.
    from dataset_page import sections  # type: ignore[import-not-found]

    return sections


def _snapshot_with(montages: dict[str, dict]) -> DatasetSnapshot:
    """Build an offline snapshot pre-populated with the given montages."""
    return DatasetSnapshot(
        rows=pd.DataFrame([{"dataset": ds_id} for ds_id in montages] or [{}]),
        aggregations={},
        montages=montages,
        source="live",
        fetched_at=datetime.now(timezone.utc),
    )


def test_electrodes_section_renders_iframe_when_snapshot_has_montage(section_module):
    """Happy path: snapshot has the dataset → RST includes the iframe
    pointing at electrodes.eegdash.org with the right montage hash.
    """
    snap = _snapshot_with(
        {
            "ds001785": {
                "label": "EEG · 63 sensors",
                "n_channels": 63,
                "modality": "eeg",
                "montage_id": "42b9e8daf4ff0e6d",
            }
        }
    )

    with patch.object(DatasetSnapshot, "build", return_value=snap):
        rst = section_module._format_electrodes_section({"dataset_id": "ds001785"})

    assert "Electrode Layout" in rst
    assert "63 channels" in rst
    assert "EEG · 63 sensors" in rst
    assert "montage=42b9e8daf4ff0e6d" in rst
    assert "electrodes.eegdash.org" in rst


def test_electrodes_section_handles_case_insensitive_dataset_id(section_module):
    """Consumer lowercases its input; the section must still find a
    montage even when the caller passes mixed-case ``dataset_id``.
    """
    snap = _snapshot_with(
        {
            "ds001785": {
                "label": "EEG · 63 sensors",
                "n_channels": 63,
                "modality": "eeg",
                "montage_id": "42b9e8daf4ff0e6d",
            }
        }
    )
    with patch.object(DatasetSnapshot, "build", return_value=snap):
        rst = section_module._format_electrodes_section({"dataset_id": "DS001785"})

    assert "63 channels" in rst
    assert "montage=42b9e8daf4ff0e6d" in rst


def test_electrodes_section_renders_placeholder_when_snapshot_empty(section_module):
    """When the snapshot has no montage for this dataset (server didn't
    join one; or the snapshot fell back to package-CSV) the section
    must render the "no layout indexed" placeholder, not crash.
    """
    snap = _snapshot_with({})

    with patch.object(DatasetSnapshot, "build", return_value=snap):
        rst = section_module._format_electrodes_section({"dataset_id": "ds_unknown"})

    assert "Electrode Layout" in rst
    assert "No scalp electrode layout is currently indexed" in rst
    # No iframe leaks through in the empty case.
    assert "electrodes.eegdash.org" not in rst


def test_electrodes_section_recovers_when_snapshot_build_raises(section_module):
    """``DatasetSnapshot.build`` blowing up (e.g. partial cache write)
    must not break the docs build — the section degrades to placeholder.
    """

    def _explode(*_, **__):
        raise RuntimeError("simulated snapshot blow-up")

    with patch.object(DatasetSnapshot, "build", side_effect=_explode):
        rst = section_module._format_electrodes_section({"dataset_id": "ds_boom"})

    assert "No scalp electrode layout is currently indexed" in rst


def test_electrodes_section_empty_when_no_dataset_id(section_module):
    """Empty ``dataset_id`` short-circuits to an empty string — the
    directive expects this for non-dataset pages.
    """
    assert section_module._format_electrodes_section({"dataset_id": ""}) == ""
    assert section_module._format_electrodes_section({}) == ""
