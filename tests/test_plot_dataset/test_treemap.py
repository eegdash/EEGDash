"""Offline test for the migrated treemap renderer."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from plot_dataset import _live, treemap


@pytest.fixture
def fake_treemap_payload():
    """Two sources, two modalities each, one population subtype each."""
    return {
        "success": True,
        "path": ["source", "modality", "clinical"],
        "nodes": [
            {
                "id": "openneuro",
                "parent": "",
                "label": "openneuro",
                "subjects": 1000,
                "hours": 200.5,
                "records": 5000,
                "datasets": 25,
            },
            {
                "id": "nemar",
                "parent": "",
                "label": "nemar",
                "subjects": 800,
                "hours": 150.0,
                "records": 4000,
                "datasets": 20,
            },
            {
                "id": "openneuro/EEG",
                "parent": "openneuro",
                "label": "EEG",
                "subjects": 700,
                "hours": 140.0,
                "records": 3500,
                "datasets": 18,
            },
            {
                "id": "openneuro/MEG",
                "parent": "openneuro",
                "label": "MEG",
                "subjects": 300,
                "hours": 60.5,
                "records": 1500,
                "datasets": 7,
            },
            {
                "id": "nemar/EEG",
                "parent": "nemar",
                "label": "EEG",
                "subjects": 800,
                "hours": 150.0,
                "records": 4000,
                "datasets": 20,
            },
        ],
        "totals": {"datasets": 45, "subjects": 1800},
    }


def test_treemap_renders_from_aggregations(fake_treemap_payload, tmp_path):
    response = _live.AggregationResponse(
        payload=fake_treemap_payload, source="live", url="http://test"
    )
    out_path = tmp_path / "dataset_treemap.html"

    with patch("plot_dataset.treemap.fetch_aggregation", return_value=response):
        result = treemap.generate_dataset_treemap(None, out_path)

    text = result.read_text(encoding="utf-8")
    assert result.exists() and result.stat().st_size > 0
    assert 'id="dataset-treemap-plot"' in text
    # Treemap trace marker.
    assert '"type": "treemap"' in text or '"type":"treemap"' in text
    # Hierarchy labels are present.
    assert "openneuro" in text
    assert "nemar" in text
    assert "EEG" in text
    # Hover shows datasets count.
    assert "datasets" in text


def test_treemap_falls_back_when_aggregation_fails(tmp_path):
    out_path = tmp_path / "dataset_treemap_empty.html"
    with patch(
        "plot_dataset.treemap.fetch_aggregation",
        side_effect=_live.AggregationFetchError("network down"),
    ):
        result = treemap.generate_dataset_treemap(None, out_path)
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "Treemap data unavailable" in text
