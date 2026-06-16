"""Offline test for the migrated bubble renderer."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from plot_dataset import _live, bubble


@pytest.fixture
def fake_bubble_payload():
    """Five datasets across two modalities; one has a non-positive x."""
    return {
        "success": True,
        "x": "subjects",
        "y": "files",
        "size": None,
        "color": "modality",
        "points": [
            {"id": "ds001", "label": "Alpha", "x": 100.0, "y": 500.0, "color": "EEG"},
            {"id": "ds002", "label": "Beta", "x": 50.0, "y": 200.0, "color": "EEG"},
            {"id": "ds003", "label": "Gamma", "x": 20.0, "y": 80.0, "color": "MEG"},
            {"id": "ds004", "label": "Delta", "x": 0.0, "y": 0.0, "color": "EEG"},
            {"id": "ds005", "label": "Epsilon", "x": None, "y": 10.0, "color": "MEG"},
        ],
    }


def test_bubble_renders_from_aggregations(fake_bubble_payload, tmp_path):
    response = _live.AggregationResponse(
        payload=fake_bubble_payload, source="live", url="http://test"
    )
    out_path = tmp_path / "dataset_bubble.html"

    with patch("plot_dataset.bubble.fetch_aggregation", return_value=response):
        result = bubble.generate_dataset_bubble(None, out_path)

    text = result.read_text(encoding="utf-8")
    assert result.exists() and result.stat().st_size > 0
    assert 'id="dataset-bubble"' in text
    # Scatter trace marker.
    assert '"type": "scatter"' in text or '"type":"scatter"' in text
    # Modalities surface as trace names.
    assert "EEG" in text
    assert "MEG" in text
    # The non-positive-x and the null-x points are dropped; labels of the
    # surviving points show up in the customdata.
    assert "Alpha" in text
    assert "Gamma" in text


def test_bubble_falls_back_when_aggregation_fails(tmp_path):
    out_path = tmp_path / "dataset_bubble_empty.html"
    with patch(
        "plot_dataset.bubble.fetch_aggregation",
        side_effect=_live.AggregationFetchError("network down"),
    ):
        result = bubble.generate_dataset_bubble(None, out_path)
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "Bubble data unavailable" in text
