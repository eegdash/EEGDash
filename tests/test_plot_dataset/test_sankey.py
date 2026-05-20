"""Offline test for the migrated sankey renderer."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from plot_dataset import _live, plot_sankey


@pytest.fixture
def fake_sankey_payload():
    """Mirror the real /aggregations/sankey response shape."""
    return {
        "success": True,
        "levels": ["source", "modality"],
        "nodes": [
            {"id": 0, "label": "openneuro", "level": 0, "value": 100},
            {"id": 1, "label": "nemar", "level": 0, "value": 50},
            {"id": 2, "label": "EEG", "level": 1, "value": 120},
            {"id": 3, "label": "MEG", "level": 1, "value": 30},
        ],
        "links": [
            {
                "source": 0,
                "target": 2,
                "value": 80,
                "subject_sum": 80,
                "dataset_count": 12,
            },
            {
                "source": 0,
                "target": 3,
                "value": 20,
                "subject_sum": 20,
                "dataset_count": 3,
            },
            {
                "source": 1,
                "target": 2,
                "value": 40,
                "subject_sum": 40,
                "dataset_count": 7,
            },
            {
                "source": 1,
                "target": 3,
                "value": 10,
                "subject_sum": 10,
                "dataset_count": 2,
            },
        ],
    }


def test_sankey_renders_from_aggregations(fake_sankey_payload, tmp_path):
    response = _live.AggregationResponse(
        payload=fake_sankey_payload, source="live", url="http://test"
    )
    out_path = tmp_path / "dataset_sankey.html"

    with patch("plot_dataset.plot_sankey.fetch_aggregation", return_value=response):
        result = plot_sankey.generate_dataset_sankey(None, out_path)

    text = result.read_text(encoding="utf-8")
    assert result.exists() and result.stat().st_size > 0
    assert 'id="dataset-sankey"' in text
    # Sankey trace marker.
    assert '"type": "sankey"' in text or '"type":"sankey"' in text
    # Node labels appear.
    assert "openneuro" in text
    assert "EEG" in text
    # link.value uses subject_sum (80 from openneuro -> EEG).
    assert "80 subjects in 12" in text or "80 subjects in 12 datasets" in text


def test_sankey_falls_back_when_aggregation_fails(tmp_path):
    out_path = tmp_path / "dataset_sankey_empty.html"
    with patch(
        "plot_dataset.plot_sankey.fetch_aggregation",
        side_effect=_live.AggregationFetchError("network down"),
    ):
        result = plot_sankey.generate_dataset_sankey(None, out_path)
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "Sankey data unavailable" in text
