"""Offline test for the migrated growth renderer."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from plot_dataset import _live, growth


@pytest.fixture
def fake_growth_payload():
    """Two modalities, three years apiece — enough to exercise stacking."""
    return {
        "success": True,
        "bucket": "year",
        "field": "dataset_created_at",
        "series": {
            "EEG": [
                {"period": "2023", "count": 10, "cumulative": 10},
                {"period": "2024", "count": 15, "cumulative": 25},
                {"period": "2025", "count": 8, "cumulative": 33},
            ],
            "MEG": [
                {"period": "2024", "count": 3, "cumulative": 3},
                {"period": "2025", "count": 2, "cumulative": 5},
            ],
        },
    }


def test_growth_renders_from_aggregations(fake_growth_payload, tmp_path):
    response = _live.AggregationResponse(
        payload=fake_growth_payload, source="live", url="http://test"
    )
    out_path = tmp_path / "dataset_growth.html"

    with patch("plot_dataset.growth.fetch_aggregation", return_value=response):
        result = growth.generate_dataset_growth(None, out_path)

    text = result.read_text(encoding="utf-8")
    assert result.exists() and result.stat().st_size > 0
    # Plotly markers present
    assert 'id="dataset-growth-plot"' in text
    assert '"type": "scatter"' in text or '"type":"scatter"' in text
    # Two modalities + the "All datasets" rollup → 3 named traces.
    assert "All datasets" in text
    assert "EEG" in text
    assert "MEG" in text


def test_growth_falls_back_to_empty_on_fetch_error(tmp_path):
    out_path = tmp_path / "dataset_growth_empty.html"
    with patch(
        "plot_dataset.growth.fetch_aggregation",
        side_effect=_live.AggregationFetchError("network down"),
    ):
        result = growth.generate_dataset_growth(None, out_path)
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "Growth data unavailable" in text
