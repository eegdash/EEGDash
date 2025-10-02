from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from eegdash.plotting.dataset_summary_adapter import (
    BubbleSizeMetric,
    DatasetSummaryRecord,
    SummaryBubbleConfig,
)
from eegdash.plotting.plotly_dataset_bubbles import generate_plotly_dataset_bubble


def _record(
    dataset: str,
    *,
    modality: str,
    n_subjects: int,
    n_records: int,
    duration_hours_total: float | None = None,
    type_subject: str | None = "Healthy",
) -> DatasetSummaryRecord:
    return DatasetSummaryRecord(
        dataset=dataset,
        modality=modality,
        n_subjects=n_subjects,
        n_records=n_records,
        duration_hours_total=duration_hours_total,
        n_tasks=2,
        type_subject=type_subject,
        experiment_type=None,
        n_sessions=1,
        n_trials=max(n_records, 1),
        trial_len=1.0,
    )


def test_generate_plotly_dataset_bubble_creates_figure(tmp_path: Path) -> None:
    records = [
        _record(
            "ds-a",
            modality="Visual",
            n_subjects=3,
            n_records=18,
            duration_hours_total=5.5,
        ),
        _record(
            "ds-b",
            modality="Auditory",
            n_subjects=2,
            n_records=12,
            duration_hours_total=2.0,
        ),
    ]
    cfg = SummaryBubbleConfig(
        limit=5,
        layout_seed=123,
        palette_override={"Visual": "#123456", "Auditory": "#abcdef"},
    )

    outfile = tmp_path / "chart.html"
    figure = generate_plotly_dataset_bubble(
        records,
        cfg,
        BubbleSizeMetric.N_RECORDS,
        outfile=outfile,
    )

    assert isinstance(figure, go.Figure)
    assert outfile.exists()
    assert len(figure.data) == 2  # one trace per modality


def test_marker_colors_and_subject_counts() -> None:
    records = [
        _record("ds-a", modality="Visual", n_subjects=4, n_records=24),
        _record("ds-b", modality="Auditory", n_subjects=3, n_records=15),
    ]
    palette = {"Visual": "#111111", "Auditory": "#222222"}
    cfg = SummaryBubbleConfig(limit=None, palette_override=palette)

    figure = generate_plotly_dataset_bubble(records, cfg, BubbleSizeMetric.N_RECORDS)

    colors = {trace.marker.color for trace in figure.data}
    assert colors == set(palette.values())

    subjects_total = sum(record.n_subjects for record in records)
    plotted_points = sum(len(trace.x) for trace in figure.data)
    assert plotted_points == subjects_total


def test_layout_does_not_overlap_subjects() -> None:
    records = [
        _record("ds-a", modality="Visual", n_subjects=10, n_records=100),
        _record("ds-b", modality="Auditory", n_subjects=10, n_records=80),
    ]
    cfg = SummaryBubbleConfig(limit=None, layout_seed=123)

    figure = generate_plotly_dataset_bubble(records, cfg)
    for trace in figure.data:
        points = list(zip(trace.x, trace.y))
        assert len(points) == len({(round(x, 3), round(y, 3)) for x, y in points})
