from __future__ import annotations

from eegdash.plotting.dataset_summary_adapter import (
    BubbleSizeMetric,
    DatasetSummaryRecord,
    LegendConfig,
    SummaryBubbleArtifacts,
    SummaryBubbleConfig,
    summary_to_bubble_kwargs,
)


def _record(
    *,
    dataset: str,
    n_records: int,
    duration_hours_total: float | None,
    modality: str = "Visual",
    n_subjects: int = 3,
    type_subject: str | None = "Healthy",
) -> DatasetSummaryRecord:
    return DatasetSummaryRecord(
        dataset=dataset,
        modality=modality,
        n_subjects=n_subjects,
        n_records=n_records,
        duration_hours_total=duration_hours_total,
        type_subject=type_subject,
        experiment_type=None,
        n_sessions=1,
        n_trials=max(n_records, 1),
        trial_len=1.0,
        n_tasks=2,
    )


def test_size_metric_selection_and_fallback():
    records = [
        _record(dataset="ds_a", n_records=100, duration_hours_total=5.0),
        _record(dataset="ds_b", n_records=0, duration_hours_total=None),
    ]

    artifacts_records: SummaryBubbleArtifacts = summary_to_bubble_kwargs(
        records,
        config=SummaryBubbleConfig(
            size_metric=BubbleSizeMetric.N_RECORDS,
            layout_seed=10,
            min_size_override=2.5,
        ),
    )
    overrides = [
        payload["size_override"] for payload in artifacts_records.bubble_kwargs
    ]
    assert overrides[0] == 100.0
    assert overrides[1] == 2.5
    assert [payload["layout_seed"] for payload in artifacts_records.bubble_kwargs] == [
        10,
        11,
    ]


def test_structured_legend_bins_records_metric():
    records = [
        _record(
            dataset=f"ds_{idx}", n_records=(idx + 1) * 10, duration_hours_total=None
        )
        for idx in range(3)
    ]
    config = SummaryBubbleConfig(
        size_metric=BubbleSizeMetric.N_RECORDS,
        include_type_subject_badges=True,
        legend_heading="Heading",
    )
    artifacts: SummaryBubbleArtifacts = summary_to_bubble_kwargs(records, config=config)
    legend = artifacts.legend_config
    assert isinstance(legend, LegendConfig)
    assert len(legend.size_bins) == 3
    assert legend.heading == "Heading"
    assert legend.modalities  # at least one modality entry
    assert legend.type_subjects  # badges requested via config
