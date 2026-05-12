import shutil
from pathlib import Path
from typing import Literal

import pytest

from eegdash.dataset import EEGDashDataset
from eegdash.schemas import create_record


@pytest.fixture
def functional_cache_dir(request: pytest.FixtureRequest) -> Path:
    path = Path.cwd() / ".functional_test_cache" / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _synthetic_record(
    *,
    dataset: str,
    extension: str,
    backend: Literal["s3", "https", "local", "nemar"] = "local",
    base: str = "s3://source-bucket/dataset",
) -> dict:
    return create_record(
        dataset=dataset,
        storage_base=base,
        storage_backend=backend,
        bids_relpath=f"sub-01/eeg/sub-01_task-rest_eeg{extension}",
        subject="01",
        task="rest",
        sampling_frequency=100.0,
        ntimes=1000,
    )


@pytest.mark.parametrize(
    ("query", "records", "expected_dataset", "expected_error"),
    [
        (
            None,
            [_synthetic_record(dataset="ds_from_record", extension=".edf")],
            "ds_from_record",
            None,
        ),
        (
            {"dataset": "ds_from_query"},
            [_synthetic_record(dataset="ds_from_record", extension=".edf")],
            "ds_from_query",
            None,
        ),
        (None, [], None, "You must provide a 'dataset' argument"),
        (None, None, None, "You must provide a 'dataset' argument"),
    ],
)
def test_dataset_constructor_dataset_source_contract(
    functional_cache_dir: Path,
    query: dict | None,
    records: list[dict] | None,
    expected_dataset: str | None,
    expected_error: str | None,
):
    kwargs = {"cache_dir": functional_cache_dir, "query": query, "records": records}
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            EEGDashDataset(download=False, **kwargs)
        return

    ds = EEGDashDataset(download=False, **kwargs)
    assert ds.query["dataset"] == expected_dataset


@pytest.mark.parametrize("initial_backend", ["local", "https"])
def test_dataset_constructor_filters_extensions_and_overrides_storage(
    functional_cache_dir: Path, initial_backend: str
):
    records = [
        _synthetic_record(
            dataset="ds_filter",
            extension=".edf",
            backend=initial_backend,
            base="s3://original-bucket/ds_filter",
        ),
        _synthetic_record(
            dataset="ds_filter",
            extension=".json",
            backend=initial_backend,
            base="s3://original-bucket/ds_filter",
        ),
    ]

    ds = EEGDashDataset(
        cache_dir=functional_cache_dir,
        records=records,
        download=False,
        s3_bucket="s3://override-bucket/custom",
    )

    assert len(ds.records) == 1
    assert ds.records[0]["extension"] == ".edf"
    assert ds.records[0]["storage"]["backend"] == "s3"
    assert ds.records[0]["storage"]["base"] == "s3://override-bucket/custom"
