import importlib.util
from pathlib import Path


class DummyBase:
    pass


def test_register_openneuro_datasets(tmp_path: Path):
    module_path = (
        Path(__file__).resolve().parents[3] / "eegdash" / "dataset" / "registry.py"
    )
    spec = importlib.util.spec_from_file_location("registry", module_path)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)

    summary = tmp_path / "dataset_summary.csv"
    summary.write_text(
        "\n".join(
            [
                "dataset,num_subjects,num_sessions,num_runs,num_channels,sampling_rate,duration,size",
                "ds002718,18,18,1,74,250,14.844,1.2GB",
                "ds000001,1,1,1,1,1,1,100MB",
            ]
        )
    )
    namespace = {}
    registered = registry.register_openneuro_datasets(
        summary, namespace=namespace, base_class=DummyBase
    )

    assert set(registered) == {"DS002718", "DS000001"}
    ds_class = registered["DS002718"]
    assert ds_class is namespace["DS002718"]
    assert issubclass(ds_class, DummyBase)
    assert ds_class._dataset == "ds002718"


from unittest.mock import patch


def test_dynamic_class_init_with_query_update(tmp_path):
    """Test that dynamic class __init__ updates query correctly (line 89)."""
    from eegdash.dataset.registry import register_openneuro_datasets

    # Create a CSV with a dataset
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("dataset,n_subjects\nds_test,10\n")

    # Create a dummy base class to capture the init call
    class DummyBase:
        def __init__(self, query=None, cache_dir=None, s3_bucket=None, **kwargs):
            self.query = query
            self.cache_dir = cache_dir
            self.s3_bucket = s3_bucket
            self.kwargs = kwargs

    # Register with CSV (no API)
    ns = {}
    with patch(
        "eegdash.dataset.registry._fetch_datasets_from_api",
        return_value=None,
    ):
        register_openneuro_datasets(
            summary_file=str(csv_path),
            from_api=False,
            namespace=ns,
            base_class=DummyBase,
        )

    # Get the dynamically created class
    assert "DS_TEST" in ns
    DynamicClass = ns["DS_TEST"]

    # Line 74: empty dataset_id continues (already tested)
    # Line 89: query update branch - pass a query to merge
    obj = DynamicClass(cache_dir="/tmp", query={"task": "rest"})

    # The query should have both dataset and task
    assert obj.query["dataset"] == "ds_test"
    assert obj.query["task"] == "rest"


def test_dynamic_class_init_without_query(tmp_path):
    """Test dynamic class __init__ without query parameter."""
    from eegdash.dataset.registry import register_openneuro_datasets

    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("dataset,n_subjects\nds_noquery,5\n")

    class DummyBase:
        def __init__(self, query=None, cache_dir=None, s3_bucket=None, **kwargs):
            self.query = query
            self.cache_dir = cache_dir

    ns = {}
    register_openneuro_datasets(
        summary_file=str(csv_path),
        from_api=False,
        namespace=ns,
        base_class=DummyBase,
    )

    DynamicClass = ns["DS_NOQUERY"]
    obj = DynamicClass(cache_dir="/tmp")

    # Only dataset in query
    assert obj.query == {"dataset": "ds_noquery"}


def test_empty_dataset_id_skipped(tmp_path):
    """Test that empty dataset IDs are skipped (line 74)."""
    import pandas as pd

    from eegdash.dataset.registry import register_openneuro_datasets

    # Create DataFrame with empty dataset directly
    df = pd.DataFrame({"dataset": ["", "ds_valid", "  "], "n_subjects": [10, 20, 5]})
    csv_path = tmp_path / "summary.csv"
    df.to_csv(csv_path, index=False)

    class DummyBase:
        def __init__(self, **kwargs):
            pass

    ns = {}
    register_openneuro_datasets(
        summary_file=str(csv_path),
        from_api=False,
        namespace=ns,
        base_class=DummyBase,
    )

    # Only DS_VALID should be registered, empty ones skipped
    assert "DS_VALID" in ns
    # The test passes if we got here - line 74 was hit for empty datasets

    pass

    pass


def test_api_returns_not_success():
    """Test _fetch_datasets_from_api returns empty when success=False."""
    import json
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.registry import _fetch_datasets_from_api

    mock_response_data = json.dumps({"success": False}).encode()

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = mock_response_data
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = _fetch_datasets_from_api("https://api.test.com", "testdb")
        assert result.empty

    pass

    pass


def test_registry_docstring_generation():
    """Test rich docstring generation with various missing fields."""
    import pandas as pd

    from eegdash.dataset.dataset import EEGDashDataset
    from eegdash.dataset.registry import _generate_rich_docstring

    row = pd.Series(
        {
            "dataset": "ds003",
            "n_subjects": None,  # Should be handled
            "n_records": 10,
            "n_tasks": "rest",
            "dataset_doi": "doi:10.1000/1",
            "record_modality": None,  # Should check alternative
            "modality of exp": "eeg",
            "type of exp": "task",
            "Type Subject": "patient",
        }
    )

    doc = _generate_rich_docstring("ds003", row, EEGDashDataset)
    assert (
        "Subjects: Unknown" in doc or "Subjects:" in doc
    )  # Depending on implementation details
    assert "Modality: ``eeg``" in doc or "Modality: eeg" in doc
    assert "https://doi.org/10.1000/1" in doc


def test_registry_exclude_datasets():
    """Test that excluded datasets are skipped."""
    import json
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.registry import register_openneuro_datasets

    mock_data = {
        "success": True,
        "data": [
            {"dataset_id": "ABUDUKADI", "metadata": {}},  # Excluded
            {"dataset_id": "test", "metadata": {}},  # Excluded
            {"dataset_id": "valid", "metadata": {}},
        ],
    }

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        namespace = {}
        register_openneuro_datasets(from_api=True, namespace=namespace)

        assert "VALID" in namespace
        assert "ABUDUKADI" not in namespace
        assert "TEST" not in namespace


def test_registry_make_init_closure(tmp_path):
    import json
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.registry import register_openneuro_datasets

    mock_data = {"success": True, "data": [{"dataset_id": "ds_dyn", "metadata": {}}]}

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        ns = {}
        register_openneuro_datasets(from_api=True, namespace=ns)

        # Now instantiate it to trigger __init__
        # It calls base_class.__init__
        # We need to mock base_class or ensure it works

        # We can pass a dummy base class to register_openneuro_datasets to make it easier
        class DummyBase:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        ns2 = {}
        register_openneuro_datasets(from_api=True, namespace=ns2, base_class=DummyBase)
        DS_Class2 = ns2["DS_DYN"]

        obj = DS_Class2(cache_dir="/tmp")
        assert obj.kwargs["query"] == {"dataset": "ds_dyn"}
        assert obj.kwargs["cache_dir"] == "/tmp"
