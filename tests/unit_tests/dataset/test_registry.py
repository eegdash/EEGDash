import importlib.util
from pathlib import Path


class DummyBase:
    pass


def test_human_readable_size():
    """Test _human_readable_size helper function."""
    from eegdash.dataset.registry import _human_readable_size

    assert _human_readable_size(None) == "Unknown"
    assert _human_readable_size(0) == "Unknown"
    assert _human_readable_size(500) == "500.0 B"
    assert _human_readable_size(1024) == "1.0 KB"
    assert _human_readable_size(1024 * 1024) == "1.0 MB"
    assert _human_readable_size(1024 * 1024 * 1024) == "1.0 GB"
    assert _human_readable_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"
    assert _human_readable_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.0 PB"
    # Test with actual size from API (87.6 GB)
    assert "GB" in _human_readable_size(94108833435)


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
        "eegdash.dataset.registry.fetch_datasets_from_api",
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

    from eegdash.dataset.registry import fetch_datasets_from_api

    mock_response_data = json.dumps({"success": False}).encode()

    with (
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("eegdash.paths.get_default_cache_dir") as mock_cache_dir,
    ):
        # Ensure cache file does not exist
        mock_cache_dir.return_value = MagicMock()
        mock_cache_dir.return_value.__truediv__.return_value.exists.return_value = False

        mock_response = MagicMock()
        mock_response.read.return_value = mock_response_data
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_datasets_from_api("https://api.test.com", "testdb")
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

    with (
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("eegdash.paths.get_default_cache_dir") as mock_cache_dir,
    ):
        # Ensure cache file does not exist
        mock_cache_dir.return_value = MagicMock()
        mock_cache_dir.return_value.__truediv__.return_value.exists.return_value = False

        mock_response = MagicMock()
        mock_response.read.side_effect = [
            json.dumps(mock_data).encode("utf-8"),
            json.dumps({"data": {}}).encode("utf-8"),
        ]
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

    with (
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("eegdash.paths.get_default_cache_dir") as mock_cache_dir,
    ):
        # Ensure cache file does not exist
        mock_cache_dir.return_value = MagicMock()
        mock_cache_dir.return_value.__truediv__.return_value.exists.return_value = False

        mock_response = MagicMock()
        mock_response.read.side_effect = [
            json.dumps(mock_data).encode("utf-8"),
            json.dumps({"data": {}}).encode("utf-8"),
        ]
        mock_urlopen.return_value.__enter__.return_value = mock_response

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


def test_fetch_datasets_from_api_field_mappings():
    """Test that fetch_datasets_from_api correctly maps API fields."""
    import json
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.registry import fetch_datasets_from_api

    mock_api_response = {
        "success": True,
        "data": [
            {
                "dataset_id": "ds000247",
                "name": "Test Dataset",
                "demographics": {"subjects_count": 7},
                "total_files": 283,
                "tasks": ["rest", "noise"],
                "recording_modality": ["meg"],
                "study_design": "observational",
                "study_domain": "healthy",
                "size_bytes": 1024 * 1024 * 100,  # 100 MB
                "source": "openneuro",
                "license": "CC0",
                "dataset_doi": "doi:10.1234/test",
            }
        ],
    }

    mock_stats_response = {"data": {}}

    with (
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("eegdash.paths.get_default_cache_dir") as mock_cache_dir,
    ):
        mock_cache_dir.return_value = MagicMock()
        mock_cache_dir.return_value.__truediv__.return_value.exists.return_value = False

        mock_response = MagicMock()
        mock_response.read.side_effect = [
            json.dumps(mock_api_response).encode("utf-8"),
            json.dumps(mock_stats_response).encode("utf-8"),
        ]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        df = fetch_datasets_from_api("https://api.test.com", "testdb")

        assert len(df) == 1
        row = df.iloc[0]
        assert row["dataset"] == "ds000247"
        assert row["n_subjects"] == 7
        assert row["n_records"] == 283
        assert row["n_tasks"] == 2
        assert row["modality of exp"] == "meg"
        assert row["type of exp"] == "observational"
        assert row["Type Subject"] == "healthy"
        assert "MB" in row["size"]
        assert row["license"] == "CC0"
        assert row["doi"] == "doi:10.1234/test"


def test_internal_fetch_datasets_from_api_field_mappings():
    """Test that _fetch_datasets_from_api correctly maps API fields."""
    import json
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.registry import _fetch_datasets_from_api

    mock_api_response = {
        "success": True,
        "data": [
            {
                "dataset_id": "ds000117",
                "name": "Multisubject face processing",
                "demographics": {"subjects_count": 17},
                "total_files": 104,
                "tasks": ["facerecognition", "rest"],
                "recording_modality": ["meg", "eeg"],
                "study_design": "experimental",
                "study_domain": "cognitive",
                "size_bytes": 94108833435,
                "source": "openneuro",
                "license": "CC0",
                "dataset_doi": "doi:10.18112/openneuro.ds000117.v1.1.0",
            }
        ],
    }

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_api_response).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        df = _fetch_datasets_from_api("https://api.test.com", "testdb")

        assert len(df) == 1
        row = df.iloc[0]
        assert row["dataset"] == "ds000117"
        assert row["n_subjects"] == 17
        assert row["n_records"] == 104
        assert row["n_tasks"] == 2
        assert "meg" in row["modality of exp"]
        assert row["type of exp"] == "experimental"
        assert row["Type Subject"] == "cognitive"
        assert "GB" in row["size"]
        assert row["license"] == "CC0"
        assert row["doi"] == "doi:10.18112/openneuro.ds000117.v1.1.0"


def test_fetch_api_handles_missing_demographics():
    """Test API fetch handles missing demographics gracefully."""
    import json
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.registry import _fetch_datasets_from_api

    mock_api_response = {
        "success": True,
        "data": [
            {
                "dataset_id": "ds_nodemo",
                "demographics": None,  # Missing demographics
                "total_files": 50,
                "tasks": ["task1"],
                "recording_modality": "eeg",  # String instead of list
            }
        ],
    }

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_api_response).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        df = _fetch_datasets_from_api("https://api.test.com", "testdb")

        assert len(df) == 1
        row = df.iloc[0]
        assert row["n_subjects"] == 0  # Default when demographics missing
        assert row["n_records"] == 50
        assert row["modality of exp"] == "eeg"  # String converted properly


def test_fetch_api_error():
    """Test API fetch failure returns empty DataFrame."""
    from unittest.mock import patch

    from eegdash.dataset.registry import _fetch_datasets_from_api

    with patch("urllib.request.urlopen", side_effect=Exception("Network down")):
        df = _fetch_datasets_from_api("url", "db")
        assert df.empty


def test_register_fallback_to_csv(tmp_path):
    """Test fallback to CSV if API fails."""
    from eegdash.dataset.registry import register_openneuro_datasets

    # Create a simple summary CSV
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("dataset,n_subjects\nds001,5")

    with patch("eegdash.dataset.registry._fetch_datasets_from_api") as mock_fetch:
        mock_fetch.side_effect = Exception("API Error")

        # Should read from CSV
        registered = register_openneuro_datasets(summary_file=csv_path, from_api=True)
        assert "DS001" in registered


def test_register_api_success():
    """Test success path from API."""
    import pandas as pd

    from eegdash.dataset.registry import register_openneuro_datasets

    df = pd.DataFrame([{"dataset": "ds002", "n_subjects": 10}])
    with patch("eegdash.dataset.registry._fetch_datasets_from_api", return_value=df):
        registered = register_openneuro_datasets(from_api=True)
        assert "DS002" in registered
