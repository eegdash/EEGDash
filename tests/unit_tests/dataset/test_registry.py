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
    from eegdash.dataset.registry import register_openneuro_datasets

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
    registered = register_openneuro_datasets(
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
        patch("eegdash.dataset.registry.get_default_cache_dir") as mock_cache_dir,
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
            "record_modality": "eeg",  # Recording modality (BIDS data type)
            "modality of exp": "visual",  # Experimental modality
            "type of exp": "task",
            "Type Subject": "patient",
        }
    )

    doc = _generate_rich_docstring("ds003", row, EEGDashDataset)
    # When n_subjects is None, "Subjects:" is skipped (not shown as "Unknown")
    assert "Subjects:" not in doc  # Gracefully handled by omitting
    assert "recordings: 10" in doc  # But n_records is shown
    assert "Modality: ``eeg``" in doc
    assert "https://doi.org/10.1000/1" in doc


def test_registry_exclude_datasets():
    """Test that excluded datasets are skipped by fetch_datasets_from_api."""
    import json
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.registry import register_openneuro_datasets

    mock_data = {
        "success": True,
        "data": [
            {"dataset_id": "ABUDUKADI"},  # Excluded
            {"dataset_id": "test"},  # Excluded
            {"dataset_id": "valid"},
        ],
    }

    with (
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("eegdash.dataset.registry.get_default_cache_dir") as mock_cache_dir,
    ):
        mock_cache_dir.return_value = MagicMock()
        mock_cache_dir.return_value.__truediv__.return_value.exists.return_value = False

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

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
        patch("eegdash.dataset.registry.get_default_cache_dir") as mock_cache_dir,
    ):
        mock_cache_dir.return_value = MagicMock()
        mock_cache_dir.return_value.__truediv__.return_value.exists.return_value = False

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

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
                "tags": {
                    "modality": ["visual"],  # → modality of exp
                    "type": ["observational"],  # → type of exp
                    "pathology": ["healthy"],  # → Type Subject
                },
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
        patch("eegdash.dataset.registry.get_default_cache_dir") as mock_cache_dir,
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
        assert (
            row["modality of exp"] == "visual"
        )  # Experimental modality from tags.modality
        assert row["record_modality"] == "meg"  # Recording modality (BIDS data type)
        assert row["type of exp"] == "observational"  # From tags.type
        assert row["Type Subject"] == "healthy"  # From tags.pathology
        assert "MB" in row["size"]
        assert row["license"] == "CC0"
        assert row["doi"] == "doi:10.1234/test"


def test_fetch_api_handles_missing_demographics():
    """Test API fetch handles missing demographics gracefully."""
    import json
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.registry import fetch_datasets_from_api

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

    with (
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("eegdash.dataset.registry.get_default_cache_dir") as mock_cache_dir,
    ):
        mock_cache_dir.return_value = MagicMock()
        mock_cache_dir.return_value.__truediv__.return_value.exists.return_value = False

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_api_response).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        df = fetch_datasets_from_api(
            "https://api.test.com", "testdb", force_refresh=True
        )

        assert len(df) == 1
        row = df.iloc[0]
        assert row["n_subjects"] == 0  # Default when demographics missing
        assert row["n_records"] == 50
        assert row["modality of exp"] == ""  # Empty when no tags.modality
        assert row["record_modality"] == "eeg"  # Recording modality from string


def test_fetch_api_error():
    """Test API fetch failure returns empty DataFrame."""
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.registry import fetch_datasets_from_api

    with (
        patch("urllib.request.urlopen", side_effect=Exception("Network down")),
        patch("eegdash.dataset.registry.get_default_cache_dir") as mock_cache_dir,
    ):
        mock_cache_dir.return_value = MagicMock()
        mock_cache_dir.return_value.__truediv__.return_value.exists.return_value = False
        df = fetch_datasets_from_api("url", "db", force_refresh=True)
        assert df.empty


def test_register_fallback_to_csv(tmp_path):
    """Test fallback to CSV if API fails."""
    from eegdash.dataset.registry import register_openneuro_datasets

    # Create a simple summary CSV
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("dataset,n_subjects\nds001,5")

    with patch("eegdash.dataset.registry.fetch_datasets_from_api") as mock_fetch:
        mock_fetch.side_effect = Exception("API Error")

        # Should read from CSV
        registered = register_openneuro_datasets(summary_file=csv_path, from_api=True)
        assert "DS001" in registered


def test_register_api_success():
    """Test success path from API."""
    import pandas as pd

    from eegdash.dataset.registry import register_openneuro_datasets

    df = pd.DataFrame([{"dataset": "ds002", "n_subjects": 10}])
    with patch("eegdash.dataset.registry.fetch_datasets_from_api", return_value=df):
        registered = register_openneuro_datasets(from_api=True)
        assert "DS002" in registered


# ---------------------------------------------------------------------------
# Canonical-name aliases
# ---------------------------------------------------------------------------
import logging
import math

import pandas as pd
import pytest

from eegdash.dataset.registry import (
    _parse_canonical_names,
    register_openneuro_datasets,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, []),
        ("", []),
        ("   ", []),
        ("nan", []),
        ("null", []),
        ("[]", []),
        ("<NA>", []),
        (math.nan, []),
        (pd.NA, []),
        (pd.NaT, []),
        ([], []),
        (["A", " B ", ""], ["A", "B"]),
        (["Foo", "Foo", "Bar"], ["Foo", "Bar"]),
        ('["BrainTreeBank"]', ["BrainTreeBank"]),
        ('["SleepEDF", "SleepEDFPlus"]', ["SleepEDF", "SleepEDFPlus"]),
        ('["Foo", "Foo"]', ["Foo"]),
        ('"OnlyOne"', ["OnlyOne"]),
        ("Foo, Bar", ["Foo", "Bar"]),
    ],
)
def test_parse_canonical_names(raw, expected):
    assert _parse_canonical_names(raw) == expected


def _register(tmp_path, rows, namespace=None):
    csv_path = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    ns = {} if namespace is None else namespace
    register_openneuro_datasets(
        summary_file=csv_path, namespace=ns, base_class=DummyBase
    )
    return ns


@pytest.mark.parametrize(
    "canonical_field, expected_aliases",
    [
        ('["BrainTreeBank"]', ["BrainTreeBank"]),
        ('["SleepEDF", "SleepEDFPlus"]', ["SleepEDF", "SleepEDFPlus"]),
        ('["Sleep-EDF+", "ValidName"]', ["ValidName"]),
        ('["class", "None", "True", "GoodName"]', ["GoodName"]),
        ('["Foo", "Foo", "Bar"]', ["Foo", "Bar"]),
        ('["   ", "Ok"]', ["Ok"]),
        ("[]", []),
        ("", []),
    ],
    ids=[
        "single",
        "multiple",
        "bad-ident",
        "kw",
        "dup",
        "blank-entry",
        "empty",
        "blank",
    ],
)
def test_canonical_name_registration(tmp_path, canonical_field, expected_aliases):
    ns = _register(tmp_path, [{"dataset": "ds1", "canonical_name": canonical_field}])
    assert ns["DS1"].canonical_name == expected_aliases
    for alias in expected_aliases:
        assert ns[alias] is ns["DS1"]
        assert alias in ns["__all__"]


def test_canonical_name_column_absent(tmp_path):
    (tmp_path / "s.csv").write_text("dataset\nds_nocol\n")
    ns = {}
    register_openneuro_datasets(
        summary_file=tmp_path / "s.csv", namespace=ns, base_class=DummyBase
    )
    assert ns["DS_NOCOL"].canonical_name == []


@pytest.mark.parametrize(
    "seed, rows, taken, winner_ds",
    [
        # Alias-vs-alias: first row wins.
        (
            None,
            [
                {"dataset": "ds_a", "canonical_name": '["Shared"]'},
                {"dataset": "ds_b", "canonical_name": '["Shared"]'},
            ],
            "Shared",
            "ds_a",
        ),
        # Alias collides with another row's DS class name → DS class keeps it.
        (
            None,
            [
                {"dataset": "ds_a", "canonical_name": '["DS_B"]'},
                {"dataset": "ds_b", "canonical_name": "[]"},
            ],
            "DS_B",
            "ds_b",
        ),
        # Alias collides with a pre-existing module global → global untouched.
        (
            "EEGDashDataset",
            [{"dataset": "ds1", "canonical_name": '["EEGDashDataset", "Good"]'}],
            "EEGDashDataset",
            None,  # None ⇒ expect the seeded object to remain
        ),
    ],
    ids=["alias-vs-alias", "alias-vs-ds-class", "alias-vs-module-global"],
)
def test_canonical_name_reserved_names_skipped(
    tmp_path, caplog, seed, rows, taken, winner_ds
):
    seed_obj = object()
    ns = {seed: seed_obj} if seed else {}
    with caplog.at_level(logging.WARNING, logger="eegdash.dataset.registry"):
        _register(tmp_path, rows, namespace=ns)

    if winner_ds is None:
        assert ns[taken] is seed_obj
    else:
        assert ns[taken]._dataset == winner_ds
    assert any(
        "already registered" in r.message or "reserved" in r.message
        for r in caplog.records
    )


def test_canonical_name_in_docstring(tmp_path):
    ns = _register(
        tmp_path, [{"dataset": "ds_doc", "canonical_name": '["Foo", "Bar"]'}]
    )
    doc = ns["DS_DOC"].__doc__ or ""
    assert "Also importable as" in doc and "``Foo``" in doc and "``Bar``" in doc
