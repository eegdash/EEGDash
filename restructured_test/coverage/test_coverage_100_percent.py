"""Comprehensive tests to achieve 100% coverage on 15 targeted files.

Target files:
1. eegdash/paths.py (1 line missing)
2. eegdash/features/decorators.py (1 line missing)
3. eegdash/dataset/base.py (1 line missing)
4. eegdash/hbn/preprocessing.py (2 lines missing)
5. eegdash/features/feature_bank/utils.py (3 lines missing)
6. eegdash/__init__.py (3 lines missing)
7. eegdash/http_api_client.py (3 lines missing)
8. eegdash/dataset/registry.py (5 lines missing)
9. eegdash/downloader.py (5 lines missing)
10. eegdash/schemas.py (6 lines missing)
11. eegdash/features/feature_bank/spectral.py (6 lines missing)
12. eegdash/local_bids.py (7 lines missing)
13. eegdash/features/feature_bank/csp.py (7 lines missing)
14. eegdash/features/feature_bank/signal.py (9 lines missing)
15. eegdash/bids_metadata.py (13 lines missing)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# 1. eegdash/paths.py - Test line 57 (MNE_DATA fallback)
# =============================================================================


def test_get_default_cache_dir_env_var(tmp_path, monkeypatch):
    """Test EEGDASH_CACHE_DIR environment variable."""
    from eegdash.paths import get_default_cache_dir

    monkeypatch.setenv("EEGDASH_CACHE_DIR", str(tmp_path / "custom_cache"))
    result = get_default_cache_dir()
    assert result == (tmp_path / "custom_cache").resolve()


def test_get_default_cache_dir_local_fallback(monkeypatch):
    """Test local .eegdash_cache fallback."""
    from eegdash.paths import get_default_cache_dir

    monkeypatch.delenv("EEGDASH_CACHE_DIR", raising=False)
    result = get_default_cache_dir()
    assert ".eegdash_cache" in str(result)


def test_get_default_cache_dir_mne_fallback(tmp_path, monkeypatch):
    """Test MNE_DATA fallback when local mkdir fails (line 57)."""
    from eegdash.paths import get_default_cache_dir

    monkeypatch.delenv("EEGDASH_CACHE_DIR", raising=False)

    # Mock Path.mkdir to raise an exception
    original_mkdir = Path.mkdir

    def failing_mkdir(self, *args, **kwargs):
        if ".eegdash_cache" in str(self):
            raise PermissionError("Cannot create directory")
        return original_mkdir(self, *args, **kwargs)

    # Mock MNE_DATA to return a valid path
    with patch("eegdash.paths.mne_get_config") as mock_mne:
        mock_mne.return_value = str(tmp_path / "mne_data")
        with patch.object(Path, "mkdir", failing_mkdir):
            result = get_default_cache_dir()
            assert "mne_data" in str(result)


# =============================================================================
# 2. eegdash/features/decorators.py - Test line 132 (directed bivariate)
# =============================================================================


def test_bivariate_feature_directed():
    """Test bivariate_feature with directed=True (line 132)."""
    from eegdash.features.decorators import bivariate_feature
    from eegdash.features.extractors import DirectedBivariateFeature

    @bivariate_feature
    def dummy_undirected(x):
        return x

    @bivariate_feature
    def dummy_directed(x):
        return x

    # Apply with directed=True
    dummy_directed_applied = bivariate_feature(lambda x: x, directed=True)

    from eegdash.features.extractors import _get_underlying_func

    kind = _get_underlying_func(dummy_directed_applied).feature_kind
    assert isinstance(kind, DirectedBivariateFeature)


def test_feature_predecessor_empty():
    """Test FeaturePredecessor with no args."""
    from eegdash.features.decorators import FeaturePredecessor

    @FeaturePredecessor()
    def my_func(x):
        return x

    from eegdash.features.extractors import _get_underlying_func

    assert _get_underlying_func(my_func).parent_extractor_type == [None]


def test_feature_kind_decorator():
    """Test FeatureKind decorator."""
    from eegdash.features.decorators import FeatureKind
    from eegdash.features.extractors import UnivariateFeature

    kind_instance = UnivariateFeature()

    @FeatureKind(kind_instance)
    def my_feature(x):
        return x

    from eegdash.features.extractors import _get_underlying_func

    assert _get_underlying_func(my_feature).feature_kind is kind_instance


# =============================================================================
# 3. eegdash/dataset/base.py - Test line 180 (__len__ with sfreq/ntimes)
# =============================================================================


def test_eegdashraw_len_with_sfreq_ntimes(tmp_path):
    """Test __len__ returns ntimes*sfreq when raw not loaded (line 180)."""
    from eegdash.dataset.base import EEGDashRaw

    # Create proper BIDS directory structure
    ds_dir = tmp_path / "ds_test"
    ds_dir.mkdir()
    sub_dir = ds_dir / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)

    record = {
        "dataset": "ds_test",
        "data_name": "ds_test_sub-01_task-rest_eeg.set",
        "bidspath": "ds_test/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "storage": {
            "backend": "local",
            "base": str(ds_dir),
            "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.set",
        },
        "subject": "01",
        "session": None,
        "run": None,
        "task": "rest",
        "modality": "eeg",
        "suffix": "eeg",
        "datatype": "eeg",
        "extension": ".set",
        "ntimes": 10.5,  # seconds
        "sfreq": 256,  # Hz
        "sampling_frequency": 256,
        "entities_mne": {"subject": "01", "task": "rest"},
        "entities": {"subject": "01", "task": "rest"},
    }

    raw_obj = EEGDashRaw(record=record, cache_dir=tmp_path)

    # Should return int(ntimes * sfreq) = int(10.5 * 256) = 2688
    assert len(raw_obj) == int(10.5 * 256)


# =============================================================================
# 4. eegdash/hbn/preprocessing.py - Test lines 87, 91 (no events warning)
# =============================================================================


def test_hbn_preprocessor_no_annotations():
    """Test warning when no eye events found (lines 87-91)."""
    import mne

    from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation

    # Create raw with no relevant annotations
    info = mne.create_info(ch_names=["EEG"], sfreq=256, ch_types=["eeg"])
    data = np.random.randn(1, 5000)
    raw = mne.io.RawArray(data, info)
    raw.set_annotations(mne.Annotations(onset=[0], duration=[1], description=["other"]))

    preprocessor = hbn_ec_ec_reannotation()
    # Should return raw unchanged with warning
    result = preprocessor.transform(raw)
    assert result is raw


def test_hbn_preprocessor_with_valid_annotations():
    """Test with valid eye annotations."""
    import mne

    from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation

    info = mne.create_info(ch_names=["EEG"], sfreq=256, ch_types=["eeg"])
    data = np.random.randn(1, 256 * 60)  # 60 seconds
    raw = mne.io.RawArray(data, info)
    raw.set_annotations(
        mne.Annotations(
            onset=[1, 25],
            duration=[1, 1],
            description=["instructed_toCloseEyes", "instructed_toOpenEyes"],
        )
    )

    preprocessor = hbn_ec_ec_reannotation()
    result = preprocessor.transform(raw)
    # Should have new annotations
    assert len(result.annotations) > 0


# =============================================================================
# 5. eegdash/features/feature_bank/utils.py - Test lines 23, 27, 35
# =============================================================================


def test_get_valid_freq_band_assertions():
    """Test assertion failures in get_valid_freq_band (lines 23, 27)."""
    from eegdash.features.feature_bank.utils import get_valid_freq_band

    fs, n = 256, 512
    # f0 = 2 * fs / n = 1.0, f1 = fs / 2 = 128

    # f_min below f0 should raise
    with pytest.raises(AssertionError):
        get_valid_freq_band(fs, n, f_min=0.5)

    # f_max above f1 should raise
    with pytest.raises(AssertionError):
        get_valid_freq_band(fs, n, f_max=150)


def test_slice_freq_band_none_limits():
    """Test slice_freq_band with None limits (line 35)."""
    from eegdash.features.feature_bank.utils import slice_freq_band

    f = np.array([1, 2, 3, 4, 5])
    x = np.array([10, 20, 30, 40, 50])

    # Both None - should return unchanged
    result = slice_freq_band(f, x)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], f)
    np.testing.assert_array_equal(result[1], x)


def test_reduce_freq_bands():
    """Test reduce_freq_bands function."""
    from eegdash.features.feature_bank.utils import reduce_freq_bands

    f = np.array([1, 2, 3, 5, 6, 10, 15, 25])
    x = np.ones_like(f, dtype=float)
    bands = {"low": (1, 6), "high": (10, 25)}  # Within f range

    result = reduce_freq_bands(f, x, bands)
    assert "low" in result
    assert "high" in result


# =============================================================================
# 6. eegdash/__init__.py - Test lines 31, 33, 38 (lazy imports)
# =============================================================================


def test_lazy_import_eegdash():
    """Test lazy import of EEGDash (line 31)."""
    import eegdash

    cls = eegdash.EEGDash
    assert cls is not None


def test_lazy_import_datasets():
    """Test lazy import of datasets (line 33)."""
    import eegdash

    cls1 = eegdash.EEGDashDataset
    cls2 = eegdash.EEGChallengeDataset
    assert cls1 is not None
    assert cls2 is not None


def test_lazy_import_preprocessing():
    """Test lazy import of preprocessing module (line 38)."""
    import eegdash

    mod = eegdash.preprocessing
    assert mod is not None


def test_lazy_import_invalid():
    """Test AttributeError for invalid attribute."""
    import eegdash

    with pytest.raises(AttributeError):
        _ = eegdash.nonexistent_attribute


def test_dir_function():
    """Test __dir__ returns expected attributes."""
    import eegdash

    attrs = dir(eegdash)
    assert "EEGDash" in attrs
    assert "EEGDashDataset" in attrs


# =============================================================================
# 7. eegdash/http_api_client.py - Test lines 73, 104-105
# =============================================================================


def test_find_with_skip_parameter():
    """Test find with skip parameter (line 73)."""
    from eegdash.http_api_client import EEGDashAPIClient

    with patch("requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"id": 1}]}
        mock_session.get.return_value = mock_response

        client = EEGDashAPIClient()
        client.find(query={"test": 1}, limit=10, skip=5)

        # Check skip was passed
        call_args = mock_session.get.call_args
        assert call_args[1]["params"]["skip"] == 5


def test_insert_many():
    """Test insert_many method (lines 104-105)."""
    from eegdash.http_api_client import EEGDashAPIClient

    with patch("requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {"insertedCount": 2}
        mock_session.post.return_value = mock_response

        client = EEGDashAPIClient()
        result = client.insert_many([{"a": 1}, {"b": 2}])
        assert result == 2


# =============================================================================
# 8. eegdash/dataset/registry.py - Test lines 61-63, 74, 89, 308
# =============================================================================


def test_register_with_api_failure_csv_fallback(tmp_path):
    """Test fallback to CSV when API fails (lines 61-63)."""
    from eegdash.dataset.registry import register_openneuro_datasets

    # Create a minimal CSV
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("dataset,n_subjects\nds001234,10\n")

    class DummyBase:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    ns = {}
    # Mock API to fail
    with patch("urllib.request.urlopen", side_effect=Exception("API down")):
        register_openneuro_datasets(
            from_api=True,
            summary_file=str(csv_path),
            namespace=ns,
            base_class=DummyBase,
        )

    assert "DS001234" in ns


def test_register_empty_dataset_id_skip(tmp_path):
    """Test that empty dataset_id rows are skipped (line 74)."""
    from eegdash.dataset.registry import register_openneuro_datasets

    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("dataset,n_subjects\n,10\nds001234,5\n")

    class DummyBase:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    ns = {}
    with patch("urllib.request.urlopen", side_effect=Exception("API down")):
        register_openneuro_datasets(
            from_api=False,
            summary_file=str(csv_path),
            namespace=ns,
            base_class=DummyBase,
        )

    # Only ds001234 should be registered, empty row skipped
    assert "DS001234" in ns
    assert len([k for k in ns if k.startswith("DS")]) == 1


def test_register_add_to_all(tmp_path):
    """Test add_to_all parameter (line 89)."""
    from eegdash.dataset.registry import register_openneuro_datasets

    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("dataset,n_subjects\nds005509,20\n")

    class DummyBase:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    ns = {"__all__": []}
    with patch("urllib.request.urlopen", side_effect=Exception("API down")):
        register_openneuro_datasets(
            from_api=False,
            summary_file=str(csv_path),
            namespace=ns,
            base_class=DummyBase,
            add_to_all=True,
        )

    assert "DS005509" in ns["__all__"]


# =============================================================================
# 9. eegdash/downloader.py - Test lines 84, 99, 130, 136-137
# =============================================================================


def test_download_file_remote_size_none(tmp_path):
    """Test download when remote size is None (line 84)."""
    from eegdash.downloader import download_s3_file

    local_path = tmp_path / "test.txt"
    local_path.write_text("existing content")

    mock_fs = MagicMock()
    mock_fs.info.side_effect = Exception("No size info")

    # File exists, remote size is None - should return early
    result = download_s3_file("s3://bucket/key", local_path, filesystem=mock_fs)
    assert result == local_path


def test_download_file_incomplete(tmp_path):
    """Test download raises on incomplete download (line 99)."""
    from eegdash.downloader import download_s3_file

    local_path = tmp_path / "test.txt"

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": 1000}  # Expected size

    def mock_get(rpath, lpath, **kwargs):
        Path(lpath).write_text("short")  # Write less than expected

    mock_fs.get = mock_get

    with pytest.raises(OSError, match="Incomplete download"):
        download_s3_file("s3://bucket/key", local_path, filesystem=mock_fs)


def test_download_files_skip_existing(tmp_path):
    """Test download_files with skip_existing (line 130)."""
    from eegdash.downloader import download_files

    existing_file = tmp_path / "existing.txt"
    existing_file.write_text("existing content")

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": len("existing content")}

    files = [("s3://bucket/existing.txt", existing_file)]

    result = download_files(files, filesystem=mock_fs, skip_existing=True)
    # Should skip and return empty list
    assert result == []


def test_remote_size_returns_none_on_exception():
    """Test _remote_size returns None on exception (lines 136-137)."""
    from eegdash.downloader import _remote_size

    mock_fs = MagicMock()
    mock_fs.info.side_effect = Exception("Error")

    result = _remote_size(mock_fs, "s3://bucket/key")
    assert result is None


def test_remote_size_invalid_size_type():
    """Test _remote_size with non-integer size."""
    from eegdash.downloader import _remote_size

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": "not_a_number"}

    result = _remote_size(mock_fs, "s3://bucket/key")
    assert result is None


# =============================================================================
# 10. eegdash/schemas.py - Test lines 74, 439, 448, 513, 515, 588
# =============================================================================


def test_create_record_minimal():
    """Test create_record with minimal required fields."""
    from eegdash.schemas import create_record

    record = create_record(
        dataset="ds001234",
        storage_base="s3://bucket",
        bids_relpath="sub-01/eeg/sub-01_eeg.set",
    )

    assert record["dataset"] == "ds001234"
    assert "storage" in record
    assert record["storage"]["base"] == "s3://bucket"


def test_create_record_with_all_fields():
    """Test create_record with all optional fields."""
    from eegdash.schemas import create_record

    record = create_record(
        dataset="ds001234",
        storage_base="s3://bucket",
        bids_relpath="sub-01/eeg/sub-01_task-rest_eeg.set",
        subject="01",
        session="01",
        task="rest",
        run="01",
        dep_keys=["key1", "key2"],
        datatype="eeg",
        suffix="eeg",
        storage_backend="s3",
    )

    # Check entities are in the record
    assert record["entities"]["subject"] == "01"
    assert record["entities"]["session"] == "01"
    assert record["entities"]["task"] == "rest"


def test_validate_record_missing_fields():
    """Test validate_record with missing fields."""
    from eegdash.schemas import validate_record

    # Minimal invalid record
    record = {}
    errors = validate_record(record)
    assert "missing: dataset" in errors
    assert "missing: bids_relpath" in errors


def test_validate_record_missing_storage():
    """Test validate_record with missing storage."""
    from eegdash.schemas import validate_record

    record = {"dataset": "ds001234", "bids_relpath": "x", "bidspath": "y"}
    errors = validate_record(record)
    assert "missing: storage" in errors


def test_sanitize_run_for_mne():
    """Test _sanitize_run_for_mne function."""
    from eegdash.schemas import _sanitize_run_for_mne

    assert _sanitize_run_for_mne(None) is None
    assert _sanitize_run_for_mne(1) == "1"
    assert _sanitize_run_for_mne("2") == "2"
    assert _sanitize_run_for_mne("abc") is None  # non-numeric
    assert _sanitize_run_for_mne("") is None


# =============================================================================
# 11. eegdash/features/feature_bank/spectral.py - Test lines 62, 68, 93-96
# =============================================================================


def test_spectral_entropy():
    """Test spectral_entropy with zeros and non-zeros (line 62, 68)."""
    from eegdash.features.feature_bank.spectral import (
        spectral_entropy,
        spectral_normalized_preprocessor,
        spectral_preprocessor,
    )

    # Create data with some zeros
    x = np.random.randn(2, 3, 256)  # batch, channels, time
    f, p = spectral_preprocessor(x, fs=256)
    f_norm, p_norm = spectral_normalized_preprocessor(f, p)

    result = spectral_entropy(f_norm, p_norm)
    assert result.shape == (2, 3)


def test_spectral_edge():
    """Test spectral_edge function (lines 93-96)."""
    from eegdash.features.feature_bank.spectral import (
        spectral_edge,
        spectral_normalized_preprocessor,
        spectral_preprocessor,
    )

    x = np.random.randn(2, 3, 256)
    f, p = spectral_preprocessor(x, fs=256)
    f_norm, p_norm = spectral_normalized_preprocessor(f, p)

    result = spectral_edge(f_norm, p_norm, edge=0.9)
    assert result.shape == (2, 3)


def test_spectral_bands_power():
    """Test spectral_bands_power function."""
    from eegdash.features.feature_bank.spectral import (
        spectral_bands_power,
        spectral_preprocessor,
    )

    # Use longer signal for better frequency resolution
    x = np.random.randn(2, 3, 1024)
    f, p = spectral_preprocessor(x, fs=256)

    # Filter to valid frequency range

    result = spectral_bands_power(f, p, bands={"alpha": (8, 12)})
    assert isinstance(result, dict)
    assert "alpha" in result


# =============================================================================
# 12. eegdash/local_bids.py - Test lines 74-77, 120, 124-125
# =============================================================================


def test_normalize_modalities_none():
    """Test _normalize_modalities with None."""
    from eegdash.local_bids import _normalize_modalities

    result = _normalize_modalities(None)
    assert result == ["eeg"]


def test_normalize_modalities_list():
    """Test _normalize_modalities with list."""
    from eegdash.local_bids import _normalize_modalities

    result = _normalize_modalities(["eeg", "fnirs"])
    assert "eeg" in result
    assert "nirs" in result  # fnirs aliased to nirs


def test_normalize_modalities_string():
    """Test _normalize_modalities with string."""
    from eegdash.local_bids import _normalize_modalities

    result = _normalize_modalities("meg")
    assert result == ["meg"]


def test_discover_local_bids_records_invalid_extension(tmp_path):
    """Test that invalid extensions are skipped (lines 120, 124-125)."""
    from eegdash.local_bids import discover_local_bids_records

    # Create minimal BIDS structure
    ds = tmp_path / "ds001234"
    ds.mkdir()
    (ds / "dataset_description.json").write_text('{"Name": "Test"}')
    sub_dir = ds / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    # Create a .json file (should be skipped)
    (sub_dir / "sub-01_task-rest_eeg.json").write_text("{}")
    # Create a valid .set file
    (sub_dir / "sub-01_task-rest_eeg.set").touch()

    with patch("eegdash.local_bids.find_matching_paths") as mock_find:
        mock_bp_json = MagicMock()
        mock_bp_json.fpath = str(sub_dir / "sub-01_task-rest_eeg.json")
        mock_bp_set = MagicMock()
        mock_bp_set.fpath = str(sub_dir / "sub-01_task-rest_eeg.set")
        mock_bp_set.datatype = "eeg"
        mock_bp_set.suffix = "eeg"
        mock_bp_set.subject = "01"
        mock_bp_set.session = None
        mock_bp_set.task = "rest"
        mock_bp_set.run = None

        mock_find.return_value = [mock_bp_json, mock_bp_set]

        records = discover_local_bids_records(ds, {"dataset": "ds001234"})
        # Only .set file should be included
        assert len(records) == 1
        assert records[0]["extension"] == ".set"


# =============================================================================
# 13. eegdash/features/feature_bank/csp.py - Test lines 17-22, 97
# =============================================================================


def test_csp_full_workflow():
    """Test CommonSpatialPattern full workflow (lines 17-22, 97)."""
    from eegdash.features.feature_bank.csp import CommonSpatialPattern

    csp = CommonSpatialPattern()
    csp.clear()

    # Create fake data: batch x channels x time
    np.random.seed(42)
    x1 = np.random.randn(50, 4, 100)  # Class 0
    x2 = np.random.randn(50, 4, 100) + 0.5  # Class 1

    x = np.vstack([x1, x2])
    y = np.array([0] * 50 + [1] * 50)

    # Partial fit
    csp.partial_fit(x, y)
    csp.fit()

    # Call - returns dict with numbered keys (channel indices)
    result = csp(x[:5], n_select=2)
    assert isinstance(result, dict)
    # Keys are numbered strings: '0', '1', etc.
    assert len(result) > 0


def test_csp_strict_criterion_error():
    """Test CSP raises when criterion too strict (line 97)."""
    from eegdash.features.feature_bank.csp import CommonSpatialPattern

    csp = CommonSpatialPattern()
    csp.clear()

    np.random.seed(42)
    x1 = np.random.randn(50, 4, 100)
    x2 = np.random.randn(50, 4, 100) + 0.5

    x = np.vstack([x1, x2])
    y = np.array([0] * 50 + [1] * 50)

    csp.partial_fit(x, y)
    csp.fit()

    # Very strict criterion that filters all weights
    with pytest.raises(RuntimeError, match="too strict"):
        csp(x[:5], crit_select=0.0001)


# =============================================================================
# 14. eegdash/features/feature_bank/signal.py - Test lines 30, 104, 110, 118-123
# =============================================================================


def test_signal_hilbert_preprocessor():
    """Test signal_hilbert_preprocessor (line 30)."""
    from eegdash.features.feature_bank.signal import signal_hilbert_preprocessor

    x = np.random.randn(2, 3, 256)
    result = signal_hilbert_preprocessor(x)
    assert result.shape == x.shape
    # Hilbert envelope should be non-negative
    assert np.all(result >= 0)


def test_signal_zero_crossings():
    """Test signal_zero_crossings with threshold (line 104)."""
    from eegdash.features.feature_bank.signal import signal_zero_crossings

    # Create signal with known zero crossings
    x = np.array([[1, -1, 1, -1, 0.5, -0.5]])
    result = signal_zero_crossings(x)
    assert result[0] > 0


def test_signal_decorrelation_time():
    """Test signal_decorrelation_time (lines 110, 118-123)."""
    from eegdash.features.feature_bank.signal import signal_decorrelation_time

    np.random.seed(42)
    x = np.random.randn(2, 3, 256)
    result = signal_decorrelation_time(x, fs=256)
    assert result.shape == (2, 3)


def test_signal_hjorth_features():
    """Test Hjorth features."""
    from eegdash.features.feature_bank.signal import (
        signal_hjorth_activity,
        signal_hjorth_complexity,
        signal_hjorth_mobility,
    )

    x = np.random.randn(2, 3, 256)

    activity = signal_hjorth_activity(x)
    mobility = signal_hjorth_mobility(x)
    complexity = signal_hjorth_complexity(x)

    assert activity.shape == (2, 3)
    assert mobility.shape == (2, 3)
    assert complexity.shape == (2, 3)


# =============================================================================
# 15. eegdash/bids_metadata.py - Test lines 211, 265, 270, 342, 387, 392, 395, 409-414
# =============================================================================


def test_merge_query_with_empty_kwargs():
    """Test merge_query returns empty warning (line 211)."""
    from eegdash import bids_metadata

    # Empty base_query and no kwargs
    result = bids_metadata.merge_query({})
    assert result == {}


def test_participants_row_for_subject_not_found(tmp_path):
    """Test when subject not in participants.tsv (line 265, 270)."""
    from eegdash import bids_metadata

    # Create participants.tsv without the subject
    tsv = tmp_path / "participants.tsv"
    tsv.write_text("participant_id\tsex\nsub-02\tM\n")

    result = bids_metadata.participants_row_for_subject(tmp_path, "01")
    assert result is None


def test_participants_row_for_subject_no_file(tmp_path):
    """Test when participants.tsv doesn't exist."""
    from eegdash import bids_metadata

    result = bids_metadata.participants_row_for_subject(tmp_path, "01")
    assert result is None
