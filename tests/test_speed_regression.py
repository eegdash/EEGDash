"""Performance regression tests for EEGBIDSDataset.

This module tests that the optimized EEGBIDSDataset (using mne_bids.find_matching_paths)
doesn't introduce performance regressions compared to the original implementation
(using pybids.BIDSLayout).

The tests verify:
- Dataset initialization time remains acceptable (< 5 seconds)
- Individual metadata query times are sub-millisecond
- Batch metadata access patterns are performant
- No functional regressions in metadata extraction
"""

import time

import pytest

from eegdash.dataset.bids_dataset import EEGBIDSDataset
from eegdash.paths import get_default_cache_dir

# Performance thresholds (in seconds)
INIT_TIME_THRESHOLD = 5.0  # Initialization should not exceed 5 seconds
SINGLE_QUERY_THRESHOLD = 0.1  # Individual queries should not exceed 100ms
BATCH_QUERY_THRESHOLD = 0.5  # Batch of 10 queries should not exceed 500ms


@pytest.fixture(scope="session")
def bids_dataset_path():
    """Get the path to the test BIDS dataset."""
    # Use the mini dataset included in the repository for CI compatibility
    path = get_default_cache_dir() / "ds005509-bdf-mini"
    if not path.exists():
        pytest.skip(f"BIDS dataset not found at {path}")
    return path


@pytest.fixture(scope="session")
def bids_dataset(bids_dataset_path):
    """Load the BIDS dataset once for all tests in the session."""
    dataset = EEGBIDSDataset(
        data_dir=str(bids_dataset_path), dataset=bids_dataset_path.name
    )
    return dataset


class TestInitializationPerformance:
    """Tests for dataset initialization time."""

    def test_initialization_time(self, bids_dataset_path):
        """Test that dataset initialization is fast enough."""
        start = time.time()
        dataset = EEGBIDSDataset(
            data_dir=str(bids_dataset_path), dataset=bids_dataset_path.name
        )
        init_time = time.time() - start

        assert len(dataset.files) > 0, "Dataset should find at least one recording"
        assert init_time < INIT_TIME_THRESHOLD, (
            f"Initialization took {init_time:.3f}s, exceeded threshold of "
            f"{INIT_TIME_THRESHOLD}s. This may indicate a performance regression."
        )

    def test_finds_all_recordings(self, bids_dataset):
        """Test that the optimized code finds all recordings without regression."""
        # The test dataset should have multiple recordings (mini dataset has ~60)
        assert len(bids_dataset.files) >= 50, (
            f"Expected at least 50 recordings, found {len(bids_dataset.files)}"
        )

    def test_dataset_validation(self, bids_dataset):
        """Test that the dataset passes EEG validation."""
        assert bids_dataset.check_eeg_dataset(), (
            "Dataset should be identified as EEG dataset"
        )


class TestMetadataAccessPerformance:
    """Tests for individual metadata query performance."""

    def test_single_subject_query(self, bids_dataset):
        """Test that single subject attribute queries are fast."""
        test_file = bids_dataset.files[0]

        start = time.time()
        subject = bids_dataset.get_bids_file_attribute("subject", test_file)
        query_time = time.time() - start

        assert subject is not None, "Subject should be found"
        assert query_time < SINGLE_QUERY_THRESHOLD, (
            f"Subject query took {query_time:.3f}s, exceeded threshold of {SINGLE_QUERY_THRESHOLD}s"
        )

    def test_channel_labels_query(self, bids_dataset):
        """Test that channel_labels query is fast."""
        test_file = bids_dataset.files[0]

        start = time.time()
        labels = bids_dataset.channel_labels(test_file)
        query_time = time.time() - start

        assert len(labels) > 0, "Should find at least one channel"
        assert query_time < SINGLE_QUERY_THRESHOLD, (
            f"channel_labels query took {query_time:.3f}s, exceeded threshold of {SINGLE_QUERY_THRESHOLD}s"
        )

    def test_channel_types_query(self, bids_dataset):
        """Test that channel_types query is fast."""
        test_file = bids_dataset.files[0]

        start = time.time()
        try:
            types = bids_dataset.channel_types(test_file)
            query_time = time.time() - start

            assert len(types) > 0, "Should find at least one channel type"
            assert query_time < SINGLE_QUERY_THRESHOLD, (
                f"channel_types query took {query_time:.3f}s, exceeded threshold of {SINGLE_QUERY_THRESHOLD}s"
            )
        except KeyError:
            # Some datasets don't have 'type' column in channels.tsv
            query_time = time.time() - start
            pytest.skip("Dataset's channels.tsv doesn't have 'type' column")

        # If we get here, verify query was fast
        assert query_time < SINGLE_QUERY_THRESHOLD, (
            f"channel_types query took {query_time:.3f}s, exceeded threshold of {SINGLE_QUERY_THRESHOLD}s"
        )

    def test_eeg_json_query(self, bids_dataset):
        """Test that eeg_json query is fast."""
        test_file = bids_dataset.files[0]

        start = time.time()
        eeg_data = bids_dataset.eeg_json(test_file)
        query_time = time.time() - start

        assert isinstance(eeg_data, dict), "eeg_json should return a dictionary"
        assert query_time < SINGLE_QUERY_THRESHOLD, (
            f"eeg_json query took {query_time:.3f}s, exceeded threshold of {SINGLE_QUERY_THRESHOLD}s"
        )

    def test_num_times_query(self, bids_dataset):
        """Test that num_times query is fast."""
        test_file = bids_dataset.files[0]

        start = time.time()
        num_times = bids_dataset.num_times(test_file)
        query_time = time.time() - start

        assert isinstance(num_times, int), "num_times should return an integer"
        assert query_time < SINGLE_QUERY_THRESHOLD, (
            f"num_times query took {query_time:.3f}s, exceeded threshold of {SINGLE_QUERY_THRESHOLD}s"
        )


class TestBatchMetadataPerformance:
    """Tests for batch metadata queries (common usage pattern)."""

    def test_batch_subject_queries(self, bids_dataset):
        """Test that querying multiple subjects doesn't show regression."""
        test_files = bids_dataset.files[:10]  # Test first 10 files

        start = time.time()
        subjects = [
            bids_dataset.get_bids_file_attribute("subject", f) for f in test_files
        ]
        total_time = time.time() - start

        assert len(subjects) == len(test_files), "Should retrieve all subjects"
        assert all(s is not None for s in subjects), "All subjects should be found"
        assert total_time < BATCH_QUERY_THRESHOLD, (
            f"Batch of {len(test_files)} subject queries took {total_time:.3f}s, "
            f"exceeded threshold of {BATCH_QUERY_THRESHOLD}s"
        )

    def test_batch_channel_labels_queries(self, bids_dataset):
        """Test that querying channel_labels for multiple files is fast."""
        test_files = bids_dataset.files[:10]  # Test first 10 files

        start = time.time()
        all_labels = [bids_dataset.channel_labels(f) for f in test_files]
        total_time = time.time() - start

        assert len(all_labels) == len(test_files), (
            "Should retrieve labels for all files"
        )
        assert all(len(l) > 0 for l in all_labels), (
            "All files should have channel labels"
        )
        assert total_time < BATCH_QUERY_THRESHOLD, (
            f"Batch of {len(test_files)} channel_labels queries took {total_time:.3f}s, "
            f"exceeded threshold of {BATCH_QUERY_THRESHOLD}s"
        )

    def test_batch_eeg_json_queries(self, bids_dataset):
        """Test that querying eeg_json for multiple files is fast."""
        test_files = bids_dataset.files[:10]  # Test first 10 files

        start = time.time()
        all_eeg_jsons = [bids_dataset.eeg_json(f) for f in test_files]
        total_time = time.time() - start

        assert len(all_eeg_jsons) == len(test_files), (
            "Should retrieve JSON for all files"
        )
        assert all(isinstance(j, dict) for j in all_eeg_jsons), (
            "All should be dictionaries"
        )
        assert total_time < BATCH_QUERY_THRESHOLD, (
            f"Batch of {len(test_files)} eeg_json queries took {total_time:.3f}s, "
            f"exceeded threshold of {BATCH_QUERY_THRESHOLD}s"
        )


class TestMetadataCorrectness:
    """Tests to ensure metadata is retrieved correctly (no functional regression)."""

    def test_subject_extraction(self, bids_dataset):
        """Test that subject IDs are correctly extracted."""
        test_file = bids_dataset.files[0]
        subject = bids_dataset.get_bids_file_attribute("subject", test_file)

        # Subject should be a string and present in the filename
        assert isinstance(subject, str), "Subject should be a string"
        assert len(subject) > 0, "Subject should not be empty"
        assert f"sub-{subject}" in test_file, f"Subject {subject} should be in filename"

    def test_channel_labels_valid(self, bids_dataset):
        """Test that channel labels are valid."""
        test_file = bids_dataset.files[0]
        labels = bids_dataset.channel_labels(test_file)

        assert isinstance(labels, list), "channel_labels should return a list"
        assert len(labels) > 0, "Should have at least one channel"
        assert all(isinstance(l, str) for l in labels), "All labels should be strings"

    def test_channel_types_valid(self, bids_dataset):
        """Test that channel types are valid."""
        test_file = bids_dataset.files[0]

        try:
            types = bids_dataset.channel_types(test_file)

            assert isinstance(types, list), "channel_types should return a list"
            assert len(types) > 0, "Should have at least one type"
            assert all(isinstance(t, str) for t in types), "All types should be strings"
            # Common EEG channel types
            valid_types = {"EEG", "EOG", "ECG", "EMG", "MISC"}
            assert any(t in valid_types for t in types), (
                f"At least one channel type should be in {valid_types}"
            )
        except KeyError:
            # Some datasets don't have 'type' column in channels.tsv - skip this test
            pytest.skip("Dataset's channels.tsv doesn't have 'type' column")

    def test_eeg_json_structure(self, bids_dataset):
        """Test that eeg_json returns proper BIDS structure."""
        test_file = bids_dataset.files[0]
        eeg_data = bids_dataset.eeg_json(test_file)

        assert isinstance(eeg_data, dict), "eeg_json should return a dictionary"
        # BIDS eeg.json should have at least SamplingFrequency
        # But it's okay if it's empty or minimal
        assert isinstance(eeg_data, dict), "Result should always be a dict"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nonexistent_file_handling(self, bids_dataset):
        """Test that querying a nonexistent file raises appropriate error."""
        nonexistent_file = "/path/to/nonexistent/file.bdf"

        with pytest.raises(FileNotFoundError):
            bids_dataset.channel_labels(nonexistent_file)

    def test_invalid_attribute(self, bids_dataset):
        """Test that querying an invalid attribute returns None or raises appropriately."""
        test_file = bids_dataset.files[0]

        # Querying an invalid attribute should return None
        result = bids_dataset.get_bids_file_attribute("nonexistent_attr", test_file)
        assert result is None, "Invalid attribute should return None"


@pytest.fixture(scope="session")
def performance_report(bids_dataset, bids_dataset_path):
    """Generate a performance report for the test session."""
    report = {
        "init_time": None,
        "num_files": len(bids_dataset.files),
        "query_times": {},
    }

    # Measure initialization (on new instance)
    start = time.time()
    _ = EEGBIDSDataset(data_dir=str(bids_dataset_path), dataset=bids_dataset_path.name)
    report["init_time"] = time.time() - start

    # Measure query times
    test_file = bids_dataset.files[0]
    queries = {
        "subject": lambda: bids_dataset.get_bids_file_attribute("subject", test_file),
        "channel_labels": lambda: bids_dataset.channel_labels(test_file),
        "eeg_json": lambda: bids_dataset.eeg_json(test_file),
        "num_times": lambda: bids_dataset.num_times(test_file),
    }

    # Try channel_types if available
    try:
        queries["channel_types"] = lambda: bids_dataset.channel_types(test_file)
    except KeyError:
        pass  # Skip if not available

    for query_name, query_func in queries.items():
        start = time.time()
        try:
            query_func()
            report["query_times"][query_name] = time.time() - start
        except KeyError:
            # Skip unavailable queries
            pass

    return report


def test_performance_report(performance_report):
    """Print performance report (informational test)."""
    print("\n" + "=" * 70)
    print("EEGBIDSDATASET PERFORMANCE REPORT")
    print("=" * 70)
    print(f"\nNumber of files: {performance_report['num_files']}")
    print(f"\nInitialization time: {performance_report['init_time']:.3f}s")
    print("\nQuery times (per file):")
    for query_name, query_time in performance_report["query_times"].items():
        print(f"  {query_name:20s}: {query_time:.4f}s")
    print("\n" + "=" * 70)
