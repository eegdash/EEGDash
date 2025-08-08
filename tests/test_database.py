from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

from eegdash.api import EEGDash, MongoDBClientSingleton


@pytest.fixture()
def eegdashObj():
    """Fixture to create an instance of EEGDashDataset."""
    from eegdash import EEGDash

    return EEGDash(is_public=True)


def test_fields(eegdashObj):
    """Test that mongodb records have the expected fields."""
    expected_fields = [
        "dataset",
        "subject",
    ]
    collection = eegdashObj.collection
    or_query = [{field: {"$exists": False}} for field in expected_fields]
    missing_count = collection.count_documents({"$or": or_query})
    assert missing_count == 0, f"Missing fields in {missing_count} records"


class TestEEGDashSingletonIntegration:
    """Test cases for EEGDash integration with MongoDB singleton."""

    def setup_method(self):
        """Setup method to clear singleton instances before each test."""
        MongoDBClientSingleton._instances.clear()

    def teardown_method(self):
        """Teardown method to clean up after each test."""
        MongoDBClientSingleton.close_all()

    @patch("mne.utils.get_config")
    @patch("eegdash.api.MongoClient")
    def test_eegdash_uses_singleton(self, mock_mongo_client, mock_get_config):
        """Test that EEGDash instances use the singleton pattern."""
        mock_get_config.return_value = "mongodb://test_connection"
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = mock_client

        # Create two EEGDash instances with same parameters
        eegdash1 = EEGDash(is_public=True, is_staging=False)
        eegdash2 = EEGDash(is_public=True, is_staging=False)

        # Should use the same underlying MongoDB connection
        assert eegdash1._EEGDash__client is eegdash2._EEGDash__client
        assert eegdash1._EEGDash__db is eegdash2._EEGDash__db
        assert eegdash1._EEGDash__collection is eegdash2._EEGDash__collection

        # MongoClient should only be called once
        mock_mongo_client.assert_called_once()

    @patch("mne.utils.get_config")
    @patch("eegdash.api.MongoClient")
    def test_eegdash_different_staging_flags(self, mock_mongo_client, mock_get_config):
        """Test that EEGDash instances with different staging flags use different connections."""
        mock_get_config.return_value = "mongodb://test_connection"
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = mock_client

        # Create EEGDash instances with different staging flags
        eegdash_prod = EEGDash(is_public=True, is_staging=False)
        eegdash_staging = EEGDash(is_public=True, is_staging=True)
        assert eegdash_prod._EEGDash__client is not eegdash_staging._EEGDash__client
        assert eegdash_prod._EEGDash__db is not eegdash_staging._EEGDash__db
        assert (
            eegdash_prod._EEGDash__collection
            is not eegdash_staging._EEGDash__collection
        )

        # Should create two different singleton instances
        assert len(MongoDBClientSingleton._instances) == 2

        # MongoClient should be called twice
        assert mock_mongo_client.call_count == 2

    def test_eegdash_close_behavior(self):
        """Test that EEGDash.close() doesn't affect singleton connections."""
        with (
            patch("mne.utils.get_config") as mock_get_config,
            patch("eegdash.api.MongoClient") as mock_mongo_client,
        ):
            mock_get_config.return_value = "mongodb://test_connection"
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection
            mock_mongo_client.return_value = mock_client

            # Create EEGDash instance
            eegdash1 = EEGDash(is_public=True, is_staging=False)

            # Close the instance
            eegdash1.close()

            # Client should not be closed
            mock_client.close.assert_not_called()

            # Create another instance - should reuse the same connection
            eegdash2 = EEGDash(is_public=True, is_staging=False)
            assert eegdash1._EEGDash__client is eegdash2._EEGDash__client

    def test_eegdash_close_all_connections(self):
        """Test that EEGDash.close_all_connections() properly closes singleton connections."""
        with (
            patch("mne.utils.get_config") as mock_get_config,
            patch("eegdash.api.MongoClient") as mock_mongo_client,
        ):
            mock_get_config.return_value = "mongodb://test_connection"
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection
            mock_mongo_client.return_value = mock_client

            # Create EEGDash instance
            _ = EEGDash(is_public=True, is_staging=False)

            # Close all connections
            EEGDash.close_all_connections()

            # Client should be closed
            mock_client.close.assert_called_once()

            # Singleton instances should be cleared
            assert len(MongoDBClientSingleton._instances) == 0

    def test_concurrent_eegdash_creation(self):
        """Test concurrent creation of EEGDash instances."""
        results = []

        def create_eegdash():
            """Worker function to create EEGDash instance."""
            with (
                patch("mne.utils.get_config") as mock_get_config,
                patch("eegdash.api.MongoClient") as mock_mongo_client,
            ):
                mock_get_config.return_value = "mongodb://test_connection"
                mock_client = MagicMock()
                mock_db = MagicMock()
                mock_collection = MagicMock()
                mock_client.__getitem__.return_value = mock_db
                mock_db.__getitem__.return_value = mock_collection
                mock_mongo_client.return_value = mock_client

                eegdash = EEGDash(is_public=True, is_staging=False)
                results.append(eegdash)

        # Set up the mock outside the threads
        with (
            patch("mne.utils.get_config") as mock_get_config,
            patch("eegdash.api.MongoClient") as mock_mongo_client,
        ):
            mock_get_config.return_value = "mongodb://test_connection"
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection
            mock_mongo_client.return_value = mock_client

            # Create multiple threads
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(
                        lambda: results.append(
                            EEGDash(is_public=True, is_staging=False)
                        )
                    )
                    for _ in range(10)
                ]

                # Wait for all to complete
                for future in as_completed(futures):
                    future.result()

            # All instances should share the same underlying connection
            if results:
                first_client = results[0]._EEGDash__client
                for eegdash in results[1:]:
                    assert eegdash._EEGDash__client is first_client

            # Should only have one singleton instance
            assert len(MongoDBClientSingleton._instances) <= 1
