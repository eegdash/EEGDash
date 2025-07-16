import pytest


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
        "task",
        "ntimes",
        "eeg_json.SamplingFrequency",
    ]
    collection = eegdashObj.collection
    or_query = [{field: {"$exists": False}} for field in expected_fields]
    missing_count = collection.count_documents({"$or": or_query})
    assert missing_count == 0, f"Missing fields in {missing_count} records"
