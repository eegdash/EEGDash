import pandas as pd
import pytest

from eegdash.bids_metadata import (
    build_query_from_kwargs,
    get_entities_from_record,
    get_entity_from_record,
    merge_query,
    participants_row_for_subject,
)


def test_get_entity_from_record():
    # v1 (flat)
    rec1 = {"subject": "01", "task": "rest"}
    assert get_entity_from_record(rec1, "subject") == "01"
    assert get_entity_from_record(rec1, "run") is None

    # v2 (nested)
    rec2 = {"entities": {"subject": "02", "task": "task"}}
    assert get_entity_from_record(rec2, "subject") == "02"
    assert get_entity_from_record(rec2, "run") is None

    # Mixed (nested priority)
    rec3 = {"entities": {"subject": "03"}, "subject": "04"}
    assert get_entity_from_record(rec3, "subject") == "03"


def test_get_entities_from_record():
    rec = {"entities": {"subject": "01", "task": "rest"}, "extra": "val"}
    ents = get_entities_from_record(rec)
    assert ents == {"subject": "01", "task": "rest"}

    rec_flat = {"subject": "01", "session": "01"}
    ents = get_entities_from_record(rec_flat)
    assert ents == {"subject": "01", "session": "01"}


def test_build_query_from_kwargs():
    # Valid
    q = build_query_from_kwargs(subject="01", task=["rest", "task"])
    assert q["subject"] == "01"
    assert q["task"] == {"$in": ["rest", "task"]}

    # Deduplication
    q = build_query_from_kwargs(session=["1", "1", "2"])
    assert q["session"] == {"$in": ["1", "2"]}

    # Invalid field
    with pytest.raises(ValueError, match="Unsupported query field"):
        build_query_from_kwargs(invalid_field="val")

    # None value
    with pytest.raises(ValueError, match="Received None"):
        build_query_from_kwargs(subject=None)

    # Empty list
    with pytest.raises(ValueError, match="Received an empty list"):
        build_query_from_kwargs(session=[])

    # List with None or empty strings (should be filtered)
    q = build_query_from_kwargs(subject=["01", None, "  ", "02"])
    assert q["subject"] == {"$in": ["01", "02"]}

    # List filtering results in empty list
    with pytest.raises(ValueError, match="Received an empty list"):
        build_query_from_kwargs(subject=[None, "   "])

    # Empty string scalar
    with pytest.raises(ValueError, match="Received an empty string"):
        build_query_from_kwargs(subject="   ")


def test_merge_query():
    # Only kwargs
    q = merge_query(subject="01")
    assert q == {"subject": "01"}

    # Only raw query
    q = merge_query(query={"subject": "02"})
    assert q == {"subject": "02"}

    # Both
    q = merge_query(query={"subject": "01"}, task="rest")
    assert q == {"$and": [{"subject": "01"}, {"task": "rest"}]}

    # Both but query=None (safe handling)
    q = merge_query(query=None, task="rest")
    assert q == {"task": "rest"}

    # Not require query
    q = merge_query(require_query=False)
    assert q == {}

    q = merge_query(query=None, require_query=False)
    assert q == {}

    # Conflicting
    with pytest.raises(ValueError, match="Conflicting constraints"):
        merge_query(query={"subject": "01"}, subject="02")

    # Empty
    with pytest.raises(ValueError, match="Query required"):
        merge_query()

    # Explicit empty
    q = merge_query(query={})
    assert q == {}


def test_participants_row_for_subject(tmp_path):
    # Create dummy participants.tsv
    tsv = tmp_path / "participants.tsv"
    df = pd.DataFrame(
        {"participant_id": ["sub-01", "sub-02"], "age": [20, 25], "sex": ["M", "F"]}
    )
    df.to_csv(tsv, sep="\t", index=False)

    # Found
    row = participants_row_for_subject(tmp_path, "01")
    assert row is not None
    assert row["participant_id"] == "sub-01"
    assert row["age"] == "20"  # read as string usually in helper

    # Found robustly
    row = participants_row_for_subject(tmp_path, "sub-02")
    assert row is not None
    assert row["participant_id"] == "sub-02"

    # Not found
    row = participants_row_for_subject(tmp_path, "03")
    assert row is None

    # No file
    row = participants_row_for_subject(tmp_path / "nonexistent", "01")
    assert row is None
