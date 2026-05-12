from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from eegdash.bids_metadata import (
    _check_constraint_conflict,
    attach_participants_extras,
    build_query_from_kwargs,
    enrich_from_participants,
    get_entities_from_record,
    get_entity_from_record,
    merge_participants_fields,
    merge_query,
    normalize_key,
    participants_extras_from_tsv,
    participants_row_for_subject,
)


@pytest.mark.parametrize(
    "record,entity,expected",
    [
        ({"subject": "01"}, "subject", "01"),
        ({"entities": {"subject": "02"}, "subject": "legacy"}, "subject", "02"),
        ({"entities": {"task": "rest"}}, "subject", None),
    ],
)
def test_get_entity_from_record(record, entity, expected):
    assert get_entity_from_record(record, entity) == expected


def test_get_entities_from_record_filters_missing_values():
    record = {"entities": {"subject": "01", "task": "rest", "run": None}}
    assert get_entities_from_record(record) == {"subject": "01", "task": "rest"}


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"subject": " 01 "}, {"subject": "01"}),
        (
            {"task": ["rest", "rest", " task ", "", None]},
            {"task": {"$in": ["rest", "task"]}},
        ),
        ({"session": ("01", "02")}, {"session": {"$in": ["01", "02"]}}),
    ],
)
def test_build_query_from_kwargs_normalization(kwargs, expected):
    assert build_query_from_kwargs(**kwargs) == expected


@pytest.mark.parametrize(
    "kwargs,error_match",
    [
        ({"unknown": "x"}, "Unsupported query field"),
        ({"subject": None}, "Received None"),
        ({"subject": "   "}, "empty string"),
        ({"task": []}, "empty list"),
        ({"task": [None, " "]}, "empty list"),
    ],
)
def test_build_query_from_kwargs_errors(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        build_query_from_kwargs(**kwargs)


@pytest.mark.parametrize(
    "query,kwargs,require_query,expected",
    [
        ({"dataset": "ds1"}, {}, True, {"dataset": "ds1"}),
        (None, {"subject": "01"}, True, {"subject": "01"}),
        ({}, {"subject": "01"}, True, {"subject": "01"}),
        (
            {"dataset": "ds1"},
            {"subject": "01"},
            True,
            {"$and": [{"dataset": "ds1"}, {"subject": "01"}]},
        ),
        (None, {}, False, {}),
    ],
)
def test_merge_query_happy_paths(query, kwargs, require_query, expected):
    assert merge_query(query=query, require_query=require_query, **kwargs) == expected


@pytest.mark.parametrize(
    "query,kwargs,error_match",
    [
        (None, {}, "Query required"),
        ({"subject": "01"}, {"subject": "02"}, "Conflicting constraints"),
    ],
)
def test_merge_query_errors(query, kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        merge_query(query=query, **kwargs)


def test_check_constraint_conflict_for_in_sets():
    with pytest.raises(ValueError, match="Conflicting constraints"):
        _check_constraint_conflict(
            {"subject": {"$in": ["01", "02"]}},
            {"subject": {"$in": ["03"]}},
            "subject",
        )


@pytest.mark.parametrize(
    "raw_key,expected",
    [("Age (Years)", "age_years"), (" participant-id ", "participant_id")],
)
def test_normalize_key(raw_key, expected):
    assert normalize_key(raw_key) == expected


def test_participants_row_for_subject_matching_and_missing(tmp_path):
    participants = tmp_path / "participants.tsv"
    participants.write_text(
        "participant_id\tparticipant\tage\nsub-01\tfoo\t20\nsub-02\tbar\t25\n"
    )

    assert participants_row_for_subject(tmp_path, "01")["age"] == "20"
    assert participants_row_for_subject(tmp_path, "sub-02")["participant"] == "bar"
    assert participants_row_for_subject(tmp_path, "99") is None
    assert participants_row_for_subject(tmp_path / "missing", "01") is None


def test_participants_row_returns_none_without_identifier_columns(tmp_path):
    (tmp_path / "participants.tsv").write_text("age\n20\n")
    assert participants_row_for_subject(tmp_path, "01") is None


def test_participants_extras_from_tsv_filters_ids_and_na_values(tmp_path):
    row = pd.Series(
        {
            "participant_id": "sub-01",
            "age": " 20 ",
            "sex": "N/A",
            "notes": "unknown",
            "site": "NYU",
        }
    )
    with patch("eegdash.bids_metadata.participants_row_for_subject", return_value=row):
        extras = participants_extras_from_tsv(tmp_path, "01")

    assert extras == {"age": "20", "site": "NYU"}


@pytest.mark.parametrize(
    "description,participants,description_fields,expected",
    [
        (
            {"age": "30"},
            {"Age": "20", "Sex": "F", "sex": "M", "hand": "R"},
            ["sex"],
            {"age": "30", "sex": "F", "hand": "R"},
        ),
        (
            {"existing": "x"},
            {"Age (Years)": "10"},
            ["age_years"],
            {"existing": "x", "age_years": "10"},
        ),
    ],
)
def test_merge_participants_fields(
    description, participants, description_fields, expected
):
    enriched = merge_participants_fields(
        description=description,
        participants_row=participants,
        description_fields=description_fields,
    )
    assert enriched == expected


@pytest.mark.parametrize(
    "description,participants",
    [
        ("not-a-dict", {"age": "20"}),
        ({"x": 1}, "not-a-dict"),
    ],
)
def test_merge_participants_fields_returns_input_for_invalid_types(
    description, participants
):
    assert merge_participants_fields(description, participants) == description


@pytest.mark.parametrize(
    "description_factory",
    [
        lambda: {},
        lambda: pd.Series({"existing": "value"}),
    ],
)
def test_attach_participants_extras_updates_raw_and_description(description_factory):
    raw = SimpleNamespace(info={"subject_info": {"participants_extras": {"age": "30"}}})
    description = description_factory()

    attach_participants_extras(raw, description, {"age": "99", "site": "NYU"})

    # setdefault: existing keys are preserved
    assert raw.info["subject_info"]["participants_extras"]["age"] == "30"
    assert raw.info["subject_info"]["participants_extras"]["site"] == "NYU"

    if isinstance(description, dict):
        assert description["site"] == "NYU"
    else:
        assert "existing" in description.index


def test_attach_participants_extras_early_return_for_empty_extras():
    raw = SimpleNamespace(info={"subject_info": {}})
    description = {}

    attach_participants_extras(raw, description, {})

    assert raw.info == {"subject_info": {}}
    assert description == {}


@pytest.mark.parametrize("subject,expected", [(None, {}), ("01", {"age": "20"})])
def test_enrich_from_participants(subject, expected, tmp_path):
    bids_path = SimpleNamespace(subject=subject)
    raw = SimpleNamespace(info={})
    description = {}

    with patch(
        "eegdash.bids_metadata.participants_extras_from_tsv", return_value={"age": "20"}
    ) as extras:
        result = enrich_from_participants(tmp_path, bids_path, raw, description)

    assert result == expected
    if subject is None:
        extras.assert_not_called()
    else:
        extras.assert_called_once()
        assert raw.info["subject_info"]["participants_extras"]["age"] == "20"
        assert description["age"] == "20"
