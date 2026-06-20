# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Warn when explicit filter values match no records (#141).

The check lives on the API side (``EEGDash.find`` via
``warn_unmatched_filter_values``) so the dataset classes stay minimal.
"""

import logging
import re
from unittest.mock import MagicMock, patch

import pytest

from eegdash import EEGDash
from eegdash.bids_metadata import warn_unmatched_filter_values


def _record(sub, task):
    return {"entities_mne": {"subject": sub, "task": task}}


# --------------------------------------------------------------------------- #
# Pure helper
# --------------------------------------------------------------------------- #
def test_helper_warns_on_unmatched_in_value(caplog):
    query = {"dataset": "ds1", "task": {"$in": ["rest", "wrongtask"]}}
    records = [_record("01", "rest")]
    with caplog.at_level(logging.WARNING):
        warn_unmatched_filter_values(query, records)
    msgs = " ".join(r.message for r in caplog.records)
    assert "wrongtask" in msgs and "matched no records" in msgs


def test_helper_no_warning_when_all_matched(caplog):
    query = {"task": {"$in": ["rest", "oddball"]}}
    records = [_record("01", "rest"), _record("02", "oddball")]
    with caplog.at_level(logging.WARNING):
        warn_unmatched_filter_values(query, records)
    assert not any("matched no records" in r.message for r in caplog.records)


def test_helper_skips_when_no_records(caplog):
    # A fully unmatched scalar yields zero records; the empty-result path
    # (a ValueError raised by the caller) handles that, so we stay quiet.
    with caplog.at_level(logging.WARNING):
        warn_unmatched_filter_values({"task": "rest"}, [])
    assert not any("matched no records" in r.message for r in caplog.records)


def test_helper_skips_regex_and_operators(caplog):
    query = {
        "task": re.compile("rest"),
        "ntimes": {"$gte": 1000},
        "$or": [{"subject": "99"}],
    }
    with caplog.at_level(logging.WARNING):
        warn_unmatched_filter_values(query, [_record("01", "rest")])
    assert not any("matched no records" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Through EEGDash.find
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_client():
    with patch("eegdash.api.get_client") as mock_get_client:
        client = MagicMock()
        mock_get_client.return_value = client
        yield client


def test_find_warns_on_unmatched_value(mock_client, caplog):
    mock_client.find.return_value = [_record("01", "rest")]
    eegdash = EEGDash()
    with caplog.at_level(logging.WARNING):
        eegdash.find(dataset="ds1", task=["rest", "wrongtask"])
    msgs = " ".join(r.message for r in caplog.records)
    assert "wrongtask" in msgs
