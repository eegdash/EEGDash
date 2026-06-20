# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Tests for compiled-regex support in build_query_from_kwargs (#135)."""

import re

import pytest

from eegdash.bids_metadata import build_query_from_kwargs


def test_regex_pattern_with_ignorecase():
    query = build_query_from_kwargs(
        dataset="ds002718", task=re.compile("rest", re.IGNORECASE)
    )
    assert query == {
        "dataset": "ds002718",
        "task": {"$regex": "rest", "$options": "i"},
    }


def test_regex_pattern_without_flags():
    assert build_query_from_kwargs(task=re.compile("rest")) == {
        "task": {"$regex": "rest"}
    }


def test_regex_pattern_maps_multiple_flags():
    query = build_query_from_kwargs(
        task=re.compile("rest", re.IGNORECASE | re.MULTILINE | re.DOTALL)
    )
    assert query == {"task": {"$regex": "rest", "$options": "ims"}}


def test_empty_regex_raises():
    with pytest.raises(ValueError, match="empty regex"):
        build_query_from_kwargs(task=re.compile(""))
