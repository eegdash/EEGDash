# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""The dataset docstrings should document the keyword filter fields (#211)."""

import pytest

from eegdash import EEGChallengeDataset, EEGDashDataset
from eegdash.const import ALLOWED_QUERY_FIELDS


@pytest.mark.parametrize("field", sorted(ALLOWED_QUERY_FIELDS - {"data_name"}))
def test_eegdashdataset_docstring_mentions_filter_fields(field):
    doc = EEGDashDataset.__doc__ or ""
    assert field in doc, f"{field!r} missing from EEGDashDataset docstring"


def test_eegdashdataset_docstring_mentions_allowed_query_fields_and_target():
    doc = EEGDashDataset.__doc__ or ""
    assert "ALLOWED_QUERY_FIELDS" in doc
    assert "target_name" in doc


def test_challenge_docstring_mentions_filters():
    doc = EEGChallengeDataset.__doc__ or ""
    assert "ALLOWED_QUERY_FIELDS" in doc
    assert "target_name" in doc
