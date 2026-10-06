"""Opt-in tiny public metadata probes; no recordings/models/sidecars downloaded."""

import os

import pytest

from eegdash import EEGDash

pytestmark = pytest.mark.skipif(
    os.getenv("EEGDASH_TEST_NEMAR_METADATA") != "1",
    reason="explicit public metadata network opt-in required",
)


def test_public_catalog_one_document():
    docs = EEGDash(backend="nemar").find_datasets(limit=1)
    assert len(docs) == 1
    assert docs[0]["provider"] == "nemar"


def test_public_rich_metadata_one_dataset():
    doc = EEGDash(backend="nemar").get_dataset("nm000132")
    assert doc["dataset_id"] == "nm000132"
    assert doc["version"].startswith("v")
    assert doc["license"] and doc["dataset_doi"] and doc["version_doi"]
