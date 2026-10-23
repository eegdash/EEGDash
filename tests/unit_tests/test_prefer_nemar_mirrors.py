"""Retiring an OpenNeuro twin must not drop its curated tags.

The 2026-08 prefer-NEMAR retirement deleted ``ds*`` docs that carried the
only copy of the LLM tags, emptying ~540 catalog entries. These tests pin
the carry-over decision; they are pure and offline.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_INGESTIONS_DIR = str(Path(__file__).resolve().parents[2] / "scripts" / "ingestions")


@pytest.fixture()
def mod():
    if _INGESTIONS_DIR not in sys.path:
        sys.path.insert(0, _INGESTIONS_DIR)
    pytest.importorskip("httpx")
    return importlib.import_module("prefer_nemar_mirrors")


_TAGGED = {
    "tags": {"pathology": ["Healthy"], "modality": ["Visual"], "type": ["Memory"]},
    "tagger_meta": {"model": "openai/gpt-5.2"},
}
_UNKNOWN = {
    "tags": {"pathology": ["Unknown"], "modality": ["Unknown"], "type": ["Unknown"]}
}


def test_carries_tags_to_untagged_mirror(mod):
    assert mod.curated_carry_over(_TAGGED, {"dataset_id": "on1"}) == _TAGGED


def test_all_unknown_mirror_counts_as_untagged(mod):
    assert mod.curated_carry_over(_TAGGED, _UNKNOWN) == _TAGGED


def test_never_overwrites_mirror_with_its_own_tags(mod):
    own = {"tags": {"pathology": ["Epilepsy"], "modality": [], "type": []}}
    assert mod.curated_carry_over(_TAGGED, own) == {}


def test_nothing_to_carry_from_untagged_twin(mod):
    assert mod.curated_carry_over(_UNKNOWN, {}) == {}
    assert mod.curated_carry_over({}, {}) == {}


def test_string_tag_values_are_recognised(mod):
    legacy = {"tags": {"pathology": "Healthy", "modality": "Unknown"}}
    assert mod.has_real_tags(legacy)
    assert mod.curated_carry_over(legacy, None) == {"tags": legacy["tags"]}


def test_openneuro_check_takes_the_doc(mod):
    assert mod.dataset_is_openneuro({"source": "OpenNeuro"})
    assert not mod.dataset_is_openneuro({"source": "nemar"})
    assert not mod.dataset_is_openneuro(None)


class _Api:
    """``/datasets/names`` lists record owners; ``/datasets/summary`` lists documents."""

    names = ["on007813"]  # ds007813's records are stored under on007813
    docs = [
        {"dataset_id": "ds007813"},
        {"dataset_id": "on007813"},
        {"dataset_id": "nm000001"},
    ]

    def __call__(self, request):
        import httpx  # noqa: PLC0415

        if request.url.path.endswith("/datasets/names"):
            return httpx.Response(200, json={"data": self.names})
        skip = int(request.url.params.get("skip", 0))
        limit = int(request.url.params.get("limit", 100))
        page = self.docs[skip : skip + limit]
        return httpx.Response(200, json={"data": page, "total": len(self.docs)})


def test_lists_dataset_documents_without_records(mod):
    # ds007813 has no records of its own, so /datasets/names never names it and
    # the twin was never retired (5 such twins lingered on 2026-10-23).
    import httpx  # noqa: PLC0415

    client = httpx.Client(transport=httpx.MockTransport(_Api()))
    ids = mod.list_dataset_ids(client, "http://api", "eegdash", page_size=2)
    assert ids == ["ds007813", "nm000001", "on007813"]
