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
