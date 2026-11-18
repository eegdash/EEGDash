"""The dataset-page Sidecars card lists the fNIRS sidecars.

The card probes one record and maps the trailing names of its
``storage.dep_keys`` to labels. It knew ``eeg.json``, ``meg.json``,
``ieeg.json`` and ``electrodes.tsv`` but neither ``nirs.json`` nor
``optodes.tsv``, so fNIRS pages listed only "events · channels ·
coordsystem". The dep_keys below are those of the live nm000175 record
(13 Nov 2026). The HTTP helper is stubbed, so the tests stay offline.
"""

from __future__ import annotations

import pytest
from tests_support.dataset_page import load_dataset_page_module

NM000175_DEP_KEYS = [
    "sub-01/nirs/sub-01_task-tapping_channels.tsv",
    "sub-01/nirs/sub-01_task-tapping_events.tsv",
    "sub-01/nirs/sub-01_optodes.tsv",
    "sub-01/nirs/sub-01_task-tapping_nirs.json",
    "sub-01/nirs/sub-01_coordsystem.json",
]

EEG_DEP_KEYS = [
    "sub-01/eeg/sub-01_task-rest_events.tsv",
    "sub-01/eeg/sub-01_task-rest_events.json",
    "sub-01/eeg/sub-01_task-rest_channels.tsv",
    "sub-01/eeg/sub-01_electrodes.tsv",
    "sub-01/eeg/sub-01_coordsystem.json",
    "sub-01/eeg/sub-01_task-rest_eeg.json",
]


@pytest.fixture
def data_loaders(monkeypatch):
    module = load_dataset_page_module("data_loaders")
    monkeypatch.delenv("EEGDASH_NO_API", raising=False)
    return module


def _stub_record(monkeypatch, module, dep_keys):
    body = {"success": True, "data": [{"storage": {"dep_keys": dep_keys}}]}
    monkeypatch.setattr(module, "_get_json", lambda _url, **_kwargs: body)


def test_fnirs_sidecars_are_listed(data_loaders, monkeypatch):
    _stub_record(monkeypatch, data_loaders, NM000175_DEP_KEYS)
    assert data_loaders._detect_sidecars_for_dataset("nm000175") == [
        "events",
        "channels",
        "optodes",
        "coordsystem",
        "nirs.json",
    ]


def test_eeg_sidecars_unchanged(data_loaders, monkeypatch):
    _stub_record(monkeypatch, data_loaders, EEG_DEP_KEYS)
    assert data_loaders._detect_sidecars_for_dataset("ds002893") == [
        "events",
        "events.json",
        "channels",
        "electrodes",
        "coordsystem",
        "eeg.json",
    ]


def test_every_label_has_a_render_slot(data_loaders):
    labels = set(data_loaders._SIDECAR_SUFFIXES.values())
    assert labels == set(data_loaders._SIDECAR_RENDER_ORDER)
