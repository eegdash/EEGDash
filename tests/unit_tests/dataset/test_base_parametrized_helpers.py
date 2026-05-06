from __future__ import annotations

import sys
import types
import urllib.error
from pathlib import Path

import filelock
import numpy as np
import pytest

if "braindecode" not in sys.modules:
    braindecode = types.ModuleType("braindecode")
    datasets_mod = types.ModuleType("braindecode.datasets")
    datasets_base_mod = types.ModuleType("braindecode.datasets.base")

    class _RawDataset:
        def __init__(self, *args, **kwargs):
            pass

    class _BaseConcatDataset:
        pass

    datasets_mod.BaseConcatDataset = _BaseConcatDataset
    datasets_base_mod.RawDataset = _RawDataset
    braindecode.datasets = datasets_mod

    sys.modules["braindecode"] = braindecode
    sys.modules["braindecode.datasets"] = datasets_mod
    sys.modules["braindecode.datasets.base"] = datasets_base_mod

from eegdash.dataset.base import (
    _SPLIT_ENTITY_RE,
    _SPLIT_PART_RE,
    _clamp_negative_annotation_durations,
    _increment_name_match,
    _iter_split_fif_candidates,
    _make_tolerant_get_sample_info,
    _nemar_fast_paths,
    _noop_filelock,
    _parse_split_fif_missing_path,
    _resolve_nemar_pointer,
    _resolve_nemar_uris,
    _resolve_one_nemar_entry,
)
from eegdash.dataset.exceptions import StorageAccessError


class _FakeAnnotations:
    def __init__(self, durations):
        self.duration = np.array(durations, dtype=float)

    def __len__(self):
        return len(self.duration)


class _FakeRaw:
    def __init__(self, durations):
        self.annotations = None if durations is None else _FakeAnnotations(durations)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "Split raw file detected but next file '/tmp/run-1_raw.fif' does not exist",
            Path("/tmp/run-1_raw.fif"),
        ),
        (
            'Split raw file detected but next file "sub-01_split-02_raw.fif" does not exist',
            Path("sub-01_split-02_raw.fif"),
        ),
        ("some unrelated error", None),
    ],
    ids=["single-quoted", "double-quoted", "no-match"],
)
def test_parse_split_fif_missing_path(message, expected):
    assert _parse_split_fif_missing_path(message) == expected


@pytest.mark.parametrize(
    ("name", "regex", "expected"),
    [
        ("sub_split-009_task_raw.fif", _SPLIT_ENTITY_RE, "sub_split-010_task_raw.fif"),
        ("raw-001.fif", _SPLIT_PART_RE, "raw-002.fif"),
    ],
    ids=["entity-split", "part-split"],
)
def test_increment_name_match_preserves_padding(name, regex, expected):
    match = regex.search(name)
    assert match is not None
    assert _increment_name_match(name, match) == expected


@pytest.mark.parametrize(
    ("current_key", "current_path", "expected_path", "expected_candidates"),
    [
        (
            "s3/sub-01/raw-01.fif",
            Path("/cache/raw-01.fif"),
            Path("/x/raw-02.fif"),
            [("s3/sub-01/raw-02.fif", Path("/cache/raw-02.fif"))],
        ),
        (
            "ds/sub_split-001_task_raw.fif",
            Path("/cache/sub_split-001_task_raw.fif"),
            None,
            [
                (
                    "ds/sub_split-002_task_raw.fif",
                    Path("/cache/sub_split-002_task_raw.fif"),
                ),
                (
                    "ds/sub_split-001_task_raw-1.fif",
                    Path("/cache/sub_split-001_task_raw-1.fif"),
                ),
            ],
        ),
        (
            "d/sub-01_task_raw.fif",
            Path("/cache/sub-01_task_raw.fif"),
            None,
            [
                (
                    "d/sub-01_task_raw-1.fif",
                    Path("/cache/sub-01_task_raw-1.fif"),
                ),
            ],
        ),
    ],
    ids=["expected-path-dedup", "entity-increment", "fallback-dash-1"],
)
def test_iter_split_fif_candidates(
    current_key, current_path, expected_path, expected_candidates
):
    assert (
        _iter_split_fif_candidates(current_key, current_path, expected_path)
        == expected_candidates
    )


@pytest.mark.parametrize(
    ("st_size", "res4", "expected", "orig_called"),
    [
        (
            8 + 4 * 10 * 2,
            {"nsamp": 10, "nchan": 2},
            {"origin": "orig"},
            True,
        ),
        (
            8 + 4 * 8 * 2,
            {"nsamp": 5, "nchan": 2},
            {
                "n_samp": 8,
                "n_samp_tot": 8,
                "block_size": 8,
                "res4_nsamp": 8,
                "n_chan": 2,
            },
            False,
        ),
    ],
    ids=["divisible-use-original", "truncated-fallback"],
)
def test_make_tolerant_get_sample_info(
    monkeypatch, st_size, res4, expected, orig_called
):
    calls = []

    def _orig(fname, _res4, _system_clock):
        calls.append(fname)
        return {"origin": "orig"}

    monkeypatch.setattr("os.path.getsize", lambda _fname: st_size)
    fn = _make_tolerant_get_sample_info(_orig)

    out = fn("fake.meg4", res4, False)

    assert out == expected
    assert bool(calls) is orig_called


@pytest.mark.parametrize(
    ("storage", "relpath", "expected"),
    [
        (
            {"annex_keys": {"raw.fif": "SHA1"}, "sidecar_inline": {}},
            "raw.fif",
            ("SHA1", None),
        ),
        (
            {"annex_keys": {}, "sidecar_inline": {"events.tsv": "a\tb"}},
            "events.tsv",
            (None, "a\tb"),
        ),
        ({}, "missing", (None, None)),
    ],
)
def test_nemar_fast_paths(storage, relpath, expected):
    assert _nemar_fast_paths(storage, relpath) == expected


@pytest.mark.parametrize(
    ("raw_bytes", "expected_annex"),
    [
        (
            b"/annex/objects/MD5E-s10--abcdef1234567890abcdef12345678.fif",
            "MD5E-s10--abcdef1234567890abcdef12345678.fif",
        ),
        (b'{"name": "sidecar"}', None),
        (b"\xff\xfe\xfd", None),
    ],
    ids=["annex-pointer", "inline-json", "non-utf8"],
)
def test_resolve_nemar_pointer_variants(monkeypatch, raw_bytes, expected_annex):
    _resolve_nemar_pointer.cache_clear()
    monkeypatch.setattr(
        "eegdash.dataset.base._fetch_nemar_pointer", lambda *_a: raw_bytes
    )

    annex_key, payload = _resolve_nemar_pointer("dsX", "sub/file")

    assert annex_key == expected_annex
    assert payload == raw_bytes


@pytest.mark.parametrize(
    (
        "stored_key",
        "stored_sidecar",
        "resolver_result",
        "expected_uri",
        "expected_text",
    ),
    [
        ("SHA999", None, None, "s3://nemar/ds/objects/SHA999", None),
        (None, "col\tval\n1\t2\n", None, None, "col\tval\n1\t2\n"),
        (None, None, ("SHA001", b"ignored"), "s3://nemar/ds/objects/SHA001", None),
        (None, None, (None, b"abc"), None, "abc"),
    ],
    ids=["fast-annex", "fast-sidecar", "resolved-annex", "resolved-inline-bytes"],
)
def test_resolve_one_nemar_entry_success_paths(
    monkeypatch,
    tmp_path,
    stored_key,
    stored_sidecar,
    resolver_result,
    expected_uri,
    expected_text,
):
    dest = tmp_path / "sub" / "sample.tsv"

    def _resolver(_dataset_id, _relpath):
        if resolver_result is None:
            raise AssertionError("resolver should not be called")
        return resolver_result

    monkeypatch.setattr("eegdash.dataset.base._resolve_nemar_pointer", _resolver)

    uri = _resolve_one_nemar_entry(
        dataset_id="ds",
        relpath="sub/sample.tsv",
        base="s3://nemar/ds",
        dest=dest,
        stored_key=stored_key,
        stored_sidecar=stored_sidecar,
        is_required=True,
    )

    assert uri == expected_uri
    if expected_text is None:
        assert not dest.exists()
    else:
        assert dest.read_text(encoding="utf-8") == expected_text


@pytest.mark.parametrize("is_required", [True, False], ids=["required", "optional"])
def test_resolve_one_nemar_entry_url_errors(monkeypatch, tmp_path, is_required):
    dest = tmp_path / "x" / "sample.tsv"
    err = urllib.error.URLError("offline")

    monkeypatch.setattr(
        "eegdash.dataset.base._resolve_nemar_pointer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(err),
    )

    if is_required:
        with pytest.raises(StorageAccessError, match="Could not resolve NEMAR pointer"):
            _resolve_one_nemar_entry(
                dataset_id="ds",
                relpath="sub/sample.tsv",
                base="s3://nemar/ds",
                dest=dest,
                stored_key=None,
                stored_sidecar=None,
                is_required=True,
            )
    else:
        with pytest.raises(urllib.error.URLError, match="offline"):
            _resolve_one_nemar_entry(
                dataset_id="ds",
                relpath="sub/sample.tsv",
                base="s3://nemar/ds",
                dest=dest,
                stored_key=None,
                stored_sidecar=None,
                is_required=False,
            )


def test_resolve_nemar_uris_with_optional_sidecar_failure(monkeypatch, tmp_path):
    record = {
        "dataset": "dsA",
        "storage": {
            "base": "s3://nemar/dsA",
            "raw_key": "sub/raw.fif",
            "annex_keys": {"sub/raw.fif": "SHA-RAW", "sub/good.tsv": "SHA-GOOD"},
        },
    }

    def _resolver(*, dataset_id, relpath, **kwargs):
        if relpath == "sub/bad.tsv":
            raise urllib.error.URLError("sidecar missing")
        if relpath == "sub/good.tsv":
            return "s3://nemar/dsA/objects/SHA-GOOD"
        return "s3://nemar/dsA/objects/SHA-RAW"

    monkeypatch.setattr("eegdash.dataset.base._resolve_one_nemar_entry", _resolver)

    raw_uri, dep_uris = _resolve_nemar_uris(
        record=record,
        raw_dest=tmp_path / "raw.fif",
        dep_keys=["sub/bad.tsv", "sub/good.tsv"],
        dep_dests=[tmp_path / "bad.tsv", tmp_path / "good.tsv"],
    )

    assert raw_uri == "s3://nemar/dsA/objects/SHA-RAW"
    assert dep_uris == ["s3://nemar/dsA/objects/SHA-GOOD"]


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"dataset": "ds", "storage": {"base": "s3://x", "raw_key": ""}},
        {"dataset": "", "storage": {"base": "s3://x", "raw_key": "a"}},
    ],
)
def test_resolve_nemar_uris_missing_required_fields(record, tmp_path):
    raw_uri, dep_uris = _resolve_nemar_uris(record, tmp_path / "raw.fif", [], [])
    assert raw_uri is None
    assert dep_uris == []


@pytest.mark.parametrize(
    "durations",
    [None, [], [0.1, 2.0], [0.1, -1.5, 0.0]],
    ids=["no-annotations", "empty", "already-valid", "contains-negative"],
)
def test_clamp_negative_annotation_durations(durations):
    raw = _FakeRaw(durations)

    _clamp_negative_annotation_durations(raw)

    if durations is None:
        assert raw.annotations is None
    elif len(durations) == 0:
        assert raw.annotations.duration.size == 0
    else:
        assert np.all(raw.annotations.duration >= 0)


@pytest.mark.parametrize("raise_inside", [False, True], ids=["normal", "exceptional"])
def test_noop_filelock_restores_original(raise_inside):
    original = filelock.FileLock

    if raise_inside:
        with pytest.raises(RuntimeError, match="boom"):
            with _noop_filelock():
                assert filelock.FileLock.__name__ == "_dummy_filelock"
                raise RuntimeError("boom")
    else:
        with _noop_filelock():
            assert filelock.FileLock.__name__ == "_dummy_filelock"

    assert filelock.FileLock is original
