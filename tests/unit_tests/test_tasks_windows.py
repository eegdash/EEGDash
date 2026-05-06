# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Unit tests for the Workstream 4 windowing convenience layer.

These tests exercise :meth:`eegdash.tasks.base.EEGTask.make_windows` and the
small helpers around it (time-string parsing, manifest hashing, sampling-rate
introspection). Network access is forbidden: braindecode windowers are
monkey-patched and we hand-craft tiny stub objects that mimic the bits of a
``BaseConcatDataset`` the helper actually inspects (``len()``,
``datasets[0].raw.info['sfreq']`` and ``datasets[i].description``).
"""

from __future__ import annotations

from typing import Any

import pytest

# --------------------------------------------------------------------------- #
# Fixtures / stubs                                                            #
# --------------------------------------------------------------------------- #


class _StubInfo(dict):
    """Tiny ``mne.Info``-shaped dict so ``info['sfreq']`` works."""


class _StubRaw:
    def __init__(self, sfreq: float = 128.0) -> None:
        self.info = _StubInfo(sfreq=sfreq)


class _StubBaseDataset:
    """Single-subject sub-dataset stand-in used by the helpers.

    Mirrors the bits of a braindecode ``BaseDataset`` that
    :func:`_resolve_sfreq` and :func:`_windows_per_subject` look at.
    """

    def __init__(self, subject: str, n_windows: int, sfreq: float = 128.0) -> None:
        self.raw = _StubRaw(sfreq=sfreq)
        self._n_windows = n_windows
        self.description = {"subject": subject}

    def __len__(self) -> int:
        return self._n_windows


class _StubConcat:
    """Concat-dataset stand-in: a list of ``_StubBaseDataset`` plus ``len``."""

    def __init__(self, sub_datasets: list[_StubBaseDataset]) -> None:
        self.datasets = sub_datasets

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)


@pytest.fixture()
def stub_concat() -> _StubConcat:
    """Two-subject concat dataset; subject A has 3 windows, subject B has 2."""
    return _StubConcat(
        [
            _StubBaseDataset(subject="A", n_windows=3, sfreq=128.0),
            _StubBaseDataset(subject="B", n_windows=2, sfreq=128.0),
        ]
    )


@pytest.fixture()
def patched_windowers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the braindecode windowers with capture-and-return stubs.

    Returns the captured-call dictionary so each test can assert on which
    function ran and which kwargs reached braindecode.
    """
    import braindecode.preprocessing as _bp

    captured: dict[str, Any] = {"events": None, "fixed": None}

    def _fake_events(concat_ds: Any, **kwargs: Any) -> _StubConcat:
        captured["events"] = {"concat": concat_ds, "kwargs": kwargs}
        return concat_ds  # echo it back so tests can inspect len/datasets.

    def _fake_fixed(concat_ds: Any, **kwargs: Any) -> _StubConcat:
        captured["fixed"] = {"concat": concat_ds, "kwargs": kwargs}
        return concat_ds

    monkeypatch.setattr(_bp, "create_windows_from_events", _fake_events)
    monkeypatch.setattr(_bp, "create_fixed_length_windows", _fake_fixed)
    return captured


def _make_fixed_task() -> Any:
    """Build a minimal ``EEGTask`` subclass declaring a fixed-window recipe."""
    from eegdash.tasks.base import EEGTask

    class _FixedTask(EEGTask):
        name = "fixed-stub"

        def metadata_query(self) -> dict[str, Any]:
            return {"dataset": "ds-stub", "subject": "A"}

        def label_definition(self) -> dict[str, Any]:
            return {
                "type": "classification",
                "num_classes": 2,
                "mapping": {"a": 0, "b": 1},
                "source": "events",
            }

        def preprocessing_recipe(self) -> list[Any]:
            return []

        def windowing_recipe(self) -> dict[str, Any]:
            return {"kind": "fixed"}

        def split_definitions(self) -> list[dict[str, Any]]:
            return []

        def metrics(self) -> dict[str, Any]:
            return {"primary": "accuracy"}

        def baseline_metadata(self) -> dict[str, Any]:
            return {}

    return _FixedTask()


# --------------------------------------------------------------------------- #
# Time-string parsing                                                         #
# --------------------------------------------------------------------------- #


def test_parse_time_to_samples_handles_seconds_ms_and_int() -> None:
    from eegdash.tasks.base import _parse_time_to_samples

    assert _parse_time_to_samples("2s", sfreq=128.0) == 256
    assert _parse_time_to_samples("500ms", sfreq=128.0) == 64
    assert _parse_time_to_samples("1.5s", sfreq=200.0) == 300
    # Integers are passed through (already in samples).
    assert _parse_time_to_samples(256, sfreq=None) == 256
    # Integer-valued floats are accepted.
    assert _parse_time_to_samples(128.0, sfreq=None) == 128


def test_parse_time_to_samples_rejects_garbage() -> None:
    from eegdash.tasks.base import _parse_time_to_samples

    with pytest.raises(ValueError):
        _parse_time_to_samples("two seconds", sfreq=128.0)
    with pytest.raises(ValueError):
        _parse_time_to_samples("2", sfreq=128.0)  # missing unit
    with pytest.raises(ValueError):
        _parse_time_to_samples(0.5, sfreq=None)  # non-integer float
    with pytest.raises(ValueError):
        # No sampling rate -> cannot translate a string spec.
        _parse_time_to_samples("2s", sfreq=None)


# --------------------------------------------------------------------------- #
# make_windows -- the documented contract                                     #
# --------------------------------------------------------------------------- #


def test_make_windows_fixed_returns_windows_and_report(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
) -> None:
    """``make_windows(engine='braindecode', kind='fixed', ...)`` should
    dispatch to ``create_fixed_length_windows`` and return a fully populated
    report when ``return_report=True``.
    """
    task = _make_fixed_task()

    result = task.make_windows(
        stub_concat,
        engine="braindecode",
        kind="fixed",
        window_size="2s",
        stride="2s",
        return_report=True,
    )

    assert isinstance(result, tuple) and len(result) == 2
    windows, report = result
    assert windows is stub_concat

    # The fixed-length windower was called, the events one was not.
    assert patched_windowers["fixed"] is not None
    assert patched_windowers["events"] is None

    # ``window_size="2s"`` at 128 Hz -> 256 samples; same for the stride.
    forwarded = patched_windowers["fixed"]["kwargs"]
    assert forwarded["window_size_samples"] == 256
    assert forwarded["window_stride_samples"] == 256

    # All required report keys are present.
    required = {
        "engine",
        "function",
        "function_kwargs",
        "library_versions",
        "n_windows",
        "manifest_hash",
    }
    missing = required - set(report)
    assert not missing, f"report missing required keys: {sorted(missing)}"

    # Sanity-check the actual values.
    assert report["engine"] == "braindecode"
    assert report["function"] == (
        "braindecode.preprocessing.create_fixed_length_windows"
    )
    assert report["function_kwargs"]["window_size_samples"] == 256
    assert report["function_kwargs"]["window_stride_samples"] == 256
    assert report["library_versions"]["braindecode"]
    assert report["n_windows"] == 5  # 3 + 2 from the stub
    assert report["sfreq"] == 128.0
    assert report["windows_per_subject"] == {"A": 3, "B": 2}
    # Manifest hash is a hex prefix; just assert shape & determinism.
    assert isinstance(report["manifest_hash"], str)
    assert len(report["manifest_hash"]) == 16
    assert all(c in "0123456789abcdef" for c in report["manifest_hash"])


def test_make_windows_return_report_false(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
) -> None:
    """``return_report=False`` should yield only the windows object."""
    task = _make_fixed_task()
    out = task.make_windows(
        stub_concat,
        kind="fixed",
        window_size=256,  # already in samples
        stride=256,
        return_report=False,
    )
    assert out is stub_concat
    assert not isinstance(out, tuple)


def test_make_windows_unsupported_engine_raises(
    stub_concat: _StubConcat,
) -> None:
    task = _make_fixed_task()
    with pytest.raises(ValueError, match="not supported"):
        task.make_windows(stub_concat, engine="numpy")


def test_make_windows_kind_mismatch_with_recipe_raises(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
) -> None:
    """Calling ``kind='events'`` on a task whose recipe says ``kind='fixed'``
    should raise a clear, named error before any windower runs.
    """
    task = _make_fixed_task()
    with pytest.raises(ValueError, match="contradicts the task's windowing_recipe"):
        task.make_windows(stub_concat, kind="events")
    # Nothing should have been called when validation failed.
    assert patched_windowers["events"] is None
    assert patched_windowers["fixed"] is None


def test_make_windows_kind_matching_recipe_routes_correctly(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
) -> None:
    """Explicit ``kind`` matching the recipe is a no-op override."""
    task = _make_fixed_task()
    windows, report = task.make_windows(stub_concat, kind="fixed")
    assert windows is stub_concat
    assert patched_windowers["fixed"] is not None
    assert patched_windowers["events"] is None
    assert report["kind"] == "fixed"


def test_make_windows_eoec_uses_events_windower(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
) -> None:
    """Sanity-check the EO/EC task: its recipe is event-based."""
    from eegdash.tasks import get_task

    task = get_task("eyes-open-closed")
    windows, report = task.make_windows(stub_concat)
    assert patched_windowers["events"] is not None
    assert patched_windowers["fixed"] is None
    assert report["function"] == (
        "braindecode.preprocessing.create_windows_from_events"
    )
    # The recipe defaults flow through unchanged when no override is passed.
    forwarded = patched_windowers["events"]["kwargs"]
    assert forwarded["mapping"] == {"eyes_open": 0, "eyes_closed": 1}
    assert forwarded["preload"] is True


def test_make_windows_eoec_kind_events_matches_recipe(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
) -> None:
    """Explicit ``kind='events'`` matches the EO/EC recipe and routes events."""
    from eegdash.tasks import get_task

    task = get_task("eyes-open-closed")
    windows, report = task.make_windows(stub_concat, kind="events")
    assert report["kind"] == "events"
    assert patched_windowers["events"] is not None
    assert patched_windowers["fixed"] is None


def test_make_windows_eoec_kind_fixed_contradicts_recipe(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
) -> None:
    """The plan example: calling ``kind='fixed'`` on an event-based task
    must raise rather than silently rewriting the recipe.
    """
    from eegdash.tasks import get_task

    task = get_task("eyes-open-closed")
    with pytest.raises(ValueError, match="contradicts the task's windowing_recipe"):
        task.make_windows(stub_concat, kind="fixed", window_size="2s", stride="2s")
    assert patched_windowers["events"] is None
    assert patched_windowers["fixed"] is None


# --------------------------------------------------------------------------- #
# Report contents                                                             #
# --------------------------------------------------------------------------- #


def test_report_function_kwargs_match_braindecode_call(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
) -> None:
    task = _make_fixed_task()
    _, report = task.make_windows(
        stub_concat,
        kind="fixed",
        window_size="2s",
        stride="1s",
        drop_last_window=True,  # extra kwarg flowed through **kwargs
    )
    bd_kwargs = patched_windowers["fixed"]["kwargs"]
    assert report["function_kwargs"] == bd_kwargs
    assert bd_kwargs["window_size_samples"] == 256
    assert bd_kwargs["window_stride_samples"] == 128
    assert bd_kwargs["drop_last_window"] is True


def test_report_manifest_hash_is_deterministic(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
) -> None:
    task = _make_fixed_task()
    _, r1 = task.make_windows(stub_concat, kind="fixed", window_size="2s", stride="2s")
    _, r2 = task.make_windows(stub_concat, kind="fixed", window_size="2s", stride="2s")
    assert r1["manifest_hash"] == r2["manifest_hash"]
    # Changing the window size should change the hash.
    _, r3 = task.make_windows(stub_concat, kind="fixed", window_size="1s", stride="1s")
    assert r1["manifest_hash"] != r3["manifest_hash"]


def test_report_library_versions_includes_braindecode(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
) -> None:
    import braindecode

    task = _make_fixed_task()
    _, report = task.make_windows(stub_concat, kind="fixed")
    assert report["library_versions"].get("braindecode") == braindecode.__version__


# --------------------------------------------------------------------------- #
# _resolve_dataset                                                             #
# --------------------------------------------------------------------------- #


def test_make_windows_falls_back_to_resolve_dataset(
    stub_concat: _StubConcat,
    patched_windowers: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``dataset`` is omitted, ``_resolve_dataset`` must be called and
    its return value forwarded into braindecode.
    """
    task = _make_fixed_task()
    calls: dict[str, Any] = {}

    def _fake_resolve(self: Any, **kw: Any) -> _StubConcat:
        calls["kwargs"] = kw
        return stub_concat

    # Patch the bound method on the instance for surgical scope.
    monkeypatch.setattr(type(task), "_resolve_dataset", _fake_resolve)

    windows, _ = task.make_windows(
        kind="fixed",
        window_size="2s",
        stride="2s",
        cache_dir="/tmp/eegdash-fake",
        n_subjects=1,
    )

    assert windows is stub_concat
    assert calls["kwargs"]["cache_dir"] == "/tmp/eegdash-fake"
    assert calls["kwargs"]["n_subjects"] == 1


def test_resolve_dataset_uses_subjects_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_resolve_dataset`` should narrow the metadata query when
    ``self.subjects`` is present and ``n_subjects`` is supplied.
    """
    captured: dict[str, Any] = {}

    def _fake_init(self: Any, *args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    from eegdash.dataset.dataset import EEGDashDataset

    monkeypatch.setattr(EEGDashDataset, "__init__", _fake_init)

    from eegdash.tasks import get_task

    task = get_task("eyes-open-closed")
    # The default EOEC task has a single default subject.
    task._resolve_dataset(cache_dir="/tmp/x", n_subjects=1, download=False)

    assert captured["kwargs"]["query"]["subject"] == task.subjects[0]
    assert captured["kwargs"]["download"] is False
    assert captured["kwargs"]["cache_dir"] == "/tmp/x"
