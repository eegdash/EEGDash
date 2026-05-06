# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Unit tests for :mod:`eegdash.tasks`.

These tests exercise the registry, the abstract interface, the manifest YAML
and the :meth:`EEGTask.make_windows` adapter without ever touching the
network: ``EEGDashDataset`` is replaced with a lightweight stub and the
braindecode windowers are monkey-patched to return a sentinel object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------- #
# Registry / get_task                                                         #
# --------------------------------------------------------------------------- #


def test_get_task_returns_eyes_open_closed_instance() -> None:
    """``get_task('eyes-open-closed')`` should return ``EyesOpenClosed``."""
    from eegdash.tasks import EyesOpenClosed, get_task

    task = get_task("eyes-open-closed")
    assert isinstance(task, EyesOpenClosed)
    assert task.name == "eyes-open-closed"


def test_list_tasks_contains_all_planned_names() -> None:
    """All five planned task names must be registered (concrete or stubs)."""
    from eegdash.tasks import list_tasks

    expected = {
        "eyes-open-closed",
        "visual-p300",
        "auditory-oddball",
        "age-regression",
        "eeg2025-pfactor",
    }
    assert expected.issubset(set(list_tasks()))


@pytest.mark.parametrize(
    ("name", "tutorial_marker"),
    [
        ("visual-p300", "plot_50_visual_p300"),
        ("auditory-oddball", "plot_55_auditory_oddball"),
        ("age-regression", "plot_60_age_regression"),
        ("eeg2025-pfactor", "plot_70_eeg2025_pfactor"),
    ],
)
def test_stub_tasks_raise_not_implemented(name: str, tutorial_marker: str) -> None:
    """Deferred tasks must raise ``NotImplementedError`` naming the tutorial."""
    from eegdash.tasks import get_task

    with pytest.raises(NotImplementedError) as exc_info:
        get_task(name)
    message = str(exc_info.value)
    assert name in message
    assert tutorial_marker in message


def test_get_task_unknown_name_raises_keyerror() -> None:
    from eegdash.tasks import get_task

    with pytest.raises(KeyError):
        get_task("not-a-real-task")


# --------------------------------------------------------------------------- #
# Label / metadata schemas                                                    #
# --------------------------------------------------------------------------- #


def test_label_definition_schema() -> None:
    """``label_definition`` must expose the documented schema for EO/EC."""
    from eegdash.tasks import get_task

    task = get_task("eyes-open-closed")
    label = task.label_definition()

    assert label["type"] == "classification"
    assert label["num_classes"] == 2
    assert label["mapping"] == {"eyes_open": 0, "eyes_closed": 1}
    assert label["source"] == "events"
    # Class names are useful for plotting / reporting and should be present.
    assert label["class_names"] == ["eyes_open", "eyes_closed"]


def test_metadata_query_uses_hbn_release_9() -> None:
    from eegdash.tasks import get_task

    task = get_task("eyes-open-closed")
    query = task.metadata_query()

    assert query["dataset"] == "ds005514"
    assert query["task"] == "RestingState"
    assert "subject" in query


def test_split_definitions_metrics_baseline_present() -> None:
    """Splits / metrics / baseline structures must have the documented keys."""
    from eegdash.tasks import get_task

    task = get_task("eyes-open-closed")

    splits = task.split_definitions()
    assert isinstance(splits, list)
    assert any(s.get("strategy") == "stratified_train_test" for s in splits)

    metrics = task.metrics()
    assert metrics["primary"] == "balanced_accuracy"
    assert "accuracy" in metrics["secondary"]

    baseline = task.baseline_metadata()
    assert baseline["model"] == "braindecode.ShallowFBCSPNet"
    assert baseline["hyperparameters"]["n_outputs"] == 2


# --------------------------------------------------------------------------- #
# Manifest YAML                                                                #
# --------------------------------------------------------------------------- #


def _manifest_path() -> Path:
    import eegdash.tasks as tasks_pkg

    return Path(tasks_pkg.__file__).parent / "manifests" / "eoec_hbn.yaml"


def test_manifest_yaml_parses_with_required_keys() -> None:
    yaml = pytest.importorskip("yaml")

    text = _manifest_path().read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)

    required_keys = {
        "name",
        "datasets",
        "filters",
        "labels",
        "preprocessing",
        "windowing",
        "splits",
        "metrics",
        "baseline",
        "citations",
        "licensing",
    }
    missing = required_keys - set(parsed)
    assert not missing, f"manifest missing required keys: {sorted(missing)}"
    assert parsed["name"] == "eyes-open-closed"
    assert parsed["labels"]["mapping"] == {"eyes_open": 0, "eyes_closed": 1}


def test_eyes_open_closed_manifest_path_points_at_yaml() -> None:
    from eegdash.tasks import get_task

    task = get_task("eyes-open-closed")
    assert task.manifest_path is not None
    assert Path(task.manifest_path).exists()
    assert Path(task.manifest_path).suffix == ".yaml"


# --------------------------------------------------------------------------- #
# make_windows: braindecode adapter                                            #
# --------------------------------------------------------------------------- #


def _stub_concat_dataset() -> object:
    """Return a sentinel object that stands in for a ``BaseConcatDataset``."""

    class _StubConcat:
        # the windowers only need *something* to forward; they are mocked.
        pass

    return _StubConcat()


def test_make_windows_uses_create_windows_from_events_for_eoec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``make_windows`` should dispatch to ``create_windows_from_events``.

    We monkey-patch the braindecode functions on the public
    ``braindecode.preprocessing`` namespace because :meth:`make_windows`
    re-imports them lazily through ``from braindecode.preprocessing import``.
    """
    import braindecode
    import braindecode.preprocessing as _bp

    captured: dict[str, Any] = {}

    def _fake_events(concat_ds: Any, **kwargs: Any) -> str:
        captured["fn"] = "events"
        captured["kwargs"] = kwargs
        captured["concat_ds"] = concat_ds
        return "fake-windows"

    def _fake_fixed(concat_ds: Any, **kwargs: Any) -> str:
        captured["fn"] = "fixed"
        captured["kwargs"] = kwargs
        captured["concat_ds"] = concat_ds
        return "fake-windows"

    monkeypatch.setattr(_bp, "create_windows_from_events", _fake_events)
    monkeypatch.setattr(_bp, "create_fixed_length_windows", _fake_fixed)

    from eegdash.tasks import get_task

    task = get_task("eyes-open-closed")
    concat = _stub_concat_dataset()

    windows, report = task.make_windows(concat, return_report=True)

    assert windows == "fake-windows"
    assert captured["fn"] == "events"
    assert captured["concat_ds"] is concat
    assert captured["kwargs"]["trial_stop_offset_samples"] == 256
    assert captured["kwargs"]["preload"] is True
    assert captured["kwargs"]["mapping"] == {"eyes_open": 0, "eyes_closed": 1}

    assert report["engine"] == "braindecode"
    assert report["kind"] == "events"
    assert report["function"] == (
        "braindecode.preprocessing.create_windows_from_events"
    )
    assert report["kwargs"] == captured["kwargs"]
    assert report["package_versions"]["braindecode"] == braindecode.__version__


def test_make_windows_kind_override_routes_to_fixed_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing ``kind='fixed'`` should call ``create_fixed_length_windows``.

    The fixed-length windower does not accept the ``trial_*_offset_samples``
    kwargs that the EO/EC recipe stores, so this test additionally exercises
    the per-call kwargs override path: callers pass exactly the kwargs the
    target windower understands.
    """
    import braindecode.preprocessing as _bp

    captured: dict[str, Any] = {}

    def _fake_events(concat_ds: Any, **kwargs: Any) -> str:
        captured["fn"] = "events"
        return "events-windows"

    def _fake_fixed(concat_ds: Any, **kwargs: Any) -> str:
        captured["fn"] = "fixed"
        captured["kwargs"] = kwargs
        return "fixed-windows"

    monkeypatch.setattr(_bp, "create_windows_from_events", _fake_events)
    monkeypatch.setattr(_bp, "create_fixed_length_windows", _fake_fixed)

    from eegdash.tasks.base import EEGTask

    class _FixedTask(EEGTask):
        name = "fixed-test"

        def metadata_query(self) -> dict[str, Any]:
            return {}

        def label_definition(self) -> dict[str, Any]:
            return {"type": "classification", "num_classes": 2}

        def preprocessing_recipe(self) -> list[Any]:
            return []

        def windowing_recipe(self) -> dict[str, Any]:
            return {
                "kind": "fixed",
                "window_size_samples": 256,
                "drop_last_window": True,
            }

        def split_definitions(self) -> list[dict[str, Any]]:
            return []

        def metrics(self) -> dict[str, Any]:
            return {"primary": "accuracy"}

        def baseline_metadata(self) -> dict[str, Any]:
            return {}

    task = _FixedTask()
    windows = task.make_windows(_stub_concat_dataset(), return_report=False)
    assert windows == "fixed-windows"
    assert captured["fn"] == "fixed"
    assert captured["kwargs"]["window_size_samples"] == 256
    assert captured["kwargs"]["drop_last_window"] is True


def test_make_windows_rejects_unknown_engine() -> None:
    from eegdash.tasks import get_task

    task = get_task("eyes-open-closed")
    with pytest.raises(ValueError):
        task.make_windows(_stub_concat_dataset(), engine="numpy")


# --------------------------------------------------------------------------- #
# EEGDashDataset is referenced via metadata_query but not invoked directly.    #
# Confirm the query is shaped so it can be passed straight into the dataset    #
# constructor without any network access.                                      #
# --------------------------------------------------------------------------- #


def test_metadata_query_compatible_with_eegdashdataset_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EEGDashDataset(query=task.metadata_query(), ...)`` must accept the dict.

    We patch ``EEGDashDataset.__init__`` to a no-op so we can construct it
    without contacting the registry / network and only verify the query
    payload propagates intact.
    """
    from eegdash.dataset.dataset import EEGDashDataset
    from eegdash.tasks import get_task

    captured: dict[str, Any] = {}

    def _fake_init(self: Any, *args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(EEGDashDataset, "__init__", _fake_init)

    task = get_task("eyes-open-closed")
    query = task.metadata_query()
    EEGDashDataset(cache_dir="/tmp/does-not-exist", query=query, download=False)

    assert captured["kwargs"]["query"] == query
    assert captured["kwargs"]["download"] is False
