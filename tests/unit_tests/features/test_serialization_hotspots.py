import json
from functools import partial
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from eegdash.features import serialization as ser


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"_": 1, "1": {"-2": {"a": 3}}}, {"": 1, 1: {-2: {"a": 3}}}),
        ({"+3": 2, "x": {"_": 7}}, {3: 2, "x": {"": 7}}),
    ],
)
def test_adjust_dict_types_converts_numeric_and_underscore_keys(raw, expected):
    assert ser._adjust_dict_types(raw) == expected


def test_func_from_dict_errors_for_unknown_name():
    with pytest.raises(ValueError, match="not found in feature bank"):
        ser._func_from_dict({"name": "not_a_real_feature"})


def test_func_from_dict_builds_partial_for_args_and_kwargs():
    fn = ser._func_from_dict({"name": "signal_quantile", "kwargs": {"q": 0.5}})
    assert isinstance(fn, partial)
    out = fn(np.array([[1.0, 2.0, 3.0]]))
    assert out.shape == (1,)


def test_feature_extractor_from_dict_raises_on_malformed_feature_dict():
    bad = {"feature_extractors": {"f": {"unexpected": 1}}}
    with pytest.raises(ValueError, match="must contain either a 'feature_extractors'"):
        ser.feature_extractor_from_dict(bad)


def test_load_feature_extractor_from_json_calls_factory(monkeypatch, tmp_path: Path):
    payload = {"feature_extractors": {"m": {"name": "signal_mean"}}}
    cfg = tmp_path / "f.json"
    cfg.write_text(json.dumps(payload), encoding="utf-8")

    sentinel = object()
    monkeypatch.setattr(ser, "feature_extractor_from_dict", lambda d: sentinel)
    assert ser.load_feature_extractor_from_json(cfg) is sentinel


def test_load_feature_extractor_from_yaml_calls_factory(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "f.yaml"
    cfg.write_text(
        "feature_extractors:\n  m:\n    name: signal_mean\n", encoding="utf-8"
    )

    sentinel = object()
    monkeypatch.setattr(ser, "feature_extractor_from_dict", lambda d: sentinel)
    assert ser.load_feature_extractor_from_yaml(cfg) is sentinel


def test_load_feature_extractor_from_hocon_calls_factory(monkeypatch, tmp_path: Path):
    payload = {"feature_extractors": {"m": {"name": "signal_mean"}}}
    cfg = tmp_path / "f.conf"
    cfg.write_text("feature_extractors { m { name = signal_mean } }", encoding="utf-8")

    fake_mod = ModuleType("pyhocon")

    class _FakeFactory:
        @staticmethod
        def parse_file(_path):
            return payload

    fake_mod.ConfigFactory = _FakeFactory
    monkeypatch.setitem(__import__("sys").modules, "pyhocon", fake_mod)
    sentinel = object()
    monkeypatch.setattr(ser, "feature_extractor_from_dict", lambda d: sentinel)
    assert ser.load_feature_extractor_from_hocon(cfg) is sentinel
