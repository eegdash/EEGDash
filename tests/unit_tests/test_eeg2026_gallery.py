import importlib
import runpy
import socket
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("script", "metric_name", "held_out_axis", "upper_bound", "group_names"),
    [
        (
            "tutorial_track_1_eeg_to_image.py",
            "top-5 accuracy",
            "stimulus",
            1.0,
            ("stimulus",),
        ),
        (
            "tutorial_track_2_bci.py",
            "balanced accuracy",
            "session",
            1.0,
            ("sessions",),
        ),
        (
            "tutorial_track_3_sleep_onset.py",
            "balanced-bin MAE (s)",
            "subject + device",
            np.inf,
            ("subjects", "devices"),
        ),
        (
            "tutorial_track_4_emg_to_text.py",
            "character error rate",
            "user",
            np.inf,
            ("users",),
        ),
    ],
)
def test_eeg2026_tutorial_runs_offline_without_group_leakage(
    script, metric_name, held_out_axis, upper_bound, group_names, monkeypatch
):
    def deny_network(*_args, **_kwargs):
        raise AssertionError("tutorial attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    namespace = runpy.run_path(ROOT / "examples" / "eeg2026" / script)
    repeated = runpy.run_path(ROOT / "examples" / "eeg2026" / script)

    assert namespace["metric_name"] == metric_name
    assert namespace["held_out_axis"] == held_out_axis
    score = float(namespace["score"])
    assert np.isfinite(score)
    assert 0.0 <= score <= upper_bound
    assert namespace["score"] == repeated["score"]
    assert not namespace["split_overlap"]
    for group_name in group_names:
        group = namespace[group_name]
        assert not set(group[namespace["train"]]) & set(group[namespace["test"]])
    if script == "tutorial_track_2_bci.py":
        assert namespace["subject_overlap"] == set(range(8))
    if script == "tutorial_track_4_emg_to_text.py":
        assert namespace["edit_distance"]("kitten", "sitting") == 3
    plt.close("all")
