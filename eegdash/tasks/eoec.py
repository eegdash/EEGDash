# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Eyes-open / eyes-closed classification task.

This task wraps the legacy ``examples/core/tutorial_eoec.py`` notebook into
the high-level ``eegdash.tasks`` interface described in
``docs/tutorial_restructure_plan.md`` (Workstream 2). It uses the Healthy
Brain Network (HBN) RestingState recordings exposed via OpenNeuro
``ds005514`` (HBN Release 9), with the canonical preprocessing pipeline that
re-annotates the eyes-open / eyes-closed segments and resamples the data to
128 Hz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import EEGTask

# Default OpenNeuro dataset id and subject. ``ds005514`` is HBN Release 9, a
# small and well-curated slice that already contains both eyes-open and
# eyes-closed event annotations. ``NDARDB033FW5`` is the same subject used by
# the legacy tutorial and the existing unit tests, which keeps tutorial
# runtime small enough for documentation builds.
_DEFAULT_OPENNEURO_DATASET = "ds005514"
_DEFAULT_TASK = "RestingState"
_DEFAULT_SUBJECT = "NDARDB033FW5"

# 24-channel selection used by the legacy tutorial / paper baseline.
_DEFAULT_CHANNELS: tuple[str, ...] = (
    "E22",
    "E9",
    "E33",
    "E24",
    "E11",
    "E124",
    "E122",
    "E29",
    "E6",
    "E111",
    "E45",
    "E36",
    "E104",
    "E108",
    "E42",
    "E55",
    "E93",
    "E58",
    "E52",
    "E62",
    "E92",
    "E96",
    "E70",
    "Cz",
)

_DEFAULT_RESAMPLE_HZ = 128
_DEFAULT_BANDPASS = (1.0, 55.0)
_DEFAULT_WINDOW_SAMPLES = 256  # 2 s at 128 Hz

_LABEL_MAPPING: dict[str, int] = {"eyes_open": 0, "eyes_closed": 1}

_MANIFEST_PATH = Path(__file__).resolve().parent / "manifests" / "eoec_hbn.yaml"


class EyesOpenClosed(EEGTask):
    """Resting-state eyes-open vs eyes-closed two-class classification.

    Parameters
    ----------
    dataset : str, default ``"ds005514"``
        OpenNeuro dataset identifier. Defaults to HBN Release 9.
    subjects : str | list[str] | None
        Subject identifier(s) to filter by. Defaults to a single fast subject
        (``"NDARDB033FW5"``) so that tutorials run in seconds; pass a list to
        widen the selection.
    task : str, default ``"RestingState"``
        BIDS task label.
    mini : bool, default ``True``
        Whether to honour the eegdash mini-release semantics. The flag is
        currently informational; the registry layer uses it to wire up
        ``EEGChallengeDataset`` instances when relevant.
    channels : list[str] | None
        Override list of channels to keep. Defaults to the 24 channels used
        in the published baseline.
    resample_hz : int, default ``128``
        Target sampling frequency.
    bandpass : tuple[float, float], default ``(1.0, 55.0)``
        Band-pass cutoff frequencies (low, high) in Hz.
    window_size_samples : int, default ``256``
        Window length in samples (defaults to 2 s at 128 Hz).

    """

    name: str = "eyes-open-closed"
    manifest_path: str | None = str(_MANIFEST_PATH)

    def __init__(
        self,
        dataset: str = _DEFAULT_OPENNEURO_DATASET,
        subjects: str | list[str] | None = None,
        task: str = _DEFAULT_TASK,
        mini: bool = True,
        channels: list[str] | None = None,
        resample_hz: int = _DEFAULT_RESAMPLE_HZ,
        bandpass: tuple[float, float] = _DEFAULT_BANDPASS,
        window_size_samples: int = _DEFAULT_WINDOW_SAMPLES,
    ) -> None:
        self.dataset = dataset
        if subjects is None:
            subjects = [_DEFAULT_SUBJECT]
        elif isinstance(subjects, str):
            subjects = [subjects]
        self.subjects: list[str] = list(subjects)
        self.task = task
        self.mini = mini
        self.channels: list[str] = (
            list(channels) if channels is not None else list(_DEFAULT_CHANNELS)
        )
        self.resample_hz = resample_hz
        self.bandpass = (float(bandpass[0]), float(bandpass[1]))
        self.window_size_samples = window_size_samples

    # ------------------------------------------------------------------ #
    # EEGTask interface                                                  #
    # ------------------------------------------------------------------ #

    def metadata_query(self) -> dict[str, Any]:
        """Return the EEGDash query that selects HBN EO/EC recordings."""
        query: dict[str, Any] = {
            "dataset": self.dataset,
            "task": self.task,
        }
        if len(self.subjects) == 1:
            query["subject"] = self.subjects[0]
        else:
            query["subject"] = {"$in": list(self.subjects)}
        return query

    def label_definition(self) -> dict[str, Any]:
        """Two-class classification labels derived from event annotations."""
        return {
            "type": "classification",
            "num_classes": 2,
            "mapping": dict(_LABEL_MAPPING),
            "source": "events",
            "reannotation": "hbn_ec_ec_reannotation",
            "class_names": ["eyes_open", "eyes_closed"],
        }

    def preprocessing_recipe(self) -> list[Any]:
        """Build the canonical HBN EO/EC preprocessing pipeline.

        The recipe mirrors ``examples/core/tutorial_eoec.py``:

        1. ``hbn_ec_ec_reannotation`` -- replace instruction markers with
           regularly spaced ``eyes_open`` / ``eyes_closed`` annotations.
        2. ``pick_channels`` -- restrict to the 24-channel baseline montage.
        3. ``resample`` to ``self.resample_hz``.
        4. ``filter`` between ``self.bandpass[0]`` and ``self.bandpass[1]``.
        """
        # Imports are local so that ``import eegdash.tasks`` stays lightweight
        # and so that the tests can import :class:`EyesOpenClosed` without
        # pulling in MNE / braindecode if those are not needed.
        from braindecode.preprocessing import Preprocessor

        from ..hbn.preprocessing import hbn_ec_ec_reannotation

        return [
            hbn_ec_ec_reannotation(),
            Preprocessor("pick_channels", ch_names=list(self.channels)),
            Preprocessor("resample", sfreq=self.resample_hz),
            Preprocessor("filter", l_freq=self.bandpass[0], h_freq=self.bandpass[1]),
        ]

    def windowing_recipe(self) -> dict[str, Any]:
        """Return arguments for :func:`create_windows_from_events`."""
        return {
            "kind": "events",
            "trial_start_offset_samples": 0,
            "trial_stop_offset_samples": int(self.window_size_samples),
            "preload": True,
            "mapping": dict(_LABEL_MAPPING),
        }

    def split_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "stratified_holdout",
                "strategy": "stratified_train_test",
                "test_size": 0.2,
                "random_state": 42,
                "stratify_on": "target",
            },
            {
                "name": "cross_subject",
                "strategy": "leave_subjects_out",
                "notes": ("Use this split when more than one subject is requested."),
            },
        ]

    def metrics(self) -> dict[str, Any]:
        return {
            "primary": "balanced_accuracy",
            "secondary": ["accuracy", "roc_auc_ovr"],
            "reporting": ["confusion_matrix"],
        }

    def baseline_metadata(self) -> dict[str, Any]:
        return {
            "model": "braindecode.ShallowFBCSPNet",
            "hyperparameters": {
                "n_chans": len(self.channels),
                "n_outputs": 2,
                "n_times": int(self.window_size_samples),
                "final_conv_length": "auto",
            },
            "reference_accuracy": 0.85,
            "reference_subjects": [_DEFAULT_SUBJECT],
            "source_tutorial": "examples/core/tutorial_eoec.py",
        }
