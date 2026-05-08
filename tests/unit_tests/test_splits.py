# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Unit tests for ``eegdash.splits``.

The fixtures here build a synthetic 20-subject x 2-session x 5-window metadata
frame so the tests run without any real EEG data. The MOABB-vs-sklearn engine
selection is exercised in both directions where possible.
"""

from __future__ import annotations

import io
import json
import sys

import numpy as np
import pandas as pd
import pytest

from eegdash.splits import (
    LeakageError,
    assert_no_leakage,
    describe_split,
    get_splitter,
    k_fold,
    majority_baseline,
    median_baseline,
    train_test_split,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_metadata(
    n_subjects: int = 20,
    n_sessions: int = 2,
    n_windows: int = 5,
    n_classes: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a 20 x 2 x 5 = 200 row metadata frame with synthetic labels."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for subject in range(n_subjects):
        for session in range(n_sessions):
            for window in range(n_windows):
                rows.append(
                    {
                        "subject": f"sub-{subject:02d}",
                        "session": f"ses-{session:02d}",
                        "run": "run-01",
                        "dataset": "synthetic",
                        "sample_id": f"sub-{subject:02d}__ses-{session:02d}__r{window:03d}",
                        "target": int(rng.integers(0, n_classes)),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def metadata() -> pd.DataFrame:
    return _build_metadata()


# ---------------------------------------------------------------------------
# get_splitter
# ---------------------------------------------------------------------------


def test_get_splitter_cross_subject_returns_usable_splitter(metadata):
    splitter = get_splitter("cross_subject", random_state=42)
    folds = list(splitter.split(metadata["target"].to_numpy(), metadata))
    assert len(folds) >= 2
    for train_idx, test_idx in folds:
        train_subjects = set(metadata.iloc[list(train_idx)]["subject"])
        test_subjects = set(metadata.iloc[list(test_idx)]["subject"])
        assert not (train_subjects & test_subjects), "Subjects leaked!"


def test_get_splitter_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown splitter"):
        get_splitter("nonsense")


def test_get_splitter_cross_subject_with_n_splits(metadata):
    """cross_subject splitter with n_splits=4 produces subject-disjoint folds."""
    splitter = get_splitter("cross_subject", n_splits=4)
    folds = list(splitter.split(metadata["target"].to_numpy(), metadata))
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        train_subjects = set(metadata.iloc[list(train_idx)]["subject"])
        test_subjects = set(metadata.iloc[list(test_idx)]["subject"])
        assert not (train_subjects & test_subjects)


# ---------------------------------------------------------------------------
# k_fold + train_test_split
# ---------------------------------------------------------------------------


def test_k_fold_yields_disjoint_masks(metadata):
    splitter = get_splitter("cross_subject", n_splits=4)
    folds = list(k_fold(metadata, splitter=splitter, target="target"))
    assert len(folds) == 4
    for train_mask, test_mask in folds:
        assert isinstance(train_mask, np.ndarray)
        assert train_mask.dtype == bool
        assert test_mask.dtype == bool
        assert train_mask.sum() + test_mask.sum() == len(metadata.index)
        assert not np.any(train_mask & test_mask)


def test_k_fold_is_deterministic(metadata):
    splitter_a = get_splitter("cross_subject", n_splits=4)
    folds_a = list(k_fold(metadata, splitter=splitter_a, target="target"))
    splitter_b = get_splitter("cross_subject", n_splits=4)
    folds_b = list(k_fold(metadata, splitter=splitter_b, target="target"))
    for (tr_a, te_a), (tr_b, te_b) in zip(folds_a, folds_b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(te_a, te_b)


def test_train_test_split_returns_first_fold(metadata):
    train_mask, test_mask = train_test_split(
        metadata, group="subject", target="target", seed=42
    )
    assert isinstance(train_mask, np.ndarray)
    assert train_mask.dtype == bool
    assert int(train_mask.sum()) + int(test_mask.sum()) == len(metadata.index)


def test_k_fold_default_group_aware_splitter(metadata):
    """Without splitter=, k_fold builds a group-aware MOABB splitter."""
    folds = list(k_fold(metadata, n_folds=5, group="subject", target="target"))
    assert len(folds) >= 1
    for train_mask, test_mask in folds:
        train_subjects = set(metadata.loc[train_mask, "subject"])
        test_subjects = set(metadata.loc[test_mask, "subject"])
        assert not (train_subjects & test_subjects), "subjects leaked across fold"


# ---------------------------------------------------------------------------
# assert_no_leakage -- behavior + JSON line emission
# ---------------------------------------------------------------------------


def _capture_stdout(callable_, *args, **kwargs) -> tuple[object, str]:
    """Run ``callable_`` and capture everything it prints to stdout."""
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        result = callable_(*args, **kwargs)
    finally:
        sys.stdout = saved
    return result, buf.getvalue()


def _mask_pair(
    metadata: pd.DataFrame, train_idx, test_idx
) -> tuple[np.ndarray, np.ndarray]:
    n = len(metadata.index)
    tr = np.zeros(n, dtype=bool)
    tr[list(train_idx)] = True
    te = np.zeros(n, dtype=bool)
    te[list(test_idx)] = True
    return tr, te


def test_assert_no_leakage_passes_for_clean_split(metadata):
    splitter = get_splitter("cross_subject", n_splits=4)
    folds = list(k_fold(metadata, splitter=splitter, target="target"))
    overlap, captured = _capture_stdout(
        assert_no_leakage, folds, metadata, by="subject"
    )
    assert overlap == 0
    # The JSON line must appear exactly as the validator E5.42 expects.
    lines = [line for line in captured.splitlines() if "leakage_report" in line]
    assert lines, f"No leakage_report line in stdout; captured: {captured!r}"
    parsed = json.loads(lines[0])
    assert parsed == {"leakage_report": {"overlap": 0, "by": "subject"}}


def test_assert_no_leakage_detects_subject_overlap(metadata):
    """A hand-crafted 'leaky' fold pair must raise LeakageError."""
    # Put the same subject in train and test of the only fold.
    leaky = [_mask_pair(metadata, range(50), range(40, 60))]
    with pytest.raises(LeakageError):
        # Capture stdout to keep tests quiet but still verify line emission.
        _capture_stdout(assert_no_leakage, leaky, metadata, by="subject")


def test_assert_no_leakage_emits_line_even_on_overlap(metadata):
    """The JSON line must be printed *before* raising, for E5.42."""
    leaky = [_mask_pair(metadata, range(50), range(40, 60))]
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        try:
            assert_no_leakage(leaky, metadata, by="subject")
        except LeakageError:
            pass
    finally:
        sys.stdout = saved
    captured = buf.getvalue()
    parsed = json.loads(
        [line for line in captured.splitlines() if "leakage_report" in line][0]
    )
    assert parsed["leakage_report"]["by"] == "subject"
    assert parsed["leakage_report"]["overlap"] >= 1


# ---------------------------------------------------------------------------
# describe_split
# ---------------------------------------------------------------------------


def test_describe_split_basic(metadata):
    splitter = get_splitter("cross_subject", n_splits=4)
    folds = list(k_fold(metadata, splitter=splitter, target="target"))
    summary = describe_split(folds, metadata, target="target", print_report=False)
    assert summary["n_folds"] == 4
    assert summary["coverage"]["n_subjects"] == 20
    assert len(summary["per_fold"]) == 4
    for fold_stats in summary["per_fold"]:
        assert fold_stats["n_train"] > 0
        assert fold_stats["n_test"] > 0


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def test_majority_baseline_classification():
    y_train = np.array([0, 0, 0, 1, 1])
    y_test = np.array([0, 0, 0, 0, 1])
    out = majority_baseline(y_train, y_test)
    assert out["metric"] == "accuracy"
    assert out["chance_level"] == pytest.approx(4 / 5)  # 4 zeros in test
    # Predicting 0 (train mode) on test gives 4/5 correct.
    assert out["baseline_score"] == pytest.approx(4 / 5)


def test_median_baseline_regression():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_test = np.array([2.0, 3.0, 4.0])
    out = median_baseline(y_train, y_test)
    assert out["metric"] == "r2"
    assert out["chance_level"] == 0.0
    # Prediction = median(train) = 3 -> ss_res = 1+0+1 = 2; ss_tot = 1+0+1 = 2.
    # R^2 = 1 - 2/2 = 0
    assert out["baseline_score"] == pytest.approx(0.0)


def test_baselines_handle_empty_test_gracefully():
    y_train = np.array([0, 1, 1])
    out = majority_baseline(y_train, np.array([]))
    assert np.isnan(out["chance_level"])
    out = median_baseline(np.array([1.0, 2.0]), np.array([]))
    assert np.isnan(out["chance_level"])
