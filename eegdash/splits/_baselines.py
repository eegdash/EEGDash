# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Trivial baselines used by the E5.43 chance-level reporting rule."""

from __future__ import annotations

from collections import Counter
from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence, np.ndarray]


def _to_numpy(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def majority_baseline(
    y_train: ArrayLike, y_test: ArrayLike
) -> dict[str, Union[float, str]]:
    """Predict the most frequent training label for every test sample.

    The chance level reported is ``max(p_class)`` of the *test* set: the score
    a constant predictor of the test mode would achieve. The baseline score is
    the accuracy of predicting the *training* mode on the test set, which can
    differ when class proportions shift.

    Returns
    -------
    dict
        ``{"chance_level": float, "baseline_score": float, "metric": "accuracy"}``.

    """
    y_train_arr = _to_numpy(y_train)
    y_test_arr = _to_numpy(y_test)
    if y_train_arr.size == 0:
        raise ValueError("y_train is empty; cannot fit a majority baseline.")
    if y_test_arr.size == 0:
        return {
            "chance_level": float("nan"),
            "baseline_score": float("nan"),
            "metric": "accuracy",
        }

    train_counts = Counter(y_train_arr.tolist())
    train_mode, _ = train_counts.most_common(1)[0]

    test_counts = Counter(y_test_arr.tolist())
    n_test = float(y_test_arr.size)
    chance_level = max(test_counts.values()) / n_test
    baseline_score = float(np.mean(y_test_arr == train_mode))

    return {
        "chance_level": float(chance_level),
        "baseline_score": float(baseline_score),
        "metric": "accuracy",
    }


def median_baseline(
    y_train: ArrayLike, y_test: ArrayLike
) -> dict[str, Union[float, str]]:
    """Predict the training median for every test sample.

    The chance level reported is the R^2 of predicting the *test* mean (which
    is 0 by definition). The baseline score is the R^2 of predicting the train
    median on the test set.
    """
    y_train_arr = _to_numpy(y_train).astype(float)
    y_test_arr = _to_numpy(y_test).astype(float)
    if y_train_arr.size == 0:
        raise ValueError("y_train is empty; cannot fit a median baseline.")
    if y_test_arr.size == 0:
        return {
            "chance_level": float("nan"),
            "baseline_score": float("nan"),
            "metric": "r2",
        }

    train_median = float(np.median(y_train_arr))
    test_mean = float(np.mean(y_test_arr))

    ss_res = float(np.sum((y_test_arr - train_median) ** 2))
    ss_tot = float(np.sum((y_test_arr - test_mean) ** 2))
    if ss_tot == 0.0:
        # Degenerate case: constant test target. The R^2 score is undefined;
        # report 0 (no improvement over the mean).
        baseline_score = 0.0
    else:
        baseline_score = 1.0 - ss_res / ss_tot

    return {
        "chance_level": 0.0,  # R^2 of the test-set mean predictor.
        "baseline_score": float(baseline_score),
        "metric": "r2",
    }


__all__ = ["majority_baseline", "median_baseline"]
