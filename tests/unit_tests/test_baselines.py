# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Unit tests for the Workstream 5 baseline recipe classes.

The recipes are minimal wrappers around sklearn / Braindecode models. These
tests exercise the uniform ``.fit().score()`` interface on small synthetic
data, and verify that optional dependencies (``lightgbm``, ``torch``,
``braindecode``) report friendly ``ImportError`` messages when missing instead
of failing at module load.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from eegdash.splits import (
    EEGNetv4Baseline,
    LightGBMBaseline,
    LogisticRegressionBaseline,
    RidgeRegressionBaseline,
    ShallowFBCSPNetBaseline,
)

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


@pytest.fixture()
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(80, 6))
    coef = rng.normal(size=(6,))
    y_train = X_train @ coef + 0.1 * rng.normal(size=(80,))
    X_test = rng.normal(size=(40, 6))
    y_test = X_test @ coef + 0.1 * rng.normal(size=(40,))
    return X_train, y_train, X_test, y_test


@pytest.fixture()
def classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(80, 6))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = rng.normal(size=(40, 6))
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    return X_train, y_train, X_test, y_test


@pytest.fixture()
def eeg_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tiny (n_samples, n_channels, n_times) tensor with 2 classes."""
    rng = np.random.default_rng(2)
    n_chans, n_times = 8, 256
    X_train = rng.normal(size=(20, n_chans, n_times)).astype(np.float32)
    y_train = (X_train.mean(axis=(1, 2)) > 0).astype(int)
    X_test = rng.normal(size=(12, n_chans, n_times)).astype(np.float32)
    y_test = (X_test.mean(axis=(1, 2)) > 0).astype(int)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_all_baselines_can_be_instantiated() -> None:
    """Each class constructs without touching optional deps."""
    assert RidgeRegressionBaseline(random_state=0).random_state == 0
    assert LogisticRegressionBaseline(random_state=0).random_state == 0
    assert LightGBMBaseline(random_state=0).random_state == 0
    assert ShallowFBCSPNetBaseline(random_state=0).random_state == 0
    assert EEGNetv4Baseline(random_state=0).random_state == 0


def test_requires_attribute_documents_optional_deps() -> None:
    assert RidgeRegressionBaseline.requires == ("scikit-learn",)
    assert LogisticRegressionBaseline.requires == ("scikit-learn",)
    assert LightGBMBaseline.requires == ("lightgbm",)
    assert ShallowFBCSPNetBaseline.requires == ("braindecode", "torch")
    assert EEGNetv4Baseline.requires == ("braindecode", "torch")


# ---------------------------------------------------------------------------
# Sklearn-backed recipes
# ---------------------------------------------------------------------------


def test_ridge_regression_baseline_fit_score(regression_data) -> None:
    X_train, y_train, X_test, y_test = regression_data
    model = RidgeRegressionBaseline(random_state=42, alpha=1.0).fit(X_train, y_train)
    out = model.score(X_test, y_test)
    assert set(out) >= {"score", "r2", "mse", "chance_level", "metric"}
    assert out["metric"] == "r2"
    assert isinstance(out["score"], float)
    # Ridge should beat the median-of-train chance level on a linear problem.
    assert out["score"] > out["chance_level"]
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape


def test_logistic_regression_baseline_fit_score(classification_data) -> None:
    X_train, y_train, X_test, y_test = classification_data
    model = LogisticRegressionBaseline(random_state=42).fit(X_train, y_train)
    out = model.score(X_test, y_test)
    assert set(out) >= {"score", "accuracy", "chance_level", "metric"}
    assert out["metric"] == "accuracy"
    assert 0.0 <= out["score"] <= 1.0
    assert 0.0 <= out["chance_level"] <= 1.0
    # On a linearly separable signal the model should beat the majority chance.
    assert out["score"] >= out["chance_level"]
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape


def test_logistic_regression_baseline_score_before_fit_raises() -> None:
    model = LogisticRegressionBaseline(random_state=0)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.score(np.zeros((4, 3)), np.zeros(4))


# ---------------------------------------------------------------------------
# LightGBM (optional)
# ---------------------------------------------------------------------------


def test_lightgbm_baseline_runs_or_skips(classification_data) -> None:
    """If lightgbm is installed, the recipe trains; otherwise it raises ImportError."""
    X_train, y_train, X_test, y_test = classification_data
    model = LightGBMBaseline(random_state=42, n_estimators=20)
    has_lgb = importlib.util.find_spec("lightgbm") is not None
    if not has_lgb:
        with pytest.raises(ImportError, match="lightgbm"):
            model.fit(X_train, y_train)
        return
    model.fit(X_train, y_train)
    out = model.score(X_test, y_test)
    assert set(out) >= {"score", "accuracy", "chance_level", "metric"}
    assert out["metric"] == "accuracy"
    assert 0.0 <= out["score"] <= 1.0


# ---------------------------------------------------------------------------
# Braindecode (optional)
# ---------------------------------------------------------------------------


def _braindecode_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("braindecode") is not None
    )


def test_shallow_fbcspnet_baseline_runs_or_skips(eeg_data) -> None:
    X_train, y_train, X_test, y_test = eeg_data
    model = ShallowFBCSPNetBaseline(random_state=0, epochs=1, batch_size=4)
    if not _braindecode_available():
        with pytest.raises(ImportError):
            model.fit(X_train, y_train)
        return
    model.fit(X_train, y_train)
    out = model.score(X_test, y_test)
    assert set(out) >= {"score", "accuracy", "chance_level", "metric"}
    assert out["metric"] == "accuracy"
    assert 0.0 <= out["score"] <= 1.0
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape


def test_eegnet_v4_baseline_runs_or_skips(eeg_data) -> None:
    X_train, y_train, X_test, y_test = eeg_data
    model = EEGNetv4Baseline(random_state=0, epochs=1, batch_size=4)
    if not _braindecode_available():
        with pytest.raises(ImportError):
            model.fit(X_train, y_train)
        return
    model.fit(X_train, y_train)
    out = model.score(X_test, y_test)
    assert set(out) >= {"score", "accuracy", "chance_level", "metric"}
    assert out["metric"] == "accuracy"
    assert 0.0 <= out["score"] <= 1.0
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape


def test_braindecode_baseline_rejects_2d_input(eeg_data) -> None:
    """Braindecode recipes must reject improperly shaped feature matrices."""
    if not _braindecode_available():
        pytest.skip("braindecode/torch not installed")
    _, _, _, _ = eeg_data
    model = ShallowFBCSPNetBaseline(random_state=0, epochs=1, batch_size=4)
    with pytest.raises(ValueError, match="3D input"):
        model.fit(np.zeros((10, 32)), np.zeros(10, dtype=int))
