import numpy as np
import pytest

from eegdash.features.feature_bank.dimensionality import (
    dimensionality_detrended_fluctuation_analysis,
    dimensionality_higuchi_fractal_dim,
    dimensionality_hurst_exp,
    dimensionality_katz_fractal_dim,
    dimensionality_petrosian_fractal_dim,
)


@pytest.fixture
def signals():
    np.random.seed(42)
    n = 2000  # Enough samples for stable estimation
    t = np.linspace(0, 10, n)

    # Sine wave (smooth, low complexity)
    sine = np.sin(2 * np.pi * 5 * t)

    # White noise (high complexity, H~0.5 for R/S, alpha~0.5 for DFA)
    white = np.random.randn(n)

    # Brownian noise (random walk, alpha~1.5 for DFA)
    brown = np.cumsum(white)

    # Pre-shape (1, n)
    return {"sine": sine[None, :], "white": white[None, :], "brown": brown[None, :]}


def test_fractal_dimensions_relative(signals):
    # Higuchi, Petrosian, Katz should generally be higher for noise than sine wave
    sine = signals["sine"]
    white = signals["white"]

    # Higuchi
    higuchi_sine = dimensionality_higuchi_fractal_dim(sine)
    higuchi_white = dimensionality_higuchi_fractal_dim(white)
    # Check execution
    assert higuchi_sine.shape == (1,)
    assert higuchi_white.shape == (1,)

    # Petrosian
    petrosian_sine = dimensionality_petrosian_fractal_dim(sine)
    dimensionality_petrosian_fractal_dim(white)
    assert petrosian_sine.shape == (1,)

    # Katz
    katz_sine = dimensionality_katz_fractal_dim(sine)
    dimensionality_katz_fractal_dim(white)
    assert katz_sine.shape == (1,)


def test_hurst_exp(signals):
    white = signals["white"]
    h_white = dimensionality_hurst_exp(white)
    assert h_white.shape == (1,)


def test_dfa(signals):
    white = signals["white"]
    brown = signals["brown"]

    alpha_white = dimensionality_detrended_fluctuation_analysis(white)
    alpha_brown = dimensionality_detrended_fluctuation_analysis(brown)

    assert alpha_white.shape == (1,)
    assert alpha_brown.shape == (1,)
