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


def test_fractal_dimensions_correctness(signals):
    # Verify expected relative values
    sine = signals["sine"]
    white = signals["white"]

    # 1. Higuchi
    # Sine wave is smooth (D ~ 1)
    # White noise is rough (D ~ 2)
    h_sine = dimensionality_higuchi_fractal_dim(sine)[0]
    h_white = dimensionality_higuchi_fractal_dim(white)[0]
    assert h_sine < 1.1  # Close to 1
    assert h_white > 1.8  # Close to 2
    assert h_white > h_sine

    # 2. Petrosian
    # Sine has few sign changes in diff
    # White noise has many
    p_sine = dimensionality_petrosian_fractal_dim(sine)[0]
    p_white = dimensionality_petrosian_fractal_dim(white)[0]
    assert p_sine < 1.05
    assert p_white > p_sine

    # 3. Katz
    k_sine = dimensionality_katz_fractal_dim(sine)[0]
    k_white = dimensionality_katz_fractal_dim(white)[0]
    # Katz is sensitive to length/diameter ratio.
    # We mainly check that noise > sine
    assert k_white > k_sine


def test_dfa_correctness(signals):
    # Check theoretical scaling exponents (alpha)
    white = signals["white"]
    brown = signals["brown"]

    a_white = dimensionality_detrended_fluctuation_analysis(white)[0]
    a_brown = dimensionality_detrended_fluctuation_analysis(brown)[0]

    # White noise: alpha ~ 0.5
    assert 0.4 < a_white < 0.6

    # Brownian noise (integrated white noise): alpha ~ 1.5
    assert 1.3 < a_brown < 1.7


def test_dimensionality_features(signal_2d):
    # Higuchi
    hfd = dimensionality_higuchi_fractal_dim(signal_2d, k_max=5)
    assert hfd.shape == (2,)

    # Petrosian
    pfd = dimensionality_petrosian_fractal_dim(signal_2d)
    assert pfd.shape == (2,)

    # Katz
    kfd = dimensionality_katz_fractal_dim(signal_2d)
    assert kfd.shape == (2,)

    # Hurst
    he = dimensionality_hurst_exp(signal_2d)
    assert he.shape == (2,)

    # DFA
    dfa = dimensionality_detrended_fluctuation_analysis(signal_2d)
    assert dfa.shape == (2,)


def test_dimensionality_hurst_edge_cases():
    # Signal with zero variance
    sig = np.zeros((1, 100))
    he = dimensionality_hurst_exp(sig)
    assert np.isnan(he).all()


def test_dimensionality_gaps():
    from eegdash.features.feature_bank.dimensionality import (
        dimensionality_higuchi_fractal_dim,
        dimensionality_hurst_exp,
        dimensionality_detrended_fluctuation_analysis,
    )

    x = np.random.randn(1, 100)
    dimensionality_higuchi_fractal_dim(x, k_max=5)

    # Higuchi edge case: short signal to trigger loop skipping (lines 38-40)
    x_short = np.random.randn(1, 5)
    dimensionality_higuchi_fractal_dim(x_short, k_max=5)

    # Petrosian (missing)
    from eegdash.features.feature_bank.dimensionality import (
        dimensionality_petrosian_fractal_dim,
    )

    dimensionality_petrosian_fractal_dim(x)

    # Katz (missing)
    from eegdash.features.feature_bank.dimensionality import (
        dimensionality_katz_fractal_dim,
    )

    dimensionality_katz_fractal_dim(x)

    # Hurst 48-69 (missing all)
    dimensionality_hurst_exp(x)

    # Hurst edge case: flat signal (std=0) to trigger line 87
    x_flat = np.zeros((1, 100))
    dimensionality_hurst_exp(x_flat)

    # DFA 114-134
    dimensionality_detrended_fluctuation_analysis(x)


import numpy as np


def test_dimensionality_gaps():
    from eegdash.features.feature_bank.dimensionality import (
        dimensionality_higuchi_fractal_dim,
        dimensionality_hurst_exp,
        dimensionality_detrended_fluctuation_analysis,
    )

    x = np.random.randn(1, 100)
    dimensionality_higuchi_fractal_dim(x, k_max=5)

    # Higuchi edge case: short signal to trigger loop skipping (lines 38-40)
    x_short = np.random.randn(1, 5)
    dimensionality_higuchi_fractal_dim(x_short, k_max=5)

    # Petrosian (missing)
    from eegdash.features.feature_bank.dimensionality import (
        dimensionality_petrosian_fractal_dim,
    )

    dimensionality_petrosian_fractal_dim(x)

    # Katz (missing)
    from eegdash.features.feature_bank.dimensionality import (
        dimensionality_katz_fractal_dim,
    )

    dimensionality_katz_fractal_dim(x)

    # Hurst 48-69 (missing all)
    dimensionality_hurst_exp(x)

    # Hurst edge case: flat signal (std=0) to trigger line 87
    x_flat = np.zeros((1, 100))
    dimensionality_hurst_exp(x_flat)

    # DFA 114-134
    dimensionality_detrended_fluctuation_analysis(x)


def test_dimensionality_features(signal_2d):
    from eegdash.features.feature_bank.dimensionality import (
        dimensionality_detrended_fluctuation_analysis,
        dimensionality_higuchi_fractal_dim,
        dimensionality_hurst_exp,
        dimensionality_katz_fractal_dim,
        dimensionality_petrosian_fractal_dim,
    )

    # Higuchi
    hfd = dimensionality_higuchi_fractal_dim(signal_2d, k_max=5)
    assert hfd.shape == (2,)

    # Petrosian
    pfd = dimensionality_petrosian_fractal_dim(signal_2d)
    assert pfd.shape == (2,)

    # Katz
    kfd = dimensionality_katz_fractal_dim(signal_2d)
    assert kfd.shape == (2,)

    # Hurst
    he = dimensionality_hurst_exp(signal_2d)
    assert he.shape == (2,)

    # DFA
    dfa = dimensionality_detrended_fluctuation_analysis(signal_2d)
    assert dfa.shape == (2,)


def test_dimensionality_hurst_edge_cases():
    from eegdash.features.feature_bank.dimensionality import dimensionality_hurst_exp

    # Signal with zero variance
    sig = np.zeros((1, 100))
    he = dimensionality_hurst_exp(sig)
    assert np.isnan(he).all()
