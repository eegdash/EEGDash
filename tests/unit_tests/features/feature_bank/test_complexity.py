import numpy as np
import pytest

from eegdash.features.feature_bank.complexity import (
    complexity_approx_entropy,
    complexity_detrended_fluctuation_analysis,
    complexity_entropy_preprocessor,
    complexity_hurst_exp,
    complexity_lempel_ziv,
    complexity_sample_entropy,
    complexity_svd_entropy,
)


@pytest.fixture
def sine_wave():
    t = np.linspace(0, 10, 100)
    return np.sin(2 * np.pi * t)


@pytest.fixture
def random_noise():
    np.random.seed(42)
    return np.random.rand(100)


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


def test_complexity_approx_entropy(sine_wave, random_noise):
    # Preprocess
    counts_m_sin, counts_mp1_sin = complexity_entropy_preprocessor(sine_wave[None, :])
    counts_m_noise, counts_mp1_noise = complexity_entropy_preprocessor(
        random_noise[None, :]
    )

    # Calculate
    apen_sin = complexity_approx_entropy(counts_m_sin, counts_mp1_sin)
    apen_noise = complexity_approx_entropy(counts_m_noise, counts_mp1_noise)

    # Approximate entropy should be lower for sine wave than random noise
    assert apen_sin < apen_noise


def test_complexity_sample_entropy(sine_wave, random_noise):
    # Preprocess
    counts_m_sin, counts_mp1_sin = complexity_entropy_preprocessor(sine_wave[None, :])
    counts_m_noise, counts_mp1_noise = complexity_entropy_preprocessor(
        random_noise[None, :]
    )

    # Calculate
    sampen_sin = complexity_sample_entropy(counts_m_sin, counts_mp1_sin)
    sampen_noise = complexity_sample_entropy(counts_m_noise, counts_mp1_noise)

    # Sample entropy should be lower for sine wave than random noise
    assert sampen_sin < sampen_noise


def test_complexity_svd_entropy(sine_wave, random_noise):
    svd_en_sin = complexity_svd_entropy(sine_wave[None, :])
    svd_en_noise = complexity_svd_entropy(random_noise[None, :])

    # SVD entropy should be lower for sine wave than random noise
    assert svd_en_sin < svd_en_noise


def test_complexity_lempel_ziv(sine_wave, random_noise):
    # Constant signal
    constant = np.ones(100)
    lz_const = complexity_lempel_ziv(constant[None, :])
    # Should be close to 0
    assert lz_const[0] < 0.2

    # Sine wave vs Random
    lz_sin = complexity_lempel_ziv(sine_wave[None, :])
    lz_noise = complexity_lempel_ziv(random_noise[None, :])

    assert lz_sin < lz_noise


def test_complexity_lempel_ziv_correctness():
    # Test specific patterns

    # 1. Constant signal
    # Binary: all 0s (or all 1s if >=thresh). LZC should be low.
    x_const = np.zeros((1, 100))
    lzc_const = complexity_lempel_ziv(x_const, threshold=0.1, normalize=True)
    # Normalized LZC approaches 0 for constant/periodic
    assert lzc_const < 0.15

    # 2. Alternating 0 1 0 1 ...
    x_alt = np.tile([0, 1], 50).reshape(1, -1).astype(float)
    lzc_alt = complexity_lempel_ziv(x_alt, threshold=0.5, normalize=True)
    # Simple periodic pattern, low complexity
    assert lzc_alt < 0.2

    # 3. Random noise
    np.random.seed(42)
    x_rand = np.random.randn(1, 100)
    lzc_rand = complexity_lempel_ziv(x_rand, normalize=True)
    # Random sequence should have high LZC
    assert lzc_rand > lzc_alt
    assert lzc_rand > 0.8  # Normalized LZC for random seq -> 1.0 asymptotically


def test_complexity_features(signal_2d):
    # Test entropy preprocessor
    counts_m, counts_mp1 = complexity_entropy_preprocessor(signal_2d, m=2, r=0.2, l=1)
    assert counts_m.shape == (2, 99)
    assert counts_mp1.shape == (2, 98)

    # Test approx entropy
    ae = complexity_approx_entropy(counts_m, counts_mp1)
    assert ae.shape == (2,)

    # Test sample entropy
    se = complexity_sample_entropy(counts_m, counts_mp1)
    assert se.shape == (2,)

    # Test SVD entropy
    svde = complexity_svd_entropy(signal_2d, m=10, tau=1)
    assert svde.shape == (2,)

    # Test Lempel-Ziv
    lz = complexity_lempel_ziv(signal_2d, normalize=True)
    assert lz.shape == (2,)

    # Test Lempel-Ziv with threshold
    lz_t = complexity_lempel_ziv(signal_2d, threshold=0.5, normalize=False)
    assert lz_t.shape == (2,)

    # Hurst
    he = complexity_hurst_exp(signal_2d)
    assert he.shape == (2,)

    # DFA
    dfa = complexity_detrended_fluctuation_analysis(signal_2d)
    assert dfa.shape == (2,)


def test_complexity_lempel_ziv_gap():
    from eegdash.features.feature_bank.complexity import complexity_lempel_ziv
    # Ensure raw python execution if possible
    # if it is a dispatcher, we might need to call py_func to trace coverage IF jit was supposed to be disabled but wasn't?
    # But we want to fix the root cause (env var).

    x = np.array([[1, 0, 1, 0, 1, 0]])  # Simple pattern
    # Test branches:
    _lz = complexity_lempel_ziv(x, threshold=0.5)
    _lz_none = complexity_lempel_ziv(x, threshold=None)

    # 105: normalize
    _lz_norm = complexity_lempel_ziv(x, normalize=True)
    _lz_raw = complexity_lempel_ziv(x, normalize=False)

    # We need a complex signal to trigger the "else" branches in Lempel Ziv (lines 93-...)
    # Random signal usually does it
    rng = np.random.default_rng(42)
    x_complex = rng.random((1, 50))
    complexity_lempel_ziv(x_complex)


def test_hurst_exp(signals):
    white = signals["white"]
    h_white = complexity_hurst_exp(white)
    assert h_white.shape == (1,)


def test_dimensionality_hurst_edge_cases():
    # Signal with zero variance
    sig = np.zeros((1, 100))
    he = complexity_hurst_exp(sig)
    assert np.isnan(he).all()


def test_dfa(signals):
    white = signals["white"]
    brown = signals["brown"]

    alpha_white = complexity_detrended_fluctuation_analysis(white)
    alpha_brown = complexity_detrended_fluctuation_analysis(brown)

    assert alpha_white.shape == (1,)
    assert alpha_brown.shape == (1,)


def test_dfa_correctness(signals):
    # Check theoretical scaling exponents (alpha)
    white = signals["white"]
    brown = signals["brown"]

    a_white = complexity_detrended_fluctuation_analysis(white)[0]
    a_brown = complexity_detrended_fluctuation_analysis(brown)[0]

    # White noise: alpha ~ 0.5
    assert 0.4 < a_white < 0.6

    # Brownian noise (integrated white noise): alpha ~ 1.5
    assert 1.3 < a_brown < 1.7


def test_complexity_other_functions_gap():
    from eegdash.features.feature_bank.complexity import (
        complexity_approx_entropy,
        complexity_detrended_fluctuation_analysis,
        complexity_entropy_preprocessor,
        complexity_hurst_exp,
        complexity_sample_entropy,
        complexity_svd_entropy,
    )

    x = np.random.randn(1, 50)

    # Preprocessor directly (usually called by decorators but good to test output)
    c_m, c_mp1 = complexity_entropy_preprocessor(x)

    # Approx Entropy
    # The function expects counts, so we pass them.
    complexity_approx_entropy(c_m, c_mp1)

    # Sample Entropy
    complexity_sample_entropy(c_m, c_mp1)

    # SVD Entropy
    complexity_svd_entropy(x, m=2, tau=1)

    # Hurst 48-69 (missing all)
    complexity_hurst_exp(x)

    # Hurst edge case: flat signal (std=0) to trigger line 87
    x_flat = np.zeros((1, 100))
    complexity_hurst_exp(x_flat)

    # DFA 114-134
    complexity_detrended_fluctuation_analysis(x)
