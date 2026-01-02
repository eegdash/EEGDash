import numpy as np
import pytest

from eegdash.features.feature_bank.complexity import (
    _create_embedding,
    complexity_approx_entropy,
    complexity_entropy_preprocessor,
    complexity_lempel_ziv,
    complexity_sample_entropy,
    complexity_svd_entropy,
)


def test_create_embedding():
    x = np.array([1, 2, 3, 4, 5])
    dim = 2
    lag = 1
    # Expected: [[1, 2], [2, 3], [3, 4], [4, 5]]
    embedding = _create_embedding(x, dim, lag)
    expected = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    np.testing.assert_array_equal(embedding, expected)

    lag = 2
    # Expected: [[1, 2], [3, 4]]
    embedding = _create_embedding(x, dim, lag)
    expected = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(embedding, expected)


@pytest.fixture
def sine_wave():
    t = np.linspace(0, 10, 100)
    return np.sin(2 * np.pi * t)


@pytest.fixture
def random_noise():
    np.random.seed(42)
    return np.random.rand(100)


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
