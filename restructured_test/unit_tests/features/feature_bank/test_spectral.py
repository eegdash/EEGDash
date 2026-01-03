import numpy as np
from scipy.signal import welch

from eegdash.features.feature_bank.spectral import (
    spectral_hjorth_complexity,
    spectral_normalized_preprocessor,
    spectral_edge,
    spectral_preprocessor,
)


def test_spectral_hjorth_complexity_fix():
    # Example from issue report
    fs = 250
    t = np.arange(0, 10, 1 / fs)
    # signal = sin(2π·10·t) + 0.5·sin(2π·20·t)
    x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    x = x[np.newaxis, :]  # Add channel dim

    # Calculate PSD
    f, p = welch(x, fs=fs, nperseg=fs)

    # Normalize
    f_norm, p_norm = spectral_normalized_preprocessor(f, p)

    # Calculate complexity
    complexity = spectral_hjorth_complexity(f_norm, p_norm)

    # Expected value from issue report: ~1.250612
    # The previous buggy implementation returned ~200

    # Check if value is reasonable (close to 1.25)
    np.testing.assert_allclose(complexity, 1.25, atol=0.1)

    # Also verify it is NOT the huge value
    assert complexity < 10.0


def test_spectral_hjorth_complexity_shape():
    f = np.linspace(0, 50, 51)
    p = np.random.rand(5, 51)
    # Normalize
    p = p / p.sum(axis=-1, keepdims=True)

    comp = spectral_hjorth_complexity(f, p)
    assert comp.shape == (5,)


def test_spectral_edge_cases():
    # Trigger spectral_edge numba code
    f = np.linspace(0, 50, 100)
    p = np.random.rand(2, 4, 100)
    # Normalize p
    p /= p.sum(axis=-1, keepdims=True)

    se = spectral_edge(f, p, edge=0.9)
    assert se.shape == (2, 4)


def test_spectral_more():
    from eegdash.features.feature_bank.spectral import (
        spectral_bands_power,
        spectral_db_preprocessor,
        spectral_edge,
        spectral_entropy,
        spectral_hjorth_activity,
        spectral_hjorth_complexity,
        spectral_hjorth_mobility,
        spectral_moment,
        spectral_normalized_preprocessor,
        spectral_root_total_power,
        spectral_slope,
    )

    # lines 27-34: skip_outlier_noise
    # Use longer signal for better frequency resolution to satisfy frequency bands
    # f0 = 2 * fs / n. For fs=100, n=200 -> f0=1.0. Delta starts at 1.0
    data = np.random.randn(2, 4, 300)
    f, p = spectral_preprocessor(data, fs=100.0)

    # 39: normalized
    spectral_normalized_preprocessor(f, p)
    # 44: db
    spectral_db_preprocessor(f, p)
    # 50: root_total_power
    spectral_root_total_power(f, p)
    # 56: moment
    spectral_moment(f, p / p.sum(axis=-1, keepdims=True))
    # 62: activity
    spectral_hjorth_activity(f, p)
    # 68: mobility
    spectral_hjorth_mobility(f, p / p.sum(axis=-1, keepdims=True))
    # 74-77: complexity
    spectral_hjorth_complexity(f, p / p.sum(axis=-1, keepdims=True))
    # 83-86: entropy
    spectral_entropy(f, p / p.sum(axis=-1, keepdims=True))
    # 102-105: slope
    r_slope = spectral_slope(f, p + 1e-15)
    assert isinstance(r_slope, dict)
    # 115: bands_power
    spectral_bands_power(f, p)
    # 93-96: edge
    spectral_edge(f, p / p.sum(axis=-1, keepdims=True), edge=0.5)


def test_spectral_edge_gap():
    from eegdash.features.feature_bank.spectral import spectral_edge

    f = np.linspace(0, 100, 101)
    p = np.exp(-f / 10)  # 1/f-like
    # Make it 2D (batch, freqs)
    p = np.stack([p, p])

    # Missing 93-96?
    # That's the main body?
    # Let's ensure we call it
    se = spectral_edge(f, p, edge=0.95)
    assert se.shape[0] == 2
