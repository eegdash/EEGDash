import numpy as np
from scipy.signal import welch

from eegdash.features.feature_bank.spectral import (
    spectral_hjorth_complexity,
    spectral_normalized_preprocessor,
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
