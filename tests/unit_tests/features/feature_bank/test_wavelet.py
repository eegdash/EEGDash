import numpy as np
import pytest

from eegdash.features.feature_bank.wavelet import (
    wavelet_entropy,
    wavelet_kurtosis,
    wavelet_pac,
    wavelet_preprocessor,
    wavelet_relative_power,
    wavelet_skewness,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FS = 256
N_TIMES = 512
N_CHANNELS = 3
BANDS = {"theta": (4.5, 8), "alpha": (8, 12), "beta": (12, 30)}
METADATA = {"info": {"sfreq": float(FS)}}


def _sine(freq_hz, n_channels=N_CHANNELS, n_times=N_TIMES, fs=FS):
    t = np.arange(n_times) / fs
    s = np.sin(2 * np.pi * freq_hz * t)
    return np.tile(s, (n_channels, 1))


def _noise(n_channels=N_CHANNELS, n_times=N_TIMES, seed=0):
    return np.random.default_rng(seed).standard_normal((n_channels, n_times))


def _preprocess(x, bands=BANDS):
    bands_out, W, _ = wavelet_preprocessor(
        x, frequency_bands=bands, _metadata=METADATA.copy()
    )
    return bands_out, W


def _preprocess_with_meta(x, bands=BANDS, wavelet="cmor1.5-1.0"):
    return wavelet_preprocessor(
        x, frequency_bands=bands, wavelet=wavelet, _metadata=METADATA.copy()
    )


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


def test_wavelet_preprocessor_output_shape():
    x = _noise()
    bands, W, _ = wavelet_preprocessor(
        x, frequency_bands=BANDS, _metadata=METADATA.copy()
    )
    assert W.shape == (len(BANDS), N_CHANNELS, N_TIMES)


def test_wavelet_preprocessor_batched():
    x = _noise().reshape(1, N_CHANNELS, N_TIMES).repeat(4, axis=0)  # (4, 3, 512)
    bands, W, _ = wavelet_preprocessor(
        x, frequency_bands=BANDS, _metadata=METADATA.copy()
    )
    assert W.shape == (4, len(BANDS), N_CHANNELS, N_TIMES)


def test_wavelet_preprocessor_returns_complex_for_cmor():
    x = _noise()
    _, W, _ = wavelet_preprocessor(x, frequency_bands=BANDS, _metadata=METADATA.copy())
    assert np.iscomplexobj(W)


def test_wavelet_preprocessor_returns_real_for_mexh():
    x = _noise()
    _, W, _ = wavelet_preprocessor(
        x, frequency_bands=BANDS, wavelet="mexh", _metadata=METADATA.copy()
    )
    assert not np.iscomplexobj(W)


def test_wavelet_preprocessor_returns_bands_dict():
    x = _noise()
    bands, _, _ = wavelet_preprocessor(
        x, frequency_bands=BANDS, _metadata=METADATA.copy()
    )
    assert bands == BANDS


def test_wavelet_preprocessor_sets_wavelet_is_complex_true():
    _, _, meta = _preprocess_with_meta(_noise())
    assert meta["wavelet_is_complex"] is True


def test_wavelet_preprocessor_sets_wavelet_is_complex_false_for_mexh():
    _, _, meta = _preprocess_with_meta(_noise(), wavelet="mexh")
    assert meta["wavelet_is_complex"] is False


# ---------------------------------------------------------------------------
# wavelet_entropy
# ---------------------------------------------------------------------------


def test_wavelet_entropy_shape():
    bands, W = _preprocess(_noise())
    result = wavelet_entropy(bands, W)
    assert result.shape == (N_CHANNELS,)


def test_wavelet_entropy_noise_greater_than_sine():
    bands_noise, W_noise = _preprocess(_noise())
    bands_sine, W_sine = _preprocess(_sine(10))  # 10 Hz → mostly alpha
    H_noise = wavelet_entropy(bands_noise, W_noise)
    H_sine = wavelet_entropy(bands_sine, W_sine)
    assert H_noise.mean() > H_sine.mean()


# ---------------------------------------------------------------------------
# wavelet_relative_power
# ---------------------------------------------------------------------------


def test_wavelet_relative_power_shape():
    bands, W = _preprocess(_noise())
    result = wavelet_relative_power(bands, W)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(BANDS.keys())
    for v in result.values():
        assert v.shape == (N_CHANNELS,)


def test_wavelet_relative_power_sums_to_one():
    bands, W = _preprocess(_noise())
    result = wavelet_relative_power(bands, W)
    total = sum(result.values())
    np.testing.assert_allclose(total, np.ones(N_CHANNELS), atol=1e-10)


def test_wavelet_relative_power_alpha_dominates_for_10hz_sine():
    bands, W = _preprocess(_sine(10))  # 10 Hz is in alpha (8–12 Hz)
    result = wavelet_relative_power(bands, W)
    assert (result["alpha"] > result["theta"]).all()
    assert (result["alpha"] > result["beta"]).all()


# ---------------------------------------------------------------------------
# wavelet_skewness
# ---------------------------------------------------------------------------


def test_wavelet_skewness_shape():
    bands, W = _preprocess(_noise())
    result = wavelet_skewness(bands, W)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(BANDS.keys())
    for v in result.values():
        assert v.shape == (N_CHANNELS,)


# ---------------------------------------------------------------------------
# wavelet_kurtosis
# ---------------------------------------------------------------------------


def test_wavelet_kurtosis_shape():
    bands, W = _preprocess(_noise())
    result = wavelet_kurtosis(bands, W)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(BANDS.keys())
    for v in result.values():
        assert v.shape == (N_CHANNELS,)


def test_wavelet_kurtosis_returns_excess_kurtosis():
    # scipy kurtosis default is excess (Fisher) → Gaussian ≈ 0
    bands, W = _preprocess(_noise(n_channels=1, n_times=4096))
    result = wavelet_kurtosis(bands, W)
    for v in result.values():
        assert np.abs(v).mean() < 2.0  # rough sanity: not wildly non-zero for noise


# ---------------------------------------------------------------------------
# wavelet_pac
# ---------------------------------------------------------------------------


def test_wavelet_pac_raises_for_real_wavelet():
    x = _noise()
    _, W, meta = _preprocess_with_meta(x, wavelet="mexh")
    with pytest.raises(ValueError, match="complex wavelet"):
        wavelet_pac(BANDS, W, _metadata=meta)


def test_wavelet_pac_shape():
    bands, W = _preprocess(_noise())
    result = wavelet_pac(bands, W)
    band_names = list(BANDS.keys())
    expected_keys = {
        (band_names[i], band_names[j])
        for i in range(len(band_names))
        for j in range(i + 1, len(band_names))
    }
    assert set(result.keys()) == expected_keys
    for v in result.values():
        assert v.shape == (N_CHANNELS,)


def test_wavelet_pac_bounded():
    bands, W = _preprocess(_noise())
    result = wavelet_pac(bands, W)
    for v in result.values():
        assert (v >= 0).all() and (v <= 1).all()


def test_wavelet_pac_uncoupled_is_low():
    # Independent sinusoids in different bands → MI should be low
    t = np.arange(N_TIMES) / FS
    theta = np.sin(2 * np.pi * 6 * t)
    beta = np.sin(2 * np.pi * 20 * t)
    x = np.stack([theta + beta] * N_CHANNELS)
    bands, W = _preprocess(x)
    result = wavelet_pac(bands, W)
    for v in result.values():
        assert v.mean() < 0.5


# ---------------------------------------------------------------------------
# Phase-feature error for real wavelet
# ---------------------------------------------------------------------------


def test_wavelet_features_require_complex_message_is_clear():
    x = _noise()
    _, W, meta = _preprocess_with_meta(x, wavelet="mexh")
    with pytest.raises(ValueError) as exc_info:
        wavelet_pac(BANDS, W, _metadata=meta)
    assert "cmor" in str(exc_info.value)
