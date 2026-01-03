import numpy as np


def test_slice_freq_band_f_min_only():
    """Test slice_freq_band with only f_min."""
    from eegdash.features.feature_bank.utils import slice_freq_band

    f = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    f_out, p_out = slice_freq_band(f, p, f_min=5)
    assert len(f_out) == 6
    assert f_out[0] == 5


def test_slice_freq_band_f_max_only():
    """Test slice_freq_band with only f_max."""
    from eegdash.features.feature_bank.utils import slice_freq_band

    f = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    f_out, p_out = slice_freq_band(f, p, f_max=5)
    assert len(f_out) == 5
    assert f_out[-1] == 5


def test_slice_freq_band_with_multiple_arrays():
    """Test slice_freq_band with multiple data arrays."""
    from eegdash.features.feature_bank.utils import slice_freq_band

    f = np.array([1, 2, 3, 4, 5])
    p1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    p2 = np.array([1.1, 1.2, 1.3, 1.4, 1.5])

    f_out, p1_out, p2_out = slice_freq_band(f, p1, p2, f_min=2, f_max=4)
    assert len(f_out) == 3
    assert len(p1_out) == 3
    assert len(p2_out) == 3


def test_get_valid_freq_band_assertions():
    """Test assertion failures in get_valid_freq_band (lines 23, 27)."""
    from eegdash.features.feature_bank.utils import get_valid_freq_band
    import pytest

    fs, n = 256, 512
    # f0 = 2 * fs / n = 1.0, f1 = fs / 2 = 128

    # f_min below f0 should raise
    with pytest.raises(AssertionError):
        get_valid_freq_band(fs, n, f_min=0.5)

    # f_max above f1 should raise
    with pytest.raises(AssertionError):
        get_valid_freq_band(fs, n, f_max=150)


def test_slice_freq_band_none_limits():
    """Test slice_freq_band with None limits (line 35)."""
    from eegdash.features.feature_bank.utils import slice_freq_band
    import numpy as np

    f = np.array([1, 2, 3, 4, 5])
    x = np.array([10, 20, 30, 40, 50])

    # Both None - should return unchanged
    result = slice_freq_band(f, x)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], f)
    np.testing.assert_array_equal(result[1], x)


def test_reduce_freq_bands():
    """Test reduce_freq_bands function."""
    from eegdash.features.feature_bank.utils import reduce_freq_bands
    import numpy as np

    f = np.array([1, 2, 3, 5, 6, 10, 15, 25])
    x = np.ones_like(f, dtype=float)
    bands = {"low": (1, 6), "high": (10, 25)}  # Within f range

    result = reduce_freq_bands(f, x, bands)
    assert "low" in result
    assert "high" in result
