import numpy as np


def test_signal_decorrelation_time():
    """Test signal_decorrelation_time feature."""
    from eegdash.features.feature_bank.signal import signal_decorrelation_time

    # Create test signal
    x = np.random.randn(2, 100)

    # Lines 118-123: decorrelation time computation
    result = signal_decorrelation_time(x, fs=100)
    assert result.shape == (2,)


def test_signal_hjorth_complexity():
    """Test signal_hjorth_complexity feature."""
    from eegdash.features.feature_bank.signal import signal_hjorth_complexity

    x = np.random.randn(2, 100)
    result = signal_hjorth_complexity(x)
    assert result.shape == (2,)


def test_signal_hjorth_mobility():
    """Test signal_hjorth_mobility feature."""
    from eegdash.features.feature_bank.signal import signal_hjorth_mobility

    x = np.random.randn(2, 100)
    result = signal_hjorth_mobility(x)
    assert result.shape == (2,)


def test_signal_hilbert_preprocessor():
    """Test signal_hilbert_preprocessor (line 30)."""
    import numpy as np

    from eegdash.features.feature_bank.signal import signal_hilbert_preprocessor

    x = np.random.randn(2, 3, 256)
    result = signal_hilbert_preprocessor(x)
    assert result.shape == x.shape
    # Hilbert envelope should be non-negative
    assert np.all(result >= 0)


def test_signal_zero_crossings():
    """Test signal_zero_crossings with threshold (line 104)."""
    import numpy as np

    from eegdash.features.feature_bank.signal import signal_zero_crossings

    # Create signal with known zero crossings
    x = np.array([[1, -1, 1, -1, 0.5, -0.5]])
    result = signal_zero_crossings(x)
    assert result[0] > 0


def test_signal_decorrelation_time_2():
    """Test signal_decorrelation_time (lines 110, 118-123)."""
    import numpy as np

    from eegdash.features.feature_bank.signal import signal_decorrelation_time

    np.random.seed(42)
    x = np.random.randn(2, 3, 256)
    result = signal_decorrelation_time(x, fs=256)
    assert result.shape == (2, 3)


def test_signal_hjorth_features():
    """Test Hjorth features."""
    import numpy as np

    from eegdash.features.feature_bank.signal import (
        signal_hjorth_activity,
        signal_hjorth_complexity,
        signal_hjorth_mobility,
    )

    x = np.random.randn(2, 100)

    act = signal_hjorth_activity(x)
    assert act.shape == (2,)

    mob = signal_hjorth_mobility(x)
    assert mob.shape == (2,)

    comp = signal_hjorth_complexity(x)
    assert comp.shape == (2,)
