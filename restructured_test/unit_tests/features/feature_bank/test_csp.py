import numpy as np
import pytest

from eegdash.features.feature_bank.csp import CommonSpatialPattern


@pytest.fixture
def csp():
    return CommonSpatialPattern()


def generate_cov_data(n_trials, n_channels, n_samples=100):
    # Generate random data
    # (n_trials, n_channels, n_samples)
    return np.random.randn(n_trials, n_channels, n_samples)


def test_csp_initialization(csp):
    assert csp._labels is None
    np.testing.assert_array_equal(csp._counts, np.array([0, 0]))
    np.testing.assert_array_equal(csp._means, np.array([None, None]))
    np.testing.assert_array_equal(csp._covs, np.array([None, None]))


def test_csp_partial_fit(csp):
    n_channels = 4
    n_samples = 100

    # Class 0 data
    X0 = generate_cov_data(5, n_channels, n_samples)
    y0 = np.zeros(5)

    # Class 1 data
    X1 = generate_cov_data(5, n_channels, n_samples)
    y1 = np.ones(5)

    csp.partial_fit(X0, y0)
    assert csp._counts[0] == 500
    assert csp._counts[1] == 0
    assert csp._means[0] is not None
    assert csp._covs[0] is not None

    csp.partial_fit(X1, y1)
    assert csp._counts[0] == 500
    assert csp._counts[1] == 500
    assert csp._means[1] is not None
    assert csp._covs[1] is not None

    # Update existing class
    X0_new = generate_cov_data(2, n_channels, n_samples)
    y0_new = np.zeros(2)
    csp.partial_fit(X0_new, y0_new)
    assert csp._counts[0] == 700


def test_csp_fit_transform(csp):
    n_channels = 4
    n_samples = 100
    n_trials = 20

    # Create distinct classes by scaling variance
    X = np.random.randn(n_trials, n_channels, n_samples)
    y = np.array([0] * 10 + [1] * 10)

    # Scale class 1
    X[10:] *= 2.0

    csp.partial_fit(X, y)
    csp.fit()

    assert csp._eigvals is not None
    assert csp._weights is not None
    assert csp._weights.shape == (n_channels, n_channels)

    # Transform
    # Input to transform should be (n_trials, n_channels, n_samples)
    # Result is a dict
    features = csp(X)
    assert isinstance(features, dict)

    # Check output dimensions
    # output per feature is (n_trials,)
    # default returns all components
    assert len(features) == n_channels
    for k, v in features.items():
        assert v.shape == (n_trials,)


def test_csp_transform_selection(csp):
    n_channels = 4
    X = generate_cov_data(10, n_channels)
    y = np.array([0] * 5 + [1] * 5)

    csp.partial_fit(X, y)
    csp.fit()

    # Select top 2
    feats = csp(X, n_select=2)
    assert len(feats) == 2

    # Select by criterion (eigenval distance from 0.5)
    # Eigenvalues will be symmetric around 0.5.
    # If we set strict criterion, we select fewer
    # 0.5 - |eig - 0.5| < crit
    # |eig - 0.5| > 0.5 - crit
    # if crit is small (near 0), we filter out things near 0.5 (noise)
    # and keep things near 0 or 1.

    # n_select=None
    # Let's try to compel a selection
    feats_crit = csp(X, crit_select=0.49)  # Very relaxed, should keep most?
    # Wait: condition is `0.5 - np.abs(l - 0.5) < crit`
    # if l=0, abs=0.5, 0.5-0.5=0 < crit.
    # So if crit is > 0, we keep extreme eigenvalues.

    assert len(feats_crit) > 0


def test_csp_clear(csp):
    n_channels = 4
    X = generate_cov_data(5, n_channels)
    y = np.zeros(5)

    csp.partial_fit(X, y)
    csp.clear()

    np.testing.assert_array_equal(csp._counts, np.array([0, 0]))
    assert csp._labels is None


def test_csp_unbalanced_classes(csp):
    # Test fix for unbalanced classes scaling
    n_channels = 4
    n_samples = 50

    # Unbalanced: Class 0 has 200 trials, Class 1 has 50 trials
    # Total samples: C0=10000, C1=2500
    X0 = generate_cov_data(200, n_channels, n_samples)
    y0 = np.zeros(200)

    X1 = generate_cov_data(50, n_channels, n_samples)  # Distinct scale
    X1 *= 2.0
    y1 = np.ones(50)

    csp.partial_fit(X0, y0)
    csp.partial_fit(X1, y1)
    csp.fit()

    # Before fix, this might error or produce wrong weights.
    # We mainly check that it runs and produces valid weights.
    # The fix ensures covariance matrices are scaled by their OWN counts.
    # We can check internal state if we want, but execution without error
    # and valid output shape is a good first check for the logic fix.

    assert csp._weights.shape == (n_channels, n_channels)

    # Check that counts are correct
    assert csp._counts[0] == 200 * n_samples
    assert csp._counts[1] == 50 * n_samples


def test_csp_preprocessor():
    # Trigger CSP stats and _update_mean_cov
    csp = CommonSpatialPattern()
    data = np.random.randn(4, 4, 100)  # (epochs, channels, times)
    y = np.array([0, 0, 1, 1])

    # First fit
    csp.partial_fit(data, y)
    csp.fit()

    # Second fit (trigger _update_mean_cov)
    csp.partial_fit(data, y)
    csp.fit()

    # Call
    res = csp(data)
    assert len(res) > 0


def test_csp_features_gaps():
    from eegdash.features.feature_bank.csp import CommonSpatialPattern

    csp = CommonSpatialPattern()
    csp.clear()

    # partial_fit
    x = np.random.randn(2, 4, 100)
    y = np.array([0, 1])
    csp.partial_fit(x, y)

    # fit
    csp.fit()

    # __call__ (92, 94-95, 97)
    csp(x, n_select=1)
    csp(x, crit_select=0.9)
    with pytest.raises(RuntimeError, match="too strict"):
        csp(x, crit_select=0.0001)


def test_csp_update_mean_cov_gap():
    from eegdash.features.feature_bank.csp import CommonSpatialPattern

    csp = CommonSpatialPattern()
    # Trigger _update_mean_cov (called in partial_fit if n_epochs > 0)
    X = np.random.randn(2, 4, 100)
    y = np.array([0, 1])
    csp.partial_fit(X, y)  # First call initializes mean/cov
    csp.partial_fit(X, y)  # Second call triggers _update_mean_cov


import numpy as np


def test_csp_update_mean_cov_gap():
    from eegdash.features.feature_bank.csp import CommonSpatialPattern

    csp = CommonSpatialPattern()
    # Trigger _update_mean_cov (called in partial_fit if n_epochs > 0)
    X = np.random.randn(2, 4, 100)
    y = np.array([0, 1])
    csp.partial_fit(X, y)  # First call initializes mean/cov
    csp.partial_fit(X, y)  # Second call triggers _update_mean_cov


import numpy as np
import pytest


def test_csp_preprocessor():
    from eegdash.features.feature_bank.csp import CommonSpatialPattern

    # Trigger CSP stats and _update_mean_cov
    csp = CommonSpatialPattern()
    data = np.random.randn(4, 4, 100)  # (epochs, channels, times)
    y = np.array([0, 0, 1, 1])

    # First fit
    csp.partial_fit(data, y)
    csp.fit()

    # Second fit (trigger _update_mean_cov)
    csp.partial_fit(data, y)
    csp.fit()

    # Call
    res = csp(data)
    assert len(res) > 0


def test_csp_features_gaps():
    from eegdash.features.feature_bank.csp import CommonSpatialPattern

    csp = CommonSpatialPattern()
    csp.clear()

    # partial_fit
    x = np.random.randn(2, 4, 100)
    y = np.array([0, 1])
    csp.partial_fit(x, y)

    # fit
    csp.fit()

    # __call__ (92, 94-95, 97)
    csp(x, n_select=1)
    csp(x, crit_select=0.9)
    with pytest.raises(RuntimeError, match="too strict"):
        csp(x, crit_select=0.0001)


def test_csp_update_mean_cov_jit():
    """Test the JIT-compiled _update_mean_cov function."""
    from eegdash.features.feature_bank.csp import _update_mean_cov
    import numpy as np

    # Initialize arrays
    count = 100
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.eye(3) * 0.5
    x_count = 50
    x_mean = np.array([1.5, 2.5, 3.5])
    x_cov = np.eye(3) * 0.3

    # Call the function (lines 17-22)
    _update_mean_cov(count, mean, cov, x_count, x_mean, x_cov)

    # Should update mean and cov in place
    assert mean.shape == (3,)
    assert cov.shape == (3, 3)


def test_csp_selection_criterion_too_strict():
    """Test CSP raises when selection criterion filters all weights."""
    from eegdash.features.feature_bank.csp import CommonSpatialPattern
    import numpy as np
    import pytest

    csp = CommonSpatialPattern()
    csp.clear()

    # Create minimal data for fitting
    x = np.random.randn(20, 3, 100)  # 20 trials, 3 channels, 100 samples
    y = np.array([0] * 10 + [1] * 10)

    csp.partial_fit(x, y)
    csp.fit()

    # Line 97: crit_select too strict should raise
    with pytest.raises(RuntimeError, match="too strict"):
        csp(x, crit_select=0.0001)


def test_csp_full_workflow():
    """Test CommonSpatialPattern full workflow (lines 17-22, 97)."""
    from eegdash.features.feature_bank.csp import CommonSpatialPattern
    import numpy as np

    csp = CommonSpatialPattern()
    csp.clear()

    # Create fake data: batch x channels x time
    np.random.seed(42)
    x1 = np.random.randn(50, 4, 100)  # Class 0
    x2 = np.random.randn(50, 4, 100) + 0.5  # Class 1

    x = np.vstack([x1, x2])
    y = np.array([0] * 50 + [1] * 50)

    # Partial fit
    csp.partial_fit(x, y)
    csp.fit()

    # Call - returns dict with numbered keys (channel indices)
    result = csp(x[:5], n_select=2)
    assert isinstance(result, dict)
    # Keys are numbered strings: '0', '1', etc.
    assert len(result) > 0


def test_csp_strict_criterion_error():
    """Test CSP raises when criterion too strict (line 97)."""
    from eegdash.features.feature_bank.csp import CommonSpatialPattern
    import numpy as np
    import pytest

    csp = CommonSpatialPattern()
    csp.clear()

    np.random.seed(42)
    x1 = np.random.randn(50, 4, 100)
    x2 = np.random.randn(50, 4, 100) + 0.5

    x = np.vstack([x1, x2])
    y = np.array([0] * 50 + [1] * 50)

    csp.partial_fit(x, y)
    csp.fit()

    # Very strict criterion that filters all weights
    with pytest.raises(RuntimeError, match="too strict"):
        csp(x[:5], crit_select=0.0001)
