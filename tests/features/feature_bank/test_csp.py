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
