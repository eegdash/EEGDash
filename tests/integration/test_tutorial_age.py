"""Exercise the actual age tutorial against observed HBN participant targets."""

import runpy
from pathlib import Path

import matplotlib
import numpy as np
import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.network
@pytest.mark.slow
def test_tutorial_age_uses_observed_participant_targets():
    """Opt in with -m 'network and slow'; downloads about 596 MB if uncached."""
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    try:
        namespace = runpy.run_path(ROOT / "examples/applied/project_age_regression.py")
        identities = namespace["identities"]
        observed = (
            namespace["participants"].loc[identities, "age"].to_numpy(dtype=float)
        )
        np.testing.assert_array_equal(namespace["y"], observed)
        assert len(set(identities)) == len(identities) == 6
        assert namespace["X"].shape[0] == len(observed)
        for name in ("predicted", "baseline"):
            values = namespace[name]
            assert values.shape == observed.shape and np.isfinite(values).all()
        # Each baseline prediction must use only the other participants' ages.
        expected = (observed.sum() - observed) / (len(observed) - 1)
        np.testing.assert_allclose(namespace["baseline"], expected)
    finally:
        plt.close("all")
