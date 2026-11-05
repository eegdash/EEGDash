"""Tests for the neural-avalanche feature bank module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from eegdash.features.feature_bank.avalanches import (
    _avalanche_sizes_durations,
    _fit_truncated_power_law,
    avalanche_alpha_exponent,
    avalanche_bin_preprocessor,
    avalanche_branching_parameter,
    avalanche_dcc,
    avalanche_detect_preprocessor,
    avalanche_events_preprocessor,
    avalanche_exponents_preprocessor,
    avalanche_gamma_exponent,
    avalanche_standardize_mad_preprocessor,
    avalanche_standardize_preprocessor,
    avalanche_tau_exponent,
)


def _metadata(n_channels=2, sfreq=1000.0, batch_size=1, **extra):
    md = {
        "info": {
            "sfreq": sfreq,
            "ch_names": [f"ch{i}" for i in range(n_channels)],
        },
        "batch_size": batch_size,
    }
    md.update(extra)
    return md


# ── Standardization ───────────────────────────────────────────────────────────


def test_standardize_preprocessor_zscores_each_channel():
    x = np.array([[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]])
    z = avalanche_standardize_preprocessor(x)

    assert z.shape == x.shape
    np.testing.assert_allclose(z.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=-1), 1.0, atol=1e-12)
    # both channels are the same signal up to an affine map -> same z-scores
    np.testing.assert_allclose(z[0, 0], z[0, 1])


def test_standardize_preprocessor_handles_flat_channel():
    x = np.zeros((1, 1, 8))
    z = avalanche_standardize_preprocessor(x)

    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(z, 0.0)


def test_standardize_mad_preprocessor_is_robust_to_outliers():
    base = np.concatenate([np.arange(-10.0, 11.0), np.array([1e6])])
    x = base[None, None, :]

    z_mad = avalanche_standardize_mad_preprocessor(x)
    z_std = avalanche_standardize_preprocessor(x)

    assert z_mad.shape == x.shape
    # the outlier swamps the standard deviation but not the MAD
    assert abs(z_std[0, 0, -1]) < 5
    assert abs(z_mad[0, 0, -1]) > 100


def test_standardize_mad_preprocessor_handles_flat_channel():
    x = np.ones((1, 1, 8))
    z = avalanche_standardize_mad_preprocessor(x)
    assert np.all(np.isfinite(z))


# ── Event detection ───────────────────────────────────────────────────────────


def test_events_preprocessor_marks_only_the_excursion_peak():
    z = np.zeros((1, 2, 20))
    z[0, 0, 5:9] = [4.0, 5.0, 6.0, 4.0]  # one excursion, peak at sample 7
    z[0, 1, 12] = -5.0  # negative excursion is detected too

    events = avalanche_events_preprocessor(z, k=3.0)

    assert events.dtype == np.uint8
    assert events.shape == z.shape
    assert events.sum() == 2
    assert events[0, 0, 7] == 1
    assert events[0, 1, 12] == 1


def test_events_preprocessor_does_not_merge_across_channels():
    z = np.zeros((1, 3, 10))
    z[0, :, 4] = 5.0  # simultaneous crossing on every channel

    events = avalanche_events_preprocessor(z, k=3.0)

    assert events.sum() == 3
    np.testing.assert_array_equal(events[0, :, 4], [1, 1, 1])


def test_events_preprocessor_without_crossings_is_empty():
    z = np.zeros((2, 2, 16))
    events = avalanche_events_preprocessor(z, k=3.0)
    assert events.sum() == 0


# ── Binning ───────────────────────────────────────────────────────────────────


def test_bin_preprocessor_sums_channels_and_bins_time():
    events = np.zeros((1, 2, 10), dtype=np.uint8)
    events[0, 0, [0, 1]] = 1
    events[0, 1, 5] = 1

    binned, md = avalanche_bin_preprocessor(
        events, _metadata=_metadata(sfreq=1000.0), bin_size_ms=2.0
    )

    assert md["bin_size_samples"] == 2
    assert md["bin_size_sec"] == pytest.approx(0.002)
    assert md["n_bins"] == 5
    np.testing.assert_array_equal(binned, [[2, 0, 1, 0, 0]])


def test_bin_preprocessor_records_the_realized_bin_width():
    """The request is rounded to whole samples, so the realized width differs."""
    events = np.zeros((1, 1, 1000), dtype=np.uint8)
    _, md = avalanche_bin_preprocessor(
        events, _metadata=_metadata(sfreq=256.0), bin_size_ms=5.0
    )

    # 5 ms at 256 Hz is 1.28 samples, which rounds down to 1
    assert md["bin_size_samples"] == 1
    assert md["bin_size_sec"] == pytest.approx(1 / 256.0)
    assert md["bin_size_sec"] != pytest.approx(0.005)  # not the requested width


def test_bin_preprocessor_discards_trailing_partial_bin():
    events = np.ones((1, 1, 7), dtype=np.uint8)
    binned, md = avalanche_bin_preprocessor(
        events, _metadata=_metadata(sfreq=1000.0), bin_size_ms=2.0
    )
    assert md["n_bins"] == 3  # 7 // 2, last sample dropped
    assert binned.shape == (1, 3)


def test_bin_preprocessor_raises_for_subsample_bin():
    events = np.zeros((1, 1, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least 1 sample is required"):
        avalanche_bin_preprocessor(
            events, _metadata=_metadata(sfreq=200.0), bin_size_ms=1.0
        )


def test_bin_preprocessor_raises_when_bin_exceeds_window():
    events = np.zeros((1, 1, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="longer than the 10-sample window"):
        avalanche_bin_preprocessor(
            events, _metadata=_metadata(sfreq=1000.0), bin_size_ms=100.0
        )


# ── Avalanche detection ───────────────────────────────────────────────────────


def test_detect_preprocessor_finds_runs_and_pads():
    binned = np.array(
        [
            [0, 2, 1, 0, 3, 0, 5, 5],  # trailing run touches the edge -> dropped
            [0, 1, 0, 0, 0, 0, 0, 0],  # a single avalanche -> padded
        ]
    )
    out, starts, ends, counts = avalanche_detect_preprocessor(binned)

    np.testing.assert_array_equal(out, binned)
    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_array_equal(starts, [[1, 4], [1, -1]])
    np.testing.assert_array_equal(ends, [[2, 4], [1, -1]])


def test_detect_preprocessor_drops_leading_edge_avalanche():
    binned = np.array([[3, 1, 0, 2, 0]])
    _, starts, ends, counts = avalanche_detect_preprocessor(binned)

    np.testing.assert_array_equal(counts, [1])
    np.testing.assert_array_equal(starts[0, :1], [3])
    np.testing.assert_array_equal(ends[0, :1], [3])


def test_detect_preprocessor_handles_silent_window():
    binned = np.zeros((2, 6), dtype=int)
    _, starts, ends, counts = avalanche_detect_preprocessor(binned)

    np.testing.assert_array_equal(counts, [0, 0])
    assert starts.shape == (2, 1)  # never zero-width
    assert np.all(starts == -1)
    assert np.all(ends == -1)


def test_sizes_are_not_overcounted_by_the_trailing_segment():
    """Regression: ``np.add.reduceat`` absorbs the discarded edge avalanche."""
    binned = np.array([0, 2, 1, 0, 3, 0, 5, 5])
    starts, ends = np.array([1, 4]), np.array([2, 4])

    sizes, durations = _avalanche_sizes_durations(binned, starts, ends)

    np.testing.assert_allclose(sizes, [3.0, 3.0])
    np.testing.assert_array_equal(durations, [2, 1])
    # the naive implementation this replaces would have returned [3, 13]
    assert not np.allclose(sizes, np.add.reduceat(binned, starts))


# ── Branching parameter ───────────────────────────────────────────────────────


def test_branching_parameter_known_value():
    binned = np.array([[0, 4, 2, 0, 6, 3, 0, 0]])
    _, starts, ends, counts = avalanche_detect_preprocessor(binned)

    sigma = avalanche_branching_parameter(binned, starts, ends, counts)

    assert sigma.shape == (1,)
    np.testing.assert_allclose(sigma, [0.5])  # mean(2/4, 3/6)


def test_branching_parameter_is_nan_without_avalanches():
    binned = np.zeros((1, 6), dtype=int)
    _, starts, ends, counts = avalanche_detect_preprocessor(binned)

    sigma = avalanche_branching_parameter(binned, starts, ends, counts)
    assert np.isnan(sigma).all()


# ── Power-law fitting ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("exponent", [2.0, 2.5, 3.0])
def test_fit_truncated_power_law_recovers_generating_exponent(exponent):
    rng = np.random.default_rng(42)
    data = rng.zipf(exponent, size=20000).astype(float)

    fit = _fit_truncated_power_law(data, system_size=40, n_min=500)

    assert fit["exponent"] == pytest.approx(exponent, abs=0.1)
    assert fit["ks"] < 0.05
    assert fit["n_included"] > 500


def test_fit_truncated_power_law_returns_nan_below_n_min():
    fit = _fit_truncated_power_law(np.array([1.0, 2.0, 3.0]), system_size=10)

    assert np.isnan(fit["exponent"])
    assert fit["n_included"] == 0


def test_fit_truncated_power_law_raises_for_bad_system_size():
    with pytest.raises(ValueError, match="must be a positive number"):
        _fit_truncated_power_law(np.arange(1.0, 10.0), system_size=0)


# ── Exponents preprocessor ────────────────────────────────────────────────────


def _binned_with_avalanches(n_avalanches=40, seed=0):
    """Build a binned array holding `n_avalanches` separated runs."""
    rng = np.random.default_rng(seed)
    bins = [0]
    for _ in range(n_avalanches):
        bins.extend(rng.integers(1, 5, size=rng.integers(1, 4)).tolist())
        bins.append(0)
    bins.append(0)
    return np.array([bins])


def test_exponents_preprocessor_returns_three_arrays():
    binned = _binned_with_avalanches()
    out = avalanche_detect_preprocessor(binned)
    md = _metadata(n_channels=4, n_bins=binned.shape[1])

    with pytest.warns(UserWarning):
        alpha, tau, gamma = avalanche_exponents_preprocessor(*out, _metadata=md)

    for arr in (alpha, tau, gamma):
        assert arr.shape == (1,)
    assert np.isfinite(gamma).all()  # gamma is not gated by n_min


def test_exponents_preprocessor_warns_for_short_windows():
    binned = _binned_with_avalanches(n_avalanches=10)
    out = avalanche_detect_preprocessor(binned)
    md = _metadata(n_channels=4, n_bins=binned.shape[1])

    with pytest.warns(UserWarning, match="fewer than n_min=500 avalanches"):
        alpha, tau, _ = avalanche_exponents_preprocessor(*out, _metadata=md)

    assert np.isnan(alpha).all()
    assert np.isnan(tau).all()


def test_exponents_preprocessor_does_not_warn_when_n_min_is_met():
    binned = _binned_with_avalanches(n_avalanches=30)
    out = avalanche_detect_preprocessor(binned)
    md = _metadata(n_channels=4, n_bins=binned.shape[1])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        alpha, _, _ = avalanche_exponents_preprocessor(*out, _metadata=md, n_min=5)

    assert np.isfinite(alpha).all()


def test_exponents_preprocessor_supports_lab_t_max():
    binned = _binned_with_avalanches(n_avalanches=30)
    out = avalanche_detect_preprocessor(binned)
    md = _metadata(n_channels=4, n_bins=binned.shape[1])

    _, tau, _ = avalanche_exponents_preprocessor(
        *out, _metadata=md, n_min=5, t_max_method="lab"
    )
    assert tau.shape == (1,)


def test_exponents_preprocessor_rejects_unknown_t_max_method():
    binned = _binned_with_avalanches(n_avalanches=5)
    out = avalanche_detect_preprocessor(binned)
    md = _metadata(n_channels=4, n_bins=binned.shape[1])

    with pytest.raises(ValueError, match="t_max_method must be 'max' or 'lab'"):
        avalanche_exponents_preprocessor(*out, _metadata=md, t_max_method="nope")


# ── Exponent features ─────────────────────────────────────────────────────────


def test_exponent_features_select_their_own_value():
    alpha = np.array([1.5, 2.0])
    tau = np.array([2.0, 3.0])
    gamma = np.array([1.1, 1.2])

    np.testing.assert_array_equal(avalanche_alpha_exponent(alpha, tau, gamma), alpha)
    np.testing.assert_array_equal(avalanche_tau_exponent(alpha, tau, gamma), tau)
    np.testing.assert_array_equal(avalanche_gamma_exponent(alpha, tau, gamma), gamma)


# ── DCC post-processing ───────────────────────────────────────────────────────


def _exponent_frame():
    return pd.DataFrame(
        {
            "alpha": [1.5, 3.0, 1.0],
            "tau": [2.0, 3.0, 2.0],
            "gamma": [2.5, 1.0, 1.0],
        }
    )


def test_avalanche_dcc_known_values():
    out = avalanche_dcc(_exponent_frame())

    # predicted gamma = (tau - 1) / (alpha - 1) -> 2.0, 1.0, inf
    np.testing.assert_allclose(out["dcc"][:2], [0.5, 0.0])
    assert np.isnan(out["dcc"][2])  # alpha == 1 -> NaN, not a crash


def test_avalanche_dcc_returns_a_copy_by_default():
    df = _exponent_frame()
    out = avalanche_dcc(df)

    assert "dcc" in out.columns
    assert "dcc" not in df.columns


def test_avalanche_dcc_inplace_mutates_and_returns_none():
    df = _exponent_frame()
    result = avalanche_dcc(df, inplace=True)

    assert result is None
    assert "dcc" in df.columns


def test_avalanche_dcc_accepts_custom_column_names():
    df = _exponent_frame().rename(
        columns={"alpha": "ava_alpha", "tau": "ava_tau", "gamma": "ava_gamma"}
    )
    out = avalanche_dcc(
        df,
        alpha_col="ava_alpha",
        tau_col="ava_tau",
        gamma_col="ava_gamma",
        out_col="ava_dcc",
    )
    np.testing.assert_allclose(out["ava_dcc"][:2], [0.5, 0.0])


def test_avalanche_dcc_raises_for_missing_columns():
    df = _exponent_frame().drop(columns=["tau"])

    with pytest.raises(KeyError, match="Column\\(s\\) not found"):
        avalanche_dcc(df)


def test_avalanche_dcc_error_suggests_close_matches():
    df = _exponent_frame().rename(columns={"tau": "taus"})

    with pytest.raises(KeyError, match="did you mean"):
        avalanche_dcc(df)


def test_avalanche_dcc_propagates_nan():
    df = _exponent_frame()
    df.loc[0, "gamma"] = np.nan
    out = avalanche_dcc(df)
    assert np.isnan(out["dcc"][0])


# ── Wiring through FeatureExtractor ───────────────────────────────────────────


def _avalanche_extractor():
    from eegdash.features import FeatureExtractor

    exponents = FeatureExtractor(
        {
            "alpha": avalanche_alpha_exponent,
            "tau": avalanche_tau_exponent,
            "gamma": avalanche_gamma_exponent,
        },
        preprocessor=avalanche_exponents_preprocessor,
    )
    detected = FeatureExtractor(
        {"sigma": avalanche_branching_parameter, 0: exponents},
        preprocessor=avalanche_detect_preprocessor,
    )
    binned = FeatureExtractor({0: detected}, preprocessor=avalanche_bin_preprocessor)
    events = FeatureExtractor({0: binned}, preprocessor=avalanche_events_preprocessor)
    return FeatureExtractor(
        {0: events}, preprocessor=avalanche_standardize_preprocessor
    )


def test_feature_extractor_graph_runs_end_to_end():
    rng = np.random.default_rng(1)
    batch, n_channels = 3, 8
    x = rng.standard_normal((batch, n_channels, 4000))
    md = _metadata(n_channels=n_channels, sfreq=200.0, batch_size=batch)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = _avalanche_extractor()(x, _metadata=md)

    assert sorted(result) == ["alpha", "gamma", "sigma", "tau"]
    for value in result.values():
        assert np.shape(value) == (batch,)

    # the extracted table is what avalanche_dcc is designed to consume
    out = avalanche_dcc(pd.DataFrame(result))
    assert "dcc" in out.columns
    assert len(out) == batch


def test_feature_extractor_rejects_a_wrongly_wired_feature():
    from eegdash.features import FeatureExtractor

    with pytest.raises(TypeError, match="expects one of"):
        FeatureExtractor(
            {"alpha": avalanche_alpha_exponent},
            preprocessor=avalanche_detect_preprocessor,
        )
