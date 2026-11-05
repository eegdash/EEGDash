r"""Avalanche Feature Extraction
============================

This module provides functions to compute neural-avalanche metrics.

Data Shape Convention
---------------------
This module follows a **Network-Level** convention:

* **Input:** ``(batch, channels, time)``
* **Output:** ``(batch,)``

Avalanches are a property of the network as a whole: the binary event array
is summed across channels before binning, so every metric collapses both the
channel and the time dimensions and returns one value per window. All metric
functions are therefore :func:`~eegdash.features.decorators.multivariate_feature`.

Window Length
-------------
The power-law fits behind :func:`avalanche_alpha_exponent` and
:func:`avalanche_tau_exponent` are maximum-likelihood estimates over the
observed avalanche distribution, and they require many avalanches to be
meaningful. The default ``n_min=500`` reflects that.

Typical 2-10 s EEG windows contain far fewer avalanches than this, so
**these exponents are whole-recording-scale quantities**. Prefer long windows
(or whole recordings) when extracting them, and expect a ``UserWarning``
naming the shortfall when the windows are too short.

:func:`avalanche_branching_parameter` has no such requirement and remains
meaningful on short windows.

Deviation from Criticality
--------------------------
The Deviation from Criticality Coefficient (DCC) combines three exponents and
therefore cannot be a node in the extraction graph. It ships instead as the
post-processing function :func:`avalanche_dcc`, which is applied to the
`~eegdash.features.datasets.FeaturesDataset` produced after feature extraction.
"""

import difflib
import warnings

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize_scalar
from scipy.stats import median_abs_deviation

from ..decorators import (
    feature_predecessor,
    metadata_preprocessor,
    multivariate_feature,
    preprocessor_output_type,
)
from ..output_types import SignalOutputType

__all__ = [
    "StandardizedSignalType",
    "avalanche_standardize_preprocessor",
    "avalanche_standardize_mad_preprocessor",
    "avalanche_events_preprocessor",
    "avalanche_bin_preprocessor",
    "avalanche_detect_preprocessor",
    "avalanche_exponents_preprocessor",
    "avalanche_branching_parameter",
    "avalanche_alpha_exponent",
    "avalanche_tau_exponent",
    "avalanche_gamma_exponent",
    "avalanche_dcc",
]

#: Structuring element connecting voxels along the time axis only, so that
#: :func:`scipy.ndimage.label` never merges threshold crossings that occur on
#: different channels or in different windows.
_CHANNEL_AXIS_FILTER = np.zeros((3, 3, 3), dtype=np.int8)
_CHANNEL_AXIS_FILTER[1, 1, :] = 1


class StandardizedSignalType(SignalOutputType):
    r"""Output type for a per-channel standardized signal.

    Marks the output of :func:`avalanche_standardize_preprocessor` and
    :func:`avalanche_standardize_mad_preprocessor` so that the two are
    interchangeable parents.

    """


def _nll(
    exponent: float, x_vals_logs: np.ndarray, sum_log_data: float, n: int
) -> float:
    r"""Negative log-likelihood of a discrete truncated power law.

    The model is :math:`P(x) = x^{-\gamma} / Z` with
    :math:`Z(\gamma) = \sum_{k=x_{min}}^{x_{max}} k^{-\gamma}`, so the
    negative log-likelihood of ``n`` observations is
    :math:`\gamma \sum_i \ln x_i + n \ln Z`.

    Parameters
    ----------
    exponent : float
        The power-law exponent being optimized.
    x_vals_logs : ndarray
        ``log`` of the discrete support of the fitting range.
    sum_log_data : float
        Sum of ``log`` of the observed data points in the fitting range.
    n : int
        Number of observations in the fitting range.

    Returns
    -------
    float
        The negative log-likelihood, or ``np.inf`` if the fit is degenerate.

    """
    if exponent <= 1:
        return np.inf
    z = np.sum(np.exp(-exponent * x_vals_logs))
    if z <= 0 or not np.isfinite(z):
        return np.inf
    return exponent * sum_log_data + n * np.log(z)


def _fit_truncated_power_law(
    data: np.ndarray,
    system_size: int,
    n_min: int = 500,
    cutoff_search_step: int = 1,
    xmin: int = 1,
) -> dict:
    r"""Fit a discrete truncated power law by KS-distance minimization.

    Implements the Clauset et al. [1]_ maximum-likelihood procedure with a
    finite-size cutoff: for every candidate upper cutoff in
    ``[0.8 * system_size, 1.5 * system_size]`` the exponent is estimated by
    maximum likelihood, and the ``(exponent, cutoff)`` pair minimizing the
    Kolmogorov-Smirnov distance between the empirical and theoretical CDFs
    is returned.

    Parameters
    ----------
    data : ndarray
        One-dimensional array of avalanche sizes or durations.
    system_size : int
        Physical upper limit of the system, used to centre the cutoff grid
        (the channel count for sizes, the maximum duration for durations).
    n_min : int, optional
        Minimum number of observations required for a fit (default: 500).
        Fewer observations yield an all-NaN result.
    cutoff_search_step : int, optional
        Step of the cutoff grid search (default: 1).
    xmin : int, optional
        Lower bound of the fitting range (default: 1).

    Returns
    -------
    dict
        Keys ``exponent``, ``xmin``, ``cutoff``, ``ks`` and ``n_included``.
        All are ``np.nan`` (``n_included`` is 0) when the fit fails.

    Raises
    ------
    ValueError
        If `system_size` is not a positive number.

    References
    ----------
    .. [1] Clauset, A., Shalizi, C.R. and Newman, M.E.J., 2009. Power-law
           distributions in empirical data. SIAM Review, 51(4), pp.661-703.

    """
    if system_size is None or system_size <= 0:
        raise ValueError(
            f"system_size must be a positive number, got {system_size!r}. "
            f"It is the physical upper bound of the fitting range (the channel "
            f"count for sizes, the maximum duration for durations)."
        )

    nan_result = {
        "exponent": np.nan,
        "xmin": np.nan,
        "cutoff": np.nan,
        "ks": np.nan,
        "n_included": 0,
    }

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data >= 1)]
    if data.size < n_min:
        return nan_result

    data = np.sort(data)
    max_data = data[-1]
    cutoffs = [
        c
        for c in range(
            int(0.8 * system_size), int(1.5 * system_size) + 1, cutoff_search_step
        )
        if c <= max_data
    ]
    if not cutoffs:
        cutoffs = [int(max_data)]

    best = dict(nan_result)
    best_ks = np.inf
    start_idx = np.searchsorted(data, xmin, side="left")
    log_cumsum = np.insert(np.cumsum(np.log(data)), 0, 0.0)

    for cutoff in cutoffs:
        end_idx = np.searchsorted(data, cutoff, side="right")
        n = end_idx - start_idx
        if n < n_min:
            continue
        support = np.arange(xmin, cutoff + 1)
        x_vals_logs = np.log(support)
        sum_log_data = log_cumsum[end_idx] - log_cumsum[start_idx]
        res = minimize_scalar(
            _nll,
            bounds=(1.0001, 10),
            method="bounded",
            args=(x_vals_logs, sum_log_data, n),
        )
        exponent = float(res.x)
        pdf = np.exp(-exponent * x_vals_logs)
        pdf /= pdf.sum()
        cdf_theory = np.cumsum(pdf)
        cdf_emp = np.searchsorted(data[start_idx:end_idx], support, side="right") / n
        ks = float(np.max(np.abs(cdf_emp - cdf_theory)))
        if ks < best_ks:
            best_ks = ks
            best = {
                "exponent": exponent,
                "xmin": xmin,
                "cutoff": cutoff,
                "ks": ks,
                "n_included": int(n),
            }

    return best


def _avalanche_sizes_durations(bins, starts, ends):
    r"""Compute the size and duration of every avalanche in one window.

    The size of an avalanche is the total activity summed over its bins; its
    duration is the number of bins it spans. Sizes are obtained from a
    cumulative sum so the cost is linear in the number of bins rather than
    quadratic in the number of avalanches.

    Parameters
    ----------
    bins : ndarray
        Binned network activity of a single window, shape ``(n_bins,)``.
    starts : ndarray
        Start bin index of each avalanche.
    ends : ndarray
        End bin index (inclusive) of each avalanche.

    Returns
    -------
    sizes : ndarray
        Total activity of each avalanche, shape ``starts.shape``.
    durations : ndarray
        Number of bins spanned by each avalanche, shape ``starts.shape``.

    """
    cumulative = np.concatenate(([0.0], np.cumsum(bins, dtype=float)))
    sizes = cumulative[ends + 1] - cumulative[starts]
    durations = (ends - starts + 1).astype(np.intp)
    return sizes, durations


@feature_predecessor()
@preprocessor_output_type(StandardizedSignalType)
def avalanche_standardize_preprocessor(x, /, *, epsilon=1e-10):
    r"""Standardize each channel by z-scoring it over time.

    Calculated as:

    .. math:: z = \frac{x - \mu}{\sigma}

    where :math:`\mu` and :math:`\sigma` are taken over the time axis of each
    channel independently.

    Parameters
    ----------
    x : ndarray
        The input signal (shape: batch, channels, time).
    epsilon : float, optional
        Value substituted for a zero standard deviation to avoid dividing by
        zero on flat channels (default: 1e-10).

    Returns
    -------
    ndarray
        The standardized signal. Shape is ``x.shape``.

    See Also
    --------
    avalanche_standardize_mad_preprocessor : Robust alternative.

    """
    mu = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True)
    sigma = np.where(sigma == 0, epsilon, sigma)
    return (x - mu) / sigma


@feature_predecessor()
@preprocessor_output_type(StandardizedSignalType)
def avalanche_standardize_mad_preprocessor(x, /, *, epsilon=1e-10):
    r"""Standardize each channel by its median absolute deviation.

    Calculated as:

    .. math:: z = \frac{x - \mathrm{median}(x)}{\mathrm{MAD}(x)}

    Parameters
    ----------
    x : ndarray
        The input signal (shape: batch, channels, time).
    epsilon : float, optional
        Value substituted for a zero MAD to avoid dividing by zero on flat
        channels (default: 1e-10).

    Returns
    -------
    ndarray
        The standardized signal. Shape is ``x.shape``.

    See Also
    --------
    avalanche_standardize_preprocessor : Standard z-score.

    """
    median = np.median(x, axis=-1, keepdims=True)
    robust_sd = median_abs_deviation(x, axis=-1, scale="normal")[..., None]
    robust_sd = np.where(robust_sd == 0, epsilon, robust_sd)
    return (x - median) / robust_sd


@feature_predecessor(StandardizedSignalType)
def avalanche_events_preprocessor(x, /, *, k=3.0):
    r"""Detect discrete events as peaks of suprathreshold excursions.

    Samples whose absolute standardized amplitude exceeds ``k`` are grouped
    into connected components. Each component then contributes exactly one
    event, placed at the sample where the absolute amplitude peaks.

    Parameters
    ----------
    x : ndarray
        The standardized signal (shape: batch, channels, time).
    k : float, optional
        Threshold in units of the standardization, i.e. standard deviations
        or scaled MADs (default: 3.0).

    Returns
    -------
    ndarray
        Binary event array of dtype ``uint8``, with a 1 at the peak of each
        detected excursion and 0 elsewhere. Shape is ``x.shape``.

    Notes
    -----
    Marking only the peak, rather than every suprathreshold sample, keeps one
    long excursion from being counted as many events and makes the event rate
    insensitive to the sampling frequency.

    References
    ----------
    .. [1] Shriki, O. et al., 2013. Neuronal avalanches in the resting MEG of
           the human brain. Journal of Neuroscience, 33(16), pp.7079-7090.

    """
    abs_data = np.abs(x)
    mask = (abs_data > k).astype(np.uint8)
    labels, num_features = ndi.label(mask, structure=_CHANNEL_AXIS_FILTER)

    binary_data = np.zeros(x.shape, dtype=np.uint8)
    if num_features > 0:
        positions = ndi.maximum_position(
            abs_data, labels, index=np.arange(1, num_features + 1)
        )
        i_batch, i_channel, i_time = zip(*positions)
        binary_data[i_batch, i_channel, i_time] = 1
    return binary_data


@feature_predecessor(avalanche_events_preprocessor)
@metadata_preprocessor
def avalanche_bin_preprocessor(x, /, *, _metadata, bin_size_ms=5.0):
    r"""Collapse events across channels and bin them in time.

    The binary event array is summed over channels to give the network
    activity, which is then partitioned into non-overlapping bins. A trailing
    partial bin is discarded.

    Parameters
    ----------
    x : ndarray
        The binary event array (shape: batch, channels, time).
    _metadata : dict
        Metadata dictionary. ``_metadata["info"]["sfreq"]`` is read, and the
        keys ``"bin_size_samples"``, ``"bin_size_sec"`` and ``"n_bins"`` are
        added.
    bin_size_ms : float, optional
        Target bin width in milliseconds (default: 5.0). It is rounded to the
        nearest whole number of samples, so the realized width may differ from
        the request; it is recorded in ``_metadata["bin_size_samples"]`` and
        ``_metadata["bin_size_sec"]``.

    Returns
    -------
    binned : ndarray
        Network activity per bin, shape ``(batch, n_bins)``.
    _metadata : dict
        Metadata updated with ``"bin_size_samples"`` (the realized bin width
        in samples), ``"bin_size_sec"`` (the same width in seconds) and
        ``"n_bins"``.

    Raises
    ------
    ValueError
        If `bin_size_ms` rounds to less than one sample at the recording's
        sampling frequency, or if it is longer than the window itself.

    Notes
    -----
    The bin width sets the time scale at which separate events are considered
    part of the same cascade, so it directly controls how many avalanches are
    found. Widening it merges cascades and produces fewer, larger avalanches.

    Avalanche durations are counted in bins downstream, so
    ``_metadata["bin_size_sec"]`` is what converts them to seconds. The
    exponent :math:`\tau` is unaffected, being scale-invariant.

    """
    fs = _metadata["info"]["sfreq"]
    n_samples = x.shape[-1]

    bin_size_samples = int(np.round(bin_size_ms * fs / 1000.0))
    if bin_size_samples < 1:
        raise ValueError(
            f"bin_size_ms={bin_size_ms} rounds to {bin_size_samples} samples at "
            f"fs={fs} Hz, but at least 1 sample is required. Use a bin width of "
            f"at least {1000.0 / fs:.3f} ms."
        )

    n_bins = n_samples // bin_size_samples
    if n_bins == 0:
        raise ValueError(
            f"bin_size_ms={bin_size_ms} ({bin_size_samples} samples at fs={fs} Hz) "
            f"is longer than the {n_samples}-sample window, so no complete bin "
            f"fits. Use a smaller bin width or longer windows."
        )

    network_activity = x.sum(axis=-2)
    trimmed = network_activity[:, : n_bins * bin_size_samples]
    binned = trimmed.reshape(-1, n_bins, bin_size_samples).sum(axis=-1)

    _metadata["bin_size_samples"] = bin_size_samples
    _metadata["bin_size_sec"] = bin_size_samples / fs
    _metadata["n_bins"] = n_bins
    return binned, _metadata


@feature_predecessor(avalanche_bin_preprocessor)
def avalanche_detect_preprocessor(x, /):
    r"""Locate avalanches as maximal runs of active bins.

    An avalanche is a maximal run of consecutive non-empty bins, bracketed by
    an empty bin on each side. Avalanches touching the first or last bin of a
    window are discarded, because the window boundary truncates them to an
    unknown extent.

    Parameters
    ----------
    x : ndarray
        Binned network activity, shape ``(batch, n_bins)``.

    Returns
    -------
    binned : ndarray
        The input, passed through for downstream metrics.
    starts : ndarray
        Start bin index of each avalanche, shape ``(batch, max_avalanches)``,
        right-padded with -1.
    ends : ndarray
        End bin index of each avalanche, inclusive, padded with -1.
    counts : ndarray
        Number of valid avalanches per window, shape ``(batch,)``.

    Notes
    -----
    ``starts`` and ``ends`` are rectangular because a window batch must share
    one array shape. The padding is **not** masked arithmetically: every
    consumer must slice with ``starts[i, : counts[i]]``.

    """
    batch_size, n_bins = x.shape
    starts_list, ends_list = [], []
    counts = np.zeros(batch_size, dtype=np.intp)

    for i in range(batch_size):
        is_active = x[i] > 0
        if not is_active.any():
            starts_list.append(np.empty(0, dtype=np.intp))
            ends_list.append(np.empty(0, dtype=np.intp))
            continue

        diffs = np.diff(is_active.astype(np.intp), prepend=0, append=0)
        s = np.flatnonzero(diffs == 1)
        e = np.flatnonzero(diffs == -1) - 1

        if s.size and s[0] == 0:
            s, e = s[1:], e[1:]
        if e.size and e[-1] == n_bins - 1:
            s, e = s[:-1], e[:-1]

        starts_list.append(s)
        ends_list.append(e)
        counts[i] = s.size

    n_columns = max(int(counts.max()) if counts.size else 0, 1)
    starts = np.full((batch_size, n_columns), -1, dtype=np.intp)
    ends = np.full((batch_size, n_columns), -1, dtype=np.intp)
    for i, (s, e) in enumerate(zip(starts_list, ends_list)):
        if s.size:
            starts[i, : s.size] = s
            ends[i, : s.size] = e

    return x, starts, ends, counts


@feature_predecessor(avalanche_detect_preprocessor)
def avalanche_exponents_preprocessor(
    binned,
    starts,
    ends,
    counts,
    /,
    *,
    _metadata,
    n_min=500,
    t_max_method="max",
    min_unique_durations=3,
    epsilon=1e-10,
):
    r"""Fit the three criticality exponents of every window.

    Computes :math:`\alpha`, :math:`\tau` and :math:`\gamma` in a single pass
    so that the avalanche sizes, durations and the power-law fits are each
    evaluated once and shared by the three metric functions downstream.

    Parameters
    ----------
    binned : ndarray
        Binned network activity, shape ``(batch, n_bins)``.
    starts, ends : ndarray
        Padded avalanche index arrays from
        :func:`avalanche_detect_preprocessor`.
    counts : ndarray
        Number of valid avalanches per window.
    _metadata : dict
        Metadata dictionary. ``_metadata["info"]["ch_names"]`` supplies the
        system size of the size fit, and ``_metadata["n_bins"]`` is used by
        the ``"lab"`` value of `t_max_method`.
    n_min : int, optional
        Minimum number of avalanches required before :math:`\alpha` and
        :math:`\tau` are fitted (default: 500). See the module's
        **Window Length** section.
    t_max_method : {'max', 'lab'}, optional
        Upper bound of the duration fitting range. ``"max"`` uses the longest
        observed duration; ``"lab"`` uses the theoretical bound
        :math:`\sqrt{S_{max} / \bar{a}}` derived from the largest avalanche
        size and the mean per-bin activity.
    min_unique_durations : int, optional
        Minimum number of distinct durations required to regress
        :math:`\gamma` (default: 3).
    epsilon : float, optional
        Floor on the mean activity in the ``"lab"`` bound (default: 1e-10).

    Returns
    -------
    alpha : ndarray
        Size exponents, shape ``(batch,)``.
    tau : ndarray
        Duration exponents, shape ``(batch,)``.
    gamma : ndarray
        Size-duration scaling exponents, shape ``(batch,)``.

    Raises
    ------
    ValueError
        If `t_max_method` is not ``"max"`` or ``"lab"``.

    Warns
    -----
    UserWarning
        If any window holds fewer than `n_min` avalanches, in which case its
        :math:`\alpha` and :math:`\tau` are ``np.nan``.

    """
    if t_max_method not in ("max", "lab"):
        raise ValueError(
            f"t_max_method must be 'max' or 'lab', got {t_max_method!r}. "
            f"'max' uses the longest observed duration; 'lab' uses the "
            f"theoretical bound from the largest size and the mean activity."
        )

    batch_size = binned.shape[0]
    n_channels = len(_metadata["info"]["ch_names"])
    n_bins = _metadata.get("n_bins", binned.shape[1])

    alpha = np.full(batch_size, np.nan, dtype=float)
    tau = np.full(batch_size, np.nan, dtype=float)
    gamma = np.full(batch_size, np.nan, dtype=float)

    for i in range(batch_size):
        n_avalanches = int(counts[i])
        if n_avalanches == 0:
            continue
        s = starts[i, :n_avalanches]
        e = ends[i, :n_avalanches]
        sizes, durations = _avalanche_sizes_durations(binned[i], s, e)

        alpha[i] = _fit_truncated_power_law(sizes, system_size=n_channels, n_min=n_min)[
            "exponent"
        ]

        if t_max_method == "max":
            t_max = int(durations.max())
        else:
            mean_activity = binned[i].sum() / n_bins
            t_max = max(1, int(np.sqrt(sizes.max() / max(mean_activity, epsilon))))
        tau[i] = _fit_truncated_power_law(durations, system_size=t_max, n_min=n_min)[
            "exponent"
        ]

        gamma[i] = _scaling_exponent(sizes, durations, min_unique_durations)

    _warn_if_windows_too_short(counts, n_min)
    return alpha, tau, gamma


def _scaling_exponent(sizes, durations, min_unique_durations):
    r"""Regress the size-duration scaling exponent of one window.

    Fits a straight line to :math:`\log_{10}` mean size against
    :math:`\log_{10}` duration over the distinct observed durations. This is
    an ordinary least-squares fit, not a maximum-likelihood estimate.

    Parameters
    ----------
    sizes : ndarray
        Avalanche sizes of one window.
    durations : ndarray
        Avalanche durations of one window.
    min_unique_durations : int
        Minimum number of distinct durations required for the regression.

    Returns
    -------
    float
        The scaling exponent, or ``np.nan`` if there are too few distinct
        durations, a mean size is zero, or the fit is degenerate.

    """
    unique_durations = np.unique(durations)
    if unique_durations.size < min_unique_durations:
        return np.nan

    mean_sizes = np.array([sizes[durations == t].mean() for t in unique_durations])
    if np.any(mean_sizes == 0):
        return np.nan

    try:
        slope = np.polyfit(
            np.log10(unique_durations.astype(float)), np.log10(mean_sizes), 1
        )[0]
    except np.linalg.LinAlgError:
        return np.nan
    return float(slope)


def _warn_if_windows_too_short(counts, n_min):
    r"""Warn once when windows hold too few avalanches to fit an exponent.

    Parameters
    ----------
    counts : ndarray
        Number of avalanches per window.
    n_min : int
        The minimum required by :func:`_fit_truncated_power_law`.

    """
    n_short = int(np.count_nonzero(counts < n_min))
    if n_short == 0:
        return
    warnings.warn(
        f"{n_short} of {counts.size} window(s) contain fewer than n_min={n_min} "
        f"avalanches (observed range: {int(counts.min())}-{int(counts.max())}), "
        f"so their alpha and tau exponents are NaN. These exponents are "
        f"whole-recording-scale quantities: use longer windows, widen "
        f"bin_size_ms to merge cascades, or lower n_min if you accept a less "
        f"reliable fit.",
        UserWarning,
        stacklevel=3,
    )


@feature_predecessor(avalanche_detect_preprocessor)
@multivariate_feature
def avalanche_branching_parameter(binned, starts, ends, counts, /):
    r"""Branching parameter (:math:`\sigma`) of the avalanche activity.

    Estimates how many events a single event triggers in the next time bin,
    averaged over the avalanches of a window [1]_:

    .. math:: \sigma = \left\langle \frac{n_{d}}{n_{a}} \right\rangle

    where :math:`n_a` is the activity in an avalanche's first bin and
    :math:`n_d` the activity in its second. A critical system has
    :math:`\sigma \approx 1`; subcritical dynamics give :math:`\sigma < 1`
    and supercritical dynamics :math:`\sigma > 1`.

    Parameters
    ----------
    binned : ndarray
        Binned network activity, shape ``(batch, n_bins)``.
    starts, ends : ndarray
        Padded avalanche index arrays from
        :func:`avalanche_detect_preprocessor`. Only `starts` is used.
    counts : ndarray
        Number of valid avalanches per window.

    Returns
    -------
    ndarray
        :math:`\sigma` per window, ``np.nan`` where a window holds no
        avalanche. Shape is ``(batch,)``.

    Notes
    -----
    Unlike the power-law exponents this estimate needs only a handful of
    avalanches, so it stays meaningful on short windows. Single-bin
    avalanches contribute an exact ratio of 0 and so bias :math:`\sigma`
    downwards when the bin width is large relative to the event rate.

    References
    ----------
    .. [1] Beggs, J.M. and Plenz, D., 2003. Neuronal avalanches in neocortical
           circuits. Journal of Neuroscience, 23(35), pp.11167-11177.

    """
    batch_size = binned.shape[0]
    sigma = np.full(batch_size, np.nan, dtype=float)

    for i in range(batch_size):
        n_avalanches = int(counts[i])
        if n_avalanches == 0:
            continue
        s = starts[i, :n_avalanches]
        ancestors = binned[i, s].astype(float)
        descendants = binned[i, s + 1].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma[i] = float(np.nanmean(descendants / ancestors))

    return sigma


@feature_predecessor(avalanche_exponents_preprocessor)
@multivariate_feature
def avalanche_alpha_exponent(alpha, tau, gamma, /):
    r"""Power-law exponent (:math:`\alpha`) of the avalanche size distribution.

    The avalanche sizes of a critical system follow
    :math:`P(S) \sim S^{-\alpha}`, with :math:`\alpha \approx 1.5` for the
    mean-field branching process [1]_. The exponent is estimated by maximum
    likelihood over a truncated discrete power law, with the channel count as
    the system size.

    Parameters
    ----------
    alpha, tau, gamma : ndarray
        The exponents computed by
        :func:`avalanche_exponents_preprocessor`, shape ``(batch,)`` each.

    Returns
    -------
    ndarray
        :math:`\alpha` per window. Shape is ``(batch,)``.

    Notes
    -----
    Returns ``np.nan`` for any window holding fewer than ``n_min`` avalanches;
    see the module's **Window Length** section.

    References
    ----------
    .. [1] Beggs, J.M. and Plenz, D., 2003. Neuronal avalanches in neocortical
           circuits. Journal of Neuroscience, 23(35), pp.11167-11177.

    See Also
    --------
    avalanche_tau_exponent : Duration counterpart.
    avalanche_dcc : Combines the three exponents.

    """
    return alpha


@feature_predecessor(avalanche_exponents_preprocessor)
@multivariate_feature
def avalanche_tau_exponent(alpha, tau, gamma, /):
    r"""Power-law exponent (:math:`\tau`) of the avalanche duration distribution.

    The avalanche durations of a critical system follow
    :math:`P(T) \sim T^{-\tau}`, with :math:`\tau \approx 2` for the
    mean-field branching process [1]_. The exponent is estimated by maximum
    likelihood over a truncated discrete power law.

    Parameters
    ----------
    alpha, tau, gamma : ndarray
        The exponents computed by
        :func:`avalanche_exponents_preprocessor`, shape ``(batch,)`` each.

    Returns
    -------
    ndarray
        :math:`\tau` per window. Shape is ``(batch,)``.

    Notes
    -----
    Durations are fitted in units of bins, so the exponent does not depend on
    the bin width. Returns ``np.nan`` for any window holding fewer than
    ``n_min`` avalanches; see the module's **Window Length** section.

    References
    ----------
    .. [1] Beggs, J.M. and Plenz, D., 2003. Neuronal avalanches in neocortical
           circuits. Journal of Neuroscience, 23(35), pp.11167-11177.

    See Also
    --------
    avalanche_alpha_exponent : Size counterpart.
    avalanche_dcc : Combines the three exponents.

    """
    return tau


@feature_predecessor(avalanche_exponents_preprocessor)
@multivariate_feature
def avalanche_gamma_exponent(alpha, tau, gamma, /):
    r"""Scaling exponent (:math:`\gamma`) between avalanche size and duration.

    At criticality the mean avalanche size grows as a power of its duration,
    :math:`\langle S \rangle (T) \sim T^{\gamma}` [1]_. The exponent is the
    slope of an ordinary least-squares fit in log-log space, not a
    maximum-likelihood estimate.

    Parameters
    ----------
    alpha, tau, gamma : ndarray
        The exponents computed by
        :func:`avalanche_exponents_preprocessor`, shape ``(batch,)`` each.

    Returns
    -------
    ndarray
        :math:`\gamma` per window. Shape is ``(batch,)``.

    Notes
    -----
    Unlike :math:`\alpha` and :math:`\tau` this estimate is not gated by
    ``n_min``; it needs only ``min_unique_durations`` distinct durations.
    It is nonetheless unreliable on short windows, where few durations are
    observed.

    References
    ----------
    .. [1] Sethna, J.P., Dahmen, K.A. and Myers, C.R., 2001. Crackling noise.
           Nature, 410(6825), pp.242-250.

    See Also
    --------
    avalanche_dcc : Compares this exponent with the one predicted by
        :math:`\alpha` and :math:`\tau`.

    """
    return gamma


def avalanche_dcc(
    df,
    /,
    *,
    alpha_col="alpha",
    tau_col="tau",
    gamma_col="gamma",
    out_col="dcc",
    inplace=False,
):
    r"""Add the Deviation from Criticality Coefficient to a feature table.

    The crackling-noise relation predicts the scaling exponent from the size
    and duration exponents; the DCC is how far the observed exponent falls
    from that prediction [1]_:

    .. math::

        \mathrm{DCC} = \left| \gamma_{obs}
                       - \frac{\tau - 1}{\alpha - 1} \right|

    A value near 0 indicates a system close to criticality.

    This is a **post-processing step, not a feature**. The DCC combines three
    exponents, but a feature in this framework has exactly one predecessor and
    its result is terminal, so the coefficient cannot be a node in the
    extraction graph. Apply it to the table produced after extraction:

    .. code-block:: python

        features = extract_features(windows_ds, extractor)
        df = avalanche_dcc(features.to_dataframe())

    Parameters
    ----------
    df : pandas.DataFrame
        A feature table holding the three exponent columns.
    alpha_col, tau_col, gamma_col : str, optional
        Names of the exponent columns. The defaults assume the
        :class:`~eegdash.features.extractors.FeatureExtractor` was keyed with
        ``"alpha"``, ``"tau"`` and ``"gamma"``; feature column names come from
        those dictionary keys, not from the function names, so pass the names
        you used.
    out_col : str, optional
        Name of the column to add (default: ``"dcc"``).
    inplace : bool, optional
        If True, add the column to `df` and return None. If False (default),
        operate on a copy and return it.

    Returns
    -------
    pandas.DataFrame or None
        The table with `out_col` added, or None when ``inplace=True``.

    Raises
    ------
    KeyError
        If any of the three exponent columns is missing from `df`.

    Notes
    -----
    Windows whose :math:`\alpha` is exactly 1 yield ``np.nan`` rather than a
    division error. An all-NaN output column almost always means the windows
    were too short for the power-law fits rather than that the data is
    non-critical; see the module's **Window Length** section.

    References
    ----------
    .. [1] Ma, Z. et al., 2019. Cortical circuit dynamics are homeostatically
           tuned to criticality in vivo. Neuron, 104(4), pp.655-664.

    See Also
    --------
    avalanche_alpha_exponent, avalanche_tau_exponent, avalanche_gamma_exponent

    """
    missing = [c for c in (alpha_col, tau_col, gamma_col) if c not in df.columns]
    if missing:
        available = list(df.columns)
        hints = []
        for column in missing:
            close = difflib.get_close_matches(column, available, n=3)
            hints.append(f"{column!r}{f' (did you mean {close}?)' if close else ''}")
        raise KeyError(
            f"Column(s) not found in the DataFrame: {', '.join(hints)}. "
            f"Available columns: {available}. Feature column names come from the "
            f"keys of the FeatureExtractor dictionary, so pass alpha_col, tau_col "
            f"and gamma_col explicitly if you used different keys."
        )

    out = df if inplace else df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = (out[tau_col] - 1.0) / (out[alpha_col] - 1.0)
        out[out_col] = (out[gamma_col] - predicted).abs()
    out[out_col] = out[out_col].replace([np.inf, -np.inf], np.nan)

    return None if inplace else out
