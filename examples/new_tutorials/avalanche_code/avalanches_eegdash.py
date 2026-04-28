r"""Avalanche feature extraction — EEGDash-compatible implementation.

Preprocessor chain (each step produces input for the next):

    signal (batch, ch, time)
        │  avalanche_preprocessor
        ▼
    binary events (batch, ch, time) uint8
        │  bin_avalanches          [adds bin_size_sec + n_bins to _metadata]
        ▼
    network activity (batch, n_bins) float
        │  detect_avalanches
        ▼
    padded index arrays  →  branching_parameter
                         →  alpha_exponent
                         →  tau_exponent
                         →  gamma_exponent

Post-processing utility (not an EEGDash feature — features cannot depend
on other features in the current API):

    dcc(alphas, taus, gammas)  →  Deviation from Criticality Coefficient

Usage
-----
::

    from functools import partial
    from eegdash.features import FeatureExtractor, extract_features
    from avalanches_eegdash import (
        avalanche_preprocessor, bin_avalanches, detect_avalanches,
        branching_parameter, alpha_exponent, tau_exponent, gamma_exponent,
        dcc,
    )

    extractor = FeatureExtractor(
        {"": FeatureExtractor(
            {"": FeatureExtractor(
                {
                    "branching": branching_parameter,
                    "alpha": partial(alpha_exponent, n_channels=n_ch),
                    "tau": tau_exponent,
                    "gamma": gamma_exponent,
                },
                preprocessor=detect_avalanches,
            )},
            preprocessor=partial(bin_avalanches, bin_size_samples=bin_size_samples),
        )},
        preprocessor=partial(avalanche_preprocessor, k=3.0),
    )
    features_ds = extract_features(windows_ds, extractor, batch_size=512, n_jobs=-1)
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize_scalar
from scipy.stats import median_abs_deviation

from eegdash.features import (
    feature_predecessor,
    metadata_preprocessor,
    multivariate_feature,
    preprocessor_output_type,
)
from eegdash.features.output_types import SignalOutputType

# ── Custom output types ───────────────────────────────────────────────────────


class StandardizedSignalType(SignalOutputType):
    """Standardized signal: (batch, channels, time) float."""

# ── Constants ─────────────────────────────────────────────────────────────────

CHANNEL_AXIS_FILTER = np.zeros((3, 3, 3), dtype=np.int8)
CHANNEL_AXIS_FILTER[1, 1, :] = 1

# ── Power-law fitting utilities ───────────────────────────────────────────────


def _nll(exponent: float,
         x_vals_logs: np.ndarray,
         sum_log_data: float,
         n: int) -> float:
    r"""Negative log-likelihood for a discrete truncated power-law.

    Parameters
    ----------
    exponent : float
        Power-law exponent :math:`\gamma` being optimised.
    x_vals_logs : np.ndarray
        ``log`` of the discrete x values in the fitting range.
    sum_log_data : float
        Sum of ``log`` of the observed data points.
    n : int
        Number of data points in the fitting range.

    Returns
    -------
    float
        Negative log-likelihood; ``np.inf`` if the fit is degenerate.

    """
    if exponent <= 1:
        return np.inf
    Z = np.sum(np.exp(-exponent * x_vals_logs))
    if Z <= 0 or not np.isfinite(Z):
        return np.inf
    return exponent * sum_log_data + n * np.log(Z)


def _fit_truncated_power_law(
    data: np.ndarray,
    system_size: int,
    n_min: int = 500,
    cutoff_search_step: int = 1,
    xmin: int = 1,
) -> dict:
    r"""Truncated MLE power-law fit using KS-distance minimisation.

    Implements the Fekete-style algorithm (Clauset et al. 2009 with
    finite-size corrections). Searches for the optimal fitting window
    ``[xmin, xmax]`` by minimising the Kolmogorov-Smirnov distance.

    Parameters
    ----------
    data : np.ndarray
        1-D array of avalanche metrics (sizes or durations).
    system_size : int
        Physical upper limit of the system (e.g., number of channels).
    n_min : int
        Minimum number of points required for a reliable fit.
    cutoff_search_step : int
        Step size for the cutoff grid search.
    xmin : int
        Lower bound of the fitting range (≥ 1).

    Returns
    -------
    dict
        ``exponent``, ``xmin``, ``cutoff``, ``ks``, ``n_included``.
        All set to ``np.nan`` / 0 if the fit fails.

    """
    assert system_size is not None and system_size > 0

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data >= 1)]

    _nan = {"exponent": np.nan, "xmin": np.nan, "cutoff": np.nan, "ks": np.nan, "n_included": 0}
    if data.size < n_min:
        return _nan

    data = np.sort(data)
    maxdata = data[-1]
    lower_bound = int(0.8 * system_size)
    upper_bound = int(1.5 * system_size)
    cutoff_valid = [c for c in range(lower_bound, upper_bound + 1, cutoff_search_step) if c <= maxdata]
    if not cutoff_valid:
        cutoff_valid = [int(maxdata)]

    best_ks = np.inf
    best_params = dict(_nan)
    start_idx = np.searchsorted(data, xmin, side="left")
    log_data_cumsum = np.insert(np.cumsum(np.log(data)), 0, 0.0)

    for cutoff in cutoff_valid:
        end_idx = np.searchsorted(data, cutoff, side="right")
        n = end_idx - start_idx
        if n < n_min:
            continue
        x_vals_logs = np.log(np.arange(xmin, cutoff + 1))
        sum_log_data = log_data_cumsum[end_idx] - log_data_cumsum[start_idx]
        res = minimize_scalar(_nll, bounds=(1.0001, 10), method="bounded",
                              args=(x_vals_logs, sum_log_data, n))
        exp = float(res.x)
        pdf = np.exp(-exp * x_vals_logs)
        pdf /= pdf.sum()
        cdf_theory = np.cumsum(pdf)
        cdf_emp = np.searchsorted(data[start_idx:end_idx], np.arange(xmin, cutoff + 1), side="right") / n
        ks = np.max(np.abs(cdf_emp - cdf_theory))
        if ks < best_ks:
            best_ks = ks
            best_params = {"exponent": exp, "xmin": xmin, "cutoff": cutoff, "ks": ks, "n_included": n}

    return best_params


# ── Preprocessors ─────────────────────────────────────────────────────────────

@feature_predecessor(SignalOutputType)
@preprocessor_output_type(StandardizedSignalType)
def standardize_signal(x, /, *,
                       epsilon: float = 1e-10,):
    r"""Standardize signal by z-scoring each channel independently."""
    mu = x.mean(axis=2, keepdims=True)
    sigma = x.std(axis=2, keepdims=True)
    sigma[sigma == 0] = epsilon
    return (x - mu) / sigma


@feature_predecessor(SignalOutputType)
@preprocessor_output_type(StandardizedSignalType)
def standardize_signal_mad(x, /, *,
                           epsilon: float = 1e-10,):
    r"""Standardize signal by MAD-normalising each channel independently."""
    med = np.median(x, axis=2, keepdims=True)
    robust_sd = median_abs_deviation(x, axis=2, scale="normal", keepdims=True)
    robust_sd[robust_sd == 0] = epsilon
    return (x - med) / robust_sd


@feature_predecessor(StandardizedSignalType)
def detect_events_preprocessor(x, /, *,
                               k: float = 3.0,):
    r"""Detect events by thresholding and peak-picking.

    Operates on a 3-D batch. Each channel is z-scored (or MAD-normalised)
    independently; threshold crossings are labelled as connected components
    along the **time axis only**; the sample with the highest absolute
    z-score within each component is marked as the event peak.

    Parameters
    ----------
    x : np.ndarray, shape (batch, channels, time)
        Raw signal batch.
    k : float
        Threshold multiplier (default 3 σ or 3 MAD units).
    _metadata : dict
        EEGDash metadata dict (passed through unchanged).

    Returns
    -------
    binary_data : np.ndarray, shape (batch, channels, time), dtype uint8
        1 at the absolute peak of each detected event, 0 elsewhere.
    _metadata : dict
        Unchanged metadata (required by ``@metadata_preprocessor``).

    """
    abs_data = np.abs(x)
    mask = (abs_data > k).astype(np.uint8)
    labels, num_features = ndi.label(mask, structure=CHANNEL_AXIS_FILTER)

    binary_data = np.zeros(x.shape, dtype=np.uint8)
    if num_features > 0:
        positions = ndi.maximum_position(abs_data, labels, index=np.arange(1, num_features + 1))
        binary_data[zip(*positions)] = 1 # TODO: verify

    return binary_data

@feature_predecessor(detect_events_preprocessor)
@metadata_preprocessor
def bin_avalanches(binary_data, /, *,
                   bin_size_msec: float = 5.0,
                   _metadata: dict):
    r"""Bin network-level avalanche events into contiguous time bins.

    Sums the binary event array across channels to obtain network activity,
    then partitions it into non-overlapping bins of ``bin_size_samples``
    samples. Adds ``bin_size_sec`` and ``n_bins`` to ``_metadata`` so
    downstream features can use them.

    Parameters
    ----------
    binary_data : np.ndarray, shape (batch, channels, time), dtype uint8
        Output of :func:`avalanche_preprocessor`.
    bin_size_msec : float, optional
        Desired size of each bin in milliseconds. If not compatible with the sampling frequency,
        the actual bin size will be adjusted to the nearest integer number of samples.
    _metadata : dict
        EEGDash metadata dict; updated in-place copy is returned.

    Returns
    -------
    binned_array : np.ndarray, shape (batch, n_bins)
        Summed network activity per bin.
    _metadata : dict
        Updated with ``"bin_size_sec"`` and ``"n_bins"`` keys.

    """
    fs = _metadata["info"]["sfreq"]
    n_samples = binary_data.shape[-1]

    bin_size_samples = int(np.round(bin_size_msec * fs / 1000.0))
    if bin_size_samples < 1:
        raise ValueError(f"bin_size_msec={bin_size_msec} is too small for the sampling frequency ({fs} Hz)." +
                         "\nUse a larger bin size or check the sampling frequency.")

    actual_n_bins = n_samples // bin_size_samples
    if actual_n_bins == 0:
        raise ValueError(
            f"bin_size_samples={bin_size_samples} is larger than the window "
            f"length ({n_samples} samples). Use a smaller bin size."
        )

    # sum across channels → network activity (batch, time)
    network_activity = binary_data.sum(axis=-2)
    trimmed = network_activity[:, : actual_n_bins * bin_size_samples]
    binned_array = trimmed.reshape(-1, actual_n_bins, bin_size_samples).sum(axis=-1)

    _metadata["bin_size_samples"] = bin_size_samples
    _metadata["n_bins"] = actual_n_bins
    return binned_array, _metadata


@feature_predecessor(bin_avalanches)
def detect_avalanches(binned_array, /):
    r"""Detect avalanche start and end indices in the binned activity.

    An avalanche is a maximal contiguous run of active bins (activity > 0).
    Edge avalanches (touching the first or last bin) are discarded to avoid
    boundary artefacts.

    Returns padded index arrays so all windows share the same shape.
    Padding value is ``-1``; valid entries are ``indices[:n]``.

    Parameters
    ----------
    binned_array : np.ndarray, shape (batch, n_bins)
        Output of :func:`bin_avalanches`.

    Returns
    -------
    binned_array : np.ndarray, shape (batch, n_bins)
        Passed through unchanged (needed by downstream features).
    starts : np.ndarray, shape (batch, max_avalanches), dtype int
        Start bin index of each avalanche; padded with ``-1``.
    ends : np.ndarray, shape (batch, max_avalanches), dtype int
        End bin index (inclusive) of each avalanche; padded with ``-1``.
    counts : np.ndarray, shape (batch,), dtype int
        Actual number of avalanches per window.

    """
    batch_size, n_bins = binned_array.shape
    starts_list, ends_list = [], []
    counts = np.zeros(batch_size, dtype=np.intp) # empty?

    for i in range(batch_size):
        bins = binned_array[i]
        is_active = (bins > 0)

        if not is_active.any():
            starts_list.append(np.empty(0, dtype=np.intp))
            ends_list.append(np.empty(0, dtype=np.intp))
            continue

        diffs = np.diff(is_active, prepend=0, append=0)
        s = np.where(diffs == 1)[0]
        e = np.where(diffs == -1)[0] - 1  # inclusive end

        # discard edge-touching avalanches
        if len(s) and s[0] == 0:
            s, e = s[1:], e[1:]
        if len(e) and e[-1] == n_bins - 1:
            s, e = s[:-1], e[:-1]

        starts_list.append(s)
        ends_list.append(e)
        counts[i] = len(s)

    max_av = int(counts.max()) if counts.any() else 0
    pad = max(max_av, 1)  # always at least 1 column so shape is never (batch, 0)
    starts = np.full((batch_size, pad), -1, dtype=np.intp)
    ends = np.full((batch_size, pad), -1, dtype=np.intp)
    for i, (s, e) in enumerate(zip(starts_list, ends_list)):
        n = len(s)
        if n:
            starts[i, :n] = s
            ends[i, :n] = e

    return binned_array, starts, ends, counts


# ── Feature functions ─────────────────────────────────────────────────────────


@feature_predecessor(detect_avalanches)
@multivariate_feature
def branching_parameter(binned_array,
                        starts,
                        counts, /,):
    r"""Branching parameter (σ) of the avalanche activity.

    Estimates how many events in bin *t+1* are triggered by one event in
    bin *t* (on average across all avalanches in a window).

    Parameters
    ----------
    binned_array : np.ndarray, shape (batch, n_bins)
    starts, ends : np.ndarray, shape (batch, max_avalanches)
        Padded index arrays from :func:`detect_avalanches`.
    counts : np.ndarray, shape (batch,)
        Number of valid avalanches per window.
    method : {'naive', 'weighted'}
        * ``'naive'``: mean of per-avalanche ratios :math:`n_d / n_a`.
        * ``'weighted'``: ratio of sums :math:`\sum n_d / \sum n_a`.

    Returns
    -------
    np.ndarray, shape (batch,)
        σ per window; ``np.nan`` for windows with no avalanches.

    References
    ----------
    Beggs & Plenz, *J. Neurosci.* 23(35), 11167–11177 (2003).

    """
    batch_size = binned_array.shape[0]
    sigmas = np.full(batch_size, np.nan, dtype=float)

    for i in range(batch_size):
        n = int(counts[i])
        if n == 0:
            continue
        s = starts[i, :n]
        n_a = binned_array[i, s].astype(float)
        n_d = binned_array[i, s + 1].astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
                sigmas[i] = float(np.nanmean(n_d / n_a))

    return sigmas


@feature_predecessor(detect_avalanches)
@multivariate_feature
def alpha_exponent(
    binned_array,
    starts,
    ends,
    counts,
    /, *,
    ks_threshold: float = 0.1,
    _metadata: dict = None,
):
    r"""Alpha exponent of the avalanche *size* distribution.

    Fits a truncated discrete power-law to the distribution of avalanche
    sizes (total activity summed over all bins within each avalanche).

    Parameters
    ----------
    n_channels : int, optional
        Physical system size (number of channels). Used as the upper bound
        of the fitting range. Inferred from ``_metadata`` if not provided.
    ks_threshold : float
        Maximum KS distance for the fit to be considered reliable.

    Returns
    -------
    np.ndarray, shape (batch,)
        α per window; ``np.nan`` if the fit is unreliable or has too few
        avalanches.

    """
    n_channels = len(_metadata["info"]["ch_names"])

    batch_size = binned_array.shape[0]
    alphas = np.full(batch_size, np.nan, dtype=float)

    for i in range(batch_size):
        n_av = int(counts[i])
        if n_av == 0:
            continue
        s = starts[i, :n_av]
        e = ends[i, :n_av]
        sizes = np.array([binned_array[i, s[a] : e[a] + 1].sum() for a in range(n_av)])
        fit = _fit_truncated_power_law(sizes, system_size=n_channels)
        if not np.isnan(fit["ks"]) and fit["ks"] <= ks_threshold:
            alphas[i] = fit["exponent"]

    return alphas


@feature_predecessor(detect_avalanches)
@multivariate_feature
def tau_exponent(
    binned_array,
    starts,
    ends,
    counts,
    /,
    *,
    t_max_method: str = "max",
    ks_threshold: float = 0.1,
    _metadata: dict = None,
):
    r"""Tau exponent of the avalanche *duration* distribution.

    Fits a truncated discrete power-law to the distribution of avalanche
    durations (number of active bins per avalanche).

    Parameters
    ----------
    t_max_method : {'max', 'lab'}
        How to determine the upper bound of the fitting range.

        * ``'max'``: maximum observed duration.
        * ``'lab'``: theoretical bound based on max size and mean activity.
    ks_threshold : float
        Maximum KS distance for the fit to be considered reliable.

    Returns
    -------
    np.ndarray, shape (batch,)
        τ per window; ``np.nan`` if the fit is unreliable.

    Notes
    -----
    Fitting is performed on discrete bin counts. The resulting exponent is
    scale-invariant and valid for physical time units.

    """
    batch_size = binned_array.shape[0]
    taus = np.full(batch_size, np.nan, dtype=float)
    n_bins_meta = _metadata.get("n_bins", binned_array.shape[1]) if _metadata else binned_array.shape[1]

    for i in range(batch_size):
        n = int(counts[i])
        if n == 0:
            continue
        s = starts[i, :n]
        e = ends[i, :n]
        durations = (e - s + 1).astype(int)

        if t_max_method == "max":
            t_max = int(durations.max())
        elif t_max_method == "lab":
            sizes = np.array([binned_array[i, s[j] : e[j] + 1].sum() for j in range(n)])
            mean_activity = binned_array[i].sum() / n_bins_meta
            t_max = max(1, int(np.sqrt(sizes.max() / max(mean_activity, 1e-10))))
        else:
            raise ValueError(f"Unsupported t_max_method: {t_max_method!r}")

        fit = _fit_truncated_power_law(durations, system_size=t_max)
        if not np.isnan(fit["ks"]) and fit["ks"] <= ks_threshold:
            taus[i] = fit["exponent"]

    return taus


@feature_predecessor(detect_avalanches)
@multivariate_feature
def gamma_exponent(
    binned_array,
    starts,
    ends,
    counts,
    /,
    *,
    min_unique_durations: int = 3,
):
    r"""Gamma exponent of the size–duration scaling relationship.

    Estimates γ from the log-log linear regression of mean avalanche size
    against duration (continuous power-law fit, not MLE).

    Parameters
    ----------
    min_unique_durations : int
        Minimum number of distinct durations required for a reliable fit.

    Returns
    -------
    np.ndarray, shape (batch,)
        γ per window; ``np.nan`` if too few distinct durations or fit fails.

    """
    batch_size = binned_array.shape[0]
    gammas = np.full(batch_size, np.nan, dtype=float)

    for i in range(batch_size):
        n = int(counts[i])
        if n == 0:
            continue
        s = starts[i, :n]
        e = ends[i, :n]
        sizes = np.array([binned_array[i, s[j] : e[j] + 1].sum() for j in range(n)])
        durations = (e - s + 1).astype(int)

        unique_dur = np.unique(durations)
        if len(unique_dur) < min_unique_durations:
            continue

        avg_sizes = np.array([sizes[durations == t].mean() for t in unique_dur])
        if np.any(avg_sizes <= 0):
            continue  # log10 of non-positive values is undefined

        log_t = np.log10(unique_dur.astype(float))
        log_s = np.log10(avg_sizes)

        try:
            poly = np.polyfit(log_t, log_s, 1, full=True)
            gammas[i] = poly[0][0]
        except np.linalg.LinAlgError:
            pass

    return gammas


# ── Post-processing utility ───────────────────────────────────────────────────


def dcc(alphas: np.ndarray, taus: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    r"""Deviation from Criticality Coefficient.

    Quantifies how close the system is to a critical state by comparing
    the observed γ with the predicted value from α and τ:

    .. math::

        \text{DCC} = \left| \gamma_{\text{obs}}
                     - \frac{\tau - 1}{\alpha - 1} \right|

    A value near 0 indicates criticality.

    .. note::

        This is a **post-processing utility**, not an EEGDash feature.
        Features cannot depend on other features in the current API.
        Call this function after :func:`~eegdash.features.extract_features`
        on the resulting DataFrame columns.

    Parameters
    ----------
    alphas, taus, gammas : np.ndarray
        Arrays of exponents, one value per recording window.

    Returns
    -------
    np.ndarray
        DCC values; ``np.nan`` wherever any input is ``np.nan``.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_pred = (taus - 1.0) / (alphas - 1.0)
        return np.abs(gammas - gamma_pred)
