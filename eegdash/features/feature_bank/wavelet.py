r"""Wavelet Feature Extraction
============================

This module computes wavelet-domain features via the Continuous Wavelet
Transform (CWT), using PyWavelets.

Data Shape Convention
---------------------
This module follows a **Time-Last** convention:

* **Input:** ``(..., time)``
* **Output:** ``(...,)``

The preprocessor returns coefficients of shape
``(n_bands, *leading_dims, n_times)`` — one CWT slice per frequency band,
all leading dimensions (trials, channels) preserved.

Wavelet Selection
-----------------
The default wavelet ``'cmor1.5-1.0'`` (complex Morlet) returns complex
coefficients and supports all features including phase-based ones (PLV, PAC,
coherence). Real wavelets (``'mexh'``, ``'db4'``, ``'sym4'``) work for
energy-only features; phase features raise a ``ValueError``.
"""

import numba as nb
import numpy as np
import pywt
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

from ..decorators import (
    bivariate_feature,
    channel_pairer_undirected,
    feature_predecessor,
    metadata_preprocessor,
    univariate_feature,
)
from . import utils
from .utils import requires_complex_wavelet

__all__ = [
    "wavelet_preprocessor",
    "wavelet_connectivity_preprocessor",
    "wavelet_entropy",
    "wavelet_relative_power",
    "wavelet_skewness",
    "wavelet_kurtosis",
    "wavelet_pac",
    "wavelet_plv",
    "wavelet_coherence",
]

_DEFAULT_WAVELET = "cmor1.5-1.0"


def _band_center_freq(f_low, f_high):
    return np.sqrt(f_low * f_high)


@nb.njit(cache=True, fastmath=True)
def _pac_bin_accumulate(phi_flat, A_flat, bin_edges, K):
    n_leading, n_times = phi_flat.shape
    mean_A = np.zeros((n_leading, K))
    counts = np.zeros((n_leading, K))
    for s in range(n_leading):
        for t in range(n_times):
            k = np.searchsorted(bin_edges, phi_flat[s, t]) - 1
            if k < 0:
                k = 0
            if k >= K:
                k = K - 1
            mean_A[s, k] += A_flat[s, t]
            counts[s, k] += 1
    for s in range(n_leading):
        for k in range(K):
            if counts[s, k] > 0:
                mean_A[s, k] /= counts[s, k]
    return mean_A


# ---------------------------------------------------------------------------
# Preprocessors
# ---------------------------------------------------------------------------


@feature_predecessor()
@metadata_preprocessor
def wavelet_preprocessor(
    x,
    /,
    frequency_bands=utils.DEFAULT_FREQ_BANDS,
    wavelet=_DEFAULT_WAVELET,
    *,
    _metadata,
):
    r"""Compute per-band CWT coefficients.

    Parameters
    ----------
    x : ndarray
        Input signal of shape ``(*leading_dims, n_times)``.
    frequency_bands : dict
        Mapping of band name → ``(f_low, f_high)`` in Hz.
        Defaults to ``DEFAULT_FREQ_BANDS`` (delta/theta/alpha/beta).
    wavelet : str
        PyWavelets wavelet name. Defaults to ``'cmor1.5-1.0'`` (complex Morlet).
        Complex wavelets enable phase features; real wavelets do not.
    _metadata : dict
        Must contain ``_metadata["info"]["sfreq"]``.

    Returns
    -------
    bands : dict
        The ``frequency_bands`` argument, passed through for downstream features.
    W : ndarray
        CWT coefficients of shape ``(n_bands, *leading_dims, n_times)``.
        Complex-valued for complex wavelets; real-valued otherwise.
    _metadata : dict
        Updated metadata with ``wavelet_is_complex`` set.

    """
    sfreq = _metadata["info"]["sfreq"]
    fc = pywt.central_frequency(wavelet)
    scales = np.array(
        [
            fc * sfreq / _band_center_freq(f_low, f_high)
            for f_low, f_high in frequency_bands.values()
        ]
    )
    coefs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / sfreq, axis=-1)
    if x.ndim > 2:  # batched: pywt puts n_bands first; move it behind n_batches
        coefs = np.moveaxis(coefs, 0, 1)  # (n_batches, n_bands, n_ch, n_t)
    _metadata["wavelet_is_complex"] = np.iscomplexobj(coefs)
    return frequency_bands, coefs, _metadata


@feature_predecessor(wavelet_preprocessor)
@channel_pairer_undirected
def wavelet_connectivity_preprocessor(
    bands,
    W,
    /,
    *,
    _metadata,
):
    r"""Compute per-band CWT coefficients for all unique channel pairs.

    Receives output of :func:`wavelet_preprocessor` and splits the channel
    axis into pairs.

    Parameters
    ----------
    bands : dict
        Band names → frequency ranges, passed through from ``wavelet_preprocessor``.
    W : ndarray
        CWT coefficients, shape ``(n_bands, n_ch, n_t)`` (unbatched) or
        ``(n_batches, n_bands, n_ch, n_t)`` (batched).
    _metadata : dict
        Must contain ``_metadata["ch_pair_iterator"]``.

    Returns
    -------
    bands : dict
        The ``bands`` argument, passed through.
    Wx : ndarray
        Coefficients for the first channel of each pair.
        Shape matches W with the channel axis replaced by n_pairs.
    Wy : ndarray
        Coefficients for the second channel of each pair, same shape as ``Wx``.

    """
    idx_x, idx_y = _metadata["ch_pair_iterator"].get_pair_iterators()
    # channel axis is always -2 in both batched and unbatched W
    Wx = W[..., idx_x, :]
    Wy = W[..., idx_y, :]
    return bands, Wx, Wy


# ---------------------------------------------------------------------------
# Univariate features
# ---------------------------------------------------------------------------


@feature_predecessor(wavelet_preprocessor)
@univariate_feature
def wavelet_entropy(bands, W, /):
    r"""Wavelet entropy (Rosso et al., 2001): :math:`-\sum_j p_j \log p_j`.

    Treats the relative per-band energy as a probability distribution and
    returns its Shannon entropy. Low entropy = energy concentrated in one
    band (narrow-band signal); high entropy = energy spread across bands
    (broadband / disordered signal).

    Parameters
    ----------
    bands : dict
        Band names → frequency ranges (passed through from preprocessor).
    W : ndarray
        CWT coefficients, shape ``(n_bands, *leading_dims, n_times)``.

    Returns
    -------
    ndarray
        Shape ``(*leading_dims,)`` — one scalar per channel.

    """
    bands_axis = 0 if W.ndim == 3 else 1
    E = (W.real**2 + W.imag**2).mean(axis=-1)
    p = E / E.sum(axis=bands_axis, keepdims=True)
    safe_log = np.where(p > 0, np.log(p), 0.0)
    return -(p * safe_log).sum(axis=bands_axis)


@feature_predecessor(wavelet_preprocessor)
@univariate_feature
def wavelet_relative_power(bands, W, /):
    r"""Per-band energy as a fraction of total energy across all bands.

    Parameters
    ----------
    bands : dict
        Band names → frequency ranges.
    W : ndarray
        CWT coefficients, shape ``(n_bands, *leading_dims, n_times)``.

    Returns
    -------
    dict
        ``{band_name: ndarray}`` where each array has shape
        ``(*leading_dims,)``.

    """
    bands_axis = 0 if W.ndim == 3 else 1
    E = (W.real**2 + W.imag**2).mean(axis=-1)
    total = E.sum(axis=bands_axis, keepdims=True)
    return {
        name: np.take(E, i, axis=bands_axis) / np.squeeze(total, axis=bands_axis)
        for i, name in enumerate(bands)
    }


@feature_predecessor(wavelet_preprocessor)
@univariate_feature
def wavelet_skewness(bands, W, /):
    r"""Skewness of ``|W|`` along the time axis, per band.

    Parameters
    ----------
    bands : dict
        Band names → frequency ranges.
    W : ndarray
        CWT coefficients, shape ``(n_bands, *leading_dims, n_times)``.

    Returns
    -------
    dict
        ``{band_name: ndarray}`` where each array has shape
        ``(*leading_dims,)``.

    """
    bands_axis = 0 if W.ndim == 3 else 1
    return {
        name: scipy_skew(np.abs(np.take(W, i, axis=bands_axis)), axis=-1)
        for i, name in enumerate(bands)
    }


@feature_predecessor(wavelet_preprocessor)
@univariate_feature
def wavelet_kurtosis(bands, W, /):
    r"""Excess kurtosis of ``|W|`` along the time axis, per band.

    Parameters
    ----------
    bands : dict
        Band names → frequency ranges.
    W : ndarray
        CWT coefficients, shape ``(n_bands, *leading_dims, n_times)``.

    Returns
    -------
    dict
        ``{band_name: ndarray}`` where each array has shape
        ``(*leading_dims,)``.

    """
    bands_axis = 0 if W.ndim == 3 else 1
    return {
        name: scipy_kurtosis(np.abs(np.take(W, i, axis=bands_axis)), axis=-1)
        for i, name in enumerate(bands)
    }


@feature_predecessor(wavelet_preprocessor)
@univariate_feature
@requires_complex_wavelet
def wavelet_pac(bands, W, /, *, _metadata=None):
    r"""Wavelet Phase-Amplitude Coupling via Modulation Index (Tort et al., 2010).

    For every ordered pair (low-frequency phase band, high-frequency amplitude
    band), computes the Modulation Index:

    .. math::

        MI = \frac{\log K - H\!\left(\bar{A}_{high}(\phi_k)\right)}{\log K}

    where :math:`\bar{A}_{high}(\phi_k)` is the mean amplitude of the high
    band at each of the :math:`K=18` phase bins of the low band, and :math:`H`
    is the Shannon entropy of that distribution. :math:`MI \in [0, 1]`;
    zero means no coupling, one means perfect coupling.

    Requires complex wavelet coefficients (e.g. ``'cmor1.5-1.0'``).

    Parameters
    ----------
    bands : dict
        Band names → frequency ranges.
    W : ndarray
        Complex CWT coefficients, shape ``(n_bands, *leading_dims, n_times)``.

    Returns
    -------
    dict
        ``{(phase_band_name, amp_band_name): ndarray}`` where each array has
        shape ``(*leading_dims,)``.

    """
    band_names = list(bands.keys())
    bands_axis = 0 if W.ndim == 3 else 1
    K = 18
    bin_edges = np.linspace(-np.pi, np.pi, K + 1)
    # leading shape: everything except bands_axis and time
    leading_shape = tuple(
        s for ax, s in enumerate(W.shape) if ax != bands_axis and ax != W.ndim - 1
    )
    result = {}
    for i, phase_name in enumerate(band_names):
        phi = np.angle(np.take(W, i, axis=bands_axis))  # (*leading, n_times)
        for j, amp_name in enumerate(band_names):
            if j <= i:
                continue
            A = np.abs(np.take(W, j, axis=bands_axis))  # (*leading, n_times)
            phi_flat = phi.reshape(-1, W.shape[-1])
            A_flat = A.reshape(-1, W.shape[-1])
            mean_A_flat = _pac_bin_accumulate(phi_flat, A_flat, bin_edges, K)
            mean_A = mean_A_flat.reshape(*leading_shape, K)
            p_bin = mean_A / np.maximum(mean_A.sum(axis=-1, keepdims=True), 1e-15)
            safe_log = np.where(p_bin > 0, np.log(p_bin), 0.0)
            H = -(p_bin * safe_log).sum(axis=-1)  # (*leading)
            result[(phase_name, amp_name)] = (np.log(K) - H) / np.log(K)
    return result


# ---------------------------------------------------------------------------
# Bivariate features
# ---------------------------------------------------------------------------


# TODO: add analytic_signal_connectivity_preprocessor as a second predecessor once
# that module is pulled from GitHub (see gal_notes.md item 1).
@feature_predecessor(wavelet_connectivity_preprocessor)
@bivariate_feature
@requires_complex_wavelet
def wavelet_plv(bands, Wx, Wy, /, *, _metadata=None):
    r"""Wavelet Phase Locking Value between channel pairs.

    .. math::

        wPLV(f) = \left|\frac{1}{T}\sum_t e^{i(\phi_x(f,t) - \phi_y(f,t))}\right|

    Values in :math:`[0, 1]`: 0 = no phase locking, 1 = perfect phase locking.
    More accurate than filter + Hilbert PLV because the complex Morlet CWT
    avoids bandpass filter ringing.

    Requires complex wavelet coefficients (e.g. ``'cmor1.5-1.0'``).

    Parameters
    ----------
    bands : dict
        Band names → frequency ranges.
    Wx : ndarray
        Complex CWT for channel x, shape ``(n_bands, n_trials, n_pairs, n_times)``.
    Wy : ndarray
        Complex CWT for channel y, same shape as ``Wx``.

    Returns
    -------
    dict
        ``{band_name: ndarray}`` where each array has shape
        ``(n_trials, n_pairs)``.

    """
    bands_axis = 0 if Wx.ndim == 3 else 1
    mag_x = np.sqrt(Wx.real**2 + Wx.imag**2)
    mag_y = np.sqrt(Wy.real**2 + Wy.imag**2)
    wx_n = Wx / np.maximum(mag_x, 1e-15)
    wy_n = Wy / np.maximum(mag_y, 1e-15)
    plv = np.abs(np.mean(wx_n * np.conj(wy_n), axis=-1))
    return {name: np.take(plv, i, axis=bands_axis) for i, name in enumerate(bands)}


@feature_predecessor(wavelet_connectivity_preprocessor)
@bivariate_feature
@requires_complex_wavelet
def wavelet_coherence(bands, Wx, Wy, /, *, _metadata=None):
    r"""Time-averaged wavelet coherence between channel pairs.

    .. math::

        C_{xy}(s) = \frac{|\langle W_x(s,t)\overline{W_y(s,t)}\rangle_t|^2}
                         {\langle|W_x(s,t)|^2\rangle_t \cdot
                          \langle|W_y(s,t)|^2\rangle_t}

    Values in :math:`[0, 1]`. This is a simplified (non-smoothed) estimate;
    for a fully regularised estimate, scale-smoothing can be added later.

    Requires complex wavelet coefficients (e.g. ``'cmor1.5-1.0'``).

    Parameters
    ----------
    bands : dict
        Band names → frequency ranges.
    Wx : ndarray
        Complex CWT for channel x, shape ``(n_bands, n_trials, n_pairs, n_times)``.
    Wy : ndarray
        Complex CWT for channel y, same shape as ``Wx``.

    Returns
    -------
    dict
        ``{band_name: ndarray}`` where each array has shape
        ``(n_trials, n_pairs)``.

    """
    bands_axis = 0 if Wx.ndim == 3 else 1
    cross = Wx * np.conj(Wy)
    num = np.abs(cross.mean(axis=-1)) ** 2
    denom = (Wx.real**2 + Wx.imag**2).mean(axis=-1) * (Wy.real**2 + Wy.imag**2).mean(
        axis=-1
    )
    coherence = num / np.maximum(denom, 1e-15)
    return {
        name: np.take(coherence, i, axis=bands_axis) for i, name in enumerate(bands)
    }
