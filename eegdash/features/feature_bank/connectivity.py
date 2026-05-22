r"""Connectivity Feature Extraction
===============================

This module computes bivariate connectivity features based on the complex
coherency between pairs of channels.

Data Shape Convention
---------------------
This module follows a **Time-Last** convention:

* **Input:** ``(..., time)``
* **Output:** ``(...,)``

All functions collapse the last dimension (time), returning an ndarray of
features corresponding to the leading dimensions (e.g., subjects, channels).
"""

from itertools import chain

import numba as nb
import numpy as np
from scipy.signal import csd, hilbert

from ..decorators import (
    bivariate_feature,
    channel_pairer_undirected,
    feature_predecessor,
)
from . import utils
from .signal import signal_filter_preprocessor
from .spectral import (
    spectral_db_preprocessor,
    spectral_normalized_preprocessor,
    spectral_preprocessor,
)

__all__ = [
    "connectivity_coherency_preprocessor",
    "connectivity_phase_diff_preprocessor",
    "connectivity_temporal_correlation",
    "connectivity_spectral_correlation",
    "connectivity_magnitude_square_coherence",
    "connectivity_imaginary_coherence",
    "connectivity_lagged_coherence",
    "connectivity_phase_locking_value",
    "connectivity_corrected_imaginary_phase_locking_value",
    "connectivity_phase_lag_index",
    "connectivity_weighted_phase_lag_index",
    "connectivity_directed_phase_lag_index",
]


@nb.njit(cache=True, fastmath=True)
def _channel_pairs_correlations(x, idx_x, idx_y):
    """Compute the correlation between channel pairs.

    Parameters
    ----------
    x : numpy.ndarray
        The input of shape (n_trials, n_channels, n_samples).

    Returns
    -------
    numpy.ndarray
        The channel pairwise Pearson correlation of shape (n_trials, n_pairs).

    """
    corr = np.empty((*x.shape[:-2], len(idx_x)))
    for i in np.ndindex(x.shape[:-2]):
        c = np.corrcoef(x[i])
        for j, (l, m) in enumerate(zip(idx_x, idx_y)):
            corr[i][j] = c[l, m]
    return corr


@feature_predecessor()
@channel_pairer_undirected
@bivariate_feature
def connectivity_temporal_correlation(x, /, *, _metadata):
    """Compute the temporal correlation between channel pairs.

    Parameters
    ----------
    x : numpy.ndarray
        The input signal of shape (n_trials, n_channels, n_times).

    Returns
    -------
    numpy.ndarray
        The channel pairwise temporal Pearson correlation of shape
        (n_trials, n_pairs).

    """
    idx_x, idx_y = _metadata["ch_pair_iterator"].get_pair_iterators()
    return _channel_pairs_correlations(x, idx_x, idx_y)


@feature_predecessor(
    spectral_preprocessor,
    spectral_normalized_preprocessor,
    spectral_db_preprocessor,
)
@channel_pairer_undirected
@bivariate_feature
def connectivity_spectral_correlation(f, p, /, *, _metadata):
    """Compute the spectral correlation between channel pairs.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Power Spectral Density (PSD).

    Returns
    -------
    numpy.ndarray
        The channel pairwise Pearson correlation between power spectra of shape
        (n_trials, n_pairs).

    """
    idx_x, idx_y = _metadata["ch_pair_iterator"].get_pair_iterators()
    return _channel_pairs_correlations(p, idx_x, idx_y)


@feature_predecessor()
@channel_pairer_undirected
@utils.spectral_kwargs
def connectivity_coherency_preprocessor(x, /, *, _metadata, f_min, f_max, **kwargs):
    r"""Compute Complex Coherency for all unique channel pairs.

    The Complex Coherency is calculated by estimating the Cross-Spectral
    Densities (CSD) between pairs of channels and normalizing it by the
    auto-spectral densities.

    Parameters
    ----------
    x : ndarray
        The input signal of shape (n_trials, n_channels, n_times).
    **kwargs : dict
        Supports any :func:`scipy.signal.csd` arguments like 'nperseg'
        and 'noverlap'.

    Returns
    -------
    f : ndarray
        Frequency vector of shape (n_frequencies,).
    c : ndarray
        Complex coherency array of shape (n_trials, n_pairs, n_frequencies).
        Values are complex numbers where:

        - Absolute value :math:`|c|` is the coherence magnitude (0 to 1).
        - Angle :math:`\arg(c)` is the phase lag.

    """
    n = x.shape[1]
    idx_x, idx_y = _metadata["ch_pair_iterator"].get_pair_iterators()
    ix, iy = list(chain(range(n), idx_x)), list(chain(range(n), idx_y))
    f, s = csd(x[:, ix], x[:, iy], **kwargs)
    f_min, f_max = utils.get_valid_freq_band(kwargs["fs"], x.shape[-1], f_min, f_max)
    f, s = utils.slice_freq_band(f, s, f_min=f_min, f_max=f_max)
    p, sxy = np.split(s, [n], axis=1)
    sxx, syy = p[:, idx_x].real, p[:, idx_y].real
    c = sxy / np.sqrt(sxx * syy)
    return f, c


@feature_predecessor(connectivity_coherency_preprocessor)
@bivariate_feature
def connectivity_magnitude_square_coherence(f, c, /, bands=utils.DEFAULT_FREQ_BANDS):
    r"""Calculate Magnitude Squared Coherence (MSC).

    MSC measures the linear correlation between two signals in the frequency
    domain. It is defined as the squared magnitude of the complex coherency,
    :math:`|c|^2`, where :math:`c` is the complex coherency.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    c : ndarray
        Complex coherency array.
    bands : dict, optional
        Frequency bands to aggregate (defaults to DEFAULT_FREQ_BANDS).

    Returns
    -------
    dict
        Mean MSC for each frequency band.

    References
    ----------
    `Brainstorm - Connectivity <https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity>`_

    """
    coher = c.real**2 + c.imag**2
    return utils.reduce_freq_bands(f, coher, bands, np.mean)


@feature_predecessor(connectivity_coherency_preprocessor)
@bivariate_feature
def connectivity_imaginary_coherence(f, c, /, bands=utils.DEFAULT_FREQ_BANDS):
    r"""Calculate Imaginary Coherence (iCOH).

    Imaginary coherence captures only the non-zero phase-lagged
    synchronization. It is defined as :math:`\operatorname{Im}(c)`,
    where :math:`c` is the complex coherency.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    c : ndarray
        Complex coherency array.
    bands : dict, optional
        Frequency bands to aggregate.

    Returns
    -------
    dict
        Mean Imaginary Coherence for each frequency band.

    References
    ----------
    `Brainstorm - Connectivity <https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity>`_

    """
    coher = c.imag
    return utils.reduce_freq_bands(f, coher, bands, np.mean)


@feature_predecessor(connectivity_coherency_preprocessor)
@bivariate_feature
def connectivity_lagged_coherence(f, c, /, bands=utils.DEFAULT_FREQ_BANDS):
    r"""Calculate Lagged Coherence.

    Lagged coherence further refines the synchronization measure by
    normalizing the imaginary part of the coherency, effectively removing
    all instantaneous (zero-lag) contributions. It is defined as
    :math:`\operatorname{Im}(c)/\sqrt{1 - \left(\operatorname{Re}(c)\right)^2}`,
    where :math:`c` is the complex coherency.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    c : ndarray
        Complex coherency array.
    bands : dict, optional
        Frequency bands to aggregate.

    Returns
    -------
    dict
        Mean Lagged Coherence for each frequency band.

    References
    ----------
    `Brainstorm - Connectivity <https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity>`_

    """
    coher = c.imag / np.sqrt(1 - c.real**2)
    return utils.reduce_freq_bands(f, coher, bands, np.mean)


@feature_predecessor(signal_filter_preprocessor)
@channel_pairer_undirected
def connectivity_phase_diff_preprocessor(x, /, *, _metadata):
    r"""Compute complex exponent of phase difference for all unique channel pairs.

    For each pair of channels :math:`l, m`, calculate:

    .. math::
        e^{i\left(\varphi_l\left(t\right) - \varphi_m\left(t\right)\right)}.

    The instantanous phases are calculated via Hilbert transform.

    .. note::
        This preprocessor should follow a narrow-band filter, otherwise the
        Hilbert transform cannot yield meaningful phases.

    Parameters
    ----------
    x : ndarray
        The input signal of shape (n_trials, n_channels, n_times).

    Returns
    -------
    ndarray
        Complex exponents of phase diffs.

    """
    a = hilbert(x, axis=-1)
    exp_phi = a / np.abs(a)
    idx_x, idx_y = _metadata["ch_pair_iterator"].get_pair_iterators()
    exp_dphi = exp_phi[..., idx_x, :] * exp_phi[..., idx_y, :].conj()
    return exp_dphi


@feature_predecessor(connectivity_phase_diff_preprocessor)
@bivariate_feature
def connectivity_phase_locking_value(exp_dphi, /):
    r"""Compute the Phase Locking Value (PLV) of each channel pair.

    .. math::
        PLV_{lm} = \left|\left\langle e^{i\left(\varphi_l\left(t\right)
                   - \varphi_m\left(t\right)\right)}\right\rangle_t\right|

    Parameters
    ----------
    exp_dphi : ndarray
        Complex exponents of phase diffs.

    Returns
    -------
    ndarray :
        The PLV of each channel pair.

    References
    ----------
    - Lachaux, JP. et al. (1999). Measuring phase synchrony in brain signals.
      Hum Brain Mapp, 8(4), 194-208.

    """
    return np.abs(exp_dphi.mean(axis=-1))


@feature_predecessor(connectivity_phase_diff_preprocessor)
@bivariate_feature
def connectivity_corrected_imaginary_phase_locking_value(exp_dphi, /):
    r"""Compute the corrected imaginary Phase Locking Value (ciPLV) of each channel pair.

    .. math::

        ciPLV_{lm} = \frac{\Im{\left(\left\langle e^{i\left(
                     \varphi_l\left(t\right) - \varphi_m\left(t\right)\right)}
                     \right\rangle_t\right)}}{\sqrt{1 - \left(\Re{\left(
                     \left\langle e^{i\left(\varphi_l\left(t\right)
                     - \varphi_m\left(t\right)\right)}\right\rangle_t\right)}
                     \right)^2}}

    Parameters
    ----------
    exp_dphi : ndarray
        Complex exponents of phase diffs.

    Returns
    -------
    ndarray :
        The ciPLV of each channel pair.

    References
    ----------
    - Bruña, R. et al. (2018). Phase locking value revisited: teaching new
      tricks to an old dog. Journal of Neural Engineering, 15(5), 056011.

    """
    mean_exp_dphi = exp_dphi.mean(axis=-1)
    return mean_exp_dphi.imag / np.sqrt(1 - mean_exp_dphi.real**2)


@feature_predecessor(connectivity_phase_diff_preprocessor)
@bivariate_feature
def connectivity_phase_lag_index(exp_dphi, /):
    r"""Compute the Phase Lag Index (PLI) of each channel pair.

    .. math::

        PLI_{lm} = \left|\left\langle\operatorname{sign}{\left(\Im{\left(
                   e^{i\left(\varphi_l\left(t\right) - \varphi_m\left(t\right)
                   \right)}\right)}\right)}\right\rangle_t\right|

    Parameters
    ----------
    exp_dphi : ndarray
        Complex exponents of phase diffs.

    Returns
    -------
    ndarray :
        The PLI of each channel pair.

    References
    ----------
    - Stam, CJ. et al. (2007). A. Phase lag index: assessment of functional
      connectivity from multi channel EEG and MEG with diminished bias from
      common sources. Hum Brain Mapp, 28(11), 1178-93.

    """
    return np.abs(np.mean(np.sign(exp_dphi.imag), axis=-1))


@feature_predecessor(connectivity_phase_diff_preprocessor)
@bivariate_feature
def connectivity_weighted_phase_lag_index(exp_dphi, /):
    r"""Compute the weighted Phase Lag Index (wPLI) of each channel pair.

    .. math::

        wPLI_{lm} = \frac{\left|\left\langle\Im{\left(e^{i\left(
                    \varphi_l\left(t\right) - \varphi_m\left(t\right)\right)}
                    \right)}\right\rangle_t\right|}{\left\langle\left|
                    \Im{\left(e^{i\left(\varphi_l\left(t\right)
                    - \varphi_m\left(t\right)\right)}\right)}\right|
                    \right\rangle_t}

    Parameters
    ----------
    exp_dphi : ndarray
        Complex exponents of phase diffs.

    Returns
    -------
    ndarray :
        The wPLI of each channel pair.

    References
    ----------
    - Vinck, M. et al. (2011). An improved index of phase-synchronization for
      electrophysiological data in the presence of volume-conduction, noise
      and sample-size bias. NeuroImage, 55(4), 1548-1565.

    """
    return np.abs(exp_dphi.imag.mean(axis=-1)) / np.abs(exp_dphi.imag).mean(axis=-1)


@feature_predecessor(connectivity_phase_diff_preprocessor)
@bivariate_feature
def connectivity_directed_phase_lag_index(exp_dphi, /):
    r"""Compute the directed Phase Lag Index (dPLI) of each channel pair.

    .. math::

        dPLI_{lm} = \left|\left\langle\Theta{\left(\Im{\left(e^{i\left(
                    \varphi_l\left(t\right) - \varphi_m\left(t\right)\right)}
                    \right)}\right)}\right\rangle_t\right|,

    where :math:`\Theta\left(\cdot\right)` is Heaviside's step function.

    Parameters
    ----------
    exp_dphi : ndarray
        Complex exponents of phase diffs.

    Returns
    -------
    ndarray :
        The dPLI of each channel pair.

    References
    ----------
    - Stam, CJ. et al. (2012). Go with the flow: Use of a directed phase lag
      index (dPLI) to characterize patterns of phase relations in a
      large-scale model of brain dynamics. NeuroImage, 62(3), 1415-1428.

    """
    return np.mean(exp_dphi.imag > 0, axis=-1)
