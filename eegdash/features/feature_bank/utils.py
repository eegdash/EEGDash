import numpy as np

__all__ = [
    "DEFAULT_FREQ_BANDS",
    "get_valid_freq_band",
    "reduce_freq_bands",
    "slice_freq_band",
]


DEFAULT_FREQ_BANDS = {
    "delta": (1, 4.5),
    "theta": (4.5, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
}


def get_valid_freq_band(fs, n, f_min=None, f_max=None):
    """Validate and return frequency boundaries based on Nyquist and resolution.

    Parameters
    ----------
    fs : float
        The sampling frequency in Hz.
    n : int
        The number of points in the signal/window.
    f_min : float, optional
        Requested minimum frequency. Defaults to 2 * resolution (f0).
    f_max : float, optional
        Requested maximum frequency. Defaults to Nyquist frequency.

    Returns
    -------
    f_min, f_max : float
        The validated frequency boundaries.

    """
    f0 = 2 * fs / n
    f1 = fs / 2
    if f_min is None:
        f_min = f0
    else:
        assert f_min >= f0
    if f_max is None:
        f_max = f1
    else:
        assert f_max <= f1
    return f_min, f_max


def slice_freq_band(f, *x, f_min=None, f_max=None):
    """Slice frequency vector and associated data arrays to a specific range.

    Parameters
    ----------
    f : ndarray
        The frequency vector.
    *x : ndarray
        One or more data arrays to be sliced along the last axis.
    f_min : float, optional
        Lower frequency bound.
    f_max : float, optional
        Upper frequency bound.

    Returns
    -------
    f_sliced : ndarray
        The truncated frequency vector.
    *x_sliced : ndarray
        The truncated data arrays.

    """
    if f_min is None and f_max is None:
        return f, *x
    else:
        f_min_idx = f >= f_min if f_min is not None else True
        f_max_idx = f <= f_max if f_max is not None else True
        idx = np.logical_and(f_min_idx, f_max_idx)
        f = f[idx]
        xl = [*x]
        for i, xi in enumerate(xl):
            xl[i] = xi[..., idx]
        return f, *xl


def reduce_freq_bands(f, x, bands, reduce_func=np.sum):
    """Reduce spectral data into discrete frequency bands.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    x : ndarray
        Spectral data. The last dimension must match `f`.
    bands : dict
        Mapping of band names to (min, max) tuples.
    reduce_func : callable, optional
        Function to aggregate the values (e.g., np.sum, np.mean). 
        Default is np.sum.

    Returns
    -------
    dict
        Dictionary where keys are band names and values are reduced arrays.
    
    """
    x_bands = dict()
    for k, lims in bands.items():
        assert isinstance(k, str)
        assert len(lims) == 2 and lims[0] <= lims[1]
        assert lims[0] >= f[0] and lims[1] <= f[-1]
        mask = np.logical_and(f >= lims[0], f < lims[1])
        xf = x[..., mask]
        x_bands[k] = reduce_func(xf, axis=-1)
    return x_bands
