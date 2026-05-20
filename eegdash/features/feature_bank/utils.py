r"""Feature Extraction Utilities
============================

This module provides the following helper functions:
- `get_valid_freq_band`: Validates and returns frequency boundaries based on Nyquist and resolution.
- `slice_freq_band`: Slices frequency vector and associated data arrays to a specific range.
- `reduce_freq_bands`: Reduces spectral data into discrete frequency bands by aggregating bins
"""

import inspect
from collections.abc import Callable

import numpy as np

from ..decorators import preprocessor_output_type, wraps
from ..output_types import AsInputOutputType

__all__ = [
    "DEFAULT_FREQ_BANDS",
    "get_valid_freq_band",
    "preprocessor_as_feature",
    "reduce_freq_bands",
    "set_spectral_default_kwargs",
    "slice_freq_band",
    "spectral_kwargs",
]


DEFAULT_FREQ_BANDS = {
    "delta": (1, 4.5),
    "theta": (4.5, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
}


@preprocessor_output_type(AsInputOutputType)
def preprocessor_as_feature(*x):
    """A pass-through feature, returning its preprocessor output as is.

    Use if the preprocessor is a feature by itself, and it should also be treated as a
    feature.

    Parameters
    ----------
    *x: tuple
        Any preprocessor output.

    Returns
    -------
    *x: tuple
        The input (as is).

    """
    if len(x) == 1:
        return x[0]
    return x


def set_spectral_default_kwargs(kwargs, metadata):
    r"""Sets default parameters for spectral preprocecssors.

    - Set the default frequency limits to the bandpass frequencies (if available).
    - Set the default sampling frequency to `freq` in MNE's info.
    - Use `window_size_in_sec` if `nperseg` is not provided. Defaults to 4 seconds.
    - Use `overlap_in_sec` if `nperseg` and `noverlap` are not provided.
       Defaults to half the window size.
    - Set the axis to -1

    Parameters
    ----------
    kwargs : dict
        A dictionary of keyword arguments.
    metadata : dict
        A dictionary of record and batch metadata.

    Returns
    -------
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    kwargs : dict
        A dictionary of keyword arguments.

    """
    f_min = (
        kwargs.pop("f_min")
        if "f_min" in kwargs
        else metadata["info"]["highpass"]
        if "highpass" in metadata["info"]
        else None
    )
    f_max = (
        kwargs.pop("f_max")
        if "f_max" in kwargs
        else metadata["info"]["lowpass"]
        if "lowpass" in metadata["info"]
        else None
    )
    if "fs" not in kwargs or kwargs["fs"] is None:
        kwargs["fs"] = metadata["info"]["sfreq"]
    if "nperseg" not in kwargs or kwargs["nperseg"] is None:
        window_size_in_sec = (
            kwargs.pop("window_size_in_sec") if "window_size_in_sec" in kwargs else 4
        )
        kwargs["nperseg"] = int(window_size_in_sec * kwargs["fs"])
        if "noverlap" not in kwargs or kwargs["noverlap"] is None:
            overlap_in_sec = (
                kwargs.pop("overlap_in_sec")
                if "overlap_in_sec" in kwargs
                else window_size_in_sec / 2
            )
            kwargs["noverlap"] = int(overlap_in_sec * kwargs["fs"])
    kwargs["axis"] = -1
    return f_min, f_max, kwargs


def spectral_kwargs(func: Callable):
    r"""A decorator for functions receiving spectral-like parameters.

    Parameters
    ----------
    func: Callable
        A function receiving spectral-like parameters.

    Returns
    -------
    Callable
        A wrapped function with extra parameters and a suitable docstring.

    """
    new_params = [
        {
            "signature": inspect.Parameter(
                name="fs",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=int | None,
            ),
            "doc": r"Sampling frequency. Defaults to `sfreq` in MNE's info. "
            + r"Do not use unless you know what you are doing.",
        },
        {
            "signature": inspect.Parameter(
                name="f_min",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=float | None,
            ),
            "doc": r"The minimum frequency. Use `None` for half the window"
            + r" length. Defaults to the highpass frequency used to MNE's"
            + r":meth:`~mne.io.Raw.filter`.",
        },
        {
            "signature": inspect.Parameter(
                name="f_max",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=float | None,
            ),
            "doc": r"The maximum frequency. Use `None` for Nyquist. "
            + r"Defaults to the lowpass frequency used to MNE's "
            + r":meth:`~mne.io.Raw.filter`.",
        },
        {
            "signature": inspect.Parameter(
                name="window_size_in_sec",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=4,
                annotation=float | None,
            ),
            "doc": r"Window size in seconds, replacing `nperseg`. "
            + r"Only used if `nperseg` is not provided. "
            r"Defaults to 4 seconds.",
        },
        {
            "signature": inspect.Parameter(
                name="overlap_in_sec",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=float | None,
            ),
            "doc": r"Window overlap in seconds, replacing `noverlap`. "
            + r"Only used if `nperseg` and `noverlap` are not provided."
            + r"defaults to half of `window_size_in_sec`.",
        },
    ]

    @wraps(func, new_args=new_params)
    def spectral_wrapper(*args, _metadata, **kwargs):
        f_min, f_max, kwargs = set_spectral_default_kwargs(kwargs, _metadata)
        return func(*args, _metadata=_metadata, f_min=f_min, f_max=f_max, **kwargs)

    return spectral_wrapper


def get_valid_freq_band(fs, n, f_min=None, f_max=None):
    r"""Validate and return frequency boundaries based on Nyquist and resolution.

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

    Raises
    ------
    AssertionError
        If `f_min` is below the minimum resolvable frequency.
    AssertionError
        If `f_max` is above the Nyquist frequency.

    Examples
    --------
    >>> get_valid_freq_band(fs=100, n=1000)
    (0.2, 50.0)
    >>> get_valid_freq_band(fs=200, n=500, f_min=1, f_max=80)
    (1, 80)

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
    r"""Slice frequency vector and associated data arrays to a specific range.

    Parameters
    ----------
    f : ndarray
        The frequency vector.
    *x : ndarray
        One or more data arrays to be sliced along the frequency axis.
        The last dimension of each array must match the length of `f`.
    f_min : float, optional
        Lower frequency bound.
    f_max : float, optional
        Upper frequency bound.

    Returns
    -------
    f : ndarray
        The cropped frequency vector.
    *xl : ndarray
        The cropped data arrays.

    Examples
    --------
    >>> # Create 0-10 Hz frequencies
    >>> freqs = np.array([0, 2, 4, 6, 8, 10])

    >>> # Create data: (2 channels, 6 frequency bins)
    >>> data = np.array([[10, 20, 30, 40, 50, 60],
    ...                  [15, 25, 35, 45, 55, 65]])

    >>> # Keep only the range 4Hz to 8Hz
    >>> f_s, d_s = slice_freq_band(freqs, data, f_min=4, f_max=8)

    >>> f_s
    array([4, 6, 8])
    >>> d_s
    array([[30, 40, 50],
           [35, 45, 55]])

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
    r"""Reduce spectral data into discrete frequency bands by aggregating bins.

    This function identifies the frequency indices belonging to specific
    bands and applies a reduction function (like sum or mean) to collapse
    the frequency axis.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    x : ndarray
        Spectral data. Can be multi-dimensional.
        The last dimension must match the length of `f`.
    bands : dict
        Mapping of band names to (min, max) frequency tuples.
    reduce_func : callable, optional
        Function to aggregate the values. Default is np.sum.

    Returns
    -------
    x_bands : dict
        Dictionary where keys are the band names from `bands` and values
        are the reduced arrays. The last dimension of the input `x`
        is removed.

    Raises
    ------
    AssertionError
        If a band name is not a string.
        If a band limit tuple does not contain exactly two values or
        if min > max.
        If the requested band limits fall outside the range of the
        available frequency vector `f`.

    Examples
    --------
    >>> f = np.array([0, 2, 4, 6, 8, 10])
    >>> x = np.array([
    ...     [1, 2, 3, 4, 5, 6],
    ...     [60, 50, 40, 30, 20, 10],
    ... ])
    >>> bands = {'low': (0, 5), 'high': (5, 11)} # check assertion
    >>> results = reduce_freq_bands(f, x, bands, reduce_func=np.sum)
    >>> results['low']
    array([6, 150])
    >>> results['high']
    array([15, 60])

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
