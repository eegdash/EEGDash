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
    x_bands = dict()
    for k, lims in bands.items():
        assert isinstance(k, str)
        assert len(lims) == 2 and lims[0] <= lims[1]
        assert lims[0] >= f[0] and lims[1] <= f[-1]
        mask = np.logical_and(f >= lims[0], f < lims[1])
        xf = x[..., mask]
        x_bands[k] = reduce_func(xf, axis=-1)
    return x_bands
