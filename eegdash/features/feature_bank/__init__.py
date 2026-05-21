r"""Feature bank public API exports.

This module consolidates and re-exports the feature extractors and
feature functions so users can import them directly from
:mod:`eegdash.features.feature_bank`.
"""

from .complexity import (
    complexity_approx_entropy,
    complexity_entropy_preprocessor,
    complexity_lempel_ziv,
    complexity_multiscale_entropy,
    complexity_sample_entropy,
    complexity_svd_entropy,
)
from .connectivity import (
    connectivity_coherency_preprocessor,
    connectivity_correlation,
    connectivity_imaginary_coherence,
    connectivity_lagged_coherence,
    connectivity_magnitude_square_coherence,
)
from .csp import CommonSpatialPattern
from .dimensionality import (
    dimensionality_detrended_fluctuation_analysis,
    dimensionality_higuchi_fractal_dim,
    dimensionality_hurst_exp,
    dimensionality_katz_fractal_dim,
    dimensionality_petrosian_fractal_dim,
)
from .pick import (
    pick_channel_pairs_preprocessor,
    pick_channels_preprocessor,
)
from .signal import (
    signal_decorrelation_time,
    signal_envelope_preprocessor,
    signal_filter_preprocessor,
    signal_hjorth_activity,
    signal_hjorth_complexity,
    signal_hjorth_mobility,
    signal_kurtosis,
    signal_line_length,
    signal_mean,
    signal_peak_to_peak,
    signal_quantile,
    signal_root_mean_square,
    signal_skewness,
    signal_std,
    signal_variance,
    signal_zero_crossings,
)
from .spectral import (
    spectral_bands_power,
    spectral_db_preprocessor,
    spectral_edge,
    spectral_entropy,
    spectral_hjorth_activity,
    spectral_hjorth_complexity,
    spectral_hjorth_mobility,
    spectral_moment,
    spectral_normalized_preprocessor,
    spectral_peak_frequency,
    spectral_preprocessor,
    spectral_root_total_power,
    spectral_slope,
)
from .utils import preprocessor_as_feature

__all__ = [
    # Complexity
    "complexity_entropy_preprocessor",
    "complexity_approx_entropy",
    "complexity_multiscale_entropy",
    "complexity_sample_entropy",
    "complexity_svd_entropy",
    "complexity_lempel_ziv",
    # Connectivity
    "connectivity_coherency_preprocessor",
    "connectivity_correlation",
    "connectivity_magnitude_square_coherence",
    "connectivity_imaginary_coherence",
    "connectivity_lagged_coherence",
    # CSP
    "CommonSpatialPattern",
    # Dimensionality
    "dimensionality_higuchi_fractal_dim",
    "dimensionality_petrosian_fractal_dim",
    "dimensionality_katz_fractal_dim",
    "dimensionality_hurst_exp",
    "dimensionality_detrended_fluctuation_analysis",
    # Pick
    "pick_channel_pairs_preprocessor",
    "pick_channels_preprocessor",
    # Signal
    "signal_filter_preprocessor",
    "signal_envelope_preprocessor",
    "signal_mean",
    "signal_variance",
    "signal_skewness",
    "signal_kurtosis",
    "signal_std",
    "signal_root_mean_square",
    "signal_peak_to_peak",
    "signal_quantile",
    "signal_zero_crossings",
    "signal_line_length",
    "signal_hjorth_activity",
    "signal_hjorth_mobility",
    "signal_hjorth_complexity",
    "signal_decorrelation_time",
    # Spectral
    "spectral_preprocessor",
    "spectral_normalized_preprocessor",
    "spectral_db_preprocessor",
    "spectral_root_total_power",
    "spectral_moment",
    "spectral_entropy",
    "spectral_peak_frequency",
    "spectral_edge",
    "spectral_slope",
    "spectral_bands_power",
    "spectral_hjorth_activity",
    "spectral_hjorth_mobility",
    "spectral_hjorth_complexity",
    # Utils
    "preprocessor_as_feature",
]
