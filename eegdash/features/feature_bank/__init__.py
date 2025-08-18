"""Feature bank public API exports.

This module consolidates and re-exports the feature extractors and feature
functions so users can import them directly from
``eegdash.features.feature_bank``.
"""

from .complexity import (
    EntropyFeatureExtractor,
    complexity_approx_entropy,
    complexity_lempel_ziv,
    complexity_sample_entropy,
    complexity_svd_entropy,
)
from .connectivity import (
    CoherenceFeatureExtractor,
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
from .signal import (
    HilbertFeatureExtractor,
    signal_decorrelation_time,
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
    DBSpectralFeatureExtractor,
    NormalizedSpectralFeatureExtractor,
    SpectralFeatureExtractor,
    spectral_bands_power,
    spectral_edge,
    spectral_entropy,
    spectral_hjorth_activity,
    spectral_hjorth_complexity,
    spectral_hjorth_mobility,
    spectral_moment,
    spectral_root_total_power,
    spectral_slope,
)

__all__ = [
    # Complexity
    "EntropyFeatureExtractor",
    "complexity_approx_entropy",
    "complexity_sample_entropy",
    "complexity_svd_entropy",
    "complexity_lempel_ziv",
    # Connectivity
    "CoherenceFeatureExtractor",
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
    # Signal
    "HilbertFeatureExtractor",
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
    "SpectralFeatureExtractor",
    "NormalizedSpectralFeatureExtractor",
    "DBSpectralFeatureExtractor",
    "spectral_root_total_power",
    "spectral_moment",
    "spectral_entropy",
    "spectral_edge",
    "spectral_slope",
    "spectral_bands_power",
    "spectral_hjorth_activity",
    "spectral_hjorth_mobility",
    "spectral_hjorth_complexity",
]
