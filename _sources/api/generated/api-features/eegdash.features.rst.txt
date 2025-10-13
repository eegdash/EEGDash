eegdash.features
================

.. automodule:: eegdash.features
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

   
   .. rubric:: Functions

   .. autosummary::
   
      bivariate_feature
      get_all_feature_extractors
      get_all_feature_kinds
      get_all_features
      get_feature_kind
      get_feature_predecessors
      load_features_concat_dataset
      extract_features
      fit_feature_extractors
      complexity_approx_entropy
      complexity_sample_entropy
      complexity_svd_entropy
      complexity_lempel_ziv
      connectivity_magnitude_square_coherence
      connectivity_imaginary_coherence
      connectivity_lagged_coherence
      dimensionality_higuchi_fractal_dim
      dimensionality_petrosian_fractal_dim
      dimensionality_katz_fractal_dim
      dimensionality_hurst_exp
      dimensionality_detrended_fluctuation_analysis
      signal_mean
      signal_variance
      signal_skewness
      signal_kurtosis
      signal_std
      signal_root_mean_square
      signal_peak_to_peak
      signal_quantile
      signal_zero_crossings
      signal_line_length
      signal_hjorth_activity
      signal_hjorth_mobility
      signal_hjorth_complexity
      signal_decorrelation_time
      spectral_root_total_power
      spectral_moment
      spectral_entropy
      spectral_edge
      spectral_slope
      spectral_bands_power
      spectral_hjorth_activity
      spectral_hjorth_mobility
      spectral_hjorth_complexity
   
   .. rubric:: Classes

   .. autosummary::
   
      FeaturesConcatDataset
      FeaturesDataset
      FeatureKind
      FeaturePredecessor
      BivariateFeature
      DirectedBivariateFeature
      FeatureExtractor
      MultivariateFeature
      TrainableFeature
      UnivariateFeature
      EntropyFeatureExtractor
      CoherenceFeatureExtractor
      CommonSpatialPattern
      HilbertFeatureExtractor
      SpectralFeatureExtractor
      NormalizedSpectralFeatureExtractor
      DBSpectralFeatureExtractor
   

