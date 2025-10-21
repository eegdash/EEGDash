Feature Package Overview
========================

.. currentmodule:: eegdash.features

The :mod:`eegdash.features` namespace re-exports feature extractors,
decorators, and dataset utilities from the underlying submodules so callers can
import the most common helpers from a single place. To avoid duplicated
documentation in the API reference, the classes themselves are documented in
their defining modules (see the links below). This page focuses on the
high-level orchestration helpers that only live in the package ``__init__``.

High-level discovery helpers
----------------------------

.. autofunction:: get_all_features
.. autofunction:: get_feature_kind
.. autofunction:: get_feature_predecessors
.. autofunction:: get_all_feature_extractors
.. autofunction:: get_all_feature_kinds

Dataset and extraction utilities
--------------------------------

.. autofunction:: extract_features
.. autofunction:: fit_feature_extractors
.. autofunction:: load_features_concat_dataset

See also
--------

- :mod:`eegdash.features.extractors` for the feature-extraction base classes
  such as :class:`~eegdash.features.extractors.FeatureExtractor`.
- :mod:`eegdash.features.datasets` for dataset wrappers like
  :class:`~eegdash.features.datasets.FeaturesConcatDataset`.
- :mod:`eegdash.features.feature_bank.*` for the concrete feature families
  (complexity, connectivity, spectral, and more).
