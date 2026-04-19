# eegdash.features.feature_bank.csp

## Common Spatial Pattern Features Extraction

This module provides the Common Spatial Pattern (CSP) feature extractor
for signal classification.

### Data Shape Convention

This module follows a **Time-Last** convention:

* **Input:** `(..., time)`
* **Output:** `(...,)`

All functions collapse the last dimension (time), returning an ndarray of
features corresponding to the leading dimensions (e.g., subjects, channels).

<!-- !! processed by numpydoc !! -->

### Classes

| `CommonSpatialPattern`()   | Common Spatial Pattern (CSP) for binary signal classification.   |
|----------------------------|------------------------------------------------------------------|

### *class* eegdash.features.feature_bank.csp.CommonSpatialPattern

Bases: `TrainableFeature`

Common Spatial Pattern (CSP) for binary signal classification.

CSP finds spatial filters that maximize the variance for one class while
minimizing it for the other. It transforms multi-channel signals into a
subspace where the differences between two conditions are most prominent.

#### \_weights

The spatial filter matrix.

* **Type:**
  ndarray

#### \_eigvals

The eigenvalues representing the variance ratio for class 0.

* **Type:**
  ndarray

#### \_means

The class-wise means used for centering.

* **Type:**
  ndarray

#### \_covs

The class-wise covariance matrices.

* **Type:**
  ndarray

### Notes

This implementation supports online learning through `partial_fit`,
allowing the model to be updated with new batches.

For a theoretical overview of Common Spatial Patterns, see the
[Wikipedia entry](https://en.wikipedia.org/wiki/Common_spatial_pattern).

<!-- !! processed by numpydoc !! -->

#### clear()

Reset the internal state of the feature extractor.

#### SEE ALSO
`clear()`

<!-- !! processed by numpydoc !! -->

#### partial_fit(x, y=None)

Incrementally update class-wise mean and covariance statistics.

* **Parameters:**
  * **x** (*ndarray*) – Input array of shape (n_epochs, n_channels, n_times).
  * **y** (*ndarray*) – Class labels for each epoch (must contain exactly two classes).
* **Raises:**
  **AssertionError** – If more than two unique labels are detected across all
  partial fits.

<!-- !! processed by numpydoc !! -->

#### *static* transform_input(x)

Reshape and transpose epoch data for matrix operations.

Converts 3D epoch data into a 2D format suitable for covariance
estimation and spatial filtering. The temporal dimension is
collapsed into the samples dimension.

* **Parameters:**
  **x** (*ndarray*) – Input array of shape (n_epochs, n_channels, n_times).
* **Returns:**
  Reshaped array of shape (n_epochs \* n_times, n_channels).
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

#### fit()

Solve the generalized eigenvalue problem to find spatial filters.

Calculates the filters $W$ such that the ratio of variances between
the two classes is maximized. Filters are sorted by their
discriminative power (distance from 0.5 eigenvalue).

#### SEE ALSO
`fit()`

### Notes

For more details on the CSP algorithm, visit the
[Wikipedia entry](https://en.wikipedia.org/wiki/Common_spatial_pattern).

<!-- !! processed by numpydoc !! -->

#### feature_kind *= <eegdash.features.kinds.MultivariateFeature object>*

#### parent_extractor_type *= [<class 'eegdash.features.output_types.SignalOutputType'>]*
