# eegdash.features.trainable

Core Trainable Feature Interface.

This module defines the interface for creating trainable features.

The module provides the base class:

- `TrainableFeature` - The interface for features requiring a
  fitting phase.

<!-- !! processed by numpydoc !! -->

### Classes

| `TrainableFeature`()   | Abstract base class for features requiring a training phase.   |
|------------------------|----------------------------------------------------------------|

### *class* eegdash.features.trainable.TrainableFeature

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Abstract base class for features requiring a training phase.

This class provides the interface for features that must be
fitted on a representative dataset before they can process new samples.

#### \_is_trained

Internal flag indicating whether the feature has completed
its training phase.

* **Type:**
  [bool](https://docs.python.org/3/library/functions.html#bool)

<!-- !! processed by numpydoc !! -->

#### *abstractmethod* clear()

Reset the internal state of the feature.

This method must be implemented by subclasses to clear any learned
parameters, statistics, or buffers.

<!-- !! processed by numpydoc !! -->

#### *abstractmethod* partial_fit(\*x, y=None)

Update the extractor’s state using a single batch of data.

This method allows for incremental learning, making it possible to
train on datasets that are too large to fit into memory at once.

* **Parameters:**
  * **\*x** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* *ndarray*) – The input data batch.
  * **y** (*ndarray* *,* *optional*) – Target labels associated with the batch, required for supervised
    feature extraction methods.

<!-- !! processed by numpydoc !! -->

#### fit()

Finalize the training of the feature extractor.

This method should be called after the entire training set has been
processed via `partial_fit()`. It transitions the object to a
“trained” state, enabling the `__call__()` method.

<!-- !! processed by numpydoc !! -->
