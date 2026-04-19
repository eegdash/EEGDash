# eegdash.features.output_types

Core Output Types.

This module defines the fundamental output types for feature preprocessors.

The module provides the classes:

- `BasePreprocessorOutputType` - The base abstract output type.
- `AsInputOutputType` - A “pass through” output type, enforcing the
  output type to match the input type.

<!-- !! processed by numpydoc !! -->

### Classes

| `AsInputOutputType`(preprocessor)          | A special class for preprocessors where the output type is the same as their input type.   |
|--------------------------------------------|--------------------------------------------------------------------------------------------|
| `BasePreprocessorOutputType`(preprocessor) | An abstract class representing a type of preprocessor output.                              |
| `SignalOutputType`(preprocessor)           | A class for preprocessors where the output type is raw-signal-like.                        |

### *class* eegdash.features.output_types.AsInputOutputType(preprocessor: Callable)

Bases: `BasePreprocessorOutputType`

A special class for preprocessors where the output type is the same
as their input type.

If used as a preprocessor predecessor, the preprocessor must not have any
other predecessors.

* **Parameters:**
  **preprocessor** (*callable*) – The underlying preprocessor callable.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.features.output_types.BasePreprocessorOutputType(preprocessor: Callable)

Bases: `ABC`, `Callable`

An abstract class representing a type of preprocessor output.

* **Parameters:**
  **preprocessor** (*callable*) – The underlying preprocessor callable.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.features.output_types.SignalOutputType(preprocessor: Callable)

Bases: `BasePreprocessorOutputType`

A class for preprocessors where the output type is raw-signal-like.

* **Parameters:**
  **preprocessor** (*callable*) – The underlying preprocessor callable.

<!-- !! processed by numpydoc !! -->
