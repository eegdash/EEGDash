# eegdash.features.output_types

Core Output Types.

This module defines the fundamental output types for feature preprocessors.

The module provides the classes:

- `BasePreprocessorOutputType` - The base abstract output type.
- `AsInputOutputType` - A “pass through” output type, enforcing the
  output type to match the input type.

<!-- !! processed by numpydoc !! -->

### Classes

| `AsInputOutputType`()          | A special class for preprocessors where the output type is the same as their input type.   |
|--------------------------------|--------------------------------------------------------------------------------------------|
| `BasePreprocessorOutputType`() | An abstract class representing a type of preprocessor output.                              |
| `SignalOutputType`()           | A class for preprocessors where the output type is signal-like.                            |

### *class* eegdash.features.output_types.AsInputOutputType

Bases: `BasePreprocessorOutputType`

A special class for preprocessors where the output type is the same
as their input type.

If used as a preprocessor predecessor, the preprocessor must not have any
other predecessors.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.features.output_types.BasePreprocessorOutputType

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

An abstract class representing a type of preprocessor output.

<!-- !! processed by numpydoc !! -->

#### *classmethod* validate_output(\*output, \_metadata)

<!-- !! processed by numpydoc !! -->

### *class* eegdash.features.output_types.SignalOutputType

Bases: `BasePreprocessorOutputType`

A class for preprocessors where the output type is signal-like.

<!-- !! processed by numpydoc !! -->

#### *classmethod* validate_output(\*output, \_metadata)

<!-- !! processed by numpydoc !! -->
