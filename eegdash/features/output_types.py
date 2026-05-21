r"""Core Output Types.

This module defines the fundamental output types for feature preprocessors.

The module provides the classes:

- :class:`BasePreprocessorOutputType` - The base abstract output type.
- :class:`AsInputOutputType` - A "pass through" output type, enforcing the
  output type to match the input type.
"""

from abc import ABC
from collections.abc import Callable

import numpy as np

__all__ = [
    "AsInputOutputType",
    "BasePreprocessorOutputType",
    "SignalOutputType",
]


class BasePreprocessorOutputType(ABC, Callable):
    """An abstract class representing a type of preprocessor output."""

    @classmethod
    def validate_output(cls, *output, _metadata):
        pass


class AsInputOutputType(BasePreprocessorOutputType):
    """A special class for preprocessors where the output type is the same
    as their input type.

    If used as a preprocessor predecessor, the preprocessor must not have any
    other predecessors.

    """

    pass


class SignalOutputType(BasePreprocessorOutputType):
    """A class for preprocessors where the output type is signal-like."""

    @classmethod
    def validate_output(cls, *output, _metadata):
        assert (
            len(output) == 1
            and isinstance(output[0], np.ndarray)
            and len(output[0].shape) == 3
            and output[0].shape[0] == _metadata["batch_size"]
            and output[0].shape[1] == len(_metadata["info"]["ch_names"])
        )
