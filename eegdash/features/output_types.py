r"""Core Output Types.

This module defines the fundamental output types for feature preprocessors.

The module provides the classes:

- :class:`BasePreprocessorOutputType` - The base abstract output type.
- :class:`AsInputOutputType` - A "pass through" output type, enforcing the
  output type to match the input type.
"""

import inspect
from abc import ABC
from collections.abc import Callable

__all__ = [
    "AsInputOutputType",
    "BasePreprocessorOutputType",
]


class BasePreprocessorOutputType(ABC):
    """An abstract class representing a type of preprocessor output.

    Parameters
    ----------
    preprocessor : callable
        The underlying preprocessor callable.

    """

    def __init__(self, preprocessor: Callable):
        super().__init__()
        self.preprocessor = preprocessor
        if "_metadata" in inspect.signature(preprocessor).parameters:
            self.__call__ = self._call_metadata
        else:
            self.__call__ = self._call

    def _call(self, *args, **kwargs):
        r"""Call the underlying preprocessor with the provided arguments."""
        return self.preprocessor(*args, **kwargs)

    def _call_metadata(self, *args, _metadata: dict, **kwargs):
        r"""Call the underlying preprocessor with the provided arguments and metadata."""
        return self.preprocessor(*args, _metadata=_metadata, **kwargs)


class AsInputOutputType(BasePreprocessorOutputType):
    """A special class for preprocessors where the output type is the same
    as their input type.

    If used as a preprocessor predecessor, the preprocessor must not have any
    other predecessors.

    Parameters
    ----------
    preprocessor : callable
        The underlying preprocessor callable.

    """

    pass
