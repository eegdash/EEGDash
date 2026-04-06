r"""Feature Metadata Decorators.

This module provides  decorators used to annotate feature extraction
functions with structural metadata. These annotations define the dependency
graph (via predecessors) and the data format (via feature kinds).

The module provides the following decorators:

- :class:`FeaturePredecessor` — Specifies the required input transformation
  for a feature.
- :class:`FeatureKind` — Defines the dimensionality of the feature output.
- :func:`univariate_feature` — Sugar for per-channel features.
- :func:`bivariate_feature` — Sugar for channel-pair features.
- :func:`multivariate_feature` — Sugar for global/all-channel features.

"""

import functools
import inspect
import os
from collections.abc import Callable
from typing import Iterable, List, Tuple, Type

from .base_utils import (
    BivariateIterator,
    channel_names_to_indices,
    get_underlying_func,
)
from .kinds import (
    BivariateFeature,
    MultivariateFeature,
    UnivariateFeature,
)
from .output_types import (
    BasePreprocessorOutputType,
)

__all__ = [
    "bivariate_feature",
    "FeatureKind",
    "FeaturePredecessor",
    "metadata_preprocessor",
    "multivariate_feature",
    "PreprocessorOutputType",
    "univariate_feature",
]

_WRAPPER_ASSIGNMENTS = [
    *functools.WRAPPER_ASSIGNMENTS,
    "parent_extractor_type",
    "feature_kind",
    "metadata_preprocessor",
]
SPHINX_BUILD = bool(os.environ.get("SPHINX_BUILD", ""))


def _update_wrapper(
    wrapper,
    wrapped,
    assigned=_WRAPPER_ASSIGNMENTS,
    updated=functools.WRAPPER_UPDATES,
):
    """Update a wrapper function to look like the wrapped function

    wrapper is the function to be updated
    wrapped is the original function
    assigned is a tuple naming the attributes assigned directly
    from the wrapped function to the wrapper function (defaults to
    eegdash.features.decorators._WRAPPER_ASSIGNMENTS)
    updated is a tuple naming the attributes of the wrapper that
    are updated with the corresponding attribute from the wrapped
    function (defaults to functools.WRAPPER_UPDATES)
    """
    if SPHINX_BUILD:
        return wrapped  # fool sphinx
    wrapped_f = get_underlying_func(wrapped)
    for attr in assigned:
        try:
            value = getattr(wrapped_f, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)
    for attr in updated:
        getattr(wrapper, attr).update(getattr(wrapped_f, attr, {}))
    wrapper.__signature__ = inspect.signature(wrapped)
    # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
    # from the wrapped function when updating __dict__
    wrapper.__wrapped__ = wrapped
    # Return the wrapper so this can be used as a decorator via partial()
    return wrapper


def _wraps(wrapped, assigned=_WRAPPER_ASSIGNMENTS, updated=functools.WRAPPER_UPDATES):
    """Decorator factory to apply update_wrapper() to a wrapper function

    Returns a decorator that invokes update_wrapper() with the decorated
    function as the wrapper argument and the arguments to wraps() as the
    remaining arguments. Default arguments are as for update_wrapper().
    This is a convenience function to simplify applying partial() to
    update_wrapper().
    """
    return functools.partial(
        _update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated
    )


class FeaturePredecessor:
    r"""Decorator to specify parent extractors for a feature function.

    This decorator attaches a list of immediate parent preprocessing steps to
    a feature extraction function. This metadata is used by the
    :class:`~eegdash.features.extractors.FeatureExtractor` to validate the
    execution tree.

    Parameters
    ----------
    *parent_extractor_type : list of callable or None
        A list of preprocessing functions that this feature immediately
        depends on. Use ``None`` to indicate that the feature can operate
        directly on raw signal arrays.

    Attributes
    ----------
    parent_extractor_type : list of callable or None
        The stored list of predecessor functions.

    Notes
    -----
    A feature can have multiple potential predecessors.

    """

    def __init__(self, *parent_extractor_type: List[Callable | None]):
        parent_func = parent_extractor_type
        if not parent_func:
            parent_func = [None]
        for p_func in parent_func:
            assert p_func is None or callable(p_func)
        self.parent_extractor_type = parent_func

    def __call__(self, func: Callable) -> Callable:
        r"""Apply the decorator to a function.

        Parameters
        ----------
        func : callable
            The feature extraction function to decorate.

        Returns
        -------
        callable
            The decorated function with the `parent_extractor_type`
            attribute attached.

        """
        f = get_underlying_func(func)
        f.parent_extractor_type = self.parent_extractor_type
        return func


class FeatureKind:
    r"""Decorator to specify the operational dimensionality of a feature.

    This decorator attaches a "feature kind" instance to a function,
    determining how the :class:`~eegdash.features.extractors.FeatureExtractor`
    should map the resulting numerical arrays to channel names.

    Parameters
    ----------
    feature_kind : ~eegdash.features.extractors.MultivariateFeature
        An instance of a feature kind class, such as
        :class:`~eegdash.features.extractors.UnivariateFeature` or
        :class:`~eegdash.features.extractors.BivariateFeature`.

    Attributes
    ----------
    feature_kind : ~eegdash.features.extractors.MultivariateFeature
        The stored kind instance used for output formatting.

    """

    def __init__(self, feature_kind: MultivariateFeature):
        self.feature_kind = feature_kind

    def __call__(self, func: Callable) -> Callable:
        r"""Apply the decorator to a function.

        Parameters
        ----------
        func : callable
            The feature extraction function to decorate.

        Returns
        -------
        callable
            The decorated function with the `feature_kind` attribute set.

        """
        f = get_underlying_func(func)
        f.feature_kind = self.feature_kind
        return func


# Syntax sugar
univariate_feature = FeatureKind(UnivariateFeature())
r"""Decorator to mark a feature as univariate.

Indicates that the feature is computed for each channel independently.
The output will be formatted as a dictionary with keys matching the
original channel names.
"""


bivariate_feature = FeatureKind(BivariateFeature())
r"""Decorator to mark a feature as bivariate.

Specifies that the feature operates on pairs of channels.
The output will be formatted as a dictionary with keys matching the
original channel name pairs.

"""


multivariate_feature = FeatureKind(MultivariateFeature())
r"""Decorator to mark a feature as multivariate.

Indicates that the feature operates on all channels simultaneously. The
output naming convention is determined by the feature's internal logic
rather than channel labels.
"""


class PreprocessorOutputType:
    r"""Decorator to specify the expected output type of a preprocessor.

    Parameters
    ----------
    output_type : Type
        The expected output type for the preprocessor.

    Raises
    ------
    ValueError
        If the provided `output_type` does not inherit from
        :class:`~eegdash.features.preprocessors.BasePreprocessorOutputType`.

    """

    def __init__(self, output_type: Type):
        if (
            not inspect.isclass(output_type)
            or not issubclass(output_type, BasePreprocessorOutputType)
            or output_type is BasePreprocessorOutputType
        ):
            raise ValueError(
                f"`output_type` must inherit from `PreprocessorOutputType`, got `{output_type}`."
            )
        self.output_type = output_type

    def __call__(self, preprocessor: Callable) -> Callable:
        r"""Apply the decorator to a preprocessor function.

        Parameters
        ----------
        preprocessor : callable
            The preprocessor function to decorate.

        Returns
        -------
        callable
            An instance of the class named after the preprocessor, inheriting from the specified
            `output_type`, with the original preprocessor function as its implementation.

        """
        preprocessor_class = type(
            preprocessor.__name__,
            (self.output_type,),
            {
                "__call__": self.output_type._call_metadata
                if "_metadata" in inspect.signature(preprocessor).parameters
                else self.output_type._call,
            },
        )
        preprocessor_instance = preprocessor_class(preprocessor)
        preprocessor_instance = _update_wrapper(preprocessor_instance, preprocessor)
        return preprocessor_instance


def metadata_preprocessor(func: Callable):
    r"""Decorator to set a feature preprocessor as a metadata preprocessor.

    A metadata preprocessor must get a keyword argument named ``"_metadata"``
    and return a copy of it as its last output argument.

    Parameters
    ----------
    func : callable
        The feature preprocessor function to decorate.

    Returns
    -------
    callable
        The decorated function with the `metadata_preprocessor` attribute set.

    """
    f = get_underlying_func(func)
    if "_metadata" not in inspect.signature(f).parameters:
        raise TypeError(
            f"{f.__name__} cannot be set as a metadata preprocessor "
            + "because it does not get a keyword argument named "
            + "``'_metadata'``"
        )
    f.metadata_preprocessor = True
    return func


class ChannelPairer:
    r"""Decorator to set a feature preprocessor as a channel pairer.

    This decorator lets a feature preprocessor get an additional ``pairs``
    keyword argument, and sets a metadata field named ``'ch_pair_iterator'``
    containing a :class:`~eegdash.features.base_utils.BivariateIterator`
    accordingly before calling the underlying preprocessor.

    Parameters
    ----------
    directed : bool
        Whether the preprocessor assumes *directed* or *undirected* bivariate
        iteration.

    """

    def __init__(self, directed: bool = False):
        self.directed = directed

    def __call__(self, func: Callable):
        @metadata_preprocessor
        @_wraps(func)
        def func_wrapper(
            *args,
            _metadata: dict,
            pairs: Iterable[Tuple[str, str]] | None = None,
            **kwargs,
        ):
            ch_names = _metadata["info"]["ch_names"]
            if pairs is None:
                pairs = len(ch_names)
            else:
                pairs = list(
                    zip(*[channel_names_to_indices(x, ch_names) for x in zip(*pairs)])
                )
            _metadata["ch_pair_iterator"] = BivariateIterator(
                pairs, directed=self.directed
            )
            f = get_underlying_func(func)
            if "_metadata" in inspect.signature(f).parameters:
                kwargs["_metadata"] = _metadata
            if hasattr(f, "metadata_preprocessor") and f.metadata_preprocessor:
                return (*func(*args, **kwargs),)
            else:
                return (*func(*args, **kwargs), _metadata)

        return func_wrapper


# Syntax sugar
channel_pairer = ChannelPairer(directed=False)
r"""Decorator to mark a feature preprocessor as an undirected channel pairer.

Specifies that the feature preprocessor operates on undirected pairs of
channels.

"""

channel_directed_pairer = ChannelPairer(directed=True)
r"""Decorator to mark a feature preprocessor as an undirected channel pairer.

Specifies that the feature preprocessor operates on undirected pairs of
channels.

"""
