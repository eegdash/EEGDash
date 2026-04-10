r"""Feature Metadata Decorators.

This module provides  decorators used to annotate feature extraction
functions with structural metadata. These annotations define the dependency
graph (via predecessors) and the data format (via feature kinds).

The module provides the following decorators:

- :func:`feature_predecessor` — Specifies the required input transformation
  for a feature.
- :func:`feature_kind` — Defines the dimensionality of the feature output.

  - :func:`univariate_feature` — Sugar for per-channel features.
  - :func:`bivariate_feature` — Sugar for per channel-pair features.
  - :func:`multivariate_feature` — Sugar for global/all-channel features.

- :func:`metadata_preprocessor` — Specifies a preprocessor returning a modified
  metadata instance.
- :func:`channel_pairer` — Specifies a preprocessor that creates channel pairs.

  - :func:`channel_pairer_undirected` — Sugar for undirected pairs.
  - :func:`channel_pairer_directed` — Sugar for directed pairs.

"""

import functools
import inspect
import os
import re
from collections.abc import Callable
from functools import partial
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
from .output_types import BasePreprocessorOutputType, SignalOutputType

__all__ = [
    "bivariate_feature",
    "channel_pairer",
    "channel_pairer_directed",
    "channel_pairer_undirected",
    "feature_kind",
    "feature_predecessor",
    "metadata_preprocessor",
    "multivariate_feature",
    "preprocessor_output_type",
    "univariate_feature",
]

_WRAPPER_ASSIGNMENTS = [
    *functools.WRAPPER_ASSIGNMENTS,
    "parent_extractor_type",
    "feature_kind",
    "metadata_preprocessor",
]
SPHINX_BUILD = bool(os.environ.get("SPHINX_BUILD", ""))


def _str_replacer(match: re.Match[str], *, new_entries: str = "") -> str:
    header, existing_params, suffix = match.groups()
    # Ensure we don't double-indent if we are appending
    return f"{header}{existing_params.rstrip()}{new_entries}{suffix}"


def _add_params(wrapper: Callable, wrapped: Callable, new_args: List[dict]) -> Callable:
    """Add new parameter to signature and docs."""
    # Update the Signature (for help() and introspection)
    sig = inspect.signature(wrapped)
    params = list(sig.parameters.values())
    args_idx, kwargs_idx = len(params) - 1, len(params) - 1
    for i, param in enumerate(params):
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            args_idx = i
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            args_idx = min(args_idx, i)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            args_idx = min(args_idx, i)
            kwargs_idx = i
    for new_param in new_args:
        param = new_param["signature"]
        param_idx = -1
        for i, p in enumerate(params):
            if p.name == param.name:
                param_idx = i
                break
        if param_idx >= 0:
            params[param_idx] = param
        elif param.kind in [
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ]:
            params.insert(args_idx, new_param["signature"])
            args_idx += 1
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            params.insert(kwargs_idx, new_param["signature"])
            kwargs_idx += 1
    wrapper.__signature__ = sig.replace(parameters=params)

    # Update the Docstring (Numpydoc Format)
    if wrapped.__doc__:
        original_doc = wrapped.__doc__
        lines = original_doc.splitlines()
        indent = ""
        if len(lines) > 1:
            # Look for the first non-empty line after the summary to find the indent
            for line in lines[1:]:
                if line.strip():
                    indent = re.match(r"^(\s*)", line).group(1)
                    break

        new_param_entry = ""
        for new_param in new_args:
            param = new_param["signature"]
            new_param_entry += f"\n{indent}{param.name} : {param.annotation}"
            if "doc" in new_param:
                new_param_entry += f"\n{indent}    {new_param['doc']}"

        # 2. Update the Docstring
        if f"\n\n{indent}Parameters\n{indent}----------\n" in original_doc:
            # Match the Parameters block, capturing the underline and the content
            # until the next section (double newline + non-space) or end of string
            pattern = rf"(Parameters\s*\n\s*{indent}-+)(.*?)(\n\s*\n\s*\S|$)"
            updated_doc = re.sub(
                pattern,
                partial(_str_replacer, new_entries=new_param_entry),
                original_doc,
                flags=re.DOTALL,
            )
        else:
            # Create a new Parameters section with correct indentation
            new_section = (
                f"\n\n{indent}Parameters\n{indent}----------\n{new_param_entry}\n"
            )
            updated_doc = original_doc.rstrip() + new_section
        wrapper.__doc__ = updated_doc
    return wrapper


def update_wrapper(
    wrapper: Callable,
    wrapped: Callable,
    assigned: list = _WRAPPER_ASSIGNMENTS,
    updated: dict = functools.WRAPPER_UPDATES,
    new_args: List[dict] = [],
) -> Callable:
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
    if SPHINX_BUILD:  # fool sphinx
        wrapped = _add_params(wrapped, wrapped, new_args)
        return wrapped
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
    if new_args:
        wrapper = _add_params(wrapper, wrapped_f, new_args)
    else:
        wrapper.__signature__ = inspect.signature(wrapped_f)
    # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
    # from the wrapped function when updating __dict__
    wrapper.__wrapped__ = wrapped
    # Return the wrapper so this can be used as a decorator via partial()
    return wrapper


def wraps(
    wrapped: Callable,
    assigned: list = _WRAPPER_ASSIGNMENTS,
    updated: dict = functools.WRAPPER_UPDATES,
    new_args: List[dict] = [],
) -> Callable:
    """Decorator factory to apply update_wrapper() to a wrapper function

    Returns a decorator that invokes update_wrapper() with the decorated
    function as the wrapper argument and the arguments to wraps() as the
    remaining arguments. Default arguments are as for update_wrapper().
    This is a convenience function to simplify applying partial() to
    update_wrapper().
    """
    return partial(
        update_wrapper,
        wrapped=wrapped,
        assigned=assigned,
        updated=updated,
        new_args=new_args,
    )


def _feature_predecessor_update(
    func: Callable,
    *,
    parent_extractor_type: List[Callable | Type],
) -> Callable:
    r"""Apply the :func:`feature_predecessor` decorator to a function.

    Parameters
    ----------
    func : callable
        The feature extraction function to decorate.
    parent_extractor_type : list of callable
        A list of preprocessing functions that this feature immediately
        depends on.
        Default is [:class:`~eegdash.features.output_types.SignalOutputType`].

    Returns
    -------
    callable
        The decorated function with the `parent_extractor_type`
        attribute attached.

    """
    parent_func = parent_extractor_type
    if not parent_func:
        parent_func = [SignalOutputType]
    for p_func in parent_func:
        assert callable(p_func) or issubclass(p_func, BasePreprocessorOutputType)
    f = get_underlying_func(func)
    f.parent_extractor_type = parent_func
    return func


def feature_predecessor(*parent_extractor_type: List[Callable]) -> Callable:
    r"""Decorator to specify parent extractors for a feature function.

    This decorator attaches a list of immediate parent preprocessing steps to
    a feature extraction function. This metadata is used by the
    :class:`~eegdash.features.extractors.FeatureExtractor` to validate the
    execution tree.

    Parameters
    ----------
    *parent_extractor_type : list of callable
        A list of preprocessing functions that this feature immediately
        depends on.
        Default is [:class:`~eegdash.features.output_types.SignalOutputType`].

    Notes
    -----
    A feature can have multiple potential predecessors.

    """
    return partial(
        _feature_predecessor_update, parent_extractor_type=parent_extractor_type
    )


def _feature_kind_update(func: Callable, *, kind: MultivariateFeature) -> Callable:
    r"""Apply the :func:`feature_kind` decorator to a function.

    Parameters
    ----------
    func : callable
        The feature extraction function to decorate.
    kind : ~eegdash.features.extractors.MultivariateFeature
        An instance of a feature kind class, such as
        :class:`~eegdash.features.extractors.UnivariateFeature` or
        :class:`~eegdash.features.extractors.BivariateFeature`.

    Returns
    -------
    callable
        The decorated function with the ``feature_kind`` attribute set.

    """
    f = get_underlying_func(func)
    f.feature_kind = kind
    return func


def feature_kind(kind: MultivariateFeature) -> Callable:
    r"""Decorator to specify the operational dimensionality of a feature.

    This decorator attaches a "feature kind" instance to a function,
    determining how the :class:`~eegdash.features.extractors.FeatureExtractor`
    should map the resulting numerical arrays to channel names.

    Parameters
    ----------
    kind : ~eegdash.features.extractors.MultivariateFeature
        An instance of a feature kind class, such as
        :class:`~eegdash.features.extractors.UnivariateFeature` or
        :class:`~eegdash.features.extractors.BivariateFeature`.

    """
    return partial(_feature_kind_update, kind=kind)


# Syntax sugar
univariate_feature = feature_kind(UnivariateFeature())
r"""Decorator to mark a feature as univariate.

Indicates that the feature is computed for each channel independently.
The output will be formatted as a dictionary with keys matching the
original channel names.
"""


bivariate_feature = feature_kind(BivariateFeature())
r"""Decorator to mark a feature as bivariate.

Specifies that the feature operates on pairs of channels.
The output will be formatted as a dictionary with keys matching the
original channel name pairs.

"""


multivariate_feature = feature_kind(MultivariateFeature())
r"""Decorator to mark a feature as multivariate.

Indicates that the feature operates on all channels simultaneously. The
output naming convention is determined by the feature's internal logic
rather than channel labels.
"""


def metadata_preprocessor(func: Callable) -> Callable:
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


def _preprocessor_output_type_wrap(
    preprocessor: Callable, *, output_type: Type
) -> Callable:
    r"""Apply the :func:`preprocessor_output_type` decorator to a preprocessor function.

    Parameters
    ----------
    preprocessor : callable
        The preprocessor function to decorate.
    output_type : Type
        The expected output type for the preprocessor. Must be a
        :class:`~eegdash.features.output_types.BasePreprocessorOutputType`.

    Returns
    -------
    callable
        An instance of the class named after the preprocessor, inheriting from
        the specified `output_type`, with the original preprocessor function as
        its implementation.

    """
    if (
        not inspect.isclass(output_type)
        or not issubclass(output_type, BasePreprocessorOutputType)
        or output_type is BasePreprocessorOutputType
    ):
        raise ValueError(
            "`output_type` must inherit from `BasePreprocessorOutputType`, "
            + f"got `{output_type}`."
        )
    preprocessor_class = type(
        preprocessor.__name__,
        (output_type,),
        {
            "__call__": output_type._call_metadata
            if "_metadata" in inspect.signature(preprocessor).parameters
            else output_type._call,
        },
    )
    preprocessor_instance = preprocessor_class(preprocessor)
    preprocessor_instance = update_wrapper(preprocessor_instance, preprocessor)
    return preprocessor_instance


def preprocessor_output_type(output_type: Type) -> Callable:
    r"""Decorator to specify the expected output type of a preprocessor.

    Parameters
    ----------
    output_type : Type
        The expected output type for the preprocessor. Must be a
        :class:`~eegdash.features.output_types.BasePreprocessorOutputType`.

    Raises
    ------
    ValueError
        If the provided `output_type` does not inherit from
        :class:`~eegdash.features.preprocessors.BasePreprocessorOutputType`.

    """
    return partial(_preprocessor_output_type_wrap, output_type=output_type)


def _channel_pairer_wrap(func: Callable, *, directed: bool = False) -> Callable:
    r"""Apply the :func:`channel_pairer` decorator to a function.

    Parameters
    ----------
    func : callable
        The preprocessor function to decorate.
    directed : bool
        Whether the preprocessor assumes *directed* or *undirected* bivariate
        iteration.

    Returns
    -------
    callable
        The decorated function with the extra ``pairs`` keyword parameter.

    """
    pairs_param = {
        "signature": inspect.Parameter(
            name="pairs",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=Iterable[Tuple[str, str]] | None,
        ),
        "doc": r"A list of channel pairs to pick.",
    }

    @metadata_preprocessor
    @wraps(func, new_args=[pairs_param])
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
        _metadata["ch_pair_iterator"] = BivariateIterator(pairs, directed=directed)
        f = get_underlying_func(func)
        if "_metadata" in inspect.signature(f).parameters:
            kwargs["_metadata"] = _metadata
        if hasattr(f, "metadata_preprocessor") and f.metadata_preprocessor:
            return (*func(*args, **kwargs),)
        else:
            return (*func(*args, **kwargs), _metadata)

    return func_wrapper


def channel_pairer(directed: bool = False) -> Callable:
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
    return partial(_channel_pairer_wrap, directed=directed)


# Syntax sugar
channel_pairer_undirected = channel_pairer(directed=False)
r"""Decorator to mark a feature preprocessor as an undirected channel pairer.

Specifies that the feature preprocessor operates on undirected pairs of
channels.

"""

channel_pairer_directed = channel_pairer(directed=True)
r"""Decorator to mark a feature preprocessor as an undirected channel pairer.

Specifies that the feature preprocessor operates on undirected pairs of
channels.

"""
