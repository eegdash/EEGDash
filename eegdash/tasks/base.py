# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Abstract base class for high-level ``eegdash.tasks`` task definitions.

A task bundles the metadata query, label definition, preprocessing recipe,
windowing recipe, split definitions, metrics, and baseline metadata that a
tutorial or downstream user needs to reproduce a benchmark.

The interface here is intentionally minimal: the plan
(``docs/tutorial_restructure_plan.md`` Workstream 2) explicitly says to
hard-code a few excellent task manifests before extracting deeper
abstractions. This base class therefore defines required hook methods and a
single thin :meth:`make_windows` adapter on top of braindecode windowers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

# The braindecode windowers are imported lazily inside :meth:`make_windows` so
# that simply importing the base class (e.g. for type checking or registry
# lookup) does not pay the import cost or fail when braindecode is missing in
# trimmed-down environments.


class EEGTask(ABC):
    """Abstract base class for an EEGDash high-level task.

    Subclasses should implement the seven abstract hooks below. ``EEGTask``
    purposefully does not own data loading: callers pass an
    ``EEGDashDataset``-compatible object into :meth:`make_windows` (and into
    any future split / metric helpers). This keeps the class trivially unit
    testable without network access.

    Attributes
    ----------
    name : str
        Registry key. Subclasses must set this as a class attribute.
    manifest_path : str | None
        Optional path (relative to the package or absolute) of the YAML
        manifest that documents the task. Used by tooling and tutorials.

    """

    #: Registry key for the task. Must be set by every subclass.
    name: str = ""

    #: Optional path to the YAML manifest documenting the task.
    manifest_path: str | None = None

    # ------------------------------------------------------------------ #
    # Required hook methods                                              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def metadata_query(self) -> dict[str, Any]:
        """Return the EEGDash metadata query that selects the task records.

        The returned mapping is the ``query`` argument that should be passed
        to :class:`eegdash.EEGDashDataset` (or the equivalent kwargs in the
        case of an ``EEGChallengeDataset``).
        """

    @abstractmethod
    def label_definition(self) -> dict[str, Any]:
        """Describe how labels are derived from the windowed dataset.

        Returns
        -------
        dict
            A mapping with at minimum the keys ``type`` (e.g. ``"classification"``
            or ``"regression"``), ``num_classes`` (or ``None`` for regression),
            ``mapping`` (label name to integer code), and ``source``
            (``"events"``, ``"metadata"``, ...).

        """

    @abstractmethod
    def preprocessing_recipe(self) -> list[Any]:
        """Return an ordered list of braindecode :class:`Preprocessor`-like steps.

        The list is consumed by :func:`braindecode.preprocessing.preprocess`.
        Returning an empty list is allowed for tasks that work on raw signals.
        """

    @abstractmethod
    def windowing_recipe(self) -> dict[str, Any]:
        """Return the windowing parameters used by :meth:`make_windows`.

        The dictionary must include a ``kind`` key (``"events"`` or
        ``"fixed"``) and the keyword arguments forwarded to the corresponding
        braindecode windower. ``kind`` is consumed by :meth:`make_windows`
        and removed from the kwargs before forwarding.
        """

    @abstractmethod
    def split_definitions(self) -> list[dict[str, Any]]:
        """Return the available split strategies for this task.

        Each entry should contain at least ``name`` and ``strategy`` keys and
        any strategy-specific hyperparameters (test size, random state,
        stratification key, ...). The :mod:`eegdash.splits` workstream is in
        charge of consuming these dictionaries.
        """

    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        """Return the metric specification for this task.

        The mapping must include ``primary`` (str) and may include
        ``secondary`` (list of str) and ``reporting`` (list of str).
        """

    @abstractmethod
    def baseline_metadata(self) -> dict[str, Any]:
        """Return reference baseline information for the task.

        Typical contents: ``model`` name, ``hyperparameters`` dictionary,
        ``reference_accuracy`` or other headline metric, and the subjects /
        splits used to compute it.
        """

    # ------------------------------------------------------------------ #
    # Concrete adapter on top of braindecode                             #
    # ------------------------------------------------------------------ #

    def make_windows(
        self,
        dataset: Any,
        engine: str = "braindecode",
        kind: Literal["fixed", "events"] | None = None,
        return_report: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Window a (preprocessed) concat-dataset using braindecode.

        Parameters
        ----------
        dataset : Any
            A ``BaseConcatDataset``-compatible object (typically the result of
            :func:`braindecode.preprocessing.preprocess` applied to an
            :class:`eegdash.EEGDashDataset`).
        engine : str, default ``"braindecode"``
            The windowing engine. Only ``"braindecode"`` is supported today;
            the parameter exists so that callers can already write
            forward-compatible code (e.g. ``engine="mne"`` later on).
        kind : {"fixed", "events"} or None
            Override for the windower selection. When ``None``, the value is
            taken from :meth:`windowing_recipe` (key ``"kind"``).
        return_report : bool, default ``True``
            When ``True``, return a ``(windows, report)`` tuple. The report
            dictionary records the engine, function name, kwargs that were
            forwarded and the relevant package version, which keeps tutorial
            output reproducible. When ``False``, only the windows are
            returned.
        **kwargs : Any
            Extra keyword arguments forwarded to the chosen braindecode
            windower. They take precedence over the values defined in
            :meth:`windowing_recipe`.

        Returns
        -------
        windows or (windows, report)
            The windowed dataset. When ``return_report`` is ``True``, the
            return value is a 2-tuple of ``(windows, report)``.

        """
        if engine != "braindecode":
            raise ValueError(
                f"engine={engine!r} is not supported yet; only 'braindecode' "
                "is implemented in eegdash.tasks."
            )

        # Lazy imports to keep the base module cheap to import.
        import braindecode
        from braindecode.preprocessing import (
            create_fixed_length_windows,
            create_windows_from_events,
        )

        recipe = dict(self.windowing_recipe())
        recipe_kind = recipe.pop("kind", None)
        # Strip the engine entry too if a subclass chose to record it there.
        recipe.pop("engine", None)
        effective_kind = kind if kind is not None else recipe_kind
        if effective_kind not in {"fixed", "events"}:
            raise ValueError(
                "windowing kind must be either 'fixed' or 'events', got "
                f"{effective_kind!r}."
            )

        # Caller-supplied kwargs win, then the recipe defaults fill the rest.
        forwarded: dict[str, Any] = dict(recipe)
        forwarded.update(kwargs)

        if effective_kind == "events":
            fn = create_windows_from_events
            canonical_name = "create_windows_from_events"
        else:
            fn = create_fixed_length_windows
            canonical_name = "create_fixed_length_windows"

        windows = fn(dataset, **forwarded)

        if not return_report:
            return windows

        report = {
            "engine": engine,
            "kind": effective_kind,
            "function": f"braindecode.preprocessing.{canonical_name}",
            "kwargs": forwarded,
            "package_versions": {"braindecode": braindecode.__version__},
        }
        return windows, report
