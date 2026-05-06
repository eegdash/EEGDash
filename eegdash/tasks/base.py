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
single :meth:`make_windows` adapter on top of braindecode windowers.

Workstream 4 of the plan (lines 2853-2898) promotes :meth:`make_windows`
from a thin stub into a task-level convenience layer that:

- Resolves a dataset (either supplied by the caller or built from
  ``task.dataset`` / ``task.subjects``).
- Parses time-string window/stride specifications (``"2s"``, ``"500ms"``,
  ``"1.5s"``, or raw integer samples).
- Forwards to :func:`braindecode.preprocessing.create_fixed_length_windows`
  or :func:`braindecode.preprocessing.create_windows_from_events`.
- Returns a ``(windows, report)`` tuple with reproducibility metadata --
  function name, kwargs, library versions, sampling rate, total window
  count, per-subject window counts, and a manifest hash.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Literal

# The braindecode windowers are imported lazily inside :meth:`make_windows` so
# that simply importing the base class (e.g. for type checking or registry
# lookup) does not pay the import cost or fail when braindecode is missing in
# trimmed-down environments.

# Sentinel returned by :func:`_resolve_sfreq` when the sampling rate cannot
# be discovered (e.g. when a stub dataset is being used in unit tests).
_UNKNOWN_SFREQ: float | None = None


# --------------------------------------------------------------------------- #
# Time-string parsing                                                         #
# --------------------------------------------------------------------------- #

# Accept ``"2s"``, ``"500ms"``, ``"1.5s"`` and raw integers. The regex below
# is intentionally strict: anything else falls through to a clear error.
_TIME_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)\s*$")


def _parse_time_to_samples(spec: str | int | float, sfreq: float | None) -> int:
    """Convert a time spec to an integer number of samples.

    Parameters
    ----------
    spec : str | int | float
        Either an integer number of samples, or a string time spec such as
        ``"2s"``, ``"500ms"`` or ``"1.5s"``.
    sfreq : float | None
        Sampling frequency in Hertz. Required when ``spec`` is a string.

    Returns
    -------
    int
        The corresponding window length in samples.

    Raises
    ------
    ValueError
        If ``spec`` is malformed or if a time string is supplied without a
        sampling rate to convert against.

    """
    if isinstance(spec, bool):  # pragma: no cover -- defensive
        # ``bool`` is a subclass of ``int``; reject it explicitly.
        raise ValueError(f"time spec must not be a bool, got {spec!r}.")
    if isinstance(spec, int):
        return int(spec)
    if isinstance(spec, float):
        # Pure floats are interpreted as samples too, but only if integer-valued.
        if not spec.is_integer():
            raise ValueError(
                f"float time spec {spec!r} is not an integer number of samples; "
                "use a string like '0.5s' if you meant a duration."
            )
        return int(spec)
    if not isinstance(spec, str):
        raise ValueError(
            f"time spec must be a string, int, or float, got {type(spec).__name__}."
        )

    match = _TIME_RE.match(spec)
    if not match:
        raise ValueError(
            f"invalid time spec {spec!r}; expected forms like '2s', '500ms' or "
            "an integer number of samples."
        )
    value = float(match.group("value"))
    unit = match.group("unit")
    if sfreq is None:
        raise ValueError(
            f"cannot convert time spec {spec!r} to samples without a sampling "
            "rate; pass an integer sample count or supply a dataset that "
            "exposes ``info['sfreq']``."
        )
    seconds = value / 1000.0 if unit == "ms" else value
    samples = int(round(seconds * float(sfreq)))
    if samples <= 0:
        raise ValueError(
            f"time spec {spec!r} resolves to {samples} samples at "
            f"{sfreq} Hz; expected a positive duration."
        )
    return samples


# --------------------------------------------------------------------------- #
# Manifest hashing                                                            #
# --------------------------------------------------------------------------- #


def _manifest_hash(payload: dict[str, Any]) -> str:
    """Return a deterministic short SHA256 hash for a windowing manifest.

    The payload is serialised with ``sort_keys=True`` and ``default=str`` so
    that non-JSON-native pieces (e.g. tuples, ``Path`` objects) hash
    consistently across runs.
    """
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Library version probe                                                       #
# --------------------------------------------------------------------------- #


def _library_versions() -> dict[str, str]:
    """Best-effort version dictionary for the report.

    We try braindecode, mne, and eegdash; missing packages are simply
    omitted so that callers always get a populated dict for the libraries
    that *are* installed.
    """
    versions: dict[str, str] = {}
    for pkg in ("braindecode", "mne", "eegdash"):
        try:
            mod = __import__(pkg)
        except Exception:  # pragma: no cover -- defensive
            continue
        version = getattr(mod, "__version__", None)
        if version is not None:
            versions[pkg] = str(version)
    return versions


# --------------------------------------------------------------------------- #
# Sampling-rate / sub-dataset introspection                                   #
# --------------------------------------------------------------------------- #


def _resolve_sfreq(dataset: Any) -> float | None:
    """Discover the sampling rate of a (concat) dataset, if possible.

    Looks at ``dataset.datasets[0]`` -- which may be either an EEGDash raw
    dataset or a windowed dataset -- and tries the standard MNE locations
    (``raw.info['sfreq']`` then ``windows.info['sfreq']``). Returns ``None``
    when nothing is available so the caller can degrade gracefully.
    """
    sub_datasets = getattr(dataset, "datasets", None)
    if not sub_datasets:
        return _UNKNOWN_SFREQ
    first = sub_datasets[0]
    raw = getattr(first, "raw", None)
    if raw is not None and hasattr(raw, "info"):
        try:
            sfreq = raw.info["sfreq"]
        except (KeyError, TypeError):
            sfreq = None
        if sfreq is not None:
            return float(sfreq)
    windows = getattr(first, "windows", None)
    if windows is not None and hasattr(windows, "info"):
        try:
            sfreq = windows.info["sfreq"]
        except (KeyError, TypeError):
            sfreq = None
        if sfreq is not None:
            return float(sfreq)
    return _UNKNOWN_SFREQ


def _windows_per_subject(windows: Any) -> dict[str, int]:
    """Return a mapping ``subject -> window count`` for a windowed dataset.

    Works on a braindecode :class:`BaseConcatDataset` of windowed datasets:
    each sub-dataset has ``len(...)`` windows and a ``description`` Series
    with at least a ``subject`` field. Returns an empty dict when the
    information is not available (e.g. unit-test stubs).
    """
    sub_datasets = getattr(windows, "datasets", None)
    if not sub_datasets:
        return {}
    counts: dict[str, int] = {}
    for sub in sub_datasets:
        try:
            n = len(sub)
        except TypeError:
            continue
        description = getattr(sub, "description", None)
        if description is not None:
            try:
                subject = str(description.get("subject", "unknown"))
            except Exception:  # pragma: no cover -- pandas-style access only
                subject = "unknown"
        else:
            subject = "unknown"
        counts[subject] = counts.get(subject, 0) + int(n)
    return counts


def _windows_total(windows: Any) -> int | None:
    """Return ``len(windows)`` if the object is sized; ``None`` otherwise."""
    try:
        return int(len(windows))
    except TypeError:
        return None


# --------------------------------------------------------------------------- #
# EEGTask                                                                     #
# --------------------------------------------------------------------------- #


class EEGTask(ABC):
    """Abstract base class for an EEGDash high-level task.

    Subclasses should implement the seven abstract hooks below. ``EEGTask``
    purposefully does not own data loading: callers may pass an
    ``EEGDashDataset``-compatible object into :meth:`make_windows` (and into
    any future split / metric helpers). When ``dataset`` is omitted,
    :meth:`make_windows` will try to build one through
    :meth:`_resolve_dataset` using ``self.dataset`` and ``self.subjects``.

    Attributes
    ----------
    name : str
        Registry key. Subclasses must set this as a class attribute.
    manifest_path : str | None
        Optional path (relative to the package or absolute) of the YAML
        manifest that documents the task. Used by tooling and tutorials.
    dataset : str
        Primary OpenNeuro / source dataset identifier the task is anchored
        to (e.g. ``"ds005514"``). Subclasses set this in ``__init__`` (or
        as a class attribute) so tutorials can introspect it without
        loading the data.
    subjects : list[str]
        Default list of subject identifiers used by the task. Tutorials
        index ``subjects[0]`` to display the canonical demo subject.
    bandpass : tuple[float, float]
        Default band-pass cut-off ``(l_freq, h_freq)`` in Hz that the
        ``preprocessing_recipe`` applies. Exposed for tutorial prose so
        the filter parameters can be reported alongside results without
        re-walking the recipe.

    """

    #: Registry key for the task. Must be set by every subclass.
    name: str = ""

    #: Optional path to the YAML manifest documenting the task.
    manifest_path: str | None = None

    #: Primary dataset identifier (set by subclasses; default is empty).
    dataset: str = ""

    #: Default subject identifiers (set by subclasses; default is empty).
    subjects: list[str] = []  # noqa: RUF012

    #: Default band-pass ``(l_freq, h_freq)`` in Hz (set by subclasses).
    bandpass: tuple[float, float] = (0.0, 0.0)

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
    # Dataset resolution helper                                          #
    # ------------------------------------------------------------------ #

    def _resolve_dataset(
        self,
        cache_dir: str | None = None,
        n_subjects: int | None = None,
        download: bool = True,
    ) -> Any:
        """Build an :class:`~eegdash.EEGDashDataset` for this task.

        Used by :meth:`make_windows` when the caller does not pass an
        explicit ``dataset`` argument. The defensive ``getattr`` calls are
        intentional: subclasses that don't follow the ``self.dataset`` /
        ``self.subjects`` convention can still override :meth:`make_windows`
        directly.

        Parameters
        ----------
        cache_dir : str | None
            Local directory where EEGDash stores BIDS sidecars and downloads.
        n_subjects : int | None
            If set, truncates ``self.subjects`` to the first ``n_subjects``
            entries before building the query. Useful for tutorials that
            intentionally run on a single subject.
        download : bool, default ``True``
            Forwarded to :class:`EEGDashDataset`. Tests may pass ``False``.

        """
        # Lazy import: keeping ``import eegdash.tasks`` cheap and dependency-free.
        from eegdash.dataset.dataset import EEGDashDataset

        query = dict(self.metadata_query())
        subjects: Any = getattr(self, "subjects", None)
        if subjects and n_subjects is not None and n_subjects > 0:
            trimmed = list(subjects)[: int(n_subjects)]
            if len(trimmed) == 1:
                query["subject"] = trimmed[0]
            else:
                query["subject"] = {"$in": trimmed}

        return EEGDashDataset(
            cache_dir=cache_dir,
            query=query,
            download=download,
        )

    # ------------------------------------------------------------------ #
    # Concrete adapter on top of braindecode                             #
    # ------------------------------------------------------------------ #

    def make_windows(
        self,
        dataset: Any | None = None,
        engine: str = "braindecode",
        kind: Literal["fixed", "events"] | None = None,
        window_size: str | int | float | None = None,
        stride: str | int | float | None = None,
        return_report: bool = True,
        cache_dir: str | None = None,
        n_subjects: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Window a (preprocessed) concat-dataset using braindecode.

        Parameters
        ----------
        dataset : Any, optional
            A ``BaseConcatDataset``-compatible object (typically the result of
            :func:`braindecode.preprocessing.preprocess` applied to an
            :class:`eegdash.EEGDashDataset`). When ``None``,
            :meth:`_resolve_dataset` will be invoked with ``cache_dir`` and
            ``n_subjects`` to build one from the task metadata.
        engine : str, default ``"braindecode"``
            The windowing engine. Only ``"braindecode"`` is supported today;
            the parameter exists so that callers can already write
            forward-compatible code (e.g. ``engine="mne"`` later on).
        kind : {"fixed", "events"} or None
            Override for the windower selection. When ``None``, the value is
            taken from :meth:`windowing_recipe` (key ``"kind"``). Passing a
            ``kind`` that contradicts the recipe (e.g. ``kind="events"`` on a
            fixed-window task) raises :class:`ValueError`.
        window_size : str | int | float | None
            Window length. Strings such as ``"2s"`` or ``"500ms"`` are
            converted using the dataset sampling rate; integers / integer
            floats are forwarded as raw sample counts. ``None`` falls back to
            the recipe's ``window_size_samples`` (or the windower default).
        stride : str | int | float | None
            Stride between consecutive windows; same parsing rules as
            ``window_size``. Only meaningful for ``kind="fixed"``.
        return_report : bool, default ``True``
            When ``True``, return a ``(windows, report)`` tuple. The report
            dictionary records the engine, function name, forwarded kwargs,
            library versions, sampling rate, total window count, per-subject
            window counts, and a manifest hash. When ``False``, only the
            windows are returned.
        cache_dir : str | None
            Forwarded to :meth:`_resolve_dataset` when ``dataset is None``.
        n_subjects : int | None
            Limit the resolved dataset to the first ``n_subjects`` subjects.
        **kwargs : Any
            Extra keyword arguments forwarded to the chosen braindecode
            windower. They take precedence over the values defined in
            :meth:`windowing_recipe`.

        Returns
        -------
        windows or (windows, report)
            The windowed dataset. When ``return_report`` is ``True``, the
            return value is a 2-tuple of ``(windows, report)``.

        Raises
        ------
        ValueError
            If ``engine`` is unsupported, if the resolved ``kind`` is not
            ``"fixed"`` or ``"events"``, or if a caller-supplied ``kind``
            contradicts the recipe (the recipe is the authoritative
            description of the task; an explicit override must match).

        """
        if engine != "braindecode":
            raise ValueError(
                f"engine={engine!r} is not supported yet; only 'braindecode' "
                "is implemented in eegdash.tasks. Pass engine='braindecode' "
                "or omit the argument."
            )

        # Lazy imports keep the base module cheap to import.
        import braindecode  # noqa: F401  -- imported for version probing
        from braindecode.preprocessing import (
            create_fixed_length_windows,
            create_windows_from_events,
        )

        recipe = dict(self.windowing_recipe())
        recipe_kind = recipe.pop("kind", None)
        # Strip the engine entry too if a subclass chose to record it there.
        recipe.pop("engine", None)

        if kind is not None and recipe_kind is not None and kind != recipe_kind:
            raise ValueError(
                f"kind={kind!r} contradicts the task's windowing_recipe() "
                f"which declares kind={recipe_kind!r}. Either remove the "
                "explicit ``kind=`` argument, or override "
                "``windowing_recipe()`` in the task subclass."
            )
        effective_kind = kind if kind is not None else recipe_kind
        if effective_kind not in {"fixed", "events"}:
            raise ValueError(
                "windowing kind must be either 'fixed' or 'events', got "
                f"{effective_kind!r}."
            )

        # Resolve the dataset before parsing time strings (the sampling rate
        # comes from the dataset).
        if dataset is None:
            dataset = self._resolve_dataset(
                cache_dir=cache_dir,
                n_subjects=n_subjects,
            )
        sfreq = _resolve_sfreq(dataset)

        # Caller-supplied kwargs win, then recipe defaults fill the rest.
        forwarded: dict[str, Any] = dict(recipe)
        forwarded.update(kwargs)

        # Time-string overrides (window_size / stride) are translated into
        # the kwarg names the chosen windower expects.
        if window_size is not None:
            forwarded["window_size_samples"] = _parse_time_to_samples(
                window_size, sfreq
            )
        if stride is not None:
            forwarded["window_stride_samples"] = _parse_time_to_samples(stride, sfreq)

        if effective_kind == "events":
            fn = create_windows_from_events
            canonical_name = "create_windows_from_events"
        else:
            fn = create_fixed_length_windows
            canonical_name = "create_fixed_length_windows"

        windows = fn(dataset, **forwarded)

        if not return_report:
            return windows

        # ----- Build the report ---------------------------------------- #
        n_windows = _windows_total(windows)
        per_subject = _windows_per_subject(windows)
        versions = _library_versions()

        manifest_payload: dict[str, Any] = {
            "task": self.name,
            "engine": engine,
            "kind": effective_kind,
            "function": f"braindecode.preprocessing.{canonical_name}",
            "function_kwargs": forwarded,
            "library_versions": versions,
            "sfreq": sfreq,
        }
        manifest_hash = _manifest_hash(manifest_payload)

        report: dict[str, Any] = {
            "engine": engine,
            "kind": effective_kind,
            "function": f"braindecode.preprocessing.{canonical_name}",
            "function_kwargs": forwarded,
            "library_versions": versions,
            "n_windows": n_windows,
            "windows_per_subject": per_subject,
            "sfreq": sfreq,
            "manifest_hash": manifest_hash,
            # Legacy aliases retained for backwards compatibility with the
            # 0.7.x report shape exercised by ``tests/unit_tests/test_tasks.py``.
            "kwargs": forwarded,
            "package_versions": versions,
        }
        return windows, report
