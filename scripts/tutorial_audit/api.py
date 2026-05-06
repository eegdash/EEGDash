"""Shared types and orchestrator entrypoint for the tutorial audit pipeline.

This module defines the lightweight, deterministic objects that every
validator in :mod:`scripts.tutorial_audit` produces and consumes. They are
intentionally stdlib-only so the *static* stage can run without nbclient,
nbformat, sphinx-gallery, Pillow, or any other heavy dependency.

The :func:`run_audit` function is the public Python-level orchestrator. It
dynamically imports the per-rule validator modules so that a missing static
or runtime module (for example because another agent has not yet committed
its file) degrades to a warning rather than a hard import failure. The
command-line orchestrator lives in :mod:`scripts.tutorial_audit.pipeline`.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

# -- Levels and shared types -------------------------------------------------

# Three severity levels. They are intentionally a Literal alias so static
# analysers can flag misuse and so the JSON serialisation stays a plain
# string rather than an enum representation.
Level = Literal["error", "warn", "info"]
LEVELS: tuple[Level, ...] = ("error", "warn", "info")


class Severity:
    """String constants for severity levels.

    Provided so callers can write ``Severity.ERROR`` instead of the raw
    string literal. The underlying type contract is still :data:`Level`.
    """

    ERROR: Level = "error"
    WARN: Level = "warn"
    INFO: Level = "info"


# -- Finding -----------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One audit observation about a tutorial.

    The shape mirrors the contract documented in
    ``docs/tutorial_implementation_strategy.md`` -- "Validator implementations
    -- Finding dataclass". Every Finding traces back to both the literature-
    anchored rubric (``cite_rubric``) and the prescriptive plan
    (``cite_plan``) so the chain from tutorial -> evidence -> source is
    auditable.
    """

    rule_id: str
    """Stable identifier such as ``"E5.42"`` or ``"DV.palette"``."""

    level: Level
    """One of ``"error" | "warn" | "info"``."""

    message: str
    """Human-readable summary of what the validator observed."""

    cite_rubric: str
    """Citation back to the compass artifact rubric, e.g. ``compass_artifact.md#E5.42``."""

    cite_plan: str
    """Citation back to the tutorial plan, e.g. ``tutorial_restructure_plan.md#L902-L920``."""

    evidence: dict[str, Any] = field(default_factory=dict)
    """Structured per-rule data (counts, hashes, paths, ...). Must be JSON-serialisable."""

    tool: str = ""
    """Name of the tool that produced the Finding, e.g. ``"ast"`` or ``"nbclient"``."""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for deterministic JSON serialisation."""
        return asdict(self)


# -- Run context -------------------------------------------------------------


@dataclass
class RunContext:
    """Bag of paths and metadata passed to validators during a single run.

    Validators take ``(path, spec)`` historically, but more elaborate runtime
    checks need access to the executed notebook and the saved figures. The
    :class:`RunContext` is provided as an optional companion so callers can
    grow the surface without changing the validator signatures.
    """

    tutorial_path: Path
    """Absolute path to the ``plot_*.py`` source file."""

    spec: dict[str, Any]
    """Parsed YAML spec from ``docs/tutorials/_spec/<id>.yaml``."""

    executed_nb_path: Optional[Path] = None
    """Path to the executed notebook, populated by runtime stage."""

    figures_dir: Optional[Path] = None
    """Directory containing the figures rendered during execution."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Free-form bag for stage-specific metadata (timings, hashes, ...)."""


# -- Run summary -------------------------------------------------------------


@dataclass
class RunSummary:
    """Aggregate of findings produced by :func:`run_audit`."""

    findings: list[Finding] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)
    tutorials_visited: list[str] = field(default_factory=list)
    errors: int = 0
    warns: int = 0
    infos: int = 0

    def update_counts(self) -> None:
        self.errors = sum(1 for f in self.findings if f.level == "error")
        self.warns = sum(1 for f in self.findings if f.level == "warn")
        self.infos = sum(1 for f in self.findings if f.level == "info")

    def to_dict(self) -> dict[str, Any]:
        self.update_counts()
        return {
            "tutorials_visited": list(self.tutorials_visited),
            "skipped_modules": list(self.skipped_modules),
            "totals": {
                "errors": self.errors,
                "warns": self.warns,
                "infos": self.infos,
                "findings": len(self.findings),
            },
            "findings": [f.to_dict() for f in self.findings],
        }


# -- Validator registries ----------------------------------------------------

# Module names that the orchestrator attempts to import. Listed explicitly
# instead of scanned from disk so that a typo in a file name surfaces as a
# warning rather than silently dropping a stage. Agent D fills these in.
STATIC_MODULES: tuple[str, ...] = (
    "scripts.tutorial_audit.static.e1_structural",
    "scripts.tutorial_audit.static.e2_pedagogical",
    "scripts.tutorial_audit.static.e3_technical",
    "scripts.tutorial_audit.static.e4_engagement",
    "scripts.tutorial_audit.static.e5_domain",
    "scripts.tutorial_audit.static.e6_diataxis",
)

RUNTIME_MODULES: tuple[str, ...] = (
    "scripts.tutorial_audit.runtime.e1_runtime",
    "scripts.tutorial_audit.runtime.e3_runtime",
    "scripts.tutorial_audit.runtime.e4_runtime",
    "scripts.tutorial_audit.runtime.e5_runtime",
    "scripts.tutorial_audit.runtime.budgets",
    "scripts.tutorial_audit.runtime.visual",
)


_log = logging.getLogger("eegdash.tutorial_audit")


def _import_optional(module_name: str, summary: RunSummary) -> Any:
    """Import a validator module; return ``None`` and record skip on failure.

    The tutorial pipeline is built incrementally by several agents; a missing
    static module should not make the whole pipeline crash. We log a warning
    and continue. Hard breakage of an existing module still raises through.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        _log.warning(
            "tutorial_audit: skipping %s (ModuleNotFoundError: %s)",
            module_name,
            exc,
        )
        summary.skipped_modules.append(module_name)
        return None
    except ImportError as exc:
        _log.warning("tutorial_audit: skipping %s (ImportError: %s)", module_name, exc)
        summary.skipped_modules.append(module_name)
        return None


def _collect_validators(module: Any) -> list[Any]:
    """Pull validator callables out of a static/runtime module.

    Convention: any top-level callable whose name starts with ``check_`` or
    ``score_`` is treated as a validator. Modules can also export an explicit
    ``VALIDATORS`` iterable; that wins if present.
    """
    if hasattr(module, "VALIDATORS"):
        return list(module.VALIDATORS)
    out: list[Any] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        if not (name.startswith("check_") or name.startswith("score_")):
            continue
        attr = getattr(module, name)
        if callable(attr):
            out.append(attr)
    return out


def _safe_call(fn: Any, ctx: RunContext, summary: RunSummary) -> list[Finding]:
    """Invoke a validator with graceful fallback for arity differences.

    Validators may declare any of:

    * ``fn(ctx: RunContext)``
    * ``fn(tutorial_path: Path, spec: dict, ctx: RunContext | None = None)``
    * ``fn(tutorial_path: Path, spec: dict)``

    The orchestrator tries the richer signatures first and falls back on
    ``TypeError``. Other exceptions are caught and recorded as skipped.
    """
    try:
        try:
            # Preferred: takes RunContext directly.
            result = fn(ctx)
        except TypeError:
            try:
                # Validator-D signature with optional ctx kwarg.
                result = fn(ctx.tutorial_path, ctx.spec, ctx=ctx)
            except TypeError:
                # Legacy two-arg form.
                result = fn(ctx.tutorial_path, ctx.spec)
    except Exception as exc:  # pragma: no cover -- defensive
        _log.exception("validator %s raised: %s", getattr(fn, "__name__", fn), exc)
        summary.skipped_modules.append(
            f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}"
        )
        return []
    if result is None:
        return []
    if isinstance(result, Finding):
        return [result]
    if isinstance(result, Iterable):
        return [f for f in result if isinstance(f, Finding)]
    return []


def run_audit(
    static: bool = True,
    runtime: bool = False,
    tutorials: Optional[Iterable[RunContext]] = None,
) -> dict[str, Any]:
    """Run the validator pipeline and return an aggregate summary dict.

    Parameters
    ----------
    static
        Run the static stage (E.1 structural, E.2 pedagogical, E.3
        technical, E.4 engagement, E.5 domain, E.6 Diataxis). Default
        ``True``.
    runtime
        Run the runtime stage (clean kernel execution, budgets, leakage
        check, visual identity). Default ``False`` because runtime modules
        require optional dependencies (nbclient, nbformat, sphinx-gallery,
        Pillow). The CLI front-end exposes these as separate ``--stage``
        choices.
    tutorials
        Iterable of :class:`RunContext` instances. When ``None`` no
        per-tutorial validation runs and the call returns an empty
        summary. The CLI fills this from spec discovery.

    Returns
    -------
    dict
        Output of :meth:`RunSummary.to_dict`.

    """
    summary = RunSummary()
    contexts = list(tutorials or [])

    # Resolve once; ``None`` modules are recorded in ``skipped_modules`` and
    # then ignored when iterating.
    static_modules = (
        [_import_optional(m, summary) for m in STATIC_MODULES] if static else []
    )
    runtime_modules = (
        [_import_optional(m, summary) for m in RUNTIME_MODULES] if runtime else []
    )

    static_validators = [
        v
        for module in static_modules
        if module is not None
        for v in _collect_validators(module)
    ]
    runtime_validators = [
        v
        for module in runtime_modules
        if module is not None
        for v in _collect_validators(module)
    ]

    for ctx in contexts:
        summary.tutorials_visited.append(ctx.tutorial_path.stem)
        for validator in static_validators:
            summary.findings.extend(_safe_call(validator, ctx, summary))
        if runtime:
            for validator in runtime_validators:
                summary.findings.extend(_safe_call(validator, ctx, summary))

    summary.update_counts()
    return summary.to_dict()


__all__ = [
    "Finding",
    "Level",
    "LEVELS",
    "RunContext",
    "RunSummary",
    "Severity",
    "STATIC_MODULES",
    "RUNTIME_MODULES",
    "run_audit",
]
