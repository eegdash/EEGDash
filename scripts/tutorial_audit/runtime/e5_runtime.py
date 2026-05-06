"""Runtime E.5 checks (subject leakage and chance-level reporting)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import nbformat  # runtime stage may import nbformat (not nbclient).

from ..api import Finding

if TYPE_CHECKING:  # pragma: no cover
    from ..api import RunContext


CLASSIFIER_FIT_RE = re.compile(r"\.fit\s*\(")
CLASSIFIER_PREDICT_RE = re.compile(r"\.predict\s*\(")
CLASSIFIER_SCORE_RE = re.compile(r"\.score\s*\(")
ACCURACY_RE = re.compile(r"\baccuracy\b|\bacc\b", re.I)
CHANCE_RE = re.compile(
    r"\bchance(\s+level)?\b|\brandom\s+baseline\b|\b1\s*/\s*n_classes\b",
    re.I,
)
LEAKAGE_LINE_RE = re.compile(r"leakage_report")


def _read_source(tutorial_path: Path) -> str:
    try:
        return tutorial_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _trains_classifier(tutorial_path: Path) -> bool:
    """Heuristic: does this tutorial train a classifier?

    True iff the source contains ``.fit(`` somewhere.
    """
    src = _read_source(tutorial_path)
    return bool(CLASSIFIER_FIT_RE.search(src))


def _has_predict_or_score(tutorial_path: Path) -> bool:
    src = _read_source(tutorial_path)
    return bool(CLASSIFIER_PREDICT_RE.search(src) or CLASSIFIER_SCORE_RE.search(src))


def _executed_nb_path(
    ctx: Optional["RunContext"], tutorial_path: Path
) -> Optional[Path]:
    """Resolve the executed notebook path from the run context if available."""
    if ctx is None:
        return None
    candidate = getattr(ctx, "executed_nb_path", None)
    if candidate is None:
        return None
    return Path(candidate)


def _iter_code_outputs(nb_path: Path) -> Iterable[str]:
    """Yield the textual content of every code-cell output in ``nb_path``."""
    nb = nbformat.read(nb_path.as_posix(), as_version=4)
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []) or []:
            otype = out.get("output_type")
            if otype == "stream":
                text = out.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                yield text
            elif otype in {"execute_result", "display_data"}:
                data = out.get("data", {}) or {}
                plain = data.get("text/plain", "")
                if isinstance(plain, list):
                    plain = "".join(plain)
                yield str(plain)
            elif otype == "error":
                tb = out.get("traceback", []) or []
                yield "\n".join(tb)


def _extract_leakage_report(text_chunks: Iterable[str]) -> Optional[dict]:
    """Find a ``{"leakage_report": {...}}`` JSON line emitted by the tutorial."""
    for chunk in text_chunks:
        if "leakage_report" not in chunk:
            continue
        for raw in chunk.splitlines():
            line = raw.strip()
            if not LEAKAGE_LINE_RE.search(line):
                continue
            for candidate in _candidate_json_substrings(line):
                try:
                    parsed = json.loads(candidate)
                except (ValueError, TypeError):
                    continue
                if isinstance(parsed, dict) and "leakage_report" in parsed:
                    return parsed["leakage_report"]
    return None


def _candidate_json_substrings(line: str) -> list[str]:
    """Pick out plausible JSON object substrings from a single output line."""
    candidates: list[str] = [line]
    start = line.find("{")
    end = line.rfind("}")
    if 0 <= start < end:
        candidates.append(line[start : end + 1])
    return candidates


def check_no_subject_leakage(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E5.42 -- if a classifier is trained, subject overlap must be 0.

    Reads ``ctx.executed_nb_path`` (an executed .ipynb) and looks for a JSON
    line of the form
    ``{"leakage_report": {"overlap": <int>, "by": "subject"}}``.
    """
    if not _trains_classifier(tutorial_path):
        return []

    nb_path = _executed_nb_path(ctx, tutorial_path)
    if nb_path is None or not nb_path.exists():
        return [
            Finding(
                rule_id="E5.42",
                level="error",
                message=(
                    "Executed notebook not available; cannot verify subject-"
                    "leakage report"
                ),
                cite_rubric="compass_artifact.md#E5.42",
                cite_plan="tutorial_restructure_plan.md#L902-L920",
                evidence={"executed_nb_path": str(nb_path) if nb_path else None},
                tool="nbformat",
            )
        ]

    report = _extract_leakage_report(_iter_code_outputs(nb_path))
    if report is None:
        return [
            Finding(
                rule_id="E5.42",
                level="error",
                message=(
                    "Tutorial trains a classifier but executed notebook does "
                    'not print a JSON line {"leakage_report": {"overlap": '
                    '..., "by": "subject"}}'
                ),
                cite_rubric="compass_artifact.md#E5.42",
                cite_plan="tutorial_restructure_plan.md#L902-L920",
                evidence={"executed_nb_path": str(nb_path)},
                tool="nbformat",
            )
        ]

    overlap = report.get("overlap")
    if not isinstance(overlap, int):
        try:
            overlap = int(overlap)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            overlap = None
    if overlap is None:
        return [
            Finding(
                rule_id="E5.42",
                level="error",
                message=("leakage_report missing an integer 'overlap' field"),
                cite_rubric="compass_artifact.md#E5.42",
                cite_plan="tutorial_restructure_plan.md#L902-L920",
                evidence={"leakage_report": report},
                tool="nbformat",
            )
        ]
    if overlap > 0:
        return [
            Finding(
                rule_id="E5.42",
                level="error",
                message=(f"Subject overlap between train and test: {overlap} subjects"),
                cite_rubric="compass_artifact.md#E5.42",
                cite_plan="tutorial_restructure_plan.md#L902-L920",
                evidence={"leakage_report": report},
                tool="nbformat",
            )
        ]
    return []


def check_chance_level_reported(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E5.43 -- when accuracy is reported, chance level must be too.

    Heuristic: the tutorial trains a classifier (``.fit(`` *and* one of
    ``.predict(`` / ``.score(``). If the executed notebook prints
    ``accuracy``-like text without ``chance``-like text nearby, warn.
    """
    if not (_trains_classifier(tutorial_path) and _has_predict_or_score(tutorial_path)):
        return []

    nb_path = _executed_nb_path(ctx, tutorial_path)
    if nb_path is None or not nb_path.exists():
        return [
            Finding(
                rule_id="E5.43",
                level="warn",
                message=(
                    "Executed notebook not available; cannot verify that "
                    "chance level is reported alongside accuracy"
                ),
                cite_rubric="compass_artifact.md#E5.43",
                cite_plan="tutorial_restructure_plan.md#L921-L940",
                evidence={"executed_nb_path": str(nb_path) if nb_path else None},
                tool="nbformat",
            )
        ]

    saw_accuracy = False
    saw_chance = False
    for chunk in _iter_code_outputs(nb_path):
        if not saw_accuracy and ACCURACY_RE.search(chunk):
            saw_accuracy = True
        if not saw_chance and CHANCE_RE.search(chunk):
            saw_chance = True
        if saw_accuracy and saw_chance:
            break

    if saw_accuracy and not saw_chance:
        return [
            Finding(
                rule_id="E5.43",
                level="warn",
                message=(
                    "Tutorial reports accuracy but no chance-level baseline "
                    "(expected one of: 'chance', 'random baseline', or "
                    "'1/n_classes')"
                ),
                cite_rubric="compass_artifact.md#E5.43",
                cite_plan="tutorial_restructure_plan.md#L921-L940",
                evidence={
                    "saw_accuracy": saw_accuracy,
                    "saw_chance": saw_chance,
                },
                tool="nbformat",
            )
        ]
    return []
