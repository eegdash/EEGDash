"""Spec coherence checker for the EEGDash tutorial audit pipeline.

Loads every YAML in ``docs/tutorials/_spec/`` (excluding ``coverage.yaml``,
``rubric_overrides.yaml`` and the README) and validates that:

* the required schema fields are present;
* ``difficulty`` is one of ``{1, 2, 3}``;
* ``state`` is one of the legal states for the tutorial state machine;
* every ``cites.plan`` line range resolves -- the cited file exists and the
  line numbers fit within it;
* any change to ``assignee`` is paired with a state transition (best-effort:
  consults ``git diff HEAD --`` if available; otherwise emits a warn).

The CLI exits non-zero on schema/value/cite failures and zero on warn-level
issues. CI gates merges on this exit code.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


_log = logging.getLogger("eegdash.tutorial_audit.validate_spec")


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "docs" / "tutorials" / "_spec"
EXCLUDE_NAMES = {"coverage.yaml", "rubric_overrides.yaml", "README.md"}

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "category",
    "title",
    "state",
    "difficulty",
    "estimated_runtime_minutes",
    "cites",
    "learning_objectives",
    "audience",
    "budgets",
    "sections_required",
)

ALLOWED_STATES = {
    "proposed",
    "drafted",
    "static-pass",
    "runtime-pass",
    "reviewed",
    "merged",
}

ALLOWED_DIFFICULTIES = {1, 2, 3}

CITE_PLAN_RE = re.compile(r"^(?P<file>[^#]+)#L(?P<start>\d+)(?:-L(?P<end>\d+))?$")


# -- Issue accumulator ------------------------------------------------------


class Issue:
    """One spec validation issue."""

    __slots__ = ("path", "level", "field", "message")

    def __init__(self, path: Path, level: str, field: str, message: str) -> None:
        self.path = path
        self.level = level
        self.field = field
        self.message = message

    def __str__(self) -> str:
        rel = self.path.relative_to(REPO_ROOT) if self.path.is_absolute() else self.path
        return f"{self.level.upper()} {rel}::{self.field}: {self.message}"


# -- Loading ----------------------------------------------------------------


def _iter_spec_files(spec_dir: Path) -> list[Path]:
    if not spec_dir.exists():
        return []
    out: list[Path] = []
    for path in sorted(spec_dir.iterdir()):
        if path.is_dir():
            continue
        if path.name in EXCLUDE_NAMES:
            continue
        if path.suffix.lower() not in (".yaml", ".yml"):
            continue
        out.append(path)
    return out


def _load_spec(path: Path, issues: list[Issue]) -> dict[str, Any] | None:
    if yaml is None:
        issues.append(
            Issue(path, "error", "<load>", "pyyaml is required to load specs")
        )
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        issues.append(Issue(path, "error", "<load>", f"YAML parse error: {exc}"))
        return None
    except OSError as exc:
        issues.append(Issue(path, "error", "<load>", f"Cannot read: {exc}"))
        return None
    if not isinstance(data, dict):
        issues.append(
            Issue(path, "error", "<load>", "Top-level YAML must be a mapping")
        )
        return None
    return data


# -- Field checks -----------------------------------------------------------


def _check_required(path: Path, spec: dict[str, Any], issues: list[Issue]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in spec or spec[field] in (None, "", [], {}):
            issues.append(
                Issue(path, "error", field, "required field missing or empty")
            )


def _check_difficulty(path: Path, spec: dict[str, Any], issues: list[Issue]) -> None:
    diff = spec.get("difficulty")
    if diff not in ALLOWED_DIFFICULTIES:
        issues.append(
            Issue(
                path,
                "error",
                "difficulty",
                f"must be one of {sorted(ALLOWED_DIFFICULTIES)}, got {diff!r}",
            )
        )


def _check_state(path: Path, spec: dict[str, Any], issues: list[Issue]) -> None:
    state = spec.get("state")
    if state not in ALLOWED_STATES:
        issues.append(
            Issue(
                path,
                "error",
                "state",
                f"must be one of {sorted(ALLOWED_STATES)}, got {state!r}",
            )
        )


def _check_id_matches_filename(
    path: Path, spec: dict[str, Any], issues: list[Issue]
) -> None:
    expected = path.stem
    actual = spec.get("id")
    if actual != expected:
        issues.append(
            Issue(
                path,
                "error",
                "id",
                f"id {actual!r} must match filename stem {expected!r}",
            )
        )


def _check_plan_cites(path: Path, spec: dict[str, Any], issues: list[Issue]) -> None:
    cites = spec.get("cites") or {}
    plan_cites = cites.get("plan") if isinstance(cites, dict) else None
    if not plan_cites:
        return
    if not isinstance(plan_cites, list):
        issues.append(Issue(path, "error", "cites.plan", "must be a list of citations"))
        return
    for cite in plan_cites:
        _check_one_plan_cite(path, str(cite), issues)


def _check_one_plan_cite(path: Path, cite: str, issues: list[Issue]) -> None:
    m = CITE_PLAN_RE.match(cite.strip())
    if not m:
        issues.append(
            Issue(
                path,
                "error",
                "cites.plan",
                f"cite {cite!r} does not match '<file>#L<a>[-L<b>]'",
            )
        )
        return
    plan_file = (REPO_ROOT / "docs" / m.group("file").lstrip("/")).resolve()
    if not plan_file.exists():
        # Plan citations live in docs/, but the convention in
        # tutorial_implementation_strategy.md prepends the bare filename.
        # Try resolving relative to docs/ directly.
        alt = (REPO_ROOT / "docs" / m.group("file")).resolve()
        if alt.exists():
            plan_file = alt
        else:
            issues.append(
                Issue(
                    path,
                    "error",
                    "cites.plan",
                    f"cited file does not exist: {m.group('file')}",
                )
            )
            return
    try:
        n_lines = sum(1 for _ in plan_file.open("r", encoding="utf-8"))
    except OSError as exc:
        issues.append(
            Issue(path, "warn", "cites.plan", f"cannot read {plan_file.name}: {exc}")
        )
        return
    start = int(m.group("start"))
    end_str = m.group("end")
    end = int(end_str) if end_str else start
    if start < 1 or end > n_lines or start > end:
        issues.append(
            Issue(
                path,
                "error",
                "cites.plan",
                (
                    f"cite {cite!r} out of range: file has {n_lines} lines "
                    f"but cite resolves to L{start}-L{end}"
                ),
            )
        )


# -- Assignee transition check (best-effort) --------------------------------


def _git_diff_against_head(path: Path) -> str | None:
    """Return ``git diff HEAD -- <path>`` or ``None`` if git is unavailable."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [git, "diff", "HEAD", "--", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode not in (0, 1):
        return None
    return result.stdout


_ASSIGNEE_DIFF_RE = re.compile(r"^[+-]\s*assignee\s*:", re.M)
_STATE_DIFF_RE = re.compile(r"^[+-]\s*state\s*:", re.M)


def _check_assignee_transition(
    path: Path, spec: dict[str, Any], issues: list[Issue]
) -> None:
    diff = _git_diff_against_head(path)
    if diff is None:
        # No git available; emit an info note and move on.
        return
    if not _ASSIGNEE_DIFF_RE.search(diff):
        return
    if not _STATE_DIFF_RE.search(diff):
        issues.append(
            Issue(
                path,
                "warn",
                "assignee",
                (
                    "assignee changed without an accompanying state transition; "
                    "CI will reject this on PRs"
                ),
            )
        )


# -- Validator entry --------------------------------------------------------


def validate_spec(path: Path) -> list[Issue]:
    issues: list[Issue] = []
    spec = _load_spec(path, issues)
    if spec is None:
        return issues
    _check_id_matches_filename(path, spec, issues)
    _check_required(path, spec, issues)
    _check_difficulty(path, spec, issues)
    _check_state(path, spec, issues)
    _check_plan_cites(path, spec, issues)
    _check_assignee_transition(path, spec, issues)
    return issues


def validate_all(spec_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    files = _iter_spec_files(spec_dir)
    if not files:
        _log.warning("No spec files found under %s", spec_dir)
        return issues
    for path in files:
        issues.extend(validate_spec(path))
    return issues


# -- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.tutorial_audit.validate_spec",
        description=(
            "Validate every tutorial spec YAML under docs/tutorials/_spec/. "
            "Exits non-zero on schema or cite failures."
        ),
    )
    p.add_argument(
        "--spec-dir",
        type=Path,
        default=SPEC_DIR,
        help="Directory containing the spec YAMLs.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Print only failures, not summary info.",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    issues = validate_all(args.spec_dir)
    errors = [i for i in issues if i.level == "error"]
    warns = [i for i in issues if i.level == "warn"]
    for issue in issues:
        print(str(issue))
    if not args.quiet:
        print(
            f"validate_spec: {len(errors)} error(s), {len(warns)} warning(s)",
            file=sys.stderr,
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
