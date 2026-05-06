r"""CLI orchestrator for the tutorial audit pipeline.

Usage:

    python -m scripts.tutorial_audit.pipeline \
        --stage {static,runtime,all} \
        [--tutorial ID] [--pattern GLOB] [--out DIR]

For each tutorial that matches the discovery rules the orchestrator:

1. resolves the tutorial id (file stem) and looks up
   ``docs/tutorials/_spec/<id>.yaml`` -- a missing spec yields a single
   ``E1.spec_present`` error finding;
2. invokes the requested validator stages via
   :func:`scripts.tutorial_audit.api.run_audit`;
3. writes a deterministic ``evidence.json`` into
   ``docs/evidence/tutorials/<id>/`` using sorted keys, 4-space indent and
   ``ensure_ascii=False``. Floats are rounded to a fixed 6-digit precision.

Determinism: identical inputs produce identical evidence.json. The only
non-deterministic field is ``ran_at`` (top-level ISO timestamp). Diff
consumers must ignore that field.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

# yaml is optional; if missing the orchestrator still runs, but every spec
# will be reported as unloadable. The whole audit pipeline depends on
# pyyaml in the project's dev extras so this is mainly a graceful-fail
# branch for fresh environments.
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

from .api import (
    Finding,
    RunContext,
    Severity,
    run_audit,
)

_log = logging.getLogger("eegdash.tutorial_audit.pipeline")


# -- Project layout ---------------------------------------------------------

# Resolve the repo root from this file's location: .../scripts/tutorial_audit/pipeline.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATTERN = "examples/tutorials/**/plot_*.py"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "evidence" / "tutorials"
SPEC_DIR = REPO_ROOT / "docs" / "tutorials" / "_spec"

EVIDENCE_FLOAT_NDIGITS = 6


# -- Spec loading -----------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning ``{}`` on error and logging the cause."""
    if yaml is None:
        _log.warning("pyyaml not installed; cannot parse %s", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        _log.error("YAML parse error in %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        _log.warning("Spec %s did not parse to a mapping", path)
        return {}
    return data


def load_spec(tutorial_id: str) -> tuple[dict[str, Any], Path]:
    """Return ``(spec, spec_path)`` for a given tutorial id.

    The path is always returned even when the file is absent, so callers
    can synthesise a missing-spec finding with a useful path.
    """
    spec_path = SPEC_DIR / f"{tutorial_id}.yaml"
    spec = _load_yaml(spec_path) if spec_path.exists() else {}
    return spec, spec_path


# -- Discovery --------------------------------------------------------------


def discover_tutorials(
    pattern: str,
    explicit_id: str | None = None,
) -> list[Path]:
    """Glob the repo for tutorial source files."""
    if explicit_id is not None:
        # Match anywhere in examples/tutorials with this stem.
        candidates = sorted(
            (REPO_ROOT / "examples" / "tutorials").glob(f"**/{explicit_id}.py")
        )
        return list(candidates)
    return sorted(REPO_ROOT.glob(pattern))


# -- Determinism helpers ----------------------------------------------------


def _round_floats(obj: Any) -> Any:
    """Recursively round floats to a fixed precision so JSON is byte-stable."""
    if isinstance(obj, float):
        return round(obj, EVIDENCE_FLOAT_NDIGITS)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def _spec_hash(spec_path: Path) -> str:
    if not spec_path.exists():
        return "sha256:absent"
    h = hashlib.sha256()
    h.update(spec_path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def _git_sha() -> str:
    """Best-effort git SHA without spawning subprocesses for speed/safety.

    Reads ``.git/HEAD`` directly. Returns ``"unknown"`` on any failure --
    we never want this to crash the pipeline.
    """
    git_dir = REPO_ROOT / ".git"
    head = git_dir / "HEAD"
    try:
        if not head.exists():
            return "unknown"
        ref_line = head.read_text(encoding="utf-8").strip()
        if ref_line.startswith("ref: "):
            ref_path = git_dir / ref_line[len("ref: ") :].strip()
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:40]
            packed = git_dir / "packed-refs"
            if packed.exists():
                target = ref_line[len("ref: ") :].strip()
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or " " not in line:
                        continue
                    sha, name = line.split(" ", 1)
                    if name.strip() == target:
                        return sha[:40]
            return "unknown"
        return ref_line[:40]
    except OSError:
        return "unknown"


# -- Stage running ----------------------------------------------------------


def _missing_spec_finding(spec_path: Path) -> Finding:
    try:
        rel = str(spec_path.relative_to(REPO_ROOT))
    except ValueError:
        rel = spec_path.name
    return Finding(
        rule_id="E1.spec_present",
        level=Severity.ERROR,
        message=f"Spec YAML not found at {rel}",
        cite_rubric="compass_artifact.md#E1",
        cite_plan="tutorial_restructure_plan.md#Spec_contract",
        evidence={"spec_path": rel},
        tool="pipeline",
    )


def _runtime_available() -> bool:
    """Check whether the runtime stage's optional deps are importable."""
    for mod in ("nbclient", "nbformat", "sphinx_gallery"):
        try:
            __import__(mod)
        except ImportError:
            return False
    return True


def _build_context(tutorial_path: Path) -> tuple[RunContext, Path, Finding | None]:
    tutorial_id = tutorial_path.stem
    spec, spec_path = load_spec(tutorial_id)
    missing = None if spec_path.exists() else _missing_spec_finding(spec_path)
    ctx = RunContext(tutorial_path=tutorial_path, spec=spec)
    return ctx, spec_path, missing


def _aggregate_for_tutorial(
    ctx: RunContext,
    spec_path: Path,
    findings: Iterable[Finding],
) -> dict[str, Any]:
    rule_results: list[dict[str, Any]] = []
    for f in findings:
        rec = {
            "rule_id": f.rule_id,
            "level": f.level,
            "result": "fail",
            "message": f.message,
            "cite_rubric": f.cite_rubric,
            "cite_plan": f.cite_plan,
            "tool": f.tool,
        }
        if f.evidence:
            rec["evidence"] = _round_floats(f.evidence)
        rule_results.append(rec)
    rule_results.sort(key=lambda r: (r["rule_id"], r["level"]))
    errors = sum(1 for r in rule_results if r["level"] == "error")
    warns = sum(1 for r in rule_results if r["level"] == "warn")
    infos = sum(1 for r in rule_results if r["level"] == "info")

    spec = ctx.spec
    return {
        "tutorial_id": ctx.tutorial_path.stem,
        "tutorial_path": str(ctx.tutorial_path.relative_to(REPO_ROOT)),
        "spec_path": str(spec_path.relative_to(REPO_ROOT))
        if spec_path.exists()
        else str(spec_path.relative_to(REPO_ROOT)),
        "spec_hash": _spec_hash(spec_path),
        "spec_state": spec.get("state", "proposed"),
        "spec_difficulty": spec.get("difficulty", 1),
        "git_sha": _git_sha(),
        "host": {
            "python": platform.python_version(),
            "platform": sys.platform,
            "ci": (
                "github-actions"
                if os.environ.get("GITHUB_ACTIONS") == "true"
                else "local"
            ),
        },
        "totals": {
            "errors": errors,
            "warns": warns,
            "infos": infos,
            "findings": len(rule_results),
        },
        "rule_results": rule_results,
        # Operational scorecard placeholder; populated by report.py once the
        # 12-dimension mapping is wired through. We commit it as ``unknown``
        # rather than omitting so deterministic diffs only flip when a real
        # change occurs.
        "scorecard": {
            "audience": "unknown",
            "structure": "unknown",
            "examples": "unknown",
            "retrieval": "unknown",
            "spacing": "unknown",
            "interleaving": "unknown",
            "feedback": "unknown",
            "data": "unknown",
            "reproducibility": "unknown",
            "accessibility": "unknown",
            "community": "unknown",
            "reuse": "unknown",
        },
    }


def _write_evidence(out_dir: Path, dossier: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "evidence.json"
    # ``ran_at`` lives outside the structural diff so reruns with no real
    # changes don't churn the file. We store it on a top-level key only.
    payload = dict(dossier)
    payload["ran_at"] = _dt.datetime.now(tz=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    body = json.dumps(payload, sort_keys=True, indent=4, ensure_ascii=False)
    if not body.endswith("\n"):
        body += "\n"
    target.write_text(body, encoding="utf-8")
    return target


# -- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.tutorial_audit.pipeline",
        description=(
            "Run the EEGDash tutorial audit pipeline (static and/or runtime "
            "stages) and emit deterministic evidence.json dossiers."
        ),
    )
    p.add_argument(
        "--stage",
        choices=("static", "runtime", "all"),
        default="static",
        help="Which validator stage(s) to run.",
    )
    p.add_argument(
        "--tutorial",
        default=None,
        help="Restrict to a single tutorial id (file stem, e.g. plot_11_leakage_safe_split).",
    )
    p.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern (relative to repo root) when --tutorial is not given.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for per-tutorial dossiers.",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress informational logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    run_static = args.stage in ("static", "all")
    run_runtime = args.stage in ("runtime", "all")

    if run_runtime and not _runtime_available():
        _log.warning(
            "runtime stage requested but optional deps "
            "(nbclient/nbformat/sphinx-gallery) are missing; skipping runtime."
        )
        run_runtime = False
        if args.stage == "runtime":
            # Nothing to do: surface a non-error exit so CI can short-circuit.
            print("runtime stage skipped: optional deps unavailable", file=sys.stderr)
            return 0

    tutorials = discover_tutorials(args.pattern, args.tutorial)
    if not tutorials:
        _log.warning(
            "No tutorials matched (pattern=%s, tutorial=%s)",
            args.pattern,
            args.tutorial,
        )
        return 0

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    total_errors = 0

    for tutorial_path in tutorials:
        ctx, spec_path, missing = _build_context(tutorial_path)
        per_tutorial: list[Finding] = []
        if missing is not None:
            per_tutorial.append(missing)

        # Always run the requested stages even when the spec is missing -- a
        # missing spec is itself a finding, not a fatal stop, so the dossier
        # still records what the validators saw.
        result = run_audit(
            static=run_static,
            runtime=run_runtime,
            tutorials=[ctx],
        )
        for raw in result["findings"]:
            per_tutorial.append(
                Finding(
                    rule_id=raw["rule_id"],
                    level=raw["level"],
                    message=raw["message"],
                    cite_rubric=raw.get("cite_rubric", ""),
                    cite_plan=raw.get("cite_plan", ""),
                    evidence=raw.get("evidence", {}),
                    tool=raw.get("tool", ""),
                )
            )

        dossier = _aggregate_for_tutorial(ctx, spec_path, per_tutorial)
        target_dir = out_dir / ctx.tutorial_path.stem
        target = _write_evidence(target_dir, dossier)
        total_errors += dossier["totals"]["errors"]
        try:
            display_path = target.relative_to(REPO_ROOT)
        except ValueError:
            display_path = target
        _log.info(
            "wrote %s (errors=%d warns=%d infos=%d)",
            display_path,
            dossier["totals"]["errors"],
            dossier["totals"]["warns"],
            dossier["totals"]["infos"],
        )
    # Pipeline command itself does not gate on errors; report.py --gate
    # enforces that. Returning zero lets CI run further reporting steps.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
