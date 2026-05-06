r"""Reporting and gating CLI for the EEGDash tutorial audit pipeline.

Usage::

    python -m scripts.tutorial_audit.report \
        [--tutorial ID] [--render-md] [--aggregate] [--gate] [--comment-pr]

* ``--render-md``  Read ``docs/evidence/tutorials/<id>/evidence.json`` and
  render ``report.md`` next to it. The report contains the 12-dimension
  scorecard and a per-rule findings table.
* ``--aggregate``  Build ``docs/evidence/tutorials/_aggregate.md`` summarising
  every dossier in the directory.
* ``--gate``       Exit non-zero if any tutorial in the aggregate has
  ``totals.errors > 0``.
* ``--comment-pr`` Print to stdout the markdown that should be posted as a
  PR comment. The actual posting is the responsibility of GitHub Actions.

This module deliberately does no network I/O; ``--comment-pr`` is a stub.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

_log = logging.getLogger("eegdash.tutorial_audit.report")


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "evidence" / "tutorials"


# Order matters: this drives the rendered scorecard table rows so two runs
# of --render-md produce a byte-stable file.
SCORECARD_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("audience", "E2.20, spec.audience"),
    ("structure", "E1.2, E1.4, E2.20"),
    ("examples", "E2.12, E2.16"),
    ("retrieval", "E2.13, E2.18"),
    ("spacing", "cross-tutorial check_concept_revisit"),
    ("interleaving", "reviewer + cross-tutorial"),
    ("feedback", "E3.27, runtime asserts"),
    ("data", "E3.23, E3.24, E4.32"),
    ("reproducibility", "E3.21-E3.30, E1.9"),
    ("accessibility", "E1.6, custom check_alt_text"),
    ("community", "reviewer"),
    ("reuse", "E1.7, evidence dossier"),
)


# -- IO helpers -------------------------------------------------------------


def _load_evidence(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        _log.warning("Missing evidence.json: %s", path)
        return None
    except json.JSONDecodeError as exc:
        _log.error("Cannot decode %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        _log.error("Evidence file %s did not parse to a mapping", path)
        return None
    return data


def _iter_dossiers(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    out: list[Path] = []
    for entry in sorted(out_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_"):
            continue
        evidence = entry / "evidence.json"
        if evidence.exists():
            out.append(evidence)
    return out


# -- Markdown rendering -----------------------------------------------------


def _md_escape(s: str) -> str:
    """Escape pipes inside markdown table cells."""
    return s.replace("|", "\\|")


def render_report_md(evidence: dict[str, Any]) -> str:
    tutorial_id = evidence.get("tutorial_id", "?")
    state = evidence.get("spec_state", "?")
    difficulty = evidence.get("spec_difficulty", "?")
    totals = evidence.get("totals", {})
    rule_results = evidence.get("rule_results", []) or []
    scorecard = evidence.get("scorecard", {}) or {}

    lines: list[str] = []
    lines.append(f"# Audit dossier -- {tutorial_id}")
    lines.append("")
    lines.append(
        f"State: `{state}` | Difficulty: `{difficulty}` | "
        f"Errors: **{totals.get('errors', 0)}** | "
        f"Warns: **{totals.get('warns', 0)}** | "
        f"Infos: **{totals.get('infos', 0)}**"
    )
    lines.append("")
    spec_path = evidence.get("spec_path")
    if spec_path:
        lines.append(f"Spec: `{spec_path}`")
    git_sha = evidence.get("git_sha")
    if git_sha:
        lines.append(f"Git SHA: `{git_sha}`")
    lines.append("")

    lines.append("## 12-dimension scorecard")
    lines.append("")
    lines.append("| Dimension | Result | Validators |")
    lines.append("| --- | :---: | --- |")
    for name, contributors in SCORECARD_DIMENSIONS:
        result = scorecard.get(name, "unknown")
        lines.append(
            f"| {name} | {_md_escape(str(result))} | {_md_escape(contributors)} |"
        )
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    if not rule_results:
        lines.append("_No findings recorded._")
    else:
        lines.append("| Rule | Level | Message | Tool |")
        lines.append("| --- | :---: | --- | --- |")
        for rec in rule_results:
            lines.append(
                "| {rule} | {level} | {msg} | {tool} |".format(
                    rule=_md_escape(str(rec.get("rule_id", ""))),
                    level=_md_escape(str(rec.get("level", ""))),
                    msg=_md_escape(str(rec.get("message", ""))),
                    tool=_md_escape(str(rec.get("tool", ""))),
                )
            )
    lines.append("")

    return "\n".join(lines) + "\n"


def render_aggregate_md(dossiers: Iterable[dict[str, Any]]) -> str:
    rows = list(dossiers)
    lines: list[str] = []
    lines.append("# Tutorial audit aggregate")
    lines.append("")
    lines.append(f"Tutorials audited: **{len(rows)}**")
    lines.append("")
    if not rows:
        lines.append("_No dossiers found._")
        lines.append("")
        return "\n".join(lines) + "\n"

    total_errors = sum(int((r.get("totals") or {}).get("errors", 0)) for r in rows)
    total_warns = sum(int((r.get("totals") or {}).get("warns", 0)) for r in rows)
    total_infos = sum(int((r.get("totals") or {}).get("infos", 0)) for r in rows)
    lines.append(
        f"Totals: **{total_errors} error(s)**, **{total_warns} warn(s)**, "
        f"**{total_infos} info(s)** across {len(rows)} dossiers."
    )
    lines.append("")

    lines.append("| Tutorial | State | Difficulty | Errors | Warns | Infos |")
    lines.append("| --- | :---: | :---: | ---: | ---: | ---: |")
    for r in sorted(rows, key=lambda d: d.get("tutorial_id", "")):
        totals = r.get("totals") or {}
        lines.append(
            "| {tid} | {state} | {diff} | {err} | {warn} | {info} |".format(
                tid=_md_escape(str(r.get("tutorial_id", "?"))),
                state=_md_escape(str(r.get("spec_state", "?"))),
                diff=_md_escape(str(r.get("spec_difficulty", "?"))),
                err=int(totals.get("errors", 0)),
                warn=int(totals.get("warns", 0)),
                info=int(totals.get("infos", 0)),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# -- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.tutorial_audit.report",
        description=(
            "Render Markdown reports and gate the build from evidence dossiers."
        ),
    )
    p.add_argument(
        "--tutorial",
        default=None,
        help="Tutorial id (file stem). Required for --render-md.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory holding per-tutorial dossiers.",
    )
    p.add_argument(
        "--render-md",
        action="store_true",
        help="Render report.md for --tutorial.",
    )
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="Render docs/evidence/tutorials/_aggregate.md.",
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero if any dossier has totals.errors > 0.",
    )
    p.add_argument(
        "--comment-pr",
        action="store_true",
        help="Print the markdown that should be posted as a PR comment.",
    )
    return p


def _do_render_md(tutorial: str | None, out_dir: Path) -> int:
    if not tutorial:
        print("--render-md requires --tutorial", file=sys.stderr)
        return 2
    evidence_path = out_dir / tutorial / "evidence.json"
    evidence = _load_evidence(evidence_path)
    if evidence is None:
        return 1
    text = render_report_md(evidence)
    target = out_dir / tutorial / "report.md"
    target.write_text(text, encoding="utf-8")
    try:
        display_path = target.relative_to(REPO_ROOT)
    except ValueError:
        display_path = target
    _log.info("wrote %s", display_path)
    return 0


def _load_all(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ev_path in _iter_dossiers(out_dir):
        ev = _load_evidence(ev_path)
        if ev is not None:
            rows.append(ev)
    return rows


def _do_aggregate(out_dir: Path) -> int:
    rows = _load_all(out_dir)
    text = render_aggregate_md(rows)
    target = out_dir / "_aggregate.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    try:
        display_path = target.relative_to(REPO_ROOT)
    except ValueError:
        display_path = target
    _log.info("wrote %s", display_path)
    return 0


def _do_gate(out_dir: Path) -> int:
    rows = _load_all(out_dir)
    bad: list[str] = []
    for r in rows:
        totals = r.get("totals") or {}
        if int(totals.get("errors", 0)) > 0:
            bad.append(str(r.get("tutorial_id", "?")))
    if bad:
        print(
            f"gate: {len(bad)} tutorial(s) have errors: {', '.join(sorted(bad))}",
            file=sys.stderr,
        )
        return 1
    return 0


def _do_comment_pr(out_dir: Path) -> int:
    rows = _load_all(out_dir)
    print(render_aggregate_md(rows))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not any([args.render_md, args.aggregate, args.gate, args.comment_pr]):
        parser.error(
            "select at least one of --render-md, --aggregate, --gate, --comment-pr"
        )

    rc = 0
    if args.render_md:
        rc = max(rc, _do_render_md(args.tutorial, args.out))
    if args.aggregate:
        rc = max(rc, _do_aggregate(args.out))
    if args.gate:
        rc = max(rc, _do_gate(args.out))
    if args.comment_pr:
        rc = max(rc, _do_comment_pr(args.out))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
