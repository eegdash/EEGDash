"""Tutorial runtime + data-size tracker.

Loads every spec in ``docs/tutorials/_spec/`` and aggregates the declared
budgets (``budgets.max_runtime_seconds``, ``budgets.max_network_mb``) per
category. Emits a markdown table at ``--out`` with per-tutorial estimates
and per-category + overall totals. Optionally compares the declared
estimates against actual measured runtimes loaded from ``--measured``
(a JSON file produced by the runtime CI stage as
``{"<tutorial_id>": {"runtime_seconds": <float>, "network_mb": <float>}}``).

Per `docs/tutorial_restructure_plan.md` Phase 5, governance section:
"Track tutorial runtime and data size."

Usage:

    python -m scripts.tutorial_audit.runtime_tracker \
        --out docs/evidence/runtime_tracker_2026-05-07.md

    python -m scripts.tutorial_audit.runtime_tracker \
        --out docs/evidence/runtime_tracker_2026-05-07.md \
        --measured ci-runtime.json

The output is deterministic: tutorials sort by category then id, totals
are stable, and the only non-deterministic line is the ``Generated`` row.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


_log = logging.getLogger("eegdash.tutorial_audit.runtime_tracker")

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "docs" / "tutorials" / "_spec"
EXCLUDE_NAMES = {"coverage.yaml", "rubric_overrides.yaml", "README.md"}


def _load_specs(spec_dir: Path) -> list[dict[str, Any]]:
    """Load every spec yaml in ``spec_dir`` and return a list of dicts."""
    if yaml is None:
        raise RuntimeError("pyyaml is required; install via `pip install pyyaml`")

    specs: list[dict[str, Any]] = []
    for path in sorted(spec_dir.glob("*.yaml")):
        if path.name in EXCLUDE_NAMES:
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - defensive
            _log.warning("skipping unloadable spec %s: %s", path.name, exc)
            continue
        if not isinstance(data, dict):
            continue
        data["__path__"] = path.name
        specs.append(data)
    return specs


def _budget(spec: dict[str, Any], key: str, default: float = 0.0) -> float:
    budgets = spec.get("budgets") or {}
    val = budgets.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _category_of(spec: dict[str, Any]) -> str:
    return str(spec.get("category", "unknown"))


def _id_of(spec: dict[str, Any]) -> str:
    return str(spec.get("id", spec.get("__path__", "?")))


def _gpu_flag(spec: dict[str, Any]) -> str:
    """One-letter flag: G if budgets.gpu_required, else C (cpu)."""
    budgets = spec.get("budgets") or {}
    if budgets.get("gpu_required") is True:
        return "G"
    if str(spec.get("hardware", "cpu")).lower().startswith("gpu"):
        return "G"
    return "C"


def _network_label(spec: dict[str, Any]) -> str:
    return str(spec.get("network", "?"))


def _format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _format_mb(mb: float) -> str:
    if mb <= 0:
        return "0 MB"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _delta(estimated: float, measured: float | None) -> str:
    if measured is None or estimated <= 0:
        return "-"
    diff = measured - estimated
    pct = diff / estimated * 100.0
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f} ({sign}{pct:.0f}%)"


def aggregate(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-category and overall totals.

    Returns a dict with keys ``categories`` (mapping category -> totals dict)
    and ``overall`` (totals dict). Each totals dict has ``count``,
    ``runtime_s`` and ``network_mb``.
    """
    by_cat: dict[str, dict[str, float]] = {}
    overall = {"count": 0.0, "runtime_s": 0.0, "network_mb": 0.0}
    for spec in specs:
        cat = _category_of(spec)
        bucket = by_cat.setdefault(
            cat, {"count": 0.0, "runtime_s": 0.0, "network_mb": 0.0}
        )
        rt = _budget(spec, "max_runtime_seconds", 0.0)
        net = _budget(spec, "max_network_mb", 0.0)
        bucket["count"] += 1
        bucket["runtime_s"] += rt
        bucket["network_mb"] += net
        overall["count"] += 1
        overall["runtime_s"] += rt
        overall["network_mb"] += net
    return {"categories": by_cat, "overall": overall}


def render_markdown(
    specs: list[dict[str, Any]],
    aggregates: dict[str, Any],
    measured: dict[str, dict[str, float]] | None = None,
    *,
    generated_at: str | None = None,
) -> str:
    """Render the tracker as markdown."""
    measured = measured or {}
    lines: list[str] = []
    lines.append("# Tutorial runtime and data-size tracker")
    lines.append("")
    lines.append(
        f"Generated: {generated_at or datetime.now(tz=timezone.utc).isoformat()}"
    )
    lines.append("")
    lines.append(
        "Source of truth: `docs/tutorials/_spec/*.yaml` -- `budgets.max_runtime_seconds`"
        " and `budgets.max_network_mb`. Per `docs/tutorial_restructure_plan.md`"
        " Phase 5 governance, every tutorial declares an upper bound for runtime"
        " and on-the-wire data so the CI matrix and the docs gallery stay green."
    )
    lines.append("")

    # Per-tutorial table sorted by (category, id).
    lines.append("## Per-tutorial estimates")
    lines.append("")
    if measured:
        lines.append(
            "| ID | Category | HW | Network | Est. runtime | Est. data | "
            "Measured runtime | Delta runtime | Measured data | Delta data |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    else:
        lines.append("| ID | Category | HW | Network | Est. runtime | Est. data |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
    sorted_specs = sorted(specs, key=lambda s: (_category_of(s), _id_of(s)))
    for spec in sorted_specs:
        sid = _id_of(spec)
        cat = _category_of(spec)
        hw = _gpu_flag(spec)
        net_label = _network_label(spec)
        rt = _budget(spec, "max_runtime_seconds", 0.0)
        net = _budget(spec, "max_network_mb", 0.0)
        row = [
            f"`{sid}`",
            cat,
            hw,
            net_label,
            _format_seconds(rt),
            _format_mb(net),
        ]
        if measured:
            m_entry = measured.get(sid) or {}
            m_rt = m_entry.get("runtime_seconds")
            m_net = m_entry.get("network_mb")
            row.extend(
                [
                    _format_seconds(m_rt) if isinstance(m_rt, (int, float)) else "-",
                    _delta(rt, m_rt if isinstance(m_rt, (int, float)) else None),
                    _format_mb(m_net) if isinstance(m_net, (int, float)) else "-",
                    _delta(net, m_net if isinstance(m_net, (int, float)) else None),
                ]
            )
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Per-category totals")
    lines.append("")
    lines.append("| Category | Count | Total runtime | Total data |")
    lines.append("| --- | --- | --- | --- |")
    for cat in sorted(aggregates["categories"].keys()):
        bucket = aggregates["categories"][cat]
        lines.append(
            f"| {cat} | {int(bucket['count'])} | {_format_seconds(bucket['runtime_s'])}"
            f" | {_format_mb(bucket['network_mb'])} |"
        )

    overall = aggregates["overall"]
    lines.append("")
    lines.append("## Overall total")
    lines.append("")
    lines.append(f"- Tutorials tracked: **{int(overall['count'])}**")
    lines.append(
        f"- Sum of declared runtimes: **{_format_seconds(overall['runtime_s'])}**"
    )
    lines.append(
        f"- Sum of declared on-the-wire data: **{_format_mb(overall['network_mb'])}**"
    )

    lines.append("")
    lines.append(
        "If a single PR pushes the overall runtime past 240 minutes or the"
        " data total past 12 GB, escalate to the docs maintainers and consider"
        " demoting one tutorial to the nightly-only stage of"
        " `.github/workflows/tutorial-audit.yml`."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate declared tutorial runtimes and data budgets from "
            "docs/tutorials/_spec/*.yaml and emit a markdown tracker."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output path for the markdown tracker",
    )
    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=SPEC_DIR,
        help="directory containing tutorial spec YAMLs (default: %(default)s)",
    )
    parser.add_argument(
        "--measured",
        type=Path,
        default=None,
        help=(
            "optional JSON file with measured runtimes per tutorial id, "
            "produced by the runtime CI stage"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress informational logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
    )

    if not args.spec_dir.is_dir():
        _log.error("spec directory does not exist: %s", args.spec_dir)
        return 2

    specs = _load_specs(args.spec_dir)
    if not specs:
        _log.error("no specs found in %s", args.spec_dir)
        return 2

    measured: dict[str, dict[str, float]] | None = None
    if args.measured is not None:
        if not args.measured.is_file():
            _log.error("--measured file does not exist: %s", args.measured)
            return 2
        try:
            measured = json.loads(args.measured.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _log.error("failed to parse --measured %s: %s", args.measured, exc)
            return 2
        if not isinstance(measured, dict):
            _log.error(
                "--measured must be a JSON object mapping tutorial ids to "
                "{'runtime_seconds': float, 'network_mb': float}"
            )
            return 2

    aggregates = aggregate(specs)
    md = render_markdown(specs, aggregates, measured=measured)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    _log.info(
        "wrote %s (tutorials=%d, total_runtime_s=%.0f, total_network_mb=%.0f)",
        args.out,
        int(aggregates["overall"]["count"]),
        aggregates["overall"]["runtime_s"],
        aggregates["overall"]["network_mb"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
