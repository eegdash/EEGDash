"""Static E.2 pedagogical checks (PRIMM scaffolding and learning objectives)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..api import Finding

if TYPE_CHECKING:  # pragma: no cover
    from ..api import RunContext


# Markers we accept for each PRIMM phase. Anchored to the start of a line
# (after optional '#' and whitespace) so we do not match prose mentions of the
# words inside running text.
PRIMM_HEADERS: dict[str, re.Pattern[str]] = {
    "predict": re.compile(r"^\*?\*?(predict|prediction)\b", re.I),
    "run": re.compile(r"^\*?\*?(run\b|let'?s run\b)", re.I),
    "investigate": re.compile(
        r"^\*?\*?(investigate|why this output|what we see|what just happened)\b",
        re.I,
    ),
    "modify": re.compile(r"^\*?\*?(modify|your turn|change|try changing|edit)\b", re.I),
    "make": re.compile(
        r"^\*?\*?(make|try it yourself|mini-?project|extension|challenge)\b",
        re.I,
    ),
}

# Heading-ish line introducing learning objectives. Accepts reST/Markdown
# variants ("Learning objectives", "Learning Objectives", "## Learning ...").
LO_HEADER_RE = re.compile(r"learning\s+objectives", re.I)
# Bullet-list marker at the start of a line after optional whitespace.
LO_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.+)$")
# Markdown block opener used by sphinx-gallery, anchored to start of line.
MD_BLOCK_RE = re.compile(r"^# %% \[markdown\]\s*$", re.M)
# Action verbs commonly used in objectives -- treat anything ending in -e
# (the imperative form) as plausible too. A whitelist covers high-frequency
# irregulars and short verbs.
ACTION_VERB_WHITELIST = {
    "build",
    "use",
    "load",
    "run",
    "split",
    "train",
    "test",
    "evaluate",
    "compare",
    "compute",
    "fit",
    "predict",
    "plot",
    "show",
    "find",
    "filter",
    "select",
    "convert",
    "save",
    "read",
    "write",
    "implement",
    "explain",
    "describe",
    "interpret",
    "identify",
    "verify",
    "assert",
    "check",
    "apply",
    "set",
    "see",
    "do",
}


def _strip_md_prefix(line: str) -> str:
    """Strip a leading ``# `` (sphinx-gallery markdown line prefix) and whitespace."""
    s = line.lstrip()
    if s.startswith("# "):
        s = s[2:]
    elif s.startswith("#"):
        s = s[1:]
    return s.strip()


def _markdown_blocks(src: str) -> list[str]:
    """Return text of every ``# %% [markdown]`` block in ``src``.

    Each block runs from a markdown delimiter to the next ``# %%`` (markdown or
    code) delimiter or end of file. Lines inside markdown blocks are typically
    prefixed by ``# `` in the .py representation.
    """
    blocks: list[str] = []
    cur: list[str] | None = None
    for line in src.splitlines():
        if MD_BLOCK_RE.match(line):
            if cur is not None:
                blocks.append("\n".join(cur))
            cur = []
            continue
        if line.startswith("# %%"):
            if cur is not None:
                blocks.append("\n".join(cur))
                cur = None
            continue
        if cur is not None:
            cur.append(_strip_md_prefix(line))
    if cur is not None:
        blocks.append("\n".join(cur))
    return blocks


def check_primm_blocks(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E2.13 -- count Predict/Run/Investigate/Modify/Make markers.

    Severity is difficulty-aware (Kalyuga 2003): error for 1-star tutorials,
    warn for 2-star, exempt for 3-star.
    """
    difficulty = int(spec.get("difficulty", 1))
    if difficulty >= 3:
        return []

    required = spec.get("primm_required") or {
        "predict": 1,
        "run": 2,
        "investigate": 1,
        "modify": 1,
        "make": 1,
    }
    src = tutorial_path.read_text(encoding="utf-8")

    counts: dict[str, int] = {kind: 0 for kind in PRIMM_HEADERS}
    for block in _markdown_blocks(src):
        for line in block.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if not stripped:
                continue
            for kind, pat in PRIMM_HEADERS.items():
                if pat.search(stripped):
                    counts[kind] += 1
                    break

    findings: list[Finding] = []
    level = "error" if difficulty == 1 else "warn"
    for kind, target in required.items():
        target_n = _parse_target(target)
        if counts.get(kind, 0) < target_n:
            findings.append(
                Finding(
                    rule_id="E2.13",
                    level=level,
                    message=(
                        f"PRIMM '{kind}' block: found {counts.get(kind, 0)}, "
                        f"need {target}"
                    ),
                    cite_rubric="compass_artifact.md#E2.13",
                    cite_plan="tutorial_restructure_plan.md#L516-L562",
                    evidence={
                        "counts": counts,
                        "required": dict(required),
                        "difficulty": difficulty,
                    },
                    tool="regex",
                )
            )
    return findings


def _parse_target(target: object) -> int:
    """Parse YAML-ish target expressions like ``2`` or ``">=2"``."""
    if isinstance(target, int):
        return target
    s = str(target).strip()
    s = s.lstrip(">=").lstrip("=").strip()
    try:
        return int(s)
    except ValueError:
        return 1


def _looks_like_action_verb(token: str) -> bool:
    """Heuristic: treat the bullet's first token as an action verb if it ends
    in 'e' (imperative form like "use", "compare", "compute") or is in the
    whitelist.
    """
    word = re.sub(r"[^A-Za-z]", "", token).lower()
    if not word:
        return False
    if word in ACTION_VERB_WHITELIST:
        return True
    # Imperative form heuristic: many action verbs end in 'e' ("compute",
    # "compare", "describe", "use", "evaluate"). Avoid matching nouns by
    # also requiring length >= 2.
    return len(word) >= 2 and word.endswith("e")


def _docstring(tutorial_path: Path) -> str:
    src = tutorial_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    return ast.get_docstring(tree) or ""


def check_learning_objectives(
    tutorial_path: Path,
    spec: dict,
    ctx: Optional["RunContext"] = None,
) -> list[Finding]:
    """E2.20 -- 3-5 learning-objective bullets, each starting with an action verb.

    The header may live in the module docstring or in the first
    ``# %% [markdown]`` block.
    """
    src = tutorial_path.read_text(encoding="utf-8")
    md_blocks = _markdown_blocks(src)
    candidates = [_docstring(tutorial_path)]
    if md_blocks:
        candidates.append(md_blocks[0])

    bullets: list[str] = []
    header_seen = False
    for text in candidates:
        if not text:
            continue
        if not LO_HEADER_RE.search(text):
            continue
        header_seen = True
        # Collect contiguous bullets after the header line.
        in_block = False
        for line in text.splitlines():
            if not in_block:
                if LO_HEADER_RE.search(line):
                    in_block = True
                continue
            stripped = line.strip()
            if not stripped:
                # Blank line ends the bullet block once we have at least one
                # bullet; otherwise tolerate it.
                if bullets:
                    break
                continue
            m = LO_BULLET_RE.match(line)
            if m:
                bullets.append(m.group(1).strip())
            elif bullets:
                # First non-bullet, non-blank line after bullets started --
                # treat as end of objectives block.
                break
        if bullets:
            break

    findings: list[Finding] = []
    if not header_seen:
        findings.append(
            Finding(
                rule_id="E2.20",
                level="error",
                message=(
                    "Module docstring or first markdown block must contain a "
                    "'Learning objectives' header followed by 3-5 bulleted "
                    "objectives"
                ),
                cite_rubric="compass_artifact.md#E2.20",
                cite_plan="tutorial_restructure_plan.md#L519-L524",
                tool="regex",
            )
        )
        return findings
    if len(bullets) < 3 or len(bullets) > 5:
        findings.append(
            Finding(
                rule_id="E2.20",
                level="warn",
                message=(
                    f"Learning objectives should have 3-5 bullets; found {len(bullets)}"
                ),
                cite_rubric="compass_artifact.md#E2.20",
                cite_plan="tutorial_restructure_plan.md#L519-L524",
                evidence={"bullet_count": len(bullets), "bullets": bullets[:8]},
                tool="regex",
            )
        )
    for bullet in bullets:
        first_word = bullet.split(maxsplit=1)[0] if bullet.split() else ""
        if not _looks_like_action_verb(first_word):
            findings.append(
                Finding(
                    rule_id="E2.20",
                    level="warn",
                    message=(
                        "Learning-objective bullet should start with an action "
                        f"verb; got {first_word!r}"
                    ),
                    cite_rubric="compass_artifact.md#E2.20",
                    cite_plan="tutorial_restructure_plan.md#L519-L524",
                    evidence={"bullet": bullet[:120]},
                    tool="regex",
                )
            )
    return findings
