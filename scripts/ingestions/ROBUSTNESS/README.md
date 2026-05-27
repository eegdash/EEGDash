# Ingestion Robustness Programme

This directory holds the analysis, plan, and process documents that
guide turning `scripts/ingestions/` into a scientifically-rigorous,
review-friendly, well-tested Python codebase in the spirit of
**MNE-Python** (Alexandre Gramfort's house style) and **scikit-learn**
(contribution / review practice).

It was written after a sister project — `eegdash-viewer` — was lifted
from "competent but under-instrumented" to "Level 6 maturity"
(property-based + mutation testing + fuzz + statistical bench + leak
gates + WCAG-AA accessibility, ~60 commits, several real bugs found and
fixed). The same approach transfers here — but **the gaps are
different**, and the order of operations matters.

## Documents

| File | What it contains |
|---|---|
| [`01-AUDIT.md`](01-AUDIT.md) | What's in the repo today: file inventory, complexity metrics, style observations, error-handling profile, the absent test suite. Honest, file:line citations. |
| [`02-STYLE_GUIDE.md`](02-STYLE_GUIDE.md) | The target style: Gramfort-flavoured NumPy/scientific-Python conventions, docstring template, naming, error patterns, logging, type hints. |
| [`03-CONTRIBUTING.md`](03-CONTRIBUTING.md) | The target review process: scikit-learn-style PR template, reviewer checklist, two-reviewer rule, deprecation policy, `whats_new.rst` discipline. |
| [`04-ROADMAP.md`](04-ROADMAP.md) | Phased plan (Phase 0 → 9), with rationale per phase, expected wall-clock, parallelisation lanes, and the artefact every phase must produce. |
| [`05-EVALUATION.md`](05-EVALUATION.md) | For each phase: **how do you verify the rationale was actually delivered** (not just executed). Distinguishes activity from outcome. |
| [`06-PARALLELIZATION.md`](06-PARALLELIZATION.md) | The DAG of phases: what blocks what, what can run concurrently, where the critical-path sits. The map for dispatching multiple agents safely. |
| [`07-DETAILS.md`](07-DETAILS.md) | Concrete deepening per phase: code snippets, fixture choices, threshold-setting heuristics, anti-patterns to refuse. |
| [`AGENT_PROMPT.md`](AGENT_PROMPT.md) | Copy-paste prompt for the autonomous agent or human contributor who actually executes this plan. |
| [`ADRs/`](ADRs/) | Accepted and proposed architecture decisions. ADR 0003 covers the separate-repository move and daily CI control-plane design. |

The ingestion domain vocabulary lives in [`../CONTEXT.md`](../CONTEXT.md).

## How to use

1. Read `01-AUDIT.md` to understand *where you start*.
2. Read `02-STYLE_GUIDE.md` and `03-CONTRIBUTING.md` to set the *bar*.
3. Use `04-ROADMAP.md` as the *sequence*; `06-PARALLELIZATION.md` to
   decide what to dispatch in parallel.
4. After each phase, run the **evaluation checks** in `05-EVALUATION.md`.
   Activity that doesn't move an evaluation metric is wasted effort.
5. Hand `AGENT_PROMPT.md` (or pieces of it) to an autonomous worker.

## Non-goals

This programme will NOT:
- Replace `eegdash` package responsibilities with ingestion-side code.
- Migrate the pipeline to a different orchestrator (Airflow, Dagster, etc.).
  Those are separate decisions.
- Force 100% test coverage. The viewer's session showed coverage > 90%
  can co-exist with mutation kill ratios < 40% — what matters is
  *tests that catch real bugs*, not lines executed.

## What "done" looks like

When the programme finishes:
- `pytest scripts/ingestions/tests/` passes with > 200 tests.
- `ruff check .` and `mypy --strict ingestions/` are clean.
- Mutation kill ratio ≥ 60% on parsers (mirroring the viewer's outcome).
- Memory ceiling for a 1000-record digest under 100 MB, asserted in CI.
- One reviewer can read a PR diff and understand the change without
  reading the entire 3000-line `3_digest.py` (because the function it
  touches is now ≤ 60 LOC).
- A regression that drops mutation kill ratio by 5pp blocks merge.

That target maps to **Level 5 of the maturity ladder**
([viewer's reference doc](../../../eegdash-viewer/docs/research/qa-benchmark-maturity-2026-05.md)).
Level 6 (continuous fuzz + RUM) is achievable but not in scope here.
