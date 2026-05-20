# Agent prompt — execute the robustness programme

This is the prompt you can hand to an autonomous coding agent (Claude
Code, Cursor agent, GitHub Copilot Workspace, or a human contractor)
to execute the programme described in this folder.

There are **three prompts**, each suited to a different scope:

1. **MEGA PROMPT** — single agent runs the whole programme (best for
   one capable autonomous worker over multi-day session).
2. **WAVE PROMPT** — coordinator launches one prompt per wave from
   `06-PARALLELIZATION.md` (best for multi-agent fan-out).
3. **PHASE PROMPT** — single phase, very focused (best when you want to
   land one phase per merge cycle and review carefully between).

Pick the scope that matches how much trust you want to place in the
worker.

---

## MEGA PROMPT (single autonomous agent, full programme)

```
You are executing the Ingestion Robustness Programme for
/Users/bruaristimunha/Projects/eegdash/scripts/ingestions/.

## Mandatory reading (in order)

Read these documents before touching any code:

1. ROBUSTNESS/README.md          (overview + entry point)
2. ROBUSTNESS/01-AUDIT.md        (what the codebase looks like today)
3. ROBUSTNESS/02-STYLE_GUIDE.md  (Gramfort-flavoured scientific Python)
4. ROBUSTNESS/03-CONTRIBUTING.md (scikit-learn-style review process)
5. ROBUSTNESS/04-ROADMAP.md      (phased plan)
6. ROBUSTNESS/05-EVALUATION.md   (how each phase is verified)
7. ROBUSTNESS/06-PARALLELIZATION.md (DAG and lane assignments)
8. ROBUSTNESS/07-DETAILS.md      (concrete code templates)

## Your task

Execute Phases 0 through 8 of ROBUSTNESS/04-ROADMAP.md in the order
specified by ROBUSTNESS/06-PARALLELIZATION.md.

For EACH phase:

(a) READ the corresponding section in 04-ROADMAP.md and 07-DETAILS.md.
(b) DO the work. Use ONE commit per logical step (don't bundle).
(c) VERIFY against the evaluation hooks in 05-EVALUATION.md. The
    evaluation is the contract — if the OUTCOME metric is not green,
    the phase is NOT done.
(d) DOCUMENT findings (bare-excepts replaced, mutants killed, bugs
    found) in the appropriate ROBUSTNESS/findings-<phase>.md file.

Phase 9 (bug hunting) runs CONTINUOUSLY in parallel after Phase 4 is
done. Pause for human review after every audit produces findings;
don't apply fixes silently.

## Constraints

- Style: NumPy docstrings (Parameters/Returns/Raises/Notes/Examples).
  numpydoc-lintable. See 02-STYLE_GUIDE.md § 1.
- No bare `except:` or `except Exception:` survives Phase 3. Every
  catch is a named exception class with a logged context. See
  02-STYLE_GUIDE.md § 5.
- No function over 80 LOC ships without an explicit comment justifying
  it.
- Type hints everywhere (`mypy --strict` clean for new files).
- One PR per phase, named `<type>(<scope>): <phase-name>` per
  03-CONTRIBUTING.md.
- DO NOT silently lower thresholds in pyproject.toml or mutmut.ini to
  hide regressions. If a test fails, that's the work, not a failure.
- DO NOT skip 05-EVALUATION.md disconfirmation checks. The whole
  point of the programme is to deliver outcomes, not activity.

## Anti-patterns to refuse (these will be reviewer-rejected)

- `try: ... except Exception: pass` (Phase 3 must remove all 85)
- `assert result is not None` as a test body (must be golden-value)
- 600-LOC functions (Phase 8 must split them under characterisation
  tests)
- print() statements (Phase 0 must replace with logger.X)
- Module-level mutable globals for "configuration" (pass as parameter)
- Sibling imports (`from _http import ...`) — use package-relative

## Tools available (install once, in Phase 0)

  pip install pytest pytest-cov pytest-benchmark hypothesis respx \
              mutmut ruff mypy

## When to stop

The programme is complete when ALL of the following hold:

- ROBUSTNESS/05-EVALUATION.md "Overall programme evaluation" checklist
  has ≥ 8/10 boxes checked.
- All phase OUTCOME metrics (not just action metrics) are green.
- A regression PR for one of the new gates would be REJECTED by CI.
- whats_new.rst exists with entries for the last 3 changes.

If you get stuck on a phase, escalate by writing a
ROBUSTNESS/blocker-<phase>.md document with: what you tried, what's
blocking, what kind of decision the human needs to make. Then move on
to the next file-disjoint phase per the DAG in 06-PARALLELIZATION.md.

Working directory: /Users/bruaristimunha/Projects/eegdash/scripts/ingestions/

GO.
```

---

## WAVE PROMPT (coordinator → multiple agents)

Use this when you want to dispatch agents in parallel per the
parallelisation waves. Replace `<WAVE-N>` with the wave number you're
on, and `<LANE>` with the specific lane assignment.

```
You are one of N agents executing Wave <N> of the Ingestion Robustness
Programme.

## Your lane

Lane: <LANE> (e.g., L1 — Phase 1 parser unit tests)

## Files you may modify

<list from ROBUSTNESS/06-PARALLELIZATION.md lane assignment table>

## Files you may NOT modify (other agents are working on them)

<list the other lanes' file sets to avoid collisions>

## Read first

- ROBUSTNESS/04-ROADMAP.md § Phase <N>
- ROBUSTNESS/05-EVALUATION.md § Phase <N>
- ROBUSTNESS/07-DETAILS.md § Phase <N>

## Your deliverable

A single PR with subject `<type>(<scope>): <phase-name>` (per
03-CONTRIBUTING.md § 2) that lands your phase per its evaluation
hooks. The PR description follows the template in 03-CONTRIBUTING.md
§ 3.

## Coordination rules

- Rebase on `main` before opening the PR (other Wave-<N> agents may
  have landed first).
- DO NOT touch files outside your lane assignment.
- If your phase finds a real bug while in flight, open a follow-up PR
  on a separate branch — don't bundle it with the phase work.
- If you need to negotiate with another agent (e.g., L3 needs to
  modify a file L8a is also touching), open a discussion comment on
  the relevant PR rather than racing.

## When you're done

Report back with:
- PR URL
- Evaluation-hook checklist with each box ticked (from 05-EVALUATION.md)
- Any blockers or findings worth a follow-up PR

Working directory: /Users/bruaristimunha/Projects/eegdash/scripts/ingestions/
```

---

## PHASE PROMPT (single phase, focused)

Use this when you want maximum control — one phase per agent
invocation, review carefully between.

```
You are executing Phase <N>: <NAME> of the Ingestion Robustness
Programme.

## Required reading

- ROBUSTNESS/04-ROADMAP.md, the "Phase <N>" section (rationale,
  deliverable, expected wall-clock)
- ROBUSTNESS/05-EVALUATION.md, the "Phase <N>" section (the evaluation
  hooks — your test for whether you actually delivered)
- ROBUSTNESS/07-DETAILS.md, the "Phase <N> details" section (code
  templates, config snippets)
- ROBUSTNESS/02-STYLE_GUIDE.md (the bar for style on any new code)

## Your scope

ONLY Phase <N>. Do not start the next phase. Do not preemptively
touch files outside the phase's lane (per 06-PARALLELIZATION.md).

## Your deliverable

A PR that:

1. Passes every evaluation-hook check for Phase <N> from
   05-EVALUATION.md (including the OUTCOME metrics, not just ACTION
   metrics).
2. Has a description following 03-CONTRIBUTING.md § 3 template.
3. Adds at most ONE new file per logical concept (don't bundle
   unrelated changes).

## What "done" means for this phase

Quote and check off the evaluation hooks at the end of your final
commit message:

  Closes #<phase-issue-number>

  Evaluation hooks (from ROBUSTNESS/05-EVALUATION.md § Phase <N>):
    ✓ <hook 1 verbatim>
    ✓ <hook 2 verbatim>
    ✓ <hook 3 verbatim>

If any hook is not green, the PR is NOT ready to merge.

## Stop conditions

Stop and ask for human input if:

- An evaluation hook fails AND you've tried two distinct approaches.
- The phase's expected wall-clock is exceeded by 2×.
- A real bug surfaces that doesn't fit in this phase's scope.

Working directory: /Users/bruaristimunha/Projects/eegdash/scripts/ingestions/
```

---

## Notes for the human running this

### Picking which prompt

- **MEGA**: when you have a capable model on a long session and trust
  its judgment for sequencing.
- **WAVE**: when you want speed via parallelism — you become the
  coordinator, dispatch waves, review between.
- **PHASE**: when you want maximum care per merge. One agent, one phase,
  one PR, one review.

### Expected duration

| Scope | Wall-clock |
|---|---|
| MEGA on a single very-capable agent | 8-12 hours of focused work |
| WAVE with 3-4 parallel agents | 4 calendar days |
| PHASE one at a time | 2-3 weeks calendar |

The slower options trade speed for fidelity. The viewer's session was
mostly WAVE-style; that's a balanced compromise.

### What to do if an agent runs off the rails

The most common failure mode is **lowering thresholds to make CI
green**. If you see a commit titled "lower mutation break threshold"
or "skip flaky test" without an accompanying findings doc that
explains WHY, reject the PR and re-dispatch the phase with a stricter
prompt:

> "Re-do Phase <N>. The previous attempt lowered the threshold to hide
> a regression. The threshold stays at <value>. Either land the work
> behind the threshold or document the OUTCOME-metric failure
> honestly."

### What to do if the agent finds a real bug

Celebrate. Then make sure:
1. The bug has a regression test (per 03-CONTRIBUTING.md).
2. The fix is in a separate PR from the phase work.
3. The phase that found the bug references the fix PR.

The viewer's session found ~6 real bugs this way. Expect similar
density here. Bugs caught by the new gates are the programme working.
```
