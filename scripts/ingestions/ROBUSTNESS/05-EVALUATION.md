# Evaluation — separating activity from delivered rationale

This is the document the entire programme turns on.

A common failure mode of test-quality work is **activity that doesn't
move an outcome metric**. Examples:

- "I added 50 tests" but they're all `assert result is not None` — no
  bug would have been caught.
- "Coverage went from 60% to 85%" but mutation kill ratio stayed
  flat — lines run but assertions are loose.
- "I refactored `digest_from_manifest` into 12 helpers" but didn't add
  tests for those helpers — the regression risk just moved, didn't shrink.

For each phase in `04-ROADMAP.md`, this document specifies:

1. **The rationale** — what we were trying to deliver
2. **The action metric** — did the work happen? (cheap to fake)
3. **The outcome metric** — did the rationale land? (hard to fake)
4. **The disconfirmation test** — what would prove the work was wasted?

If the action metric is green but the outcome metric is flat, the work
**did not land**. Restart that phase with a sharper target.

## Phase 0 — Foundation

**Rationale**: Make the codebase testable and lintable at all.

| Metric | Type | Threshold |
|---|---|---|
| `pytest --collect-only` exits 0 | action | required |
| `ruff check ingestions/` runs (any output) | action | required |
| `mypy ingestions/` runs without crashing | action | required |
| **`pytest tests/test_smoke.py` exists AND passes from outside the ingestions dir** | **outcome** | **the whole point** |
| `grep -c "^[[:space:]]*print(" ingestions/*.py` = 0 | action | required |

**Disconfirmation**: If after phase 0 you still need to `cd
scripts/ingestions/` to run any test, **the package conversion didn't
land** — fix it before anything else proceeds.

## Phase 1 — Parser unit tests

**Rationale**: When a parser regresses, a specific test fails with a
specific assertion message, and the failure points at the line that
broke.

| Metric | Type | Threshold |
|---|---|---|
| Test count per parser | action | ≥ 10 |
| Total test count for parsers | action | ≥ 60 |
| `pytest --cov=ingestions/_set_parser` line coverage | outcome | ≥ 80% |
| **`git revert HEAD~1` on the most recent parser change causes ≥ 3 tests to fail** | **outcome** | the tests actually constrain behaviour |
| % of tests using `assert result is not None` style | outcome | ≤ 5% |

**Disconfirmation**:
- If a test reads "`assert isinstance(result, dict)`" with no further
  assertion, it's a smoke test. Doesn't count.
- If reverting the parser fix doesn't break tests, the tests aren't
  guarding the fix. Re-write them.

## Phase 2 — Property tests

**Rationale**: Catch crash classes the unit tests didn't think of.

| Metric | Type | Threshold |
|---|---|---|
| 1 property per parser exists | action | required |
| Hypothesis shows ≥ 100 examples per test in `--hypothesis-show-statistics` | action | required |
| **A counter-example surfaces and is pinned as a regression** | **outcome** | **this is the only way to know it works** |
| At least 1 documented case of "property found a bug the unit tests didn't" | outcome | over time, ≥ 1 |

**Disconfirmation**: If no property test ever finds anything in a year,
either (a) the parsers are unusually robust (possible — the viewer's
parsers came back clean at 100k iter), or (b) the properties are too
weak. Investigate by lowering `numRuns`, generating more pathological
inputs, etc.

## Phase 3 — Bare-except sweep

**Rationale**: An error you can't see is an error you can't fix. Each
bare-except is a place where the pipeline lies about what happened.

| Metric | Type | Threshold |
|---|---|---|
| `ruff check --select BLE001 ingestions/` violations | action | 0 |
| Number of bare-excepts changed | action | 85 (all of them) |
| **Number of bugs found while replacing them** | **outcome** | ≥ 1 |
| Number of `logger.exception(...)` calls added | action | track |

**Disconfirmation**: If you replaced 85 bare-excepts and didn't find a
single surprise (a place where the swallowed error was masking real
data corruption), you replaced them too mechanically. Read each one,
think about what it was hiding.

## Phase 4 — Mutation testing

**Rationale**: c8 / coverage tells you "lines ran". Mutation testing
tells you "tests would catch a regression". They're not the same.

| Metric | Type | Threshold |
|---|---|---|
| `mutmut run` completes | action | required |
| Mutation kill ratio on `_set_parser.py` | outcome | ≥ 60% |
| Number of surviving mutants documented as "equivalent" vs "real gap" | outcome | tracked per cluster |
| `mutmut results` cycle time (after caching) | outcome | < 2 min |

**Disconfirmation**: If kill ratio is below 50% on a parser with > 80%
line coverage, the tests are *executing* code without *constraining*
it. Add golden-output assertions (see viewer iter-8: that's the lesson
that took ~37% to ~70%).

## Phase 5 — Network tests

**Rationale**: The pipeline depends on 7 external services. If any of
them changes its API contract, downstream consumers silently corrupt.
Tests pin the contract.

| Metric | Type | Threshold |
|---|---|---|
| Tests per service | action | ≥ 4 (200 / 404 / 5xx-retry / timeout) |
| **Re-running `pytest tests/test_network/` produces deterministic results** | **outcome** | no flakes in 10 runs |
| Real-bug catches: API contract change detected by a failing CI test | outcome | over time, ≥ 1 |
| Coverage on `_http.py` retry logic | outcome | ≥ 90% |

**Disconfirmation**: If a network test passes when the real service is
returning a different shape, the test is over-mocked. The test must
fail when the contract drifts.

## Phase 6 — Schema pre-flight

**Rationale**: Stop bad records before MongoDB. Recovery from a corrupt
database is hours of work; preventing the corrupt write is seconds.

| Metric | Type | Threshold |
|---|---|---|
| `5_inject.py --dry-run` flag exists and is documented | action | required |
| CI workflow `schema-dryrun.yml` exists | action | required |
| **At least 1 production-equivalent fixture record fails validation in CI when intentionally broken** | **outcome** | proves the gate fires |
| Time from PR open to CI gate result | outcome | ≤ 5 min |

**Disconfirmation**: If you push a malformed record to CI and the gate
doesn't fire, the gate is decorative. Fix.

## Phase 7 — Memory + bench

**Rationale**: Catch regressions before they OOM CI or page on-call.

| Metric | Type | Threshold |
|---|---|---|
| Memory test fixture covers `digest_batch(1000)` | action | required |
| Peak memory < 100 MB | outcome | required |
| Throughput p99 < 50 ms per record | outcome | required |
| **A deliberately-introduced O(N²) accumulator fails the test** | **outcome** | the gate fires on real regressions |

**Disconfirmation**: Same as schema gate — introduce a regression, see
if the test catches it. If not, the test is decorative.

## Phase 8 — Decompose digest

**Rationale**: A function with 631 LOC cannot be reviewed (a reviewer
will defer to the author rather than read it). A function with 60 LOC
can.

| Metric | Type | Threshold |
|---|---|---|
| Max function LOC in `3_digest.py` | outcome | ≤ 80 |
| Average function LOC in `3_digest.py` | outcome | ≤ 50 |
| Max cyclomatic complexity (`radon cc`) | outcome | ≤ 15 |
| **Per-helper unit-test coverage** | **outcome** | ≥ 80% |
| **Characterisation test against pre-refactor output is still green** | **outcome** | proves no behaviour drift |

**Disconfirmation**: If after refactor the characterisation test fails
"slightly", that's a bug introduced during decomposition. Revert, find
the divergence, redo.

## Phase 9 — Bug hunting

**Rationale**: Tests + lint are gates against future bugs. Audits find
the bugs that already exist.

| Metric | Type | Threshold |
|---|---|---|
| Number of audit reports produced | action | ≥ 4 (concurrency, montage, retry, path-traversal) |
| Number of file:line-cited findings per audit | action | track |
| **Number of findings that became fix PRs** | **outcome** | ≥ 50% of P1/P2 findings |
| **Number of regression tests added for each fix** | **outcome** | 1 per fix |

**Disconfirmation**: If an audit finds 5 P1 issues and only 1 becomes a
PR, the audit went in the trash. Either the findings were not real
P1s (in which case downgrade them) or the team is under-resourcing
follow-through.

## Overall programme evaluation

After all phases land, run this checklist:

```bash
# Maturity-ladder evaluation — based on the viewer's reference doc
# (eegdash-viewer/docs/research/qa-benchmark-maturity-2026-05.md).

✓ pytest passes from outside scripts/ingestions/
✓ ruff + mypy --strict pass on all ingestion modules
✓ Test count > 200
✓ Mutation kill ratio > 60% on parsers
✓ Memory ceiling < 100 MB for 1000-record batch
✓ Schema pre-flight blocks production-bad records
✓ All except blocks are named (zero BLE001 violations)
✓ Max function LOC in 3_digest.py ≤ 80
✓ whats_new.rst has entries for the last 3 user-visible changes
✓ At least one PR was rejected by CI for a real test failure
  (proves CI is more than decoration)
```

If 8/10 are checked, the programme **landed**. 5/10 means the activity
happened but the rationale didn't follow. Re-investigate.

## The meta-evaluation: did this programme help?

Six months after completion, ask:

1. How many PRs in the last 90 days were rejected by CI?
2. How many of those were "real" rejections (a regression you were glad
   was caught) vs "noise" (flaky test, irrelevant rule)?
3. How many bugs reported by downstream consumers (the viewer, the
   `eegdash` Python package) were *root-caused* in the ingestion code?

If (1) is high and (2)/(1) > 50%, the gates earn their keep.
If (3) is non-zero, the gates have gaps — the programme isn't "done".
