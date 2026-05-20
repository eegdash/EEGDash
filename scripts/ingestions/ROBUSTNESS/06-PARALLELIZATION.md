# Parallelisation — the dependency DAG and dispatch plan

Phases are listed in `04-ROADMAP.md` by sequence. They are **NOT**
linearly dependent — most of them can fan out. This document maps the
DAG explicitly so a coordinator (human or agent) can dispatch multiple
workers without merge collisions.

## The DAG

```
Phase 0 (Foundation)
   │
   │  unblocks
   │
   ├──▶  Phase 1 (Parser unit tests)  ───┐
   │                                      │
   ├──▶  Phase 5 (Network tests)         │
   │                                      │
   ├──▶  Phase 6 (Schema pre-flight)     │
   │                                      │
   ├──▶  Phase 3 (Bare-except sweep)     │
   │                                      │
   ├──▶  Phase 8a (Characterisation tests for digest)
   │                                      │
   │                                      ▼
   │      All four of these can run in parallel after Phase 0.
   │      None touches the same files.
   │
   └──▶  Phase 2 (Property tests) — needs Phase 1's fixtures
              │
              └──▶  Phase 4 (Mutation testing) — needs Phase 1+2
                          │
                          └──▶  Phase 9 (Bug hunt) — continuous from here
                                       │
                                       └──▶  Phase 8b (Decompose digest) — needs
                                             characterisation tests as safety net

Phase 7 (Memory + bench) — can start anytime after Phase 0, runs in
                            isolation, no file collisions
```

## Lane assignment for parallel dispatch

Below: which phase touches which files. **Two lanes can run in
parallel iff their file sets do not intersect.**

| Lane | Phase | Files touched |
|---|---|---|
| **L0** | Phase 0 | `pyproject.toml`, `ruff.toml`, `__init__.py`, `tests/__init__.py`, all `*.py` (import rewrite + logger), `.github/workflows/lint-and-test.yml` |
| **L1** | Phase 1 | `tests/test_set_parser.py`, `tests/test_vhdr_parser.py`, `tests/test_snirf_parser.py`, `tests/test_mef3_parser.py`, `tests/fixtures/eeg/*` |
| **L2** | Phase 2 | `tests/test_*_property.py` (new files only) |
| **L3** | Phase 3 | All `*.py` files (because bare-excepts are spread everywhere) — **conflict potential**, see § Lane scheduling |
| **L4** | Phase 4 | `mutmut.ini`, `tests/mutation/` (new) |
| **L5** | Phase 5 | `tests/test_network/test_<service>.py` |
| **L6** | Phase 6 | `5_inject.py` (add `--dry-run`), `tests/test_inject_dryrun.py`, `.github/workflows/schema-dryrun.yml` |
| **L7** | Phase 7 | `tests/test_memory.py`, `tests/test_bench.py`, `pytest.ini` (benchmark config) |
| **L8a** | Phase 8a | `tests/test_digest_characterization.py`, `tests/fixtures/digest_golden/*.json` |
| **L8b** | Phase 8b | `3_digest.py` decomposition + new files like `_digest_helpers.py` |
| **L9** | Phase 9 | Audit reports only — no code changes (each finding becomes its own follow-up PR) |

## Lane scheduling (concrete)

### Wave 1 — serial (1 person, 1 day)

```
[L0] Phase 0
```

This blocks everything else. Don't dispatch parallel workers on it —
the package conversion has to land cleanly.

### Wave 2 — parallel (4 workers, 1-2 days)

After Phase 0 lands:

```
worker A: [L1] Phase 1 (parser unit tests)
worker B: [L5] Phase 5 (network tests)
worker C: [L6] Phase 6 (schema pre-flight)
worker D: [L7] Phase 7 (memory + bench)
```

These four touch entirely disjoint file sets. They can run in parallel
without coordination beyond "rebase before merge". If using subagents,
dispatch all four in one message — same pattern as the viewer's
`dispatching-parallel-agents` skill.

### Wave 3 — semi-parallel (3 workers, 1-2 days)

After Wave 2 lands:

```
worker A: [L2] Phase 2 (property tests, needs Wave-2-A's fixtures)
worker B: [L3] Phase 3 (bare-except sweep) — coordinate with L8a
worker C: [L8a] Phase 8a (characterisation tests for digest)
```

`L3` and `L8a` both touch files under `3_digest.py`. Coordination:
either `L3` finishes its passes on `3_digest.py` first and then `L8a`
starts, or split `L3` by file so it doesn't touch `3_digest.py` until
`L8a`'s characterisation tests are green.

### Wave 4 — serial-ish (1-2 workers, 2-3 days)

```
worker A: [L4] Phase 4 (mutation testing — only meaningful with Wave 2+3 tests in place)
worker B: [L8b] Phase 8b (decompose digest under characterisation tests' safety net)
```

`L4` and `L8b` don't directly collide on files, but the decomposition
will invalidate mutation cache. Run them in order:

1. `L4` baselines mutation kill ratio on the current `3_digest.py`.
2. `L8b` decomposes — each refactor commit re-baselines mutation
   (expect score to shift, possibly drop temporarily).
3. `L4` re-runs after `L8b` lands, targets a higher floor on the
   decomposed helpers.

### Wave 5 — continuous

```
worker(s): [L9] Phase 9 audits, ongoing
```

Bug hunting is never done. Schedule one audit per sprint:

- Audit 1: `3_digest.py` concurrency
- Audit 2: `_montage.py` numerical correctness
- Audit 3: retry-loop convergence
- Audit 4: path-traversal defence
- Audit 5: ...

Each audit takes < 1 day. Findings become their own PRs (with the
regression test + the fix bundled together, per `03-CONTRIBUTING.md`).

## Coordination protocol

When dispatching multiple agents in parallel:

1. **Before dispatch**: confirm that the file sets in their lane
   assignments do not intersect. Use `git diff --name-only HEAD~N` on
   recent commits to predict collisions.
2. **During work**: each agent commits to its own branch; **no force
   pushes**.
3. **Merge order**: when multiple PRs are ready, merge in lane-letter
   order (L1 → L5 → L6 → …). This keeps the rebase chain shallow.
4. **Rebase before merge**: every PR rebases on `main` immediately
   before merging, so any conflicts surface in one author's queue, not
   downstream.

## Critical path

The longest blocking chain is:

```
Phase 0 (1d) → Phase 1 (3d) → Phase 2 (1d) → Phase 4 (2d) → Phase 8b (5d) → Phase 9-continuous
Total: 12 days serial
```

With 4-way Wave 2 parallelism, total wall-clock drops to **8-10 days**.

## What this is NOT

- This is not a Gantt chart with hard deadlines. Phases that find real
  bugs will take longer; that's the work, not a failure.
- This is not "do everything at once" — Phase 0 truly blocks, and
  L8b's decomposition genuinely needs L8a's characterisation tests
  green before it can land safely.
- This is not a substitute for code review. Parallel dispatch makes
  the work faster, but each PR still goes through the
  `03-CONTRIBUTING.md` two-reviewer rule (for high-risk files) or
  one-reviewer rule (for everything else).

## Anti-pattern to refuse

> "Let's dispatch 9 agents on 9 phases in parallel — maximum throughput!"

No. Phase dependencies are real (mutation testing requires unit tests
to mutate against; decomposition requires characterisation tests as a
safety net). Skipping the DAG produces merge conflicts at best and
silent regressions at worst.

The viewer's session had a moment exactly like this: I tried to
dispatch Phase 8 (tinybench) and Phase 9 (mutation iter 4) at the same
time, and they both touched `package.json` for new devDependencies.
We had to serialize. **Read the lane file-set table before each
dispatch.**
