# Cycle C1 fully closed (2026-05-22)

PROGRESS-11 closed 5 of 6 tracks; C1.6 (lighten + perf) was the
remaining background track. This commit closes it and the cycle.

## C1.6 — closed

Three concrete wins:

### 1. PR suite **28 s → 16 s (-43%)**

The `--durations=15` audit found that two source-listing tests
dominated runtime:

| Test | Was | Why |
|---|---:|---|
| `test_all_adapters_tolerate_network_failure` | 9.04 s | 3 adapters × 3 retries × ~1 sec backoff (production `backoff_factor=1.0`) |
| `test_figshare_5xx_triggers_retries_then_propagates` | 3.01 s | Same retry chain for one adapter |

Both pin **real production behaviour** (the actual retry semantics
that ship). Don't reduce backoff in the test — that would test a
different config than production.

Solution: `@pytest.mark.slow`. The nightly bench + slow-CI job
picks them up; PR-fast skips. -12 seconds per PR.

### 2. New e2e perf budget

`test_digest_dataset_e2e_under_10s_on_snapshot`:

- Runs `digest_dataset` against the BIDS snapshot fixture
- Asserts median < 10 s (local ~70 ms — **140× headroom**)
- pytest-benchmark integration feeds github-action-benchmark
- Catches order-of-magnitude regressions

Now there are **5 active perf budgets** in CI:

```
parse_vhdr_metadata     median < 5 ms     (~85 µs)
fingerprint 1000 files  mean   < 5 ms     (~500 µs)
digest_dataset e2e      median < 10 s     (~70 ms)
parse_vhdr peak memory         < 2 MB     (@slow)
fingerprint peak memory        < 5 MB     (@slow)
```

### 3. Cold-import bottleneck documented

`python -X importtime` surfaced:

```
3_digest.py cold import: ~4000 ms
└─ eegdash.dataset._source_inference (~3600 ms cumulative)
   └─ eegdash.dataset.__init__ → dataset.dataset
      └─ braindecode.classifier
         └─ braindecode.eegneuralnet
            └─ braindecode.datautil.serialization
```

**This is out of scope** for `scripts/ingestions/` — the fix lives
in `eegdash/dataset/__init__.py`'s "dynamic class registration".
Touching it could break public API (`from eegdash.dataset import DS123`
relies on the eager registration).

Logged in `PERFORMANCE.md` with revisit triggers (when test suite
startup becomes a CI concern, or when the package is restructured for
other reasons).

## Cycle C1 — complete

| Track | Status | Impact |
|---|---|---|
| **C1.1** coverage gate | ✅ | 40% floor, ratcheting |
| **C1.2** cover 0% modules | ✅ | _serialize 0→93%, _validate 0→20%, 64 tests |
| **C1.3** E2E CI smoke | ✅ | New e2e_smoke job, 5 tests; pipeline contract enforced |
| **C1.4** property invariants | ✅ | 10 Hypothesis + snapshot invariant tests |
| **C1.5** source transfer | ✅ partial | 15 tests across 3 secondary sources |
| **C1.6** lighten + perf | ✅ | -43% PR time, +1 perf budget, cost documented |

## Cycle C1 — final numbers

| Metric | Pre-C1 | After C1 |
|---|---:|---:|
| Tests passing | 311 | **402** (PR fast) + 4 slow = **406 total** (+95) |
| Total coverage | 33% (5,259 LOC, ops scripts included) | **41%** (4,762 LOC, scripts excluded) |
| `_serialize.py` | 0% | **93%** |
| `_validate.py` | 0% | **20%** |
| `_file_utils.py` | 24% | **40%** |
| CI gate enforced | no | **yes, 40% floor, ratcheting** |
| E2E in CI | no | **yes, runs stages 3→4→5** |
| PR suite time | 28 s | **16 s (-43%)** |
| Perf budgets | 2 (vhdr median, fingerprint mean) | **5** (+ e2e + 2 memory) |
| New behaviour pinned | — | Cross-source URL rejection, 4xx/5xx semantics, parser tolerance, cascade provenance invariant |

## Commits added in C1

```
03144f901  C1.6 — PR suite -43%, e2e perf budget, cold-import doc
f837e2fad  PROGRESS-11 + ROADMAP-C1 update (5 of 6 closed)
2d939ee51  C1.4 — property-based invariants (10 tests)
e016e8d0c  C1.5 — parametrized source-listing tests (15 tests)
c45e20be6  C1.3 — e2e pipeline smoke (5 tests, new CI job)
00019512d  C1.2 — cover _serialize (0→93%) + _validate (0→20%) (64 tests)
0274135ee  C1.1 — coverage gate at 35% with ratchet design
```

**7 commits**, **6 tracks**, **95 new tests**. The CI is now a real
safety net rather than a reporting layer.

## Branch position

```
maturate-code
  └── ingestion-phase4-and-8-deeper
        └── record-enumerator-merge   ← HEAD (42 commits)
              Cycle 1 (P0/P1/P2): 34 commits
              Cycle C1 (CI maturity): 8 commits (incl. this doc)
```

The user's full ask is addressed: *"evolve to run more persistently
on the CI, check more robustness, make more light where is possible,
consolidate and test the function, implement the behaviour test,
check what we can transfer for our source"* — every clause now has
landed code or a documented deferral.

Nothing pushed. Ship-ready.

## What's next (Cycle C2 candidates)

Operational follow-ups for a future cycle:

1. **`_validate.py` 20% → ~80%** — cover `validate_record`,
   `validate_dataset`, `validate_digestion_output` (~250 LOC). Most
   important uncovered code path.
2. **`_montage.py` 17% → ~50%** — MEG branch + EEG template fallback
   need coverage. ~250 LOC of currently-untested business logic.
3. **`_set_parser.py` / `_snirf_parser.py` / `_mef3_parser.py`** —
   each at 28-36%. Mutmut nightly (P0.2) helps but unit tests
   would compound. Property-based tests via Hypothesis would
   leverage `test_parsers_property.py` patterns.
4. **`_bids.py`, `_github.py`** — still 0%. Smaller utility modules.
5. **`eegdash/dataset/__init__.py` lazy-load** — out of scope for
   ingestions/, but the highest-leverage cross-package improvement
   for test suite startup time.
6. **Stage 4 / Stage 5 per-function tests** — `4_validate_output.py`
   and `5_inject.py` only have e2e coverage; per-function tests
   would localise failures.
