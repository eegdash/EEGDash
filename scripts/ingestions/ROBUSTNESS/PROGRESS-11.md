# Cycle C1 close-out (2026-05-22)

The user's frame: *"the code is undertest, and was mostly running
locally, we need to evolve this to run more persistently on the CI,
check more robustness, make more light where is possible, consolidate
and test the function, implement the behaviour test, check what we can
transfer for our source"*.

This document records what landed in response.

## ROADMAP-C1 status

| Track | Status | Tests added |
|---|---|---:|
| **C1.1** coverage gate in CI | ✅ DONE | (gate, no new tests) |
| **C1.2** cover 0%-coverage modules | ✅ DONE | 64 |
| **C1.3** E2E pipeline smoke in CI | ✅ DONE | 5 |
| **C1.4** property-based invariants | ✅ DONE | 10 |
| **C1.5** source transfer | ✅ partial | 15 |
| **C1.6** lighten where possible | ⏳ background | — |

**Five of six tracks closed.** C1.6 stays as an opportunistic
background task — any future PR that touches a legacy file should
check if it can be lighter.

## Coverage trajectory

| Stage | Total coverage | New tests |
|---|---:|---:|
| Cycle C1 baseline | 33% (5,259 LOC incl. ops scripts) | 311 |
| After C1.1 omit ops scripts | 36% (4,762 LOC) | 311 |
| After C1.2 (_serialize, _validate) | **39%** | 373 |
| After C1.3 (E2E) | 39% (e2e is subprocess) | 378 |
| After C1.5 (source-listing tests) | **41%** | 393 |
| After C1.4 (invariants) | 41% | **403** |

Coverage gate floor: 35 → 38 (C1.2) → 40 (C1.5). Every PR that
adds tests must bump the floor — ratchet enforcement.

## Per-module coverage (current state)

| Module | LOC | Before C1 | After C1 |
|---|---:|---:|---:|
| `_serialize.py` | 115 | **0%** | **93%** |
| `_validate.py` | 188 | **0%** | 20% |
| `_file_utils.py` | 389 | 24% | **40%** |
| `digest_telemetry.py` | 73 | 95% | 95% |
| `record_enumerator.py` | 134 | 89% | 89% |
| `source_adapter.py` | 98 | 86% | 86% |
| `_format_parser_registry.py` | 57 | 51% | 51% |
| `_vhdr_parser.py` | 147 | 64% | 64% |
| `_montage.py` | 428 | 17% | 17% |
| `_parser_utils.py` | 120 | 32% | 32% |
| `_set_parser.py` | 117 | 36% | 36% |
| `_snirf_parser.py` | 93 | 28% | 28% |
| `_mef3_parser.py` | 74 | 30% | 30% |
| `_bids.py` | 75 | 0% | 0% |
| `_github.py` | 91 | 0% | 0% |
| `_http.py` | 129 | 64% | 64% |

`_montage.py`, `_set_parser.py`, `_snirf_parser.py`, `_mef3_parser.py`,
`_parser_utils.py`, `_bids.py`, `_github.py` remain the highest-leverage
targets for the next coverage round.

## What the cycle delivers operationally

The user's specific asks, mapped:

| Ask | Closed by |
|---|---|
| "run more persistently on the CI" | C1.1 gate enforces 40% floor; C1.3 e2e job runs stages 3→4→5; mutmut nightly (P0.2 from cycle 1) |
| "check more robustness" | C1.2 covers schema gate (`_validate.py`); C1.4 property invariants |
| "make more light where is possible" | C1.1 excluded one-off ops scripts; C1.6 stays opportunistic |
| "consolidate and test the function" | C1.2 + C1.5 = 79 new direct tests on previously-untested code |
| "implement the behaviour test" | C1.3 e2e smoke + C1.4 invariants = behaviour-level coverage |
| "check what we can transfer for our source" | C1.5 — 15 tests across 3 secondary sources; 3 cross-source invariants codify the contract that transfers |

## Behaviour pinned by new tests

Beyond raw coverage, the tests captured real behavior contracts:

1. **`_validate.py` cross-source rejection** — the pre-PR-#327
   misrouting bug (OpenNeuro URL on a NEMAR dataset) is now a
   parametrized test case.
2. **Secondary adapter HTTP semantics** — 4xx returns empty; 5xx
   triggers retries and propagates `tenacity.RetryError`;
   `ConnectError` returns empty. Pinned across Figshare/Zenodo/OSF.
3. **Format-parser contract** — every registered parser returns
   `None` or `dict` on missing input, never raises.
4. **Cascade provenance invariant** — `_clamp_metadata_extremes`
   maintains `value is None iff provenance is None`. Tested via
   Hypothesis over the full input range.
5. **Pipeline contract enforcement** — Stage 3's output is accepted
   by Stage 4 (`validate --json`) and Stage 5 (`inject --dry-run`).
   `--strict` documentation is enforced (warnings → errors).

## Commits added this cycle

```
2d939ee51  C1.4 — property-based invariants (10 tests)
e016e8d0c  C1.5 — parametrized source-listing tests (15 tests)
c45e20be6  C1.3 — e2e pipeline smoke (5 tests, new CI job)
00019512d  C1.2 — cover _serialize.py (0→93%) + _validate.py (0→20%)
0274135ee  C1.1 — coverage gate at 35% with ratchet design
```

5 code commits + ROADMAP-C1 update + this doc. The cycle adds
**92 tests** total (311 → 403).

## What's still open

For a hypothetical Cycle C2:
- `_validate.py` 20% → ~80%: cover `validate_record`,
  `validate_dataset`, `validate_digestion_output` (~250 LOC).
- `_montage.py` 17%: the MEG / EEG-template-fallback branches.
- `_set_parser.py` / `_snirf_parser.py` / `_mef3_parser.py`: each
  at 28-36%; mutmut nightly (P0.2) helps here but unit tests
  would compound the value.
- `_bids.py`, `_github.py`: still 0% — utility modules with no tests.
- Source-listing: SciDB / DataRN / git-based adapters not yet
  parametrized.
- C1.6 lighten: opportunistic.

## Branch position

```
maturate-code
  └── ingestion-phase4-and-8-deeper
        └── record-enumerator-merge   ← HEAD (41 commits)
              Cycle 1 (P0/P1/P2): 34 commits
              Cycle C1 (CI maturity): 7 commits + this doc
```

Cycle C1 ship-ready. The CI is now a real safety net rather than a
reporting layer — coverage gated, e2e smoke runs on every PR,
invariants enforced.
