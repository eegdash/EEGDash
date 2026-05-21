# Next sprint plan + lessons learned

**Branch**: `record-enumerator-merge` (67 commits, nothing pushed)
**Last checkpoint**: C8 — Pydantic-settings config pattern across all 4 main-stage scripts
**Tests**: 788 PR-fast + 16 live integration (opt-in via env vars)
**Coverage**: 60%, floor enforced via CI gate, 20 visible ratchet steps
**Date**: 2026-05-22

---

## State at end of last session

### What works
- Coverage gate enforced in CI at **60%** (PR fails if it drops)
- E2E smoke runs digest → validate → inject --dry-run against snapshot fixtures
- Live integration tests (16) against `data.eegdash.org` → `eegdash_dev` (opt-in)
- Stress tests verify all 14 C6.1 BIDS fields round-trip through Gateway → MongoDB
- 4 main-stage scripts (clone/digest/validate/inject) use Pydantic-settings configs (`_*_config.py`, 100% covered each)
- Mutmut nightly CI workflow exists (P0.2) — never observed running because branch isn't pushed yet
- Telemetry stream (P1.1) writes per-Record `record_built` events when `EEGDASH_TELEMETRY_PATH` is set

### What's deferred (with ADRs)
- **ADR 0001** — source-listing Seam: 1 production source, not building Adapters until ≥ 2
- **ADR 0002** — pipeline orchestrator: no driver, contract documented in `PIPELINE-CONTRACT.md`
- **C2.5** — `eegdash/dataset/__init__.py` cold-import (3.6s braindecode chain) — cross-package, needs separate PR

### Branch arc (5 cycles, 67 commits)
```
Cycle 1 (P0/P1/P2):     34 commits — cascade-with-provenance, mutmut nightly,
                                     DigestTelemetry, per-helper tests, modality
                                     Seam, format-parser Seam, pipeline-contract
                                     deferral
Cycle C1 (CI maturity):  8 commits — coverage gate, e2e smoke, property tests,
                                     source-listing, PR suite 28s→16s
Cycle C2 (legacy):       5 commits — _serialize, _validate, _bids, _montage,
                                     _parser_utils
Cycle C3 (production):   6 commits — secondary source adapters, SNIRF synth,
                                     SET synth, _mef3 fail paths
Cycle C4 (diminishing):  4 commits — _http variants, _parser_utils network,
                                     _github PyGithub, _file_utils helpers
Cycle C5 (real data):    3 commits — real MEF3 .tmet from OpenNeuro ds003708
                                     CAUGHT REAL BUG (sfreq offset)
                                     + e2e MEF3 pipeline test
Cycle C6 (BIDS + API):   4 commits — BIDS sidecar enrichment (14 new fields),
                                     inject API integration tests (respx),
                                     live cluster tests (5), stress tests (11)
Cycle C6.2+/C7/C8:       3 commits — Pydantic-settings configs for all 4
                                     main-stage scripts
```

---

## Lessons learned (carry these forward)

### 1. Synthetic fixtures only validate the parser against itself
C5.1's MEF3 bug was invisible until we used a real `.tmet` from OpenNeuro. The
parser had hardcoded offsets `[1272, 1280, 1288, 272, 280]` that worked against
our synthetic fixtures but missed real-data sfreq at offset **8720**. This is
the **single most important finding** of the whole programme.

**Apply forward**: when adding a new format parser, ALWAYS pair the synthetic
fixture (h5py/scipy.io.savemat) with a real-data fixture from production
sources. Use `pytest.mark.skipif(fixture_missing)` + curl-recovery message in
the skip reason.

### 2. "Real driver" beats "would be nice"
The deferral pattern (ADRs 0001 + 0002 + many smaller calls) prevented
~3 cycles of speculative architectural work. Every C-series cycle had a
real driver: user concern, observed bug, production gap. The cycles that
were almost-done-then-deferred (orchestrator, source-listing Seam) would
have been net-negative had we built them.

**Apply forward**: before designing, name the driver. If you can't, write an
ADR with revisit triggers instead of code.

### 3. Refactor before you test
C6.2's "5_inject.py main() has 460 untested lines" wasn't a testing problem —
it was a design problem. C6.5/C7/C8 replaced argparse with Pydantic-settings
and the testing question dissolved (28 + 18 + 11 = 57 tests run in
milliseconds vs subprocess harness).

**Apply forward**: when the cost of testing X is high, ask whether X is the
right shape. The "test pyramid" inverts here — the right shape often makes
testing trivial.

### 4. Integration tests find what unit tests can't
C6.4's stress tests verified all 14 C6.1 BIDS fields round-trip through the
real Gateway. Without that test, we'd have shipped the C6.1 enrichment trusting
that the Gateway's Pydantic schema uses `extra="allow"`. It does — but the
verification is now permanent.

**Apply forward**: every new field type / shape going to production deserves
one integration test. Cheap; catches drift.

### 5. The 1-bug-per-real-fixture yield rate
- C5.1: real .tmet → bug
- C6.4: live cluster tests → no bug yet (but found 4xx/5xx semantics, the
  Caddy DELETE block, the records filter param shape)

Each real-data round produces operational findings even when it doesn't catch
a regression. The findings themselves justify the cycle.

### 6. Ratchet enforcement matters more than absolute coverage
Going from no gate → 35% floor was bigger than going from 35% → 60%. The
gate prevents silent regression; the absolute number is a side effect.

**Apply forward**: every cycle bumps the floor in the same commit as the
tests landed. Never leave a "we could ratchet later" comment — do it now.

### 7. The deferral pattern's actual format
ADR 0001 and 0002 share a shape that works:
- Decision (defer or build)
- Driver / lack thereof
- Anti-recommendations (what a future architecture pass should NOT propose)
- Revisit triggers (concrete events that should reopen the decision)
- Cross-references

**Apply forward**: when deferring substantial work, write the ADR. The
anti-recommendations are the load-bearing part — they prevent re-litigation.

### 8. LOC drift goes the wrong way during enrichment

The old `ROADMAP.md` (lines 173-194) said: *"the next round of leverage
is in observability, not LOC reduction"*. That call was correct for
the observability outcomes (provenance + telemetry shipped). But the
LOC table at the bottom of that doc was never re-checked. Between
C5 and C8 the over-ceiling functions grew:

| Function | Roadmap stated | Today | Δ |
|---|---:|---:|---:|
| `_extract_technical_metadata` | 140 | **244** | **+74%** |
| `extract_dataset_metadata` | 205 | **232** | +13% |
| `extract_record` | 189 | **223** | +18% |
| `digest_dataset` | 110 | **135** | +23% |

C6's BIDS-sidecar enrichment added `_extract_bids_sidecar_fields`,
`_extract_channel_status_counts`, `_extract_dataset_description_extras`
as standalone helpers (good — those are deep). But the orchestration
in `_extract_technical_metadata` absorbed +104 LOC of conditional
wiring (cascade ordering, provenance stamping per step, VHDR/FIF
special cases). The next sprint's Tier-1 #4 addresses this.

**Apply forward**: every cycle that adds depth to leaves must
re-check the root-function LOC table at close.

---

## Cluster integration details (for next sprint)

### Topology (sccn host, indexing.sccn.ucsd.edu)
```
data.eegdash.org (TLS)
└─ Caddy reverse proxy (ports 80/443)
   └─ eegdash-api (FastAPI internal :3000)
      ├─ mongodb-production (:27017)
      │   ├─ eegdash (1.3 GB, production)
      │   ├─ eegdash_dev (262 MB, test target ←)
      │   └─ eegdash_archive (empty)
      └─ eegdash-redis (rate limiting)
```

### Integration test access
```bash
# Run from a dev box with SSH access:
export EEGDASH_INTEGRATION_API_URL="https://data.eegdash.org"
export EEGDASH_INTEGRATION_ADMIN_TOKEN="<see compose>"  # rotate me
pytest -m integration

# Test data uses c6_smoke_ + c6_stress_ prefixes for orphan sweep:
ssh sccn 'docker exec mongodb-production mongosh \
  "mongodb://admin:PASS@localhost:27017/eegdash_dev?authSource=admin" \
  --quiet --eval "
    const ds = db.datasets.deleteMany({dataset_id: /^c6_(smoke|stress)_/});
    const rec = db.records.deleteMany({dataset: /^c6_(smoke|stress)_/});
    print(\"Deleted\", ds.deletedCount, \"+\", rec.deletedCount);
  "'
```

### Operational caveats (track these)
- **Production credentials are hardcoded** in `~/eegdash-competition/docker-compose.yml`.
  They leaked into commit messages during C6.3 discovery. **Rotate after this
  branch lands.** Move to `.env` file or vault.
- **Caddy blocks DELETE/PUT/PATCH** at the edge (Caddyfile ~line 180).
  The admin DELETE endpoint exists internally but isn't reachable via the
  public URL. See `INTEGRATION-TESTING.md` for workarounds.
- **The records `filter` query param** is JSON-encoded MongoDB syntax, not
  individual field params. Easy to get wrong; documented in
  `test_inject_integration_live.py:test_live_inject_record_round_trip`.

---

## Next sprint candidates (ranked by real-driver strength)

### Tier 1 — Real production drivers

1. **Push the branch + open a PR.** 67 commits is shippable. Watch mutmut
   nightly run for the first time. Watch the e2e CI smoke job land. Real
   driver: durability of the work itself.

2. **Rotate cluster credentials.** The compose file's secrets are in the
   commit log of THIS conversation. Real driver: security.

3. **Database list drift fix** (CONFIG-PATTERN.md caveat 1). The Literal
   in `_inject_config.py` and the API's `valid_databases` set can drift.
   Concrete fix: API exposes `GET /admin/valid-databases`; `InjectConfig`
   fetches at boot (with 5s cache). Real driver: production safety.

4. **`_extract_technical_metadata` depth refactor.** The cascade
   function grew 140 → 244 LOC across C5/C6 (BIDS-sidecar enrichment
   absorbed orchestration logic). Refactor extracts a
   `_metadata_cascade.py` module — small interface (`run(ctx) → result`),
   five cascade-step adapters behind it. Real driver: cascade test
   isolation (Lesson #3) + future "add 6th source" is one file.
   Snapshot tests are the gate (byte-stable required). See
   `SPRINT-2026-05-22-PLAN.md` Task 3.

### Tier 2 — Strong leverage if a driver appears

4. **Real SNIRF fixture from OpenNeuro** — ✅ **DONE 2026-05-22** in
   commit `998a28d1d`. OpenNeuro now publishes 26 fNIRS BIDS datasets;
   landed `ds007554` (CC0, 731 KB, 10 Hz, 32 ch). The probe caught a
   real bug (`_snirf_parser` never extracted `n_times`) — exactly the
   C5.1 yield pattern. See `tests/test_snirf_real_fixture.py` and
   `tests/fixtures/fnirs/LICENSE-ATTRIBUTION.md`.

5. **Stage 1 fetch consolidation** (CONFIG-PATTERN.md caveat 2). 9 per-source
   scripts with their own argparse. Worth doing when a 10th source is added.

6. **Stage 4/5 actual MongoDB write coverage**. The integration tests prove
   the API contract works. They don't prove the underlying MongoDB driver
   code in `api/main.py`. Out-of-scope here (different repo) but worth
   tracking.

7. **Cross-package `eegdash.dataset` lazy-load** (C2.5). Every cold
   import pays a 3.6 s braindecode chain (`PERFORMANCE.md`). Out of
   this repo's ingestions/ tree, but in scope for a follow-up PR.
   Real driver: every user pays this on every cold import. Tier 2
   (not 3) because the driver is universal, not hypothetical.

### Tier 3 — Speculative

8. **Pipeline `ConfigBase`** (CONFIG-PATTERN.md caveat 3). DRY 4 configs.
   Payoff too small unless a 5th stage joins.

---

## Working agreements (carry forward)

1. **Never push without explicit user say-so.** This branch has 67 commits
   and is intentionally local.
2. **Real drivers only.** Deferred work gets an ADR + anti-recommendations.
3. **Ratchet in-same-commit.** Coverage floor bumps with the tests that
   raised it.
4. **No `--no-verify`.** Pre-commit hooks (codespell, ruff, nested function
   check) are gates, not suggestions.
5. **Snapshots stay byte-stable.** Intentional updates only.
6. **Skip-guards have recovery commands.** When a fixture or env var is
   missing, the skip reason includes the exact command to recover.
7. **Production credentials never in code.** Use env vars + `.env` files
   (gitignored). Rotate if leaked.

---

## File locations (for handoff)

- Roadmaps: `scripts/ingestions/ROBUSTNESS/ROADMAP*.md`
- Progress logs: `scripts/ingestions/ROBUSTNESS/PROGRESS-*.md` (1-17)
- Cross-stage doc: `scripts/ingestions/ROBUSTNESS/CONFIG-PATTERN.md`
- BIDS audit: `scripts/ingestions/ROBUSTNESS/BIDS-GAP-AUDIT.md`
- Live test guide: `scripts/ingestions/ROBUSTNESS/INTEGRATION-TESTING.md`
- Pipeline contract: `scripts/ingestions/ROBUSTNESS/PIPELINE-CONTRACT.md`
- Perf doc: `scripts/ingestions/ROBUSTNESS/PERFORMANCE.md`
- ADRs: `scripts/ingestions/ROBUSTNESS/ADRs/{0001,0002}*.md`

## Resume recipe (for the next session)

```bash
cd /Users/bruaristimunha/Projects/eegdash
git log --oneline record-enumerator-merge ^ingestion-phase4-and-8-deeper
# 67 commits at e5e1574e5 (refactor C8)

cd scripts/ingestions
pytest -q -m "not network and not slow and not integration"
# Expect: 788 passed, 20 deselected, ~30s

# To run live integration (only with cluster access):
EEGDASH_INTEGRATION_API_URL="https://data.eegdash.org" \
EEGDASH_INTEGRATION_ADMIN_TOKEN="<token>" \
pytest -m integration
# Expect: 16 passed, ~25s
```

The next session can read this file and resume without reading the whole
prior transcript.
