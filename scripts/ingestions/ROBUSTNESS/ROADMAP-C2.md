# Cycle C2 — Deepen test coverage on legacy modules

C1 made CI a safety net for the new modules. C2 brings the legacy
modules up to the same trust bar.

## Opening signal

Per-module coverage after C1:

| Module | LOC | C1 end | C2 target |
|---|---:|---:|---:|
| `_validate.py` | 188 | 20% | **80%** |
| `_montage.py` | 428 | 17% | **50%** |
| `_set_parser.py` | 117 | 36% | 60% |
| `_snirf_parser.py` | 93 | 28% | 60% |
| `_mef3_parser.py` | 74 | 30% | 60% |
| `_parser_utils.py` | 120 | 32% | 60% |
| `_bids.py` | 75 | **0%** | **70%** |
| `_github.py` | 91 | **0%** | **60%** |

The 0%-coverage modules are the highest leverage per hour. After
those, `_validate.py`'s remaining 80% is the production-critical
schema gate path.

## Tracks (priority order)

### C2.1 — Cover `_validate.py` (20% → 80%)

**Driver**: `validate_record`, `validate_dataset`, and
`validate_digestion_output` are the schema gate that protects
MongoDB from malformed Records. C1.2 covered the URL-pattern
constants + `ValidationResult` shell. The actual validation logic
(~250 LOC) is still untested.

**Outcome**: per-function tests for the 3 validators. Cover both
happy paths and the documented error paths (missing fields, type
mismatches, suspicious values).

**Estimated effort**: 3-4 hours.

**Definition of done**: `_validate.py` >= 80%, gate floor bumps.

### C2.2 — Cover the two 0% utility modules

**Driver**: `_bids.py` (BIDS path helpers) and `_github.py` (GitHub
API wrappers) are both at 0%. They're small (75-91 LOC) — easy
wins that get the coverage gate above 50%.

**Estimated effort**: 2-3 hours.

**Definition of done**: each module >= 70%.

### C2.3 — Format-parser direct tests (set, snirf, mef3)

**Driver**: mutmut nightly (P0.2) catches mutants but localised
failures need unit tests too. Each parser is at 28-36%.

**Outcome**: per-format characterisation tests using fixtures + the
property-based test patterns from `test_parsers_property.py`.

**Estimated effort**: 4-6 hours.

**Definition of done**: each parser >= 60%.

### C2.4 — Cover `_montage.py` MEG + template branches

**Driver**: MEG sensor extraction (~190 LOC) and EEG template-
matching fallback (~150 LOC) are the biggest uncovered code paths
in `_montage.py`. The 4 TSV configs (P2.1) are now tested
transitively, but the special-cased MEG path isn't.

**Estimated effort**: 4-6 hours.

**Definition of done**: `_montage.py` >= 50%.

### C2.5 — `eegdash/dataset/__init__.py` lazy-load (cross-package)

**Driver**: documented in PERFORMANCE.md as the biggest cold-
import bottleneck (~3.6 sec of braindecode → PyTorch chain).
Out of scope for `scripts/ingestions/`; lives in eegdash's
`dataset` package.

**Estimated effort**: 2-3 hours (incl. checking API breakage).

**Definition of done**: cold-import < 1 sec; eegdash's existing
`from eegdash.dataset import DSXXXXX` still works.

## Execution order

```
C2.1 (validate)  ─┬─► coverage gate bump
C2.2 (bids+github)┘
                  └─► C2.3 (parsers) ─► C2.4 (montage)

C2.5 lazy-load — independent, do whenever
```

C2.1 + C2.2 first (highest leverage per hour). Then parsers, then
montage. C2.5 is cross-package — gated on having time for the
cross-codebase review.

## Working agreements

Same as C1:
- Snapshots stay byte-stable
- Coverage gate ratchets — never down
- ADRs for deferrals
- PROGRESS-N doc per closed track
- No `--no-verify`
