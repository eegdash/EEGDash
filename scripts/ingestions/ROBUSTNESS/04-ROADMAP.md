# Roadmap — phased plan, ~4 weeks of focused work

Each phase has a **rationale** (the why), a **deliverable** (the
artefact), and an **evaluation hook** (how we know the rationale was
delivered, not just executed — see `05-EVALUATION.md` for the full
methodology).

Phases are sequenced by **dependency**, not by importance. See
`06-PARALLELIZATION.md` for which phases can run concurrently.

## Phase 0 — Foundation (1 day, BLOCKS everything else)

| Item | Rationale | Deliverable |
|---|---|---|
| `pyproject.toml` in `scripts/ingestions/` | Declares dependencies, lint config, test config; enables `pip install -e .` for tests | Single file at ingestion root |
| `ingestions/__init__.py` | Turns the directory into a Python package; enables relative imports | Empty file is fine |
| Rename sibling imports | `from _http import X` → `from .http import X` (and rename `_http.py` → `http.py` once it's a package) | `git diff` over all .py files |
| `ruff.toml` (or `[tool.ruff]` in pyproject) | One source of truth for lint config | Single config block |
| `mypy.ini` (or `[tool.mypy]`) | Strict-mode opt-in **per file** initially, baselined errors | Config + a `mypy_baseline.txt` snapshot |
| `tests/` directory with empty `__init__.py` | Pytest collection root | One directory |
| `.github/workflows/lint-and-test.yml` | CI gate that runs `ruff` + `mypy` + `pytest` on every PR | One workflow file |
| Move all `print()` to `logging.getLogger(__name__)` | Single coherent log stream | `grep -c "^[[:space:]]*print(" *.py` returns 0 |

**Evaluation hook**:
- `pytest --collect-only` exits 0 (collection works → package layout is sane)
- `ruff check ingestions/` produces a baseline of N warnings; CI fails if a PR adds more
- `mypy --strict ingestions/` produces a baseline; same rule
- `grep -c "print(" ingestions/*.py` is 0

**Anti-pattern to refuse**: Skipping the package conversion because
"it works today". It only works because every CI invocation `cd`s into
the directory. A future test runner won't.

## Phase 1 — Parser unit tests (3 days, parallelisable per-parser)

Each parser file (`_set_parser.py`, `_vhdr_parser.py`, `_snirf_parser.py`,
`_mef3_parser.py`) is a **pure function** taking bytes and producing a
dict. Same shape as the viewer's `formats/edf.js` etc., which we tested
in iteration 1.

| Action | Tools |
|---|---|
| Commit small CC0 fixtures into `tests/fixtures/eeg/` | Reuse `eegdash-viewer/tests/fixtures/eeg/` (already vetted CC0 from OpenNeuro) |
| Per-parser test file: `tests/test_set_parser.py`, etc. | `pytest` |
| Golden-value assertions on the full returned dict | `pytest.approx` for floats, `==` for ints/strings |
| Edge-case tests: empty input, truncated header, malformed magic | Pure-function tests |

**Target**: 15-25 tests per parser. **Floor**: every parser has ≥ 10
tests including 2 edge-case tests.

**Evaluation hook**:
- `pytest tests/test_*_parser.py -v` shows all green
- `pytest --cov=ingestions/_set_parser` reports ≥ 80% line coverage
- A `git revert HEAD~1` on the latest parser commit causes ≥ 3 tests to
  fail (proves tests are tight)

**Anti-pattern to refuse**:
- Smoke tests: `def test_parse(): assert parse(fixture) is not None` →
  golden-value assertions on the full return value, not "is not None"
- Bundled tests: 10 asserts in one function

## Phase 2 — Property-based tests with Hypothesis (1 day)

The Python equivalent of `fast-check`. Single-day investment, multi-PR
payoff.

```python
from hypothesis import given, strategies as st

@given(buf=st.binary(min_size=0, max_size=8192))
def test_parse_set_metadata_never_crashes(buf):
    """parse_set_metadata may raise a documented exception, but must
    never produce SegFault, RecursionError, or non-Exception raise."""
    try:
        parse_set_metadata(buf)
    except (ValueError, KeyError, struct.error, EOFError):
        pass  # known and acceptable
```

**Target**: 1 no-crash property per parser, 1 round-trip property per
parser (if applicable: parse → serialise → parse → equal).

**Evaluation hook**:
- `pytest tests/test_*_property.py --hypothesis-show-statistics` runs
  ≥ 100 examples per property
- If a property finds a counter-example, fast-check / Hypothesis shrinks
  it; the shrunk bytes get committed as a regression fixture

## Phase 3 — Bare-except sweep (1 day, finds bugs)

The 85 bare-except clauses are a fault line. Mechanical sweep with
audit:

```bash
ruff check --select BLE001 ingestions/      # flag every bare-except
```

For each hit:

1. **Read** the try-block. What error is it actually catching?
2. **Replace** with a named except + log:
   ```python
   try:
       ...
   except (FileNotFoundError, PermissionError) as e:
       logger.warning("Could not read %s: %s", path, e)
       continue
   ```
3. **If** the answer is "all errors, this is a top-level safety net":
   keep `except Exception` but **log via `logger.exception`** so the
   traceback survives.

**Evaluation hook**:
- `ruff check --select BLE001 ingestions/` shows 0 violations
- At least 1 real bug surfaced (the viewer's session found 4; expect ≥ 1 here)
- Documented findings in `ROBUSTNESS/findings-bare-except.md`

## Phase 4 — Mutation testing with `mutmut` (2 days)

Same workflow as the viewer's Stryker iterations:

```bash
pip install mutmut
mutmut run --paths-to-mutate ingestions/_set_parser.py
mutmut results
mutmut show <id>  # see surviving mutants
```

Strategy (taught by the viewer): **start on ONE parser file, not the
whole package**. Get its kill ratio above 60% first, then expand.

**Target**: `_set_parser.py` mutation kill ratio ≥ 60% by end of phase.

**Evaluation hook**:
- `mutmut results` shows kill ratio per file
- A surviving-mutants document (analogue of
  `eegdash-viewer/docs/mutation-survivors-2026-05.md`) exists with the
  per-cluster analysis

## Phase 5 — Network-tier tests with `respx` (2 days)

`respx` is to `httpx` what `nock` is to fetch in Node. Each external
service gets a "what happens when …" test suite:

```python
@respx.mock
def test_openneuro_404_returns_none():
    respx.get("https://openneuro.org/api/datasets/ds999999").mock(
        return_value=httpx.Response(404)
    )
    result = fetch_openneuro_dataset("ds999999")
    assert result is None  # documented contract

@respx.mock
def test_openneuro_retries_on_502():
    route = respx.get("https://openneuro.org/api/datasets/ds002893").mock(
        side_effect=[
            httpx.Response(502),
            httpx.Response(502),
            httpx.Response(200, json={"id": "ds002893"}),
        ]
    )
    result = fetch_openneuro_dataset("ds002893")
    assert result["id"] == "ds002893"
    assert route.call_count == 3
```

**Target**: 4-6 tests per service × 7 services = 30+ tests.

**Evaluation hook**:
- One test per service exists for each of: 200 happy path, 404, 5xx retry
  exhaustion, network timeout
- Re-running the same fetch produces identical output (idempotence)

## Phase 6 — Schema validation pre-flight (1 day, critical)

`5_inject.py` writes to MongoDB. A bad record corrupts production.
Mirror the viewer's coverage gate:

1. Add `--dry-run` flag to `5_inject.py` that validates without writing.
2. Validate every record in a `tests/fixtures/records/` corpus against
   `RecordModel`, `DatasetModel`, `ManifestModel` via Pydantic strict mode.
3. New CI workflow: `schema-dryrun.yml` runs the dry-run on a fixture
   corpus on every PR touching `5_inject.py` or `eegdash.schemas`.

**Evaluation hook**:
- `python 5_inject.py --dry-run --input tests/fixtures/records/` exits 0
- CI workflow exists and blocks merge on failure

## Phase 7 — Memory + throughput gates (1 day)

```python
import tracemalloc

def test_digest_1000_records_under_100mb():
    tracemalloc.start()
    digest_batch(load_synthetic_records(1000))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 100 * 1024 * 1024
```

`pytest-benchmark` for throughput:

```python
def test_digest_record_p99_under_50ms(benchmark):
    rec = load_fixture("ds002893")
    benchmark.pedantic(digest_record, args=(rec,), iterations=100, rounds=5)
    assert benchmark.stats.stats.percentiles[99] < 0.050
```

**Evaluation hook**:
- Both tests exist and pass
- A subsequent change that introduces an O(N²) accumulator causes the
  memory test to fail
- A subsequent change that adds a sync HTTP call inside the digest loop
  causes the throughput test to fail

## Phase 8 — Decompose `3_digest.py` (3-5 days, big refactor)

The four mega-functions:

- `digest_from_manifest` — 631 LOC
- `extract_record` — 521 LOC
- `extract_dataset_metadata` — 360 LOC
- `digest_dataset` — 302 LOC

Strategy: **characterisation tests first**, then refactor under their
safety net.

1. For each mega-function: write a "characterisation test" that pins
   the current output against a known input fixture. Don't refactor
   yet.
2. Run the characterisation test → green.
3. Extract small helpers (`_walk_inheritance`, `_resolve_channels`,
   `_compose_record`, etc.) — same logic, just renamed and parameterised.
4. Re-run the characterisation test → still green.
5. Add unit tests for each new helper.

**Target**: no function over 80 LOC in `3_digest.py` by end of phase.

**Evaluation hook**:
- `radon cc -a ingestions/3_digest.py` reports max cyclomatic complexity
  ≤ 15 (current: very high)
- Average function LOC in `3_digest.py` ≤ 50
- Per-helper unit-test coverage ≥ 80%

## Phase 9 — Active bug-hunting (continuous after Phase 4)

The viewer's session found 4 real race conditions via a sleuth-agent
audit *after* the test infrastructure was in place. Same pattern here:

| Audit target | Tool / approach |
|---|---|
| `3_digest.py` concurrency | Read `ThreadPoolExecutor` blocks for shared mutable state. The viewer found `inFlight` clobbering; expect similar here. |
| `_montage.py` numerical correctness | Run golden-value tests with `np.testing.assert_allclose`; tolerances modality-calibrated (MNE-Python's `atol=1e-12` EEG vs `1e-20` MEG). |
| Retry-loop divergence | 5 files have their own retry logic. Consolidate into one `_retry.py` decorator. |
| Path-traversal defence | `extract_record(path)` walks `sidecars` which are external dicts. Can a crafted sidecar point at `/etc/passwd`? Test with a malicious-input case. |

**Evaluation hook**:
- Each audit produces a findings doc with file:line citations
- Real findings (not theoretical) become fix PRs with regression tests

## Phase summary table

| Phase | Title | Wall-clock | Blocks | Unblocks |
|---|---|---|---|---|
| 0 | Foundation | 1d | (all) | 1-9 |
| 1 | Parser unit tests | 3d | 4 | 4 |
| 2 | Property tests | 1d | — | 4 |
| 3 | Bare-except sweep | 1d | — | 9 |
| 4 | Mutation testing | 2d | — | 8 |
| 5 | Network tests | 2d | — | 9 |
| 6 | Schema pre-flight | 1d | — | (production) |
| 7 | Memory + bench | 1d | — | 8 |
| 8 | Decompose digest | 3-5d | 4 | (continuous) |
| 9 | Bug hunt | continuous | 0-8 | (continuous) |

See `06-PARALLELIZATION.md` for the actual dependency DAG.
