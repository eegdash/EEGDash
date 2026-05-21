# Performance — budgets, observations, and known costs

ROADMAP-C1 C1.6 — "lighten where possible" + perf profiling.

## Active perf budgets in CI

Tests in `tests/test_perf.py` enforce hard ceilings via pytest-benchmark.
The `ingestions-bench.yml` workflow runs them on every PR.

| Test | Budget | Current (local median) |
|---|---|---|
| `test_parse_vhdr_median_under_5ms` | 5 ms / call | ~85 µs |
| `test_fingerprint_throughput_1000_files` | 5 ms / call | ~500 µs |
| `test_digest_dataset_e2e_under_10s_on_snapshot` | 10 s / run | ~70 ms |

Plus the memory tests (marked `@pytest.mark.slow`):

| Test | Budget | Notes |
|---|---|---|
| `test_parse_vhdr_peak_memory_under_2mb` | 2 MB peak | tracemalloc; runs in slow CI |
| `test_fingerprint_1000_files_peak_memory_under_5mb` | 5 MB peak | tracemalloc; runs in slow CI |

The budgets are deliberately generous (10s for a 50KB digest gives
~140× headroom over the local 70ms). The point is to **catch
order-of-magnitude regressions**, not to micro-optimise.

## Slow tests deliberately excluded from PR-fast CI

CI runs `pytest -m "not network and not slow"`. The slow set:

- `tests/test_source_listing.py::test_all_adapters_tolerate_network_failure`
  — exercises `tenacity.wait_exponential` retry-backoff across 3
  adapters. ~9 sec of waiting; happens because production adapters
  use `backoff_factor=1.0`. Marking it fast would require
  monkey-patching tenacity — that would test a different config
  than production runs.
- `tests/test_source_listing.py::test_figshare_5xx_triggers_retries_then_propagates`
  — same reason at smaller scale. ~3 sec.
- Memory tests in `tests/test_perf.py` (`@pytest.mark.slow`).

Suite time impact: **28 s → 16 s (-43%)** by excluding these on the
PR fast-path. They run in the nightly bench job + on-demand.

## Known cold-start cost (out of scope to fix here)

```
3_digest.py cold import: ~4000 ms
```

Of which ~3.6 s comes from `eegdash.dataset.__init__.py` triggering
`from . import dataset as _dataset_mod`, which pulls in
`braindecode.classifier` → PyTorch + neural network stack.

The import chain:
```
3_digest.py
 └ from eegdash.dataset._source_inference import ...
   └ eegdash.dataset.__init__.py
     └ from . import dataset as _dataset_mod
       └ from braindecode.datasets import ...
         └ braindecode.classifier
           └ braindecode.eegneuralnet (~1.6 sec)
             └ braindecode.datautil.serialization (~1 sec)
```

**This is amortised in production**. The ingest pipeline is a
long-running process — 4s of import per digest run is negligible
against minutes of per-dataset processing.

The visible cost is in **test startup**: each `pytest` invocation
that imports `3_digest.py` pays the 4s. With 11 test files importing
it transitively, collection + run pays this once per worker.

### Why we're not fixing it here

The fix lives in `eegdash/dataset/__init__.py` — that file does
"dynamic class registration" by importing the `dataset.py` submodule
at package init. Breaking that import order could:

- Change the public API (currently `from eegdash.dataset import DS123`
  works because of the eager registration)
- Affect every other consumer of `eegdash.dataset` (notebooks, scripts,
  notebooks-tested-in-eegdash-CI)

ROADMAP-C1 is scoped to `scripts/ingestions/` only. A separate ADR /
PR against `eegdash/dataset/` could:

1. Lazy-load the dataset.py submodule on first attribute access
   (`__getattr__` at module level)
2. Move `_source_inference` out of `eegdash.dataset` into a sibling
   `eegdash.storage_inference` or similar so importing it doesn't
   trigger the dataset package init
3. Document this cost in eegdash's docs as a known
   "second-pip-install-then-import" delay

### When to revisit

Revisit if:
- Test suite startup becomes a CI-cost concern (currently 16 s suite
  + ~4 s setup = 20 s per run; acceptable)
- The eegdash package is split / restructured for other reasons
- A new consumer needs to use `_source_inference` without paying
  the full dataset-package init cost

## Profiling tools available

- **pytest-benchmark** — what `test_perf.py` uses; produces JSON
  output for `github-action-benchmark` trend tracking
- **tracemalloc** — memory profile (used by the `_peak_memory_*`
  tests)
- **python -X importtime** — what was used to find the braindecode
  cost above; example:
  ```bash
  python -X importtime -c 'import importlib.util; \
    spec = importlib.util.spec_from_file_location("d", "3_digest.py"); \
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)' \
    2>&1 | sort -t '|' -k 2 -n -r | head
  ```
- **mutmut** — mutation testing (nightly per P0.2)

For Stage 4 / Stage 5 profiling (validate + inject), the e2e perf
test now covers the digest budget; future rounds can add similar
budgets for the downstream stages.

## Summary of C1.6 wins

| Win | Mechanism | Impact |
|---|---|---|
| PR suite 28 s → 16 s | `@pytest.mark.slow` on 2 retry tests | -43% PR time |
| E2E perf budget | `test_digest_dataset_e2e_under_10s_on_snapshot` | regression gate |
| Cold-import cost documented | this doc | known-issue ADR |
| Ops scripts excluded from coverage | `[tool.coverage.run].omit` | cleaner signal |

The eegdash.dataset cold-import cost is the biggest known
lighten-opportunity, but the fix lives outside this scope.
