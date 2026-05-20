# Details — concrete deepening per phase

The previous documents are strategic. This one is tactical: code
snippets, configuration files, decision heuristics, and the small
choices that make the difference between "did the work" and "the work
landed."

## Phase 0 details

### `pyproject.toml` skeleton

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "eegdash-ingestions"
version = "0.1.0"
description = "Pipeline for ingesting BIDS EEG datasets into eegdash"
requires-python = ">=3.10"
authors = [{name = "Bruno Aristimunha"}]
dependencies = [
    "httpx>=0.27",
    "tenacity>=8",
    "pydantic>=2.5",
    "mne-bids>=0.14",
    "eegdash",
    "python-dotenv>=1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=4",
    "pytest-benchmark>=4",
    "hypothesis>=6.100",
    "respx>=0.20",
    "mutmut>=2.4",
    "ruff>=0.4",
    "mypy>=1.10",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["ingestions*"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = [
    "E", "F", "W",   # pycodestyle + pyflakes
    "I",             # isort
    "B",             # bugbear
    "BLE",           # blind-except (BLE001 is the big one for Phase 3)
    "UP",            # pyupgrade
    "PT",            # pytest-style
    "RUF",           # ruff-specific
    "N",             # pep8-naming
    "D",             # pydocstyle (numpy convention)
]
ignore = [
    "E501",  # line too long — let formatter handle it
    "D203",  # 1 blank line required before class docstring (conflicts with D211)
    "D213",  # multi-line summary on second line (conflicts with D212)
]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
# Tighten per-module as you go; start permissive on the big files
[[tool.mypy.overrides]]
module = "ingestions.digest"
disallow_untyped_defs = false  # 3_digest.py is the biggest, baseline first

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
addopts = "-ra --strict-markers --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "network: marks tests that require network access",
]

[tool.coverage.run]
source = ["ingestions"]
omit = ["*/tests/*", "*/fixtures/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

### File rename script (Phase 0 mechanics)

```bash
# Run from scripts/ingestions/
cd scripts/ingestions

# 1. Create package init
touch __init__.py
mkdir -p tests
touch tests/__init__.py

# 2. Rename underscore-prefixed modules so they become real submodules
#    (optional — sibling imports also work in a package, but cleaner)
for f in _*.py; do
    new="${f#_}"
    git mv "$f" "$new"
done

# 3. Rewrite imports across the whole tree
ruff check --select I --fix ingestions/  # isort fixes most

# 4. Verify
python -c "from ingestions import digest"  # should not raise
pytest --collect-only                       # should find tests/__init__.py
```

### CI workflow skeleton

```yaml
# .github/workflows/lint-and-test.yml
name: Lint + Test

on:
  pull_request:
    paths:
      - 'scripts/ingestions/**'
      - '.github/workflows/lint-and-test.yml'
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e "scripts/ingestions[dev]"
      - run: ruff check scripts/ingestions/ingestions/
      - run: ruff format --check scripts/ingestions/ingestions/
      - run: mypy scripts/ingestions/ingestions/

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e "scripts/ingestions[dev]"
      - run: cd scripts/ingestions && pytest tests/ --cov=ingestions --cov-fail-under=70
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: coverage-html
          path: scripts/ingestions/htmlcov/
```

## Phase 1 details

### Test fixture layout

```
scripts/ingestions/tests/
├── __init__.py
├── conftest.py                    # shared fixtures (Pydantic models, tmp_path helpers)
├── fixtures/
│   ├── eeg/
│   │   ├── sub-01_eeg.edf        # 32 KB, CC0 from ds002034
│   │   ├── sub-001_eeg.set       # 64 KB, CC0 from ds002893
│   │   ├── sub-xp101_eeg.vhdr    # 12 KB, CC0 from ds002336
│   │   └── README.md             # attribution
│   ├── ieeg/
│   │   └── ...                   # mirror eegdash-viewer/tests/fixtures/ieeg/
│   ├── meg/
│   │   └── test-proj.fif         # 4.5 KB, BSD-3 from mne-python
│   └── records/                  # golden Record JSONs for schema tests
│       ├── valid_record_eeg.json
│       └── valid_record_meg.json
├── test_set_parser.py
├── test_vhdr_parser.py
├── test_snirf_parser.py
├── test_mef3_parser.py
├── test_bids.py
├── test_http.py
└── ...
```

**Reuse viewer fixtures**: copy the entire `eegdash-viewer/tests/fixtures/eeg/`,
`ieeg/`, `meg/` trees verbatim. They're already CC0 / BSD with full
attribution. Don't re-derive.

### Test template

```python
# tests/test_set_parser.py
"""Tests for the EEGLAB .set header parser."""

from pathlib import Path

import pytest

from ingestions.set_parser import parse_set_metadata

FIXTURES = Path(__file__).parent / "fixtures" / "eeg"


def test_parse_set_minimal_header():
    """Parse a real .set file from ds002893 and pin every documented field."""
    buf = (FIXTURES / "sub-001_eeg.set").read_bytes()
    meta = parse_set_metadata(buf)
    # Golden values measured from the source dataset:
    assert meta["n_channels"] == 36
    assert meta["sampling_frequency"] == 250
    assert meta["channel_labels"][0] == "Fp1"
    # MNE-style tolerance — `n_samples` is exact for binary headers
    assert isinstance(meta["n_samples"], int)


@pytest.mark.parametrize("ext,n_chan,fs", [
    ("edf", 82, 512),
    ("set", 36, 250),
])
def test_parse_known_headers(ext, n_chan, fs):
    """Parametrised regression: each format's known fixture parses correctly."""
    parser = {
        "edf": "ingestions.edf_parser:parse_edf_metadata",
        "set": "ingestions.set_parser:parse_set_metadata",
    }[ext]
    # ... resolve and call ...


def test_parse_set_empty_buffer_raises_valueerror():
    """Empty input is not a smoke test — assert the EXACT exception class."""
    with pytest.raises(ValueError, match="header too short"):
        parse_set_metadata(b"")


def test_parse_set_truncated_magic_raises_valueerror():
    """Truncated input that LOOKS like a .set header but ends prematurely."""
    buf = b"\x00\x01MATLAB"  # partial magic
    with pytest.raises(ValueError):
        parse_set_metadata(buf)
```

## Phase 2 details — Hypothesis property template

```python
# tests/test_set_parser_property.py
"""Property-based tests for parse_set_metadata.

The no-crash property catches the class of bug that crashed fiff.js
in the sister viewer: a "magic" check that silently accepted any input
because the test corpus never accidentally collided with the expected
bytes.
"""

import struct
import pytest
from hypothesis import given, strategies as st, settings

from ingestions.set_parser import parse_set_metadata


@given(buf=st.binary(min_size=0, max_size=8192))
@settings(max_examples=1000, deadline=None)
def test_parse_set_never_crashes_on_random_bytes(buf):
    """parse_set_metadata MAY raise documented exceptions, but never
    SegFault, RecursionError, or non-Exception raise."""
    try:
        parse_set_metadata(buf)
    except (ValueError, KeyError, struct.error, EOFError):
        pass  # acceptable failure modes


@given(
    n_chan=st.integers(min_value=1, max_value=256),
    fs=st.sampled_from([100, 250, 500, 1000, 5000]),
)
def test_constructed_header_parses_consistently(n_chan, fs):
    """When given a header WE construct with known (n_chan, fs), parsing
    returns those exact values."""
    buf = _build_synthetic_set_header(n_chan=n_chan, fs=fs)
    meta = parse_set_metadata(buf)
    assert meta["n_channels"] == n_chan
    assert meta["sampling_frequency"] == fs
```

## Phase 3 details — bare-except triage

For each `except Exception:` hit, ask three questions:

1. **What error is it actually catching?** Read the try-block carefully.
2. **Is the swallow recoverable?** (e.g., "we tried to read a sidecar
   that doesn't exist — fine, return default") vs unrecoverable
   (e.g., "the channel count is corrupt — we cannot proceed").
3. **What would the user / next-pipeline-stage want to know?** Log it
   structured, with enough context to act on.

Decision template:

```python
# Before
try:
    json_data = json.load(open(sidecar_path))
except Exception:
    json_data = {}

# After — one of these three patterns:

# Pattern A: known-recoverable, log + default
try:
    with sidecar_path.open() as fh:
        json_data = json.load(fh)
except FileNotFoundError:
    logger.debug("Sidecar not found at %s, using empty default", sidecar_path)
    json_data = {}
except json.JSONDecodeError as e:
    logger.warning("Malformed JSON at %s line %d: %s", sidecar_path, e.lineno, e.msg)
    json_data = {}

# Pattern B: top-level guard, log + re-raise as our domain error
try:
    json_data = _load_sidecar(sidecar_path)
except OSError as e:
    raise BIDSParseError(f"sidecar {sidecar_path} unreadable") from e

# Pattern C: actually a bug — let it crash + CI catches it
# (just delete the try/except)
json_data = _load_sidecar(sidecar_path)
```

## Phase 4 details — `mutmut` config

```ini
# scripts/ingestions/mutmut.ini
[mutmut]
paths_to_mutate = ingestions/set_parser.py,ingestions/vhdr_parser.py
runner = python -m pytest scripts/ingestions/tests/ -x --tb=no -q
tests_dir = scripts/ingestions/tests
cache_dir = .mutmut-cache
```

Start with one file (`set_parser.py`). Run, look at survivors, write
tests, re-run. Target: ≥ 60% kill ratio before expanding scope.

Survivors doc structure (copy from viewer):

```markdown
# Mutation survivors — set_parser.py (2026-MM-DD)

Baseline: NN% kill ratio (K killed / T total, S survived)

## Genuinely surviving (test gap candidates)

### Mutant ID: 47
- Location: ingestions/set_parser.py:142
- Original: `if n_channels <= 0:`
- Mutated:  `if n_channels < 0:`
- Why survived: no test for n_channels == 0
- Fix: add test passing a 0-channel header, expect ValueError
```

## Phase 5 details — `respx` template per service

```python
# tests/test_network/test_openneuro.py
"""Network contract tests for OpenNeuro fetcher."""

import httpx
import pytest
import respx

from ingestions.fetchers.openneuro import fetch_dataset


OPENNEURO_BASE = "https://openneuro.org/api"


@respx.mock
def test_openneuro_happy_path_returns_dict():
    """200 OK → parsed dict shaped to our contract."""
    respx.get(f"{OPENNEURO_BASE}/datasets/ds002893").mock(
        return_value=httpx.Response(200, json={"id": "ds002893", "name": "..."})
    )
    result = fetch_dataset("ds002893")
    assert result["id"] == "ds002893"


@respx.mock
def test_openneuro_404_returns_none():
    """Missing dataset → None, not an exception (documented contract)."""
    respx.get(f"{OPENNEURO_BASE}/datasets/ds999999").mock(
        return_value=httpx.Response(404)
    )
    result = fetch_dataset("ds999999")
    assert result is None


@respx.mock
def test_openneuro_retries_on_502_then_succeeds():
    """5xx errors retry up to 3 times via tenacity."""
    route = respx.get(f"{OPENNEURO_BASE}/datasets/ds002893").mock(
        side_effect=[
            httpx.Response(502),
            httpx.Response(502),
            httpx.Response(200, json={"id": "ds002893"}),
        ]
    )
    result = fetch_dataset("ds002893")
    assert result["id"] == "ds002893"
    assert route.call_count == 3, "expected 2 retries + 1 success"


@respx.mock
def test_openneuro_gives_up_after_max_retries():
    """Persistent 5xx → eventual None + WARNING log."""
    respx.get(f"{OPENNEURO_BASE}/datasets/ds002893").mock(
        return_value=httpx.Response(503)
    )
    with pytest.raises(httpx.HTTPStatusError):
        fetch_dataset("ds002893")


@respx.mock
def test_openneuro_timeout_is_handled():
    """Network timeout → retry, then surface."""
    respx.get(f"{OPENNEURO_BASE}/datasets/ds002893").mock(
        side_effect=httpx.TimeoutException("upstream")
    )
    with pytest.raises(httpx.TimeoutException):
        fetch_dataset("ds002893")
```

## Phase 6 details — `--dry-run` for `5_inject.py`

```python
# Add at the top of main():
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate records against schema, do not write.")
    args = parser.parse_args()

    records = load_records(args.input)
    errors = []
    for rec in records:
        try:
            RecordModel.model_validate(rec)
        except ValidationError as e:
            errors.append((rec.get("path", "?"), e))

    if errors:
        logger.error("Schema validation failed for %d records", len(errors))
        for path, e in errors[:5]:
            logger.error("  %s: %s", path, e)
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry-run: %d records valid; not writing.", len(records))
        return

    # Real write path follows...
```

CI gate:

```yaml
# .github/workflows/schema-dryrun.yml
name: Schema dry-run

on:
  pull_request:
    paths:
      - 'scripts/ingestions/5_inject.py'
      - 'scripts/ingestions/_validate.py'
      - 'eegdash/schemas.py'

jobs:
  dryrun:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e "scripts/ingestions[dev]"
      - run: python scripts/ingestions/5_inject.py --dry-run \
               --input scripts/ingestions/tests/fixtures/records/
```

## Phase 7 details — memory test pattern

```python
# tests/test_memory.py
import gc
import tracemalloc

import pytest

from ingestions.digest import digest_batch
from ingestions.tests.fixtures import synthetic_records


@pytest.mark.slow
def test_digest_1000_records_peak_memory_under_100mb():
    """Catches O(N²) accumulator regressions."""
    records = synthetic_records(n=1000)
    gc.collect()
    tracemalloc.start()
    try:
        digest_batch(records)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    peak_mb = peak / 1024 / 1024
    assert peak_mb < 100, f"Peak memory {peak_mb:.1f} MB > 100 MB ceiling"
```

`pytest-benchmark` for throughput:

```python
def test_digest_record_p99_under_50ms(benchmark):
    record = synthetic_records(n=1)[0]
    benchmark.pedantic(
        digest_record,
        args=(record,),
        iterations=1,
        rounds=100,
    )
    p99 = benchmark.stats.stats.percentiles[0.99]
    assert p99 < 0.050, f"p99 latency {p99*1000:.1f} ms > 50 ms"
```

## Phase 8 details — characterisation test pattern

Before touching the 631-LOC `digest_from_manifest`:

```python
# tests/test_digest_characterization.py
"""Characterisation tests — pin current behaviour as the contract.

These run BEFORE the decomposition refactor. They lock down the
function's current output against a known input. If decomposition
changes any externally-observable behaviour, these tests fail.

DO NOT update the golden files to "fix" a failing test during
decomposition — that defeats the purpose. If the test fails, the
refactor introduced a real behaviour drift; revert and find why.
"""

import json
from pathlib import Path

import pytest

from ingestions.digest import digest_from_manifest

GOLDEN = Path(__file__).parent / "fixtures" / "digest_golden"


@pytest.mark.parametrize("dataset", ["ds002034", "ds002336", "ds002893"])
def test_digest_from_manifest_matches_golden(dataset):
    """Run digest_from_manifest on a snapshotted manifest, compare to golden."""
    manifest = json.loads((GOLDEN / f"{dataset}_manifest.json").read_text())
    expected = json.loads((GOLDEN / f"{dataset}_records.json").read_text())
    actual = digest_from_manifest(manifest)
    # Sort both to make the comparison order-independent
    actual_sorted = sorted(actual, key=lambda r: r["path"])
    expected_sorted = sorted(expected, key=lambda r: r["path"])
    assert actual_sorted == expected_sorted
```

Workflow:

1. **Before** any refactor: run the current `digest_from_manifest` on
   the manifest fixtures, save its output as the golden file.
2. **Run** the characterisation test — green.
3. **Refactor** (extract helpers, rename, split files).
4. **Re-run** the characterisation test — must stay green.
5. **Once green** post-refactor, add unit tests for each new helper.

## Phase 9 details — audit report template

```markdown
# Audit: 3_digest.py concurrency (2026-MM-DD)

**Auditor**: <name or agent ID>
**Scope**: `digest_from_manifest`, `process_datasets_with_watchdog`,
the ThreadPoolExecutor blocks.

## Findings

### F1 — Shared dict mutation from worker threads [P1]

- **Location**: ingestions/digest.py:NNN
- **Evidence**: `dataset_metrics` dict is shared; worker threads call
  `dataset_metrics[dataset_id] = ...` without a lock.
- **Trigger**: Two datasets completing within the same GIL tick. Rare
  but documented; will surface as occasional missing records in the
  output.
- **Severity**: P1 — silent data loss.
- **Suggested fix**: use `threading.Lock` or `concurrent.futures` result
  collection via `as_completed`.

### F2 — ...
```
