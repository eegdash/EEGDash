# Style Guide — Gramfort-flavoured scientific Python

The target style is **MNE-Python's house style**, established and
maintained by Alexandre Gramfort and the MNE team. It is well-tested
(MNE has been on this for a decade), it is review-friendly, it composes
with the tooling we want (`numpydoc`, `ruff`, `mypy`, `sphinx`), and it
is what neuroscientists who'll read this code already know.

Cross-reference: [MNE-Python `CONTRIBUTING.md`](https://github.com/mne-tools/mne-python/blob/main/doc/development/contributing.rst).

## 1. The shape of a function

Every public function follows this template. Internal helpers (leading
`_`) get the same shape but the docstring may be terser.

```python
def extract_record(file_path: Path, sidecars: dict[str, Any],
                   *, verbose: bool = False) -> RecordModel | None:
    """Extract a Record from a single BIDS-compliant file.

    Walks the BIDS inheritance chain to resolve metadata, parses the
    binary header for `n_channels` and `sampling_frequency`, and
    composes a `RecordModel` ready for injection into MongoDB.

    Parameters
    ----------
    file_path : Path
        Absolute path to the data file (e.g. ``…/sub-01_eeg.edf``).
        Must exist; symlinks are resolved.
    sidecars : dict
        BIDS sidecars indexed by suffix (``"eeg"``, ``"channels"``,
        ``"electrodes"``). See `_bids.walk_inheritance` for the schema.
    verbose : bool, default False
        If True, log per-channel detail at INFO level. Off by default
        because typical ingestion runs process 10⁴+ files.

    Returns
    -------
    record : RecordModel or None
        Validated record. Returns ``None`` (after logging at WARNING)
        if the file is structurally unreadable.

    Raises
    ------
    ValueError
        If `file_path` is outside the dataset root (defence against
        path-traversal via crafted sidecar references).

    Notes
    -----
    The walk follows BIDS 1.7.0 § 4.2 (file-level inheritance). When
    a sidecar value conflicts between levels the closest-to-file value
    wins, matching `mne_bids.get_entities_from_fname`.

    References
    ----------
    .. [1] BIDS Specification v1.7.0, § 4.2.

    Examples
    --------
    >>> path = Path("/data/ds002893/sub-001/eeg/sub-001_task-rest_eeg.edf")
    >>> rec = extract_record(path, sidecars={"eeg": {"SamplingFrequency": 250}})
    >>> rec.sampling_frequency
    250
    """
```

### Rules
- **NumPy docstring style** (Parameters / Returns / Raises / Notes /
  References / Examples). `numpydoc` lints this.
- Keyword-only arguments for "configuration" flags (use `*,` separator).
  Positional-only for the "subject" of the function.
- `verbose: bool = False` is the canonical MNE flag — preserves it for
  cross-tool muscle memory.
- Type hints **everywhere**, including in private helpers. `mypy
  --strict` is the goal.
- One blank line between sections in the docstring.

## 2. Naming

| Kind | Pattern | Example |
|---|---|---|
| Public function | `verb_noun_modifier` | `extract_record`, `fingerprint_from_files` |
| Internal helper | `_verb_noun` | `_parse_set_header` |
| Constant | `UPPER_CASE` | `MAX_BUFFER_SIZE` |
| Module | `snake_case.py` (no digits prefix for libs) | `bids.py`, `http.py` |
| File-numbered scripts | `N_action.py` (kept for pipeline scripts) | `3_digest.py` — OK as a top-level command |

**Avoid**:
- One-letter loop variables outside obvious math (`r`, `c` for row/col are OK in matrix code; `x` for a dataset is not).
- Type names in variable names (`record_dict`, `path_str`) — use the type system.
- Abbreviations that aren't in the BIDS or MNE glossary (`recs`, `seg`).

## 3. Imports

```python
# Standard library
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Third-party
import httpx
import numpy as np
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

# Local — relative imports from inside the package
from .bids import walk_inheritance
from .http import get_client
from .schemas import RecordModel

logger = logging.getLogger(__name__)
```

Rules:
- Three blocks (stdlib, third-party, local) separated by one blank.
- `from __future__ import annotations` at the top of every file with
  type hints (cheap, forward-compatible).
- **No sibling imports** (`from _http import …`). Use package-relative
  imports (`from .http import …`). This is Phase 0 work.
- `logger = logging.getLogger(__name__)` at module level, never inside
  functions. This is the **only** way to emit messages.

## 4. Logging > printing

```python
# WRONG
print(f"Processed {n} records")

# RIGHT
logger.info("Processed %d records", n)
```

- Always parameterise: `logger.info("X = %s", x)`, never
  `logger.info(f"X = {x}")`. The lazy form lets log aggregators
  group identical messages with different parameters.
- `logger.debug` for per-record detail. `logger.info` for per-dataset
  milestones. `logger.warning` for recoverable failures (a record we
  skipped). `logger.error` for unrecoverable. `logger.exception` (only
  inside `except` blocks) for "log the traceback and re-raise".

## 5. Errors

```python
# WRONG — swallows everything, hides programmer errors
try:
    record = extract_record(path)
except Exception:
    pass

# RIGHT — narrow except, log + decide
try:
    record = extract_record(path)
except (FileNotFoundError, PermissionError) as e:
    logger.warning("Could not read %s: %s", path, e)
    return None
except ValidationError as e:
    logger.warning("Skipping %s: schema invalid: %s", path, e)
    return None
# No catch-all → programmer errors propagate, CI fails, you find out.
```

Rule: **every `except` block names the exception type or types.** Bare
`except:` and `except Exception:` are linted out via `ruff` `BLE001`.

Custom exceptions when the wrapping is semantically meaningful:

```python
class BIDSParseError(ValueError):
    """Raised when a BIDS sidecar exists but cannot be parsed."""
```

## 6. Function length

Hard rule: **target ≤ 50 LOC per function, accept up to 80, alarm at
100.** No function above 100 LOC ships without an explicit reviewer
sign-off comment in the PR.

Why: testing functions over 50 LOC requires either (a) testing many
behaviours in one test (brittle) or (b) extensive mocking (fragile).
Refactoring after the fact is harder than splitting early.

The viewer's session showed this in reverse: `traces.js` `draw()` was
600+ LOC and required *promoting module-private helpers as debug
exports* (`_niceRound`, `_computeTimeAxisLayout`) to make it testable.
Don't bake yourself that corner.

## 7. Tests

- `pytest` only. No `unittest`, no `nose`.
- Test files named `test_<module>.py` mirroring source layout.
- Each test asserts ONE behaviour. Don't bundle six asserts into a
  "smoke test".
- Parametrise instead of looping:
  ```python
  @pytest.mark.parametrize("ext,n_chan", [
      ("edf", 36), ("set", 64), ("vhdr", 32),
  ])
  def test_parse_header_channel_count(ext, n_chan):
      ...
  ```
- Numeric assertions: `np.testing.assert_allclose(actual, expected, atol=...)`
  with a **domain-calibrated** tolerance. Document the choice:
  ```python
  # atol=1e-6 µV — single-precision sample storage rounds the 10th
  # decimal; tighter assertions would just be testing IEEE-754.
  np.testing.assert_allclose(samples, expected, atol=1e-6)
  ```

## 8. Configuration

Function signatures, not module globals. The viewer learned this the
hard way (`globalThis.__perf` was a dead module-global for two years).

```python
# WRONG
WORKER_COUNT = 8  # module-level constant
def digest_batch(records):
    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as ex:
        ...

# RIGHT — discoverable, testable, override-friendly
def digest_batch(records: list[Path], *, workers: int = 8) -> list[Record]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ...
```

## 9. Defensive copies (when crossing trust boundaries)

```python
def assemble_record(channels: list[Float32Array]) -> Record:
    """Channels is borrowed from the reader's internal buffer; copy
    before storing so the next readWindow doesn't clobber our data."""
    owned = [np.array(ch, copy=True) for ch in channels]
    return Record(channels=owned, ...)
```

Document the copy with a one-line comment naming WHY. This is the same
pattern that fixed the viewer's "channels buffer reused on next pan"
bug.

## 10. Anti-patterns we explicitly refuse

| Don't | Do |
|---|---|
| `try: ... except: pass` | Named except + log |
| `print("Processing", x)` | `logger.info("Processing %s", x)` |
| `def f(x):` with no annotations | `def f(x: Path) -> Record:` |
| `dataset_dict = {...}` (type in name) | `dataset = {...}` (type in hint) |
| `globals().get('config')` | Pass `config` as a parameter |
| Module-level mutable state | Function-local or class-instance state |
| Sibling import (`from _foo import …`) | Package-relative (`from .foo import …`) |
| Catching `Exception` to "be safe" | Catch the specific class; if you don't know it, you have a bug to find |

Every one of these has tooling that flags it: `ruff` + `mypy --strict`
catch all of them.
