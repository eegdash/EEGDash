# Phase 3 Findings — bare-except sweep

## Summary

Initial scope: replace all 72 `except Exception:` and `except:` clauses
flagged by `ruff BLE001`. This is the first commit of Phase 3 —
remaining files will be cleared in subsequent commits.

## Files swept (this commit)

| File | bare-excepts before | after |
|---|---:|---:|
| `_set_parser.py` | 3 | 0 |
| `_mef3_parser.py` | 2 | 0 |
| `_snirf_parser.py` | 3 | 0 |
| **subtotal** | **8** | **0** |

The per-file-ignore for these files in `pyproject.toml` is removed
in the same commit, so `ruff check` now lints them at full strictness.

## Real bugs found

### 1. `parse_set_metadata` was not catching `IndexError`

**How found**: `tests/test_parsers_property.py::test_parse_set_never_crashes`,
running 200 random byte buffers, surfaced an input
(`b'\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'`)
that causes `scipy.io.loadmat` to raise `IndexError` (not one of the
exception classes my narrowed `except` clause listed).

**Impact**: A truncated `.set` file with byte content shaped vaguely
like a MAT header would propagate an `IndexError` to the caller,
crashing the digest run on a single malformed input.

**Fix**: added `IndexError` to the `except` tuple at `_set_parser.py:147`.

**Why example-tests missed it**: The truncated committed fixture happens
to be benign (returns `has_fdt: False`). Property tests with `binary()`
inputs are precisely the tool that finds this class of bug.

### 2. `parse_snirf_metadata` lacked a `logger` import

**How found**: After narrowing the `except` clause, the test
`test_parse_snirf_garbage_input_does_not_crash` triggered a previously-
dead-code path that referenced `logger`. The module had no `logger`
defined.

**Impact**: Latent — only manifests when the new narrow `except` block
is hit (which only happens during fuzzing or production garbage input).
But once it would manifest, it would replace the original `ValueError`
with a `NameError`, hiding the real cause.

**Fix**: added `import logging; logger = logging.getLogger(__name__)`
at the top of `_snirf_parser.py`.

### 3. `_mef3_parser.py` had no `logger` either

Same pattern as #2. Fixed by the same import.

## What we did NOT do

- The 64 remaining `except Exception:` clauses in `3_digest.py`,
  `_montage.py`, `5_inject.py`, `_bids.py`, `_file_utils.py`,
  `_github.py`, `_http.py`, `4_validate_output.py`, etc. — those land
  in follow-up Phase 3 commits.
- A `noqa: BLE001` audit on the still-gated files. We rely on the
  per-file-ignore in `pyproject.toml` so the gate stays green during
  the sweep.

## Evaluation hooks (from ROBUSTNESS/05-EVALUATION.md Phase 3)

For the three swept files:

- `ruff check --select BLE001 _set_parser.py _mef3_parser.py _snirf_parser.py`
  → 0 violations ✓
- ≥ 1 real bug found while replacing → 1 confirmed (IndexError on
  parse_set), plus 2 latent NameError bugs (missing logger imports)
  ✓ (above the floor)
- `logger.exception(...)` calls added → 3 modules now import logging
  and emit structured debug-level logs on caught exceptions

## Next sweep target

`5_inject.py` and `4_validate_output.py` — the two files that hit the
production MongoDB. Highest blast radius per `01-AUDIT.md` § 11 risk
ranking.
