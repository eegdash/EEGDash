# Audit 3 — Path-traversal defence

**Auditor**: Phase 9 of the ingestion robustness programme (autonomous run, 2026-05-21)
**Scope**: Every code path that takes a path or filename from an external source (sidecar JSON, manifest, BIDS file) and resolves it on disk.
**Pattern**: Look for `Path(external_input)`, `open(external_input)`, or string concatenation into a filesystem path without containment checking.

## TL;DR

| Severity | Count |
|---|---:|
| P1 (potentially data-loss-adjacent) | 0 |
| P2 (defence-in-depth gap) | 2 |
| P3 (theoretical / hardening) | 1 |

The codebase has `validate_file_path` in `_parser_utils.py` which is
GOOD — but it isn't called consistently. Findings below.

## F1 — [P2] `parse_set_metadata` accepts arbitrary `set_path: str` without containment check

**Location**: `_set_parser.py:33`.

```python
def parse_set_metadata(set_path: Path | str) -> dict[str, Any] | None:
    set_path = Path(set_path)
    if not set_path.exists():
        return None
    ...
```

**Evidence**: `set_path` is passed through to `scipy.io.loadmat(str(set_path), ...)` directly. There's no check that `set_path` is contained within the dataset root.

In normal ingestion, the path comes from a manifest the pipeline itself built — so it's trusted. But the same function is callable from anywhere; a future contributor passing a user-supplied sidecar reference would have no warning that the path isn't sanitised.

**Severity**: P2. Path-traversal here doesn't directly enable arbitrary reads (scipy.io.loadmat *will* read whatever you point at, but in the ingestion context the process already has full filesystem access). The risk is that *if* a future change accepts a user-provided sidecar path, the lack of containment makes it trivially exploitable.

**Suggested fix**: Add a `dataset_root: Path` parameter and assert `set_path.resolve().is_relative_to(dataset_root.resolve())`. Apply the same pattern to `parse_vhdr_metadata`, `parse_snirf_metadata`, `parse_mef3_metadata`.

**Regression test**: `tests/test_path_traversal.py::test_parser_rejects_paths_outside_root` — call `parse_set_metadata(Path("../../etc/passwd"))` with `dataset_root=Path("/data/ds002893")`, assert `ValueError` (or whatever class the helper raises).

## F2 — [P2] `extract_vhdr_references` returns a sibling filename without validation

**Location**: `_vhdr_parser.py` (the `extract_vhdr_references` function).

**Evidence**: The function returns `datafile` and `markerfile` as STRINGS pulled directly from the `.vhdr` ConfigParser INI. A malicious `.vhdr` could contain `DataFile=../../../etc/shadow`. The current code joins the basename with `set_path.parent`, but if `datafile` itself contains `..`, the resolved path escapes the dataset.

**Severity**: P2. Same caveat as F1 — in the ingestion context, the `.vhdr` files come from trusted-by-source (OpenNeuro / NEMAR) datasets. Hardening it costs little; not hardening it is fragile.

**Suggested fix**:

```python
def _resolve_sibling(parent: Path, name: str) -> Path | None:
    """Return parent/name only if it stays inside parent (no traversal)."""
    resolved = (parent / name).resolve()
    if not resolved.is_relative_to(parent.resolve()):
        return None
    return resolved
```

Apply at every callsite that joins an external string to a directory.

**Regression test**: Feed a fixture `.vhdr` with `DataFile=../etc/passwd` to `extract_vhdr_references`, assert the result either returns `None` for that field or raises `ValueError`.

## F3 — [P3] `validate_file_path` exists but isn't used by every parser

**Location**: `_parser_utils.py` (function `validate_file_path`).

**Evidence**: The helper exists and is imported by `_mef3_parser.py`, `_snirf_parser.py`, `_vhdr_parser.py`. But `_set_parser.py` does NOT use it. Coverage check:

```
$ grep -l "validate_file_path" *.py
_mef3_parser.py
_snirf_parser.py
_vhdr_parser.py
```

`_set_parser.py` re-implements an inline existence check (`if not set_path.exists(): return None`) which is a subset of `validate_file_path`'s contract.

**Severity**: P3. Functional: both achieve the same goal. Hygiene: the helper exists to BE the canonical check; ignoring it in one file is the kind of drift that creates the F1/F2 attack surface gap later.

**Suggested fix**: Replace `_set_parser.py`'s inline check with `validate_file_path(set_path)`. Once that lands, audit every other path-input function for the same.

**Regression test**: Already covered by existing parser tests.

## What was NOT found

- No `os.system(user_input)` or `subprocess(shell=True, user_input)` patterns.
- No SQL injection vectors (the pipeline writes to MongoDB via Pydantic-validated bulk upserts).
- No `eval()` / `exec()` of user input.
- No `pickle.load(user_input)` of untrusted data.
- No XML external entity (XXE) paths — no XML parsing in the pipeline.

The audit covered ~30 functions across 6 files. F1/F2 are
defence-in-depth — fixing them adds value without changing observable
behaviour. F3 is hygiene.

The pipeline's blast radius is its MongoDB write, not the filesystem
read; that explains why path-traversal defence is light. The Phase 6
schema gate is the *real* defence against arbitrary records reaching
production.
