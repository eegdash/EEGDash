# Sprint 2026-05-22 — Perf follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining MEG montage perf gap (10 s → < 1 s on the sample) by short-circuiting unnecessary network calls, then trim Stage 2 + clean up two carry-forward drift-trap findings.

**Architecture:** Five orthogonal tasks ranked by ROI. **Task 1** is the user's insight: within a dataset, MEG records that share channel-count share device → cache montage by `(dataset_id, nchans)`. **Task 2** uses the existing `parse_annex_size` helper to skip the HEAD entirely for git-annex-managed datasets (OpenNeuro / NEMAR). **Task 3** flips `http2=True` on the pooled httpx client so the few remaining HEAD requests (Zenodo, Figshare) multiplex over one connection. **Task 4** replaces Stage 2's `Path.rglob` per-dataset with a single `os.scandir` pass. **Task 5** folds in the two non-blocking findings from the post-review sweep that we left for a follow-up touch.

**Tech Stack:** Python 3.11+, httpx (with `httpx[http2]` extra), mne_bids, pytest + respx, `_montage.py` FIF-header parser, git-annex symlink convention (`MD5E-s{size}--{hash}.ext`).

**Branch + constraints:**
- Branch `record-enumerator-merge`, 21 commits beyond `e38d26341`, last commit `d6dddb29a` (post-review fixes).
- Coverage floor at 61% in CI; must not regress and SHOULD ratchet in the same commit as new tests land.
- Snapshot tests (`test_digest_snapshot.py`, `test_pipeline_e2e*.py`) are byte-stable — any intentional update lives in the same commit + cites the reason. The Task 1 cache MUST stay byte-identical (same montage hash → same record).
- Never `--no-verify`. Never `Co-Authored-By`. Never robot attribution.

---

## Task ordering

```
Task 1  Within-dataset MEG montage cache              ← highest ROI; user's insight
Task 2  Annex-key size shortcut (skip HEAD)           ← uses existing helpers
Task 3  HTTP/2 multiplexing (httpx flag)              ← cheap defense in depth
Task 4  Stage 2 os.scandir manifest walker            ← Stage 2 hotspot (separate stage)
Task 5  Sweep follow-ups (payload drift + respx)     ← cleanup; non-blocking carry-forward
```

Tasks 1, 2, 3 each independently improve MEG digest time and compound. Task 4 is orthogonal (different stage). Task 5 is cleanup. Each task ends with a green test suite + clean commit.

---

## Task 1: Within-dataset MEG montage cache

**Files:**
- Modify: `scripts/ingestions/3_digest.py:2633-2700` (`_attach_montage_to_record`)
- Test: `scripts/ingestions/tests/test_montage_cache.py` (new)

**Why this matters:** The user's domain insight: *"MEG always has the same device inside a dataset if the number of channels is the same."* For an OpenNeuro MEG dataset with 50 runs sharing one Neuromag system, we currently extract the montage 50 times — fetching, parsing, hashing — when one extraction would have served all 50. Cache by `(dataset_id, nchans)`: first record establishes the layout, subsequent records reuse the cached `(montage_hash, montage_doc)`. On a 50-record MEG dataset the network calls drop by 49×.

**Correctness gate:** the cache MUST be a strict optimization — `montage_hash` and `montages[hash]` for any record served from cache must equal what `extract_layout` would have produced. Tests pin this.

**Snapshot tests are the gate.** The existing `tests/test_digest_snapshot.py` covers the BIDS path with a VHDR fixture (no MEG records); the new MEG fixture isn't in scope, so the cache is exercised end-to-end via the `_attach_montage_to_record` unit tests below.

- [ ] **Step 1.1: Read the current shape of `_attach_montage_to_record`**

```bash
cd /Users/bruaristimunha/Projects/eegdash
sed -n '2633,2700p' scripts/ingestions/3_digest.py
```

Confirm the signature: `(record, bids_file, dataset_dir, montages, dataset_id, digested_at) -> list[errors]`. The `record` dict is mutated; `montages` is the dataset-wide hash → doc map. The cache lives in a NEW per-call structure — it must NOT leak across datasets (different dataset_id → different cache).

- [ ] **Step 1.2: Write the failing tests**

Create `scripts/ingestions/tests/test_montage_cache.py`:

```python
"""Tests for the within-dataset MEG montage cache (Task 1 — Perf sprint).

The cache exploits the domain invariant: within a single MEG dataset,
records that share `nchans` share device → share layout. First record
extracts; subsequent records reuse the cached (hash, doc) without
re-running extract_layout or its network calls.

Cache MUST NOT leak across datasets — the cache key includes dataset_id.
Non-MEG records bypass the cache entirely (their layouts come from
electrodes.tsv, not the FIF header — sidecar reads are cheap and
not the bottleneck).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))


@pytest.fixture(scope="module")
def digest() -> ModuleType:
    """Load 3_digest.py via importlib (numeric filename)."""
    spec = importlib.util.spec_from_file_location(
        "digest_under_test", _INGEST_DIR / "3_digest.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _meg_record(nchans: int, name: str = "sub-01_run-01_meg.fif") -> dict:
    return {
        "datatype": "meg",
        "nchans": nchans,
        "bids_relpath": name,
    }


def test_first_meg_record_calls_extract_layout(
    digest: ModuleType, tmp_path: Path
) -> None:
    """First MEG record with a given nchans triggers extract_layout."""
    record = _meg_record(306)
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        return_value=("hash-306-A", {"system": "neuromag306"}),
    ) as mocked:
        errors = digest._attach_montage_to_record(
            record,
            tmp_path / "sub-01_meg.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "2026-05-22T00:00:00+00:00",
            montage_cache=cache,
        )

    assert errors == []
    assert mocked.call_count == 1
    assert record["montage_hash"] == "hash-306-A"
    assert "hash-306-A" in montages
    assert cache[("ds-meg-001", 306)] == (
        "hash-306-A",
        {
            "system": "neuromag306",
            "first_seen": "2026-05-22T00:00:00+00:00",
            "representative_dataset": "ds-meg-001",
        },
    )


def test_second_meg_record_same_nchans_reuses_cache(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Second record with same (dataset, nchans) MUST NOT call extract_layout."""
    record_a = _meg_record(306, "sub-01_run-01_meg.fif")
    record_b = _meg_record(306, "sub-01_run-02_meg.fif")
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        return_value=("hash-306-A", {"system": "neuromag306"}),
    ) as mocked:
        digest._attach_montage_to_record(
            record_a, tmp_path / "a.fif", tmp_path, montages,
            "ds-meg-001", "2026-05-22T00:00:00+00:00",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b, tmp_path / "b.fif", tmp_path, montages,
            "ds-meg-001", "2026-05-22T00:00:00+00:00",
            montage_cache=cache,
        )

    # extract_layout called ONCE despite two records.
    assert mocked.call_count == 1
    assert record_a["montage_hash"] == "hash-306-A"
    assert record_b["montage_hash"] == "hash-306-A"


def test_different_nchans_skips_cache(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Two records with different nchans → two extract_layout calls."""
    record_a = _meg_record(306)
    record_b = _meg_record(204)  # different device / channel count
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[
            ("hash-306-A", {"system": "neuromag306"}),
            ("hash-204-B", {"system": "ctf204"}),
        ],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a, tmp_path / "a.fif", tmp_path, montages,
            "ds-meg-001", "now", montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b, tmp_path / "b.fif", tmp_path, montages,
            "ds-meg-001", "now", montage_cache=cache,
        )

    assert mocked.call_count == 2
    assert record_a["montage_hash"] == "hash-306-A"
    assert record_b["montage_hash"] == "hash-204-B"


def test_cache_does_not_leak_across_datasets(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Same nchans, different dataset_id → cache miss (different cache keys)."""
    record_a = _meg_record(306)
    record_b = _meg_record(306)
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[
            ("hash-306-A", {"system": "neuromag306"}),
            ("hash-306-A", {"system": "neuromag306"}),  # different doc still hashes same
        ],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a, tmp_path / "a.fif", tmp_path, montages,
            "ds-meg-001", "now", montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b, tmp_path / "b.fif", tmp_path, montages,
            "ds-meg-002", "now", montage_cache=cache,
        )

    # extract_layout called once per dataset_id, even with same nchans.
    assert mocked.call_count == 2


def test_non_meg_record_bypasses_cache(
    digest: ModuleType, tmp_path: Path
) -> None:
    """EEG records still call extract_layout per-file — the cache only
    helps MEG where the device check is well-defined."""
    record_a = {"datatype": "eeg", "nchans": 64, "bids_relpath": "a.vhdr"}
    record_b = {"datatype": "eeg", "nchans": 64, "bids_relpath": "b.vhdr"}
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[
            ("hash-eeg-a", {"layout": "ten-twenty"}),
            ("hash-eeg-b", {"layout": "ten-twenty"}),
        ],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a, tmp_path / "a.vhdr", tmp_path, montages,
            "ds-eeg-001", "now", montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b, tmp_path / "b.vhdr", tmp_path, montages,
            "ds-eeg-001", "now", montage_cache=cache,
        )

    # Per-file extraction — cache MUST NOT have hijacked the EEG path.
    assert mocked.call_count == 2
    assert cache == {}  # no MEG entries; EEG bypasses


def test_missing_nchans_skips_cache(
    digest: ModuleType, tmp_path: Path
) -> None:
    """A MEG record with no nchans (missing metadata) must NOT be cached
    — without a channel count there's no safe key."""
    record = {"datatype": "meg", "bids_relpath": "broken.fif"}  # no nchans
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        return_value=("hash-x", {"layout": "x"}),
    ) as mocked:
        digest._attach_montage_to_record(
            record, tmp_path / "broken.fif", tmp_path, montages,
            "ds-x", "now", montage_cache=cache,
        )

    assert mocked.call_count == 1
    assert cache == {}  # no key without nchans
    assert record["montage_hash"] == "hash-x"


def test_extract_layout_returning_none_is_not_cached(
    digest: ModuleType, tmp_path: Path
) -> None:
    """If extract_layout returns None (no montage available), don't
    cache the absence — next record gets another chance."""
    record_a = _meg_record(306)
    record_b = _meg_record(306)
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[None, ("hash-306-A", {"system": "neuromag306"})],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a, tmp_path / "a.fif", tmp_path, montages,
            "ds-meg-001", "now", montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b, tmp_path / "b.fif", tmp_path, montages,
            "ds-meg-001", "now", montage_cache=cache,
        )

    # Second call still went through extract_layout because the first
    # returned None (cache only stores positive results).
    assert mocked.call_count == 2
    assert record_a["montage_hash"] is None
    assert record_b["montage_hash"] == "hash-306-A"
```

- [ ] **Step 1.3: Run — expect TypeError on the `montage_cache` kwarg**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_montage_cache.py -v --tb=short 2>&1 | tail -20
```

Expected: every test fails with `TypeError: _attach_montage_to_record() got an unexpected keyword argument 'montage_cache'`.

- [ ] **Step 1.4: Add the `montage_cache` parameter + cache logic**

In `scripts/ingestions/3_digest.py:2633`, change the function signature and body. Read the current implementation first:

```bash
sed -n '2633,2705p' scripts/ingestions/3_digest.py
```

Then replace the function definition:

```python
def _attach_montage_to_record(
    record: dict[str, Any],
    bids_file: Any,
    dataset_dir: Path,
    montages: dict[str, dict[str, Any]],
    dataset_id: str,
    digested_at: str,
    montage_cache: dict[tuple[str, int], tuple[str, dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Run ``extract_layout`` for ``bids_file``; stamp the result on ``record``.

    Side effects (intentional): mutates ``record`` (sets
    ``montage_hash``) and ``montages`` (adds the layout doc on first
    sighting of a hash). Returns a list of per-file errors (empty on
    the happy path; one entry if ``extract_layout`` raises).

    Within-dataset cache (perf-2026-05-22)
    --------------------------------------
    For MEG records, the device is fixed per-dataset by physical
    install — Neuromag-306 or CTF-275 etc. Two records in the same
    dataset with the same ``nchans`` therefore share device and
    therefore share montage. The optional ``montage_cache`` argument
    (caller-owned dict keyed by ``(dataset_id, nchans)``) lets the
    second-through-Nth record reuse the first record's extraction
    result without re-fetching the FIF metadata or re-hashing the
    layout. Non-MEG records bypass the cache (their layouts come from
    sidecar TSVs, which are cheap to read). Records without
    ``nchans`` are not cached (no safe key). ``extract_layout``
    returning None is not cached either — next record still tries.

    Phase 8 Stage-3 follow-up: extracted from the inline body of
    :func:`_enumerate_via_bids`'s for-loop to drop it under 100 LOC.
    """
    record_datatype = (record.get("datatype") or "").lower()
    errors: list[dict[str, Any]] = []

    # Within-dataset MEG cache lookup — only when the caller passed a
    # cache, the record is MEG, and nchans is set.
    cache_key: tuple[str, int] | None = None
    record_nchans = record.get("nchans")
    if (
        montage_cache is not None
        and record_datatype == "meg"
        and isinstance(record_nchans, int)
        and record_nchans > 0
    ):
        cache_key = (dataset_id, record_nchans)
        cached = montage_cache.get(cache_key)
        if cached is not None:
            cached_hash, cached_doc = cached
            record["montage_hash"] = cached_hash
            # Mirror the first-seen / representative-dataset stamping
            # so the per-dataset montages map carries the doc for this
            # hash (uploader uses $setOnInsert; safe to re-set).
            if cached_hash not in montages:
                montages[cached_hash] = cached_doc
            return errors

    try:
        layout_result = extract_layout(
            Path(str(bids_file)), dataset_dir, datatype=record_datatype
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # extract_layout (electrode-coords pipeline) can fail on missing
        # electrodes.tsv / coordsystem.json, malformed numeric fields,
        # or unsupported montage variants. Best-effort; we still emit
        # the Record without a montage hash.
        record["montage_hash"] = None
        errors.append(
            {"file": str(bids_file), "error": f"layout extraction failed: {exc}"}
        )
        return errors

    if layout_result is None:
        record["montage_hash"] = None
        return errors

    h, doc = layout_result
    record["montage_hash"] = h
    if h not in montages:
        # First time we see this hash in this dataset — stamp the
        # provenance fields that live outside the hashed content. The
        # API upsert layer uses $setOnInsert so these don't get
        # overwritten when the same hash appears in a later dataset.
        doc["first_seen"] = digested_at
        doc["representative_dataset"] = dataset_id
        montages[h] = doc

    # Memoize positive results for the next MEG record with this
    # (dataset, nchans). Use the doc we just stamped (with first_seen +
    # representative_dataset) so cache reuse produces identical state.
    if cache_key is not None:
        montage_cache[cache_key] = (h, montages[h])

    return errors
```

- [ ] **Step 1.5: Run the cache tests — expect 7 passes**

```bash
pytest tests/test_montage_cache.py -v --tb=short
```

Expected: 7 passed. The cache key, MEG/EEG dispatch, missing-nchans skip, None-not-cached, and cross-dataset isolation are all exercised.

- [ ] **Step 1.6: Wire the cache into `_enumerate_via_bids`**

Find the caller:

```bash
grep -n '_attach_montage_to_record' scripts/ingestions/3_digest.py
```

There's one production call site inside `_enumerate_via_bids` (the BIDS-walk for-loop). It already passes `record, bids_file, dataset_dir, montages, dataset_id, digested_at`. We need to declare a `montage_cache: dict = {}` ABOVE the loop (per-dataset scope) and pass it into the call.

Read the current call site, then edit:

```bash
sed -n '<find the call>p' scripts/ingestions/3_digest.py
```

Then patch the call site to add `montage_cache=meg_montage_cache` and declare the cache once before the loop:

```python
# Within-dataset MEG montage cache (perf-2026-05-22). Lives for the
# duration of this dataset's enumeration; first MEG record with a
# given nchans extracts the device-defined layout, subsequent
# same-nchans records reuse the cached (hash, doc).
meg_montage_cache: dict[tuple[str, int], tuple[str, dict[str, Any]]] = {}

# ... existing for-loop over BIDS records ...
errors.extend(
    _attach_montage_to_record(
        record,
        bids_file,
        dataset_dir,
        montages,
        dataset_id,
        digested_at,
        montage_cache=meg_montage_cache,
    )
)
```

- [ ] **Step 1.7: Snapshot tests still pass — the gate**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_digest_snapshot.py tests/test_pipeline_e2e.py tests/test_pipeline_e2e_mef3.py -v --tb=short 2>&1 | tail -15
```

Expected: 21/21 byte-identical. The VHDR fixture in the snapshots doesn't exercise the MEG path; if any of these fail, your edit drifted something unrelated — read the diff and fix.

- [ ] **Step 1.8: Full PR-fast suite**

```bash
pytest -q -m "not network and not slow and not integration" --tb=short 2>&1 | tail -5
```

Expected: 834 → 841 (+7 new cache tests). 0 failures.

- [ ] **Step 1.9: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/3_digest.py \
        scripts/ingestions/tests/test_montage_cache.py
git commit -m "perf(digest): within-dataset MEG montage cache by (dataset, nchans)

Domain insight: within a single MEG dataset, records that share
nchans share device → share layout. First MEG record with a given
nchans extracts the device-defined montage; subsequent same-nchans
records in the same dataset reuse the cached (hash, doc) without
re-fetching the FIF metadata or re-running extract_layout.

Eliminates ~50× re-extraction on typical OpenNeuro MEG datasets
(one Neuromag system, 50+ runs). Stacks with the connection pool
from 54e2ceab3 — combined effect lands the MEG montage path near
the noise floor.

- Cache key: (dataset_id, nchans). Per-dataset scope (lives for the
  duration of _enumerate_via_bids).
- Non-MEG records bypass the cache entirely (sidecar reads are
  cheap and the device-per-dataset invariant doesn't apply to EEG).
- Records without nchans don't get cached (no safe key).
- extract_layout returning None is not cached (next record retries).
- 7 new tests in test_montage_cache.py pin all 6 above invariants
  + the cross-dataset-isolation case.
- 21/21 snapshot + e2e gate byte-identical (VHDR fixture doesn't
  hit the MEG path; cache behaviour is unit-tested directly)."
```

---

## Task 2: Annex-key size shortcut — skip HEAD entirely

**Files:**
- Modify: `scripts/ingestions/_montage.py:798-860` (`_fetch_fif_metadata_via_directory`)
- Test: `scripts/ingestions/tests/test_montage_annex_key_shortcut.py` (new)

**Why this matters:** `_fetch_fif_metadata_via_directory` opens with `total = head_content_length(url, timeout=30.0)` — a network round-trip just to learn the file size. But OpenNeuro/NEMAR datasets are git-annex-managed; the file size is encoded in the annex symlink target (`MD5E-s{size}--{hash}.fif`). The helper `parse_annex_size` already exists in `_file_utils.py:725` and `get_annex_file_size` in `_file_utils.py:843`. Reading the symlink first, then falling back to HEAD only for non-annex datasets (Zenodo, Figshare), removes 100% of the HEAD round-trips on the production hot path.

- [ ] **Step 2.1: Read the existing annex helpers**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
sed -n '720,770p' _file_utils.py
sed -n '840,880p' _file_utils.py
```

Confirm the function signatures: `parse_annex_size(text: str) -> int | None` and `get_annex_file_size(path: Path) -> int`. The latter returns 0 for non-annex paths, so the caller must distinguish "0 = no annex info" from a successful 0-byte (impossible for FIF).

- [ ] **Step 2.2: Write the failing test**

Create `scripts/ingestions/tests/test_montage_annex_key_shortcut.py`:

```python
"""Tests for the annex-key size shortcut (Task 2 — Perf sprint).

OpenNeuro / NEMAR FIF files are git-annex-managed; the file size is
encoded in the symlink target (MD5E-s{size}--{hash}.fif). Reading
the symlink first lets us skip a network HEAD round-trip for every
MEG record. Non-annex datasets (Zenodo, Figshare with raw S3 URLs)
fall back to head_content_length.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from _montage import _resolve_fif_total_size


def test_returns_annex_size_when_symlink_present(tmp_path: Path) -> None:
    """Broken git-annex symlink → size parsed from symlink target."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    # Annex key format: MD5E-s{size}--{hash}.{ext}
    target = "../.git/annex/objects/aa/bb/MD5E-s4194304--abc123def.fif/MD5E-s4194304--abc123def.fif"
    fif.symlink_to(target)
    assert not fif.exists()  # broken — annex content not fetched

    with patch("_montage.head_content_length") as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 4_194_304
    mock_head.assert_not_called()  # No HEAD round-trip


def test_falls_back_to_head_when_no_annex_symlink(tmp_path: Path) -> None:
    """Plain file (no annex) → HEAD round-trip."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.write_bytes(b"\x00" * 1024)  # 1 KB regular file

    with patch("_montage.head_content_length", return_value=8_192) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    # Annex size returns 0 for non-annex, falls through to HEAD.
    assert size == 8_192
    mock_head.assert_called_once()


def test_falls_back_to_head_on_malformed_annex_key(tmp_path: Path) -> None:
    """Symlink target doesn't match MD5E-s{size}-- pattern → HEAD fallback."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.symlink_to("not-an-annex-key.fif")

    with patch("_montage.head_content_length", return_value=2_048) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 2_048
    mock_head.assert_called_once()


def test_returns_none_when_neither_annex_nor_head_succeeds(
    tmp_path: Path,
) -> None:
    """All paths exhausted → None (caller treats as transient failure)."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.write_bytes(b"")

    with patch("_montage.head_content_length", return_value=None):
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size is None


def test_zero_byte_annex_key_falls_back_to_head(tmp_path: Path) -> None:
    """An annex key reporting size=0 is nonsensical for FIF — fall back."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.symlink_to(
        "../.git/annex/objects/aa/bb/MD5E-s0--abc.fif/MD5E-s0--abc.fif"
    )

    with patch("_montage.head_content_length", return_value=5_000) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 5_000
    mock_head.assert_called_once()
```

- [ ] **Step 2.3: Run — expect ImportError on `_resolve_fif_total_size`**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_montage_annex_key_shortcut.py -v --tb=short 2>&1 | tail -10
```

Expected: `ImportError: cannot import name '_resolve_fif_total_size'`.

- [ ] **Step 2.4: Add `_resolve_fif_total_size` to `_montage.py`**

In `scripts/ingestions/_montage.py` just ABOVE `_fetch_fif_metadata_via_directory` (around line 795), add:

```python
def _resolve_fif_total_size(data_file: Path, url: str) -> int | None:
    """Discover the FIF total size, preferring the annex symlink over a HEAD.

    OpenNeuro / NEMAR FIF files are git-annex-managed; the size is
    encoded in the symlink target (``MD5E-s{size}--{hash}.fif``).
    Reading the symlink avoids the HTTPS HEAD round-trip that
    dominated the MEG digest profile before pooling landed.

    Falls back to :func:`head_content_length` when:
      - the file isn't an annex symlink (Zenodo / Figshare raw URLs);
      - the annex key doesn't follow the MD5E/SHA256E size convention;
      - the parsed size is zero (impossible for a real FIF — treat
        as malformed key and probe the wire).

    Returns ``None`` if both paths fail; caller treats that as a
    transient network failure.
    """
    from _file_utils import get_annex_file_size

    annex_size = get_annex_file_size(data_file)
    if annex_size > 0:
        return annex_size
    return head_content_length(url, timeout=30.0)
```

Note: `head_content_length` is imported lazily inside `_fetch_fif_metadata_via_directory` today; for the new helper, do the same lazy import in-function (matches the file's existing pattern + keeps top-level imports clean):

```python
def _resolve_fif_total_size(data_file: Path, url: str) -> int | None:
    """..."""
    from _file_utils import get_annex_file_size
    from _parser_utils import head_content_length

    annex_size = get_annex_file_size(data_file)
    if annex_size > 0:
        return annex_size
    return head_content_length(url, timeout=30.0)
```

For the test's `patch("_montage.head_content_length", ...)` to work, the helper must reference `head_content_length` via the `_montage` module namespace. Bind it at module level via a re-import:

At the top of `_montage.py` (find the existing imports), add (if not already present):

```python
from _parser_utils import head_content_length  # noqa: F401 — patch target
```

If `head_content_length` is already imported elsewhere in `_montage.py`, just confirm the symbol resolves through the module namespace (it will — Python module attribute lookup honours the import). Then change the helper to use the module-level reference:

```python
def _resolve_fif_total_size(data_file: Path, url: str) -> int | None:
    """..."""
    from _file_utils import get_annex_file_size

    annex_size = get_annex_file_size(data_file)
    if annex_size > 0:
        return annex_size
    return head_content_length(url, timeout=30.0)  # module-level binding
```

- [ ] **Step 2.5: Run the new tests — expect PASS**

```bash
pytest tests/test_montage_annex_key_shortcut.py -v --tb=short
```

Expected: 5 passed.

- [ ] **Step 2.6: Wire `_resolve_fif_total_size` into `_fetch_fif_metadata_via_directory`**

In `_montage.py:798`, find the line:

```python
total = head_content_length(url, timeout=30.0)
```

Replace with:

```python
total = _resolve_fif_total_size(data_file, url)
```

- [ ] **Step 2.7: Snapshot tests + full PR-fast suite — the gate**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_digest_snapshot.py tests/test_pipeline_e2e.py tests/test_pipeline_e2e_mef3.py -v --tb=short 2>&1 | tail -10
pytest -q -m "not network and not slow and not integration" --tb=short 2>&1 | tail -5
```

Expected: 21/21 snapshot+e2e byte-identical; 841 → 846 PR-fast (+5 new tests). 0 failures.

- [ ] **Step 2.8: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/_montage.py \
        scripts/ingestions/tests/test_montage_annex_key_shortcut.py
git commit -m "perf(montage): skip HEAD via git-annex key size shortcut

_fetch_fif_metadata_via_directory was calling head_content_length on
every MEG FIF just to learn the file size. But OpenNeuro / NEMAR
FIF files are git-annex symlinks whose target encodes the size
(MD5E-s{size}--{hash}.fif). _file_utils.get_annex_file_size already
parses this. New _resolve_fif_total_size helper tries annex first,
falls back to HEAD for non-annex paths (Zenodo / Figshare).

Removes 100% of HEAD round-trips on the production hot path; the
pool from 54e2ceab3 keeps the remaining few HEADs (for raw S3 URLs
on non-annex datasets) cheap. Combined with the per-(dataset,
nchans) cache from the prior commit, the MEG montage profile
collapses near the noise floor.

- New _resolve_fif_total_size in _montage.py — annex-first, HEAD
  fallback, returns None on transient network failure.
- _fetch_fif_metadata_via_directory rewires through the helper.
- 5 new tests pinning annex-present, no-annex-fallback, malformed-
  key-fallback, zero-byte-key-fallback, both-failed-returns-None.
- 21/21 snapshot + e2e gate byte-identical."
```

---

## Task 3: HTTP/2 multiplexing for the pooled client

**Files:**
- Modify: `scripts/ingestions/_parser_utils.py:53-74` (`_http_client`)
- Modify: `scripts/ingestions/pyproject.toml` (declare `httpx[http2]`)
- Test: `scripts/ingestions/tests/test_parser_utils_network.py` (add 1 test)

**Why this matters:** With Task 2 we removed most HEAD round-trips for annex datasets. For Zenodo / Figshare / SciDB, HEAD requests remain. Flipping `http2=True` on the shared client lets these HEADs share a single TCP/TLS connection via stream multiplexing — adds defense in depth without changing call sites. `h2` is a small pure-Python dependency (~40 KB wheel).

- [ ] **Step 3.1: Confirm `httpx[http2]` is available**

```bash
cd /Users/bruaristimunha/Projects/eegdash
python3 -c "import h2; print(h2.__version__)"
```

If h2 isn't installed: `pip install h2`. Note the version for the dependency declaration in Step 3.5.

- [ ] **Step 3.2: Write the failing test**

Add to `scripts/ingestions/tests/test_parser_utils_network.py` (append after the existing tests):

```python
def test_http_client_enables_http2():
    """Shared client uses HTTP/2 so multiple HEADs over the same host
    multiplex on a single connection (Task 3 — perf sprint).
    """
    client = _parser_utils._http_client()
    # httpx exposes the negotiated capability via the transport pool.
    # We can't actually open a connection here (respx mocks the
    # transport); inspect the constructor flag instead.
    # The flag lives on the pool config — `http2` was passed at
    # construction.
    # httpx.Client doesn't expose `_http2` publicly; check via the
    # underlying transport's connection pool config.
    transport = client._transport
    # AsyncHTTPTransport / HTTPTransport both keep `_pool` with
    # `http2` attribute. Use getattr-with-default so the test fails
    # with a clear message instead of AttributeError.
    pool = getattr(transport, "_pool", None)
    assert pool is not None, (
        "client._transport._pool not found — httpx internal layout may "
        "have changed; update the test"
    )
    assert getattr(pool, "_http2", False) is True, (
        "Shared httpx.Client must be constructed with http2=True so "
        "concurrent HEAD requests multiplex over one connection"
    )
```

- [ ] **Step 3.3: Run — expect FAIL**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_parser_utils_network.py::test_http_client_enables_http2 -v --tb=short
```

Expected: assertion failure on `_http2 is True` (current value: False).

- [ ] **Step 3.4: Flip the flag**

In `scripts/ingestions/_parser_utils.py:64-72`, change the `httpx.Client(...)` constructor:

```python
_HTTP_CLIENT = httpx.Client(
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=40,
        keepalive_expiry=60.0,
    ),
    follow_redirects=True,
    http2=True,  # multiplex concurrent HEADs over one connection
)
```

- [ ] **Step 3.5: Declare the `h2` dependency**

In `scripts/ingestions/pyproject.toml`, find the `dependencies = [...]` block (or `[project.optional-dependencies]` if dev-only) and ensure `httpx[http2]>=0.27` is present (or update the existing `httpx` pin). Then verify:

```bash
cd /Users/bruaristimunha/Projects/eegdash
pip install -e "scripts/ingestions[dev]" 2>&1 | tail -5
```

- [ ] **Step 3.6: Run the test — expect PASS**

```bash
pytest tests/test_parser_utils_network.py::test_http_client_enables_http2 -v --tb=short
```

Expected: PASS.

- [ ] **Step 3.7: Full suite + snapshot — the gate**

```bash
pytest tests/test_parser_utils_network.py -v --tb=short 2>&1 | tail -10
pytest tests/test_digest_snapshot.py tests/test_pipeline_e2e.py tests/test_pipeline_e2e_mef3.py -v --tb=short 2>&1 | tail -10
pytest -q -m "not network and not slow and not integration" --tb=short 2>&1 | tail -5
```

Expected: 19/19 parser_utils_network (+1 new); 21/21 snapshot+e2e byte-identical; 846 → 847 PR-fast.

- [ ] **Step 3.8: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/_parser_utils.py \
        scripts/ingestions/tests/test_parser_utils_network.py \
        scripts/ingestions/pyproject.toml
git commit -m "perf(parser_utils): enable HTTP/2 multiplexing on the shared client

After the annex-key shortcut (Task 2), HEAD requests are rare and
hit only Zenodo / Figshare / SciDB. Flipping http2=True lets the
remaining concurrent HEADs multiplex over a single TCP/TLS
connection — pure win for the non-annex paths, no-op for the rest.

- httpx.Client(http2=True) plus the h2 dependency.
- One test pinning the flag (introspects client._transport._pool).
- 21/21 snapshot + e2e gate byte-identical."
```

---

## Task 4: Stage 2 — `os.scandir` manifest walker

**Files:**
- Modify: `scripts/ingestions/_file_utils.py` (the manifest builder; `build_manifest` or similar — find via grep)
- Test: `scripts/ingestions/tests/test_file_utils.py` (extend; or new `test_manifest_scandir.py`)

**Why this matters:** Stage 2 profile shows ~2.6 M `pathlib.is_file` + 1.5 M `pathlib.rglob` + 2.8 M `pathlib.lstat` calls across 566 datasets — ~5 000 stat syscalls per dataset. A single `os.walk` / `os.scandir` pass per dataset that classifies entries in one stat batch is roughly 3-5× cheaper than `Path.rglob`.

- [ ] **Step 4.1: Find the manifest builder**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
grep -n '^def build_manifest\|^def list_local_files\|def.*manifest' _file_utils.py | head
```

Read whichever function `2_clone.py` calls. Likely candidates: `build_manifest`, `list_local_files`, `_walk_dataset`. Note the lines that call `Path.rglob`, `path.is_symlink`, `path.is_file`.

- [ ] **Step 4.2: Write a benchmark test before changing anything**

Create `scripts/ingestions/tests/test_manifest_walk_perf.py`:

```python
"""Benchmark: manifest walker speed.

Pins the expected throughput (entries/sec) of the manifest walker so
the os.scandir refactor can prove a measurable speedup AND the
post-refactor performance has a floor that catches regressions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Build a synthetic BIDS-like tree with 2000 files in 100 subjects."""
    root = tmp_path / "ds-synth"
    for s in range(100):
        sub = root / f"sub-{s:03d}" / "eeg"
        sub.mkdir(parents=True)
        for r in range(20):
            (sub / f"sub-{s:03d}_run-{r:02d}_eeg.edf").write_bytes(b"\x00" * 128)
            (sub / f"sub-{s:03d}_run-{r:02d}_eeg.json").write_text("{}")
    (root / "dataset_description.json").write_text("{}")
    (root / "participants.tsv").write_text("participant_id\n")
    return root


def test_manifest_walk_completes_under_floor(synthetic_dataset, benchmark):
    """Walking a 4000-file synthetic dataset should finish under 250 ms
    on a modern Mac. If this regresses past 500 ms, something walked
    each file twice."""
    from _file_utils import build_manifest  # adjust if the name differs

    result = benchmark(build_manifest, synthetic_dataset)
    # Either an integer file count or a dict with a 'files' / 'total_files'
    # key — adjust this assertion to the actual return shape.
    if isinstance(result, dict):
        assert result.get("total_files", 0) >= 4000
    else:
        # If the walker returns a list, len(list) == file count
        assert len(result) >= 4000

    assert benchmark.stats["mean"] < 0.5, (
        f"Manifest walk took {benchmark.stats['mean']*1000:.0f} ms on a "
        f"4000-file synthetic dataset (floor: 500 ms). Likely walked "
        f"each file twice — investigate."
    )
```

Run it now to capture the BEFORE number:

```bash
pytest tests/test_manifest_walk_perf.py -v --benchmark-only --benchmark-min-rounds=3 2>&1 | tail -20
```

Note the mean time in the output. Expected: somewhere in the 300-800 ms range based on the Stage 2 profile.

- [ ] **Step 4.3: Refactor the manifest builder to `os.scandir`**

Replace the `Path.rglob` / `path.is_*` / `lstat` pattern with a single recursive `os.scandir` walk that returns the same shape. The general transformation:

```python
# Before (illustrative):
def build_manifest(dataset_dir: Path) -> dict:
    files = []
    for p in dataset_dir.rglob("*"):
        if p.is_symlink():
            size = 0
            ...
        elif p.is_file():
            size = p.lstat().st_size
            ...
        files.append({"path": str(p.relative_to(dataset_dir)), "size": size})
    return {"files": files, "total_files": len(files)}

# After (single-pass scandir):
def build_manifest(dataset_dir: Path) -> dict:
    import os

    files = []
    stack: list[Path] = [dataset_dir]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    # entry.is_symlink() / is_file() / is_dir() use
                    # cached dirent flags — no extra stat. The fall-
                    # back stat goes via entry.stat() once per entry.
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) or entry.is_symlink():
                        size = (
                            0
                            if entry.is_symlink()
                            else entry.stat(follow_symlinks=False).st_size
                        )
                        rel = Path(entry.path).relative_to(dataset_dir)
                        files.append({"path": str(rel), "size": size})
        except (PermissionError, OSError):
            # Skip inaccessible directories — matches the prior best-
            # effort behaviour of Path.rglob.
            continue
    return {"files": files, "total_files": len(files)}
```

**Important**: read the actual current implementation BEFORE writing the new one and preserve every field of the returned dict and every classification rule. The above is a sketch.

- [ ] **Step 4.4: Re-run the benchmark — expect ≥30% improvement**

```bash
pytest tests/test_manifest_walk_perf.py -v --benchmark-only --benchmark-min-rounds=3 2>&1 | tail -20
```

If the new mean isn't materially better (say <30% improvement), the refactor missed something — the most common culprit is calling `entry.stat()` multiple times per entry. Each call costs a syscall.

- [ ] **Step 4.5: Full snapshot + suite — the gate**

```bash
pytest tests/test_digest_snapshot.py tests/test_pipeline_e2e.py tests/test_pipeline_e2e_mef3.py -v --tb=short 2>&1 | tail -10
pytest -q -m "not network and not slow and not integration" --tb=short 2>&1 | tail -5
```

Expected: 21/21 snapshot+e2e byte-identical; 847 → 848 PR-fast (+1 new benchmark test).

- [ ] **Step 4.6: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/_file_utils.py \
        scripts/ingestions/tests/test_manifest_walk_perf.py
git commit -m "perf(file_utils): rewrite manifest walk via os.scandir (3-5x faster)

Stage 2 profile showed ~5000 stat-like syscalls per dataset from
Path.rglob + per-path .is_symlink() / .is_file() / .lstat(). One
recursive os.scandir pass classifies each entry from the cached
dirent flags + at most one stat call per file.

- Single-pass walk; one stat per file via entry.stat(follow_symlinks=False).
- Symlinks treated as size=0 (matches the prior best-effort behaviour).
- Permission/OS errors skip the offending directory, like before.
- New synthetic-dataset benchmark in test_manifest_walk_perf.py
  (4000 files / 100 subjects) with a 500 ms ceiling — regression
  guard for future walkers."
```

---

## Task 5: Sweep follow-ups (drift trap + respx)

**Files:**
- Modify: `scripts/ingestions/3_digest.py:2879-2901` (`_emit_dataset_finished`)
- Modify: `scripts/ingestions/tests/test_parser_utils_network.py` (2 tests)

**Why this matters:** Two non-blocking items from the post-review sweep that we left for a follow-up touch. Now's the touch.

### 5a — Drift trap in `_emit_dataset_finished` payload

The function hard-codes 6 summary fields. When `write_dataset_outputs` adds a new summary field (`total_files` was the last addition), the helper silently fails to forward it. Lock the field list via a constant + tests so future additions can't slip past:

- [ ] **Step 5.1: Write the failing test**

Append to `scripts/ingestions/tests/test_digest_helpers.py`:

```python
def test_emit_dataset_finished_payload_includes_total_files(
    digest: ModuleType,
) -> None:
    """The summary now carries total_files (for the manifest path).
    The event payload MUST forward it so dashboards can see the raw
    input count. Pins the drift trap from the post-review sweep.
    """
    import digest_telemetry

    recorder = _RecordingEmitter()
    saved = digest_telemetry._EMITTER
    digest_telemetry._EMITTER = recorder
    try:
        digest._emit_dataset_finished(
            "ds-Z",
            {
                "status": "success",
                "record_count": 5,
                "error_count": 0,
                "digest_method": "manifest_only",
                "integrity_issues_count": 0,
                "montage_count": 0,
                "total_files": 5,  # NEW summary field — must propagate
            },
        )
    finally:
        digest_telemetry._EMITTER = saved

    assert len(recorder.events) == 1
    assert recorder.events[0].payload.get("total_files") == 5
```

- [ ] **Step 5.2: Run — expect FAIL** (no `total_files` in payload yet).

- [ ] **Step 5.3: Add `total_files` to the payload**

In `scripts/ingestions/3_digest.py:_emit_dataset_finished`, add the line:

```python
            payload={
                "status": summary.get("status"),
                "record_count": summary.get("record_count"),
                "error_count": summary.get("error_count"),
                "digest_method": summary.get("digest_method"),
                "integrity_issues_count": summary.get("integrity_issues_count"),
                "montage_count": summary.get("montage_count"),
                "total_files": summary.get("total_files"),  # NEW: manifest-path input count
            },
```

### 5b — Two tests missing `@respx.mock`

In `scripts/ingestions/tests/test_parser_utils_network.py`, the two tests at the bottom (`test_http_client_is_a_singleton`, `test_reset_http_client_for_testing_actually_rebuilds`) construct real `httpx.Client` instances. On a CI machine with `HTTPS_PROXY` set they'd attach to the real proxy transport.

- [ ] **Step 5.4: Add `@respx.mock` to both**

Decorate both function definitions:

```python
@respx.mock
def test_http_client_is_a_singleton():
    ...

@respx.mock
def test_reset_http_client_for_testing_actually_rebuilds():
    ...
```

`respx.mock` patches the transport even when no `.get(...)` / `.head(...)` calls are mocked — the client construction is then guaranteed to use respx's transport, not a proxy.

- [ ] **Step 5.5: Run all of Task 5**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_digest_helpers.py tests/test_parser_utils_network.py -v --tb=short 2>&1 | tail -15
pytest -q -m "not network and not slow and not integration" --tb=short 2>&1 | tail -5
```

Expected: all pass; total ≥849 (was 848 after Task 4, +1 for the payload-drift test).

- [ ] **Step 5.6: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/3_digest.py \
        scripts/ingestions/tests/test_digest_helpers.py \
        scripts/ingestions/tests/test_parser_utils_network.py
git commit -m "fix(ingestions): close sweep follow-ups from /code-review

Two non-blocking items the post-perf review surfaced and we left
for a follow-up touch:

1. _emit_dataset_finished now forwards total_files (added to the
   summary in the Stage-3D collapse). A new test pins the field so
   the next summary addition fails loudly instead of silently
   skipping the telemetry stream.

2. test_http_client_is_a_singleton + test_reset_http_client_for_
   testing_actually_rebuilds gain @respx.mock decorators so they
   construct httpx.Client through respx's transport rather than
   any real network / proxy. On CI hosts with HTTPS_PROXY set the
   bare clients would otherwise pin against the proxy — surfaces
   as flaky tests."
```

---

## Self-review checklist

- [ ] **Spec coverage**: Tasks 1-5 cover the user's MEG-device-per-dataset insight + the 4 remaining candidates from the post-pooling profile review.

- [ ] **Placeholder scan**: every step shows the exact code, file path, command, and expected output. The exception: Task 4's "read the actual current implementation BEFORE writing the new one" — the actual code path is too long to inline. The skeleton + the safety net (snapshot tests) keeps the engineer on the rails.

- [ ] **Type consistency**:
  - `montage_cache: dict[tuple[str, int], tuple[str, dict[str, Any]]]` consistent across Task 1's signature, tests, and caller.
  - `_resolve_fif_total_size(data_file: Path, url: str) -> int | None` consistent across Task 2's signature and tests.
  - `httpx.Client(http2=True)` flag identical in Task 3's source change and test introspection.

- [ ] **Constraint conformance**:
  - No `--no-verify` in any commit step.
  - No `Co-Authored-By`, no robot attribution.
  - Snapshot tests run after every code change (steps 1.7, 2.7, 3.7, 4.5).
  - Coverage gate checked at the end of each task (PR-fast suite invocation includes coverage gate via the `pytest-cov` plugin's default).

---

## Execution handoff

Plan saved to `scripts/ingestions/ROBUSTNESS/SPRINT-2026-05-22-PERF.md`.

### Two execution options

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks. Best for Tasks 1, 2, 4 (independent code work). Tasks 3 and 5 are small enough to run inline if preferred.

**2. Inline Execution** — execute tasks here with `superpowers:executing-plans`, batch with checkpoints. Best if you want every commit reviewed in-session.

Tasks 1, 2, 3 each compound the perf win; running them in order maximises the verifiable improvement at each step. Task 4 is independent (different stage). Task 5 is cleanup.
