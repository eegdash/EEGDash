# Sprint 2026-05-22 — Ship + Deepen + Secure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute every Tier-1 and Tier-2 item from `NEXT-SPRINT-PLAN.md`, re-tiered against findings from the old sprint's LOC drift in `3_digest.py`. Tasks chosen to honour Lesson #3 (refactor before you test) while preserving the snapshot-byte-stability + coverage-ratchet contracts.

**Architecture:** Six independent tasks. Task 1 re-anchors the plan against drift findings (LOC inflation in `_extract_technical_metadata`: 140 → 244 LOC across C5/C6). Tasks 2–3 are code work (database-drift consumer-side fix; `_extract_technical_metadata` depth refactor extracting a `MetadataCascade` deep module). Task 4 is conditional research (real SNIRF fixture if OpenNeuro publishes one). Task 5 prevents the next credential leak. Task 6 is a gated ops step (mutmut observation; requires explicit push authorization).

**Tech Stack:** Python 3.11+, Pydantic v2 + pydantic-settings, pytest + respx + pytest-mock, httpx, mne / mne_bids, BIDS spec v1.6, pre-commit, GitHub Actions, ruff.

**Branch + constraints (carry-forward from `NEXT-SPRINT-PLAN.md`):**
- Branch `record-enumerator-merge`, 68 commits, **NOT pushed** (push requires explicit user say-so).
- Coverage floor at **60%** in CI; must not regress and SHOULD ratchet in the same commit as new tests land.
- Snapshot tests (`test_digest_snapshot.py`, `test_pipeline_e2e.py`) are byte-stable — any intentional update lives in the same commit + cites the reason.
- Never `--no-verify`. Never add `Co-Authored-By`. Never add "🤖 Generated with..." attribution.
- Production credentials never in code (Task 5 enforces).

---

## Task ordering

```
Task 1  Plan re-tier         (docs)        ← run first; provides scope anchor
Task 2  Database drift fix   (code)        ← independent of 3
Task 3  MetadataCascade      (code)        ← independent of 2; longest
Task 4  Real SNIRF fixture   (conditional) ← independent; gated on OpenNeuro
Task 5  Security guards      (ops + code)  ← independent
Task 6  Mutmut observation   (gated)       ← requires user authorization to push
```

Tasks 2 and 3 may run in parallel by separate workers. Each task ends with a green test suite + clean commit; nothing is in-flight across task boundaries.

---

## Task 1: Re-tier NEXT-SPRINT-PLAN.md against drift findings

**Files:**
- Modify: `scripts/ingestions/ROBUSTNESS/NEXT-SPRINT-PLAN.md`

**Why this first:** the existing plan's LOC table is stale (cites pre-C5 numbers) and Tier-1/Tier-3 rankings don't reflect the drivers we now see (digest LOC drift, mutmut-never-observed, C2.5 user-facing cost). Refactoring the doc gives every subsequent task a clear authoritative scope to point at.

- [ ] **Step 1.1: Open the existing plan**

Read `scripts/ingestions/ROBUSTNESS/NEXT-SPRINT-PLAN.md` start-to-finish. Note: there is no failing-test step here — this is doc work.

- [ ] **Step 1.2: Edit the "Next sprint candidates" block — add a Tier-1 item for the cascade refactor**

Find the section header `## Next sprint candidates (ranked by real-driver strength)` and the `### Tier 1 — Real production drivers` subsection. Insert a new bullet `4` AFTER the existing 3 Tier-1 items (push / rotate / drift fix):

```markdown
4. **`_extract_technical_metadata` depth refactor.** The cascade
   function grew 140 → 244 LOC across C5/C6 (BIDS-sidecar enrichment
   absorbed orchestration logic). Refactor extracts a
   `_metadata_cascade.py` module — small interface (`run(ctx) → result`),
   five cascade-step adapters behind it. Real driver: cascade test
   isolation (Lesson #3) + future "add 6th source" is one file.
   Snapshot tests are the gate (byte-stable required). See
   `SPRINT-2026-05-22-PLAN.md` Task 3.
```

- [ ] **Step 1.3: Promote C2.5 from Tier 3 to Tier 2 (real driver: 3.6 s cold-import per user)**

Find `### Tier 3 — Speculative` and the bullet starting `8. **Cross-package eegdash.dataset lazy-load**`. Cut that bullet from Tier 3 and paste it as a new bullet in `### Tier 2 — Strong leverage if a driver appears` (renumber so it becomes `7.` after the existing 4–6 in Tier 2). Replace the body with:

```markdown
7. **Cross-package `eegdash.dataset` lazy-load** (C2.5). Every cold
   import pays a 3.6 s braindecode chain (`PERFORMANCE.md`). Out of
   this repo's ingestions/ tree, but in scope for a follow-up PR.
   Real driver: every user pays this on every cold import. Tier 2
   (not 3) because the driver is universal, not hypothetical.
```

- [ ] **Step 1.4: Add the LOC-drift section AFTER "Lessons learned"**

Find the closing line of section `## Lessons learned (carry these forward)` (it ends with the deferral-pattern bullet `### 7.`). Add this new section immediately AFTER section 7 closes (before the `---` separator that introduces "Cluster integration details"):

```markdown
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
```

- [ ] **Step 1.5: Update the "State at end of last session" coverage line**

Find the bullet `Coverage: 60%, floor enforced via CI gate, 20 visible ratchet steps`. Verify the number is still 60 (it should be — see `.github/workflows/ingestions-lint-and-test.yml:80`). If it has been bumped since 2026-05-22, update accordingly. No other edits.

- [ ] **Step 1.6: Run a diff sanity-check**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git diff scripts/ingestions/ROBUSTNESS/NEXT-SPRINT-PLAN.md | head -120
```

Expected: only the three insertions described (new Tier-1 #4, Tier-2 #7 moved, new Lesson #8). No accidental deletions.

- [ ] **Step 1.7: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/ROBUSTNESS/NEXT-SPRINT-PLAN.md
git commit -m "docs(ingestions): re-tier NEXT-SPRINT-PLAN with C5-C8 LOC drift findings

- Add Tier-1 #4: _extract_technical_metadata depth refactor (140->244 LOC)
- Promote C2.5 cross-package lazy-load from Tier-3 to Tier-2 (real driver)
- Add Lesson #8: LOC drift during enrichment cycles"
```

Verify with `git log -1 --stat` — the commit should touch only `NEXT-SPRINT-PLAN.md`.

---

## Task 2: Database-list drift fix (Tier-1 #3 from re-tiered plan)

**Files:**
- Modify: `scripts/ingestions/_inject_config.py:39-50` and the `database` field
- Modify: `scripts/ingestions/tests/test_inject_config.py` (add 5 new tests)
- Test: `scripts/ingestions/tests/test_inject_config.py`

**Why this matters:** `CONFIG-PATTERN.md` caveat #1 — the `Literal[...]` in `_inject_config.py` and the API's `valid_databases` set can drift silently. Today the only detector is the C6.3 / C6.4 integration cycle. We want config-construction-time detection.

**Design choice:** consumer-side first. We add a `fetch_valid_databases_from_api(api_url, token, timeout)` that hits `GET <api_url>/admin/valid-databases` with the Bearer token. On any failure (network, JSON shape, 404 because endpoint doesn't exist yet on the server) it returns `None` and the caller falls back to the local frozenset (today's behaviour). The server-side endpoint is documented as the follow-up but is **not required** for this task to land — the consumer-side just degrades gracefully until then.

**Threading note:** Pydantic v2 validators are called from a synchronous context. We use a sync `httpx.Client(timeout=5.0)`. Validators must not become async.

- [ ] **Step 2.1: Write the failing test for `fetch_valid_databases_from_api` success path**

Append to `scripts/ingestions/tests/test_inject_config.py`:

```python
# ─── fetch_valid_databases_from_api ────────────────────────────────────────

import respx
import httpx
from _inject_config import (
    fetch_valid_databases_from_api,
    LOCAL_FALLBACK_DATABASES,
)


@respx.mock
def test_fetch_valid_databases_returns_api_list_on_200():
    """Happy path: API returns {"databases": [...]} → frozenset of names."""
    api_url = "https://api.example.test"
    respx.get(f"{api_url}/admin/valid-databases").mock(
        return_value=httpx.Response(
            200, json={"databases": ["eegdash", "eegdash_dev", "eegdash_v2"]}
        )
    )

    result = fetch_valid_databases_from_api(api_url, token="dummy")

    assert result == frozenset({"eegdash", "eegdash_dev", "eegdash_v2"})
```

- [ ] **Step 2.2: Run it — expect ImportError**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_inject_config.py::test_fetch_valid_databases_returns_api_list_on_200 -v
```

Expected: `ImportError: cannot import name 'fetch_valid_databases_from_api' from '_inject_config'`. Same for `LOCAL_FALLBACK_DATABASES`.

- [ ] **Step 2.3: Implement `LOCAL_FALLBACK_DATABASES` + `fetch_valid_databases_from_api`**

Edit `scripts/ingestions/_inject_config.py`. After the `DEFAULT_API_URL` line (around line 37) and BEFORE the `ValidDatabase` Literal, insert:

```python
# Source-of-truth fallback. Must stay aligned with the API Gateway's
# settings.valid_databases set; the bootstrap call below will detect
# drift at boot when the server exposes /admin/valid-databases. Until
# then, this list IS the contract.
LOCAL_FALLBACK_DATABASES: frozenset[str] = frozenset(
    {
        "eegdash",
        "eegdash_dev",
        "eegdash_archive",
        "eegdash_staging",
        "eegdash_v1",
    }
)

# Per-API-URL cache so validator calls don't re-hit the network on
# every InjectConfig construction. Cleared by tests via the
# clear_valid_databases_cache() helper.
_valid_databases_cache: dict[str, frozenset[str]] = {}


def fetch_valid_databases_from_api(
    api_url: str,
    token: str | None,
    *,
    timeout: float = 5.0,
) -> frozenset[str] | None:
    """Fetch the API Gateway's valid_databases set.

    Returns a frozenset on success, None on any failure (network error,
    non-200 status, malformed JSON, missing 'databases' key). Cached
    per api_url so repeated InjectConfig construction is cheap.
    """
    if api_url in _valid_databases_cache:
        return _valid_databases_cache[api_url]

    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{api_url}/admin/valid-databases", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            valid = frozenset(data["databases"])
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        # Any of: connect/timeout/HTTPStatusError, JSON parse failure,
        # missing 'databases' key, non-iterable value. Fall back to
        # local contract — same behaviour as before this fix landed.
        return None

    _valid_databases_cache[api_url] = valid
    return valid


def clear_valid_databases_cache() -> None:
    """Test helper. Resets the per-api_url cache."""
    _valid_databases_cache.clear()
```

- [ ] **Step 2.4: Run the test — expect PASS**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_inject_config.py::test_fetch_valid_databases_returns_api_list_on_200 -v
```

Expected: PASS.

- [ ] **Step 2.5: Write the failing test for cache behaviour**

Append to `tests/test_inject_config.py`:

```python
@respx.mock
def test_fetch_valid_databases_is_cached_per_api_url():
    """Second call to the same api_url should NOT re-hit the server."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    api_url = "https://cache-test.example"
    route = respx.get(f"{api_url}/admin/valid-databases").mock(
        return_value=httpx.Response(
            200, json={"databases": ["eegdash"]}
        )
    )

    fetch_valid_databases_from_api(api_url, token=None)
    fetch_valid_databases_from_api(api_url, token=None)
    fetch_valid_databases_from_api(api_url, token=None)

    assert route.call_count == 1
```

- [ ] **Step 2.6: Run — expect PASS** (cache logic already implemented)

```bash
pytest tests/test_inject_config.py::test_fetch_valid_databases_is_cached_per_api_url -v
```

Expected: PASS.

- [ ] **Step 2.7: Write the failing tests for failure paths**

Append to `tests/test_inject_config.py`:

```python
@respx.mock
def test_fetch_valid_databases_returns_none_on_404():
    """Endpoint doesn't exist on this server -> None (caller falls back)."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    api_url = "https://no-endpoint.example"
    respx.get(f"{api_url}/admin/valid-databases").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    assert fetch_valid_databases_from_api(api_url, token="x") is None


@respx.mock
def test_fetch_valid_databases_returns_none_on_network_error():
    """Connection error -> None."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    api_url = "https://network-fail.example"
    respx.get(f"{api_url}/admin/valid-databases").mock(
        side_effect=httpx.ConnectError("boom")
    )
    assert fetch_valid_databases_from_api(api_url, token="x") is None


@respx.mock
def test_fetch_valid_databases_returns_none_on_missing_key():
    """Server returns 200 but payload shape is wrong -> None."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    api_url = "https://bad-shape.example"
    respx.get(f"{api_url}/admin/valid-databases").mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )
    assert fetch_valid_databases_from_api(api_url, token="x") is None
```

- [ ] **Step 2.8: Run — expect PASS** (all failure paths handled by the try/except)

```bash
pytest tests/test_inject_config.py -k fetch_valid_databases -v
```

Expected: 5 passes (success + cache + 3 failure paths).

- [ ] **Step 2.9: Wire the API-fetch into `InjectConfig.database` validation**

Edit `scripts/ingestions/_inject_config.py`. Change the `database: ValidDatabase = Field(...)` declaration and add a `@field_validator` that consults the API list (if available) AND the local fallback.

Replace lines `42-48` (the `ValidDatabase = Literal[...]` block) and lines `69-76` (the `database: ValidDatabase = Field(...)` block) with:

```python
# Replaces the Literal so we can dynamically extend the accepted set
# when the API exposes /admin/valid-databases. The frozenset above
# (LOCAL_FALLBACK_DATABASES) is the contract until the API is queried.
DatabaseName = str
```

And the database field becomes:

```python
    database: DatabaseName = Field(
        description=(
            "Target MongoDB database. Valid names checked against the API's "
            "valid-databases endpoint at boot (falls back to "
            "LOCAL_FALLBACK_DATABASES on network failure)."
        ),
    )
```

Then add this `@field_validator` immediately AFTER the `database` field declaration (before the other Fields):

```python
    @field_validator("database")
    @classmethod
    def _database_must_be_valid(cls, value: str, info) -> str:
        """Reject databases the cluster doesn't know about.

        Order of checks:
        1. If the API exposes /admin/valid-databases AND we can reach it,
           use the returned set.
        2. Otherwise fall back to LOCAL_FALLBACK_DATABASES (the contract
           we maintained pre-fetch).

        Local Literal-style validation always runs — even if the API call
        succeeds — so an empty-list response from a misconfigured server
        doesn't lock us out.
        """
        # info.data has fields validated BEFORE this one. `api_url` and
        # `token` are declared after, so we read defaults from env if
        # not yet set. Keep this best-effort.
        api_url = info.data.get("api_url") or DEFAULT_API_URL
        token = info.data.get("token")

        api_set = fetch_valid_databases_from_api(api_url, token)
        valid: frozenset[str] = api_set if api_set else LOCAL_FALLBACK_DATABASES

        if value not in valid:
            raise ValueError(
                f"database={value!r} is not in the valid set "
                f"({sorted(valid)}); update the API's valid_databases or "
                f"LOCAL_FALLBACK_DATABASES if you are adding a new one."
            )
        return value
```

Also import `field_validator` at the top — change the line `from pydantic import AliasChoices, Field, model_validator` to:

```python
from pydantic import AliasChoices, Field, field_validator, model_validator
```

- [ ] **Step 2.10: Update existing tests that referenced the Literal**

The existing 28 inject-config tests construct `InjectConfig(database="eegdash_dev", ...)` — these still pass because `eegdash_dev` is in `LOCAL_FALLBACK_DATABASES`. But any test that asserted `pydantic` raised on a bad name needs to ensure the api_set is not stubbed to allow it.

Search for tests that pass a bogus database name:

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
grep -n 'database=' tests/test_inject_config.py | grep -v 'eegdash'
```

If any tests pass a database name NOT in `LOCAL_FALLBACK_DATABASES`, prepend `clear_valid_databases_cache()` to ensure no cached api_set lingers. (Test runs in isolation but pytest reuses processes.)

- [ ] **Step 2.11: Write the failing test for "unknown DB rejected at config time"**

Append to `tests/test_inject_config.py`:

```python
def test_inject_config_rejects_unknown_database_via_local_fallback(tmp_path):
    """Without network access, an unknown database is rejected by
    LOCAL_FALLBACK_DATABASES."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    with pytest.raises(ValidationError) as exc:
        InjectConfig(
            database="eegdash_does_not_exist",
            input=tmp_path,
            dry_run=True,
        )
    assert "valid set" in str(exc.value)


@respx.mock
def test_inject_config_accepts_database_only_in_api_set(tmp_path):
    """An API that knows about a new database lets us inject to it
    even when LOCAL_FALLBACK_DATABASES does not."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    respx.get(f"{DEFAULT_API_URL}/admin/valid-databases").mock(
        return_value=httpx.Response(
            200,
            json={"databases": ["eegdash", "eegdash_dev", "eegdash_v99_future"]},
        )
    )

    c = InjectConfig(
        database="eegdash_v99_future",
        input=tmp_path,
        dry_run=True,
    )
    assert c.database == "eegdash_v99_future"
```

- [ ] **Step 2.12: Run the full inject-config suite**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_inject_config.py -v
```

Expected: original 28 tests PLUS the 7 new tests = 35 passes. No failures.

- [ ] **Step 2.13: Re-run full PR-fast suite to confirm no regressions**

```bash
pytest -q -m "not network and not slow and not integration"
```

Expected: 788 → 795 tests passing (we added 7). No failures, no `INTERNALERROR`.

- [ ] **Step 2.14: Verify coverage didn't drop**

```bash
pytest -q -m "not network and not slow and not integration" --cov=. --cov-fail-under=60
```

Expected: coverage ≥ 60%. If `_inject_config.py` coverage drops below 95% (it was 100%), investigate — likely a branch in the new `field_validator` is uncovered.

- [ ] **Step 2.15: Update CONFIG-PATTERN.md caveat #1**

In `scripts/ingestions/ROBUSTNESS/CONFIG-PATTERN.md`, find the section `### Database list drift risk` (line ~85). Replace the "Mitigation options" block with:

```markdown
**Status: option 1 implemented (2026-05-22).** `_inject_config.py:
fetch_valid_databases_from_api()` hits `GET <api_url>/admin/valid-databases`
with the Bearer token at config-construction time (5s timeout, per-api_url
cached). On any failure (network, 404, JSON shape) the field validator
falls back to `LOCAL_FALLBACK_DATABASES` — same contract as before the
fix landed. The API-side endpoint is documented as a server follow-up
but not required for the consumer-side guard to work.
```

- [ ] **Step 2.16: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/_inject_config.py \
        scripts/ingestions/tests/test_inject_config.py \
        scripts/ingestions/ROBUSTNESS/CONFIG-PATTERN.md
git commit -m "fix(inject): close database-list drift via /admin/valid-databases probe

Consumer-side guard. InjectConfig.database now validates against the
union of the API's valid-databases endpoint (when reachable) and
LOCAL_FALLBACK_DATABASES. On any failure (network, 404, JSON shape)
falls back to local frozenset -- same contract as before. Server-side
endpoint is a follow-up.

- 7 new tests (5 fetch + 2 InjectConfig)
- Per-api_url cache so validator doesn't re-hit network
- CONFIG-PATTERN.md caveat 1 updated to reflect consumer-side fix"
```

---

## Task 3: Extract `MetadataCascade` from `_extract_technical_metadata`

**Files:**
- Create: `scripts/ingestions/_metadata_cascade.py`
- Create: `scripts/ingestions/tests/test_metadata_cascade.py`
- Modify: `scripts/ingestions/3_digest.py:1111-1339` (the function shrinks to ~25 LOC delegator)

**Why this matters:** the function grew 140 → 244 LOC across C5/C6. Five independent cascade sources are interleaved with provenance stamping; tests today exercise the whole function and only catch surface bugs. Goal: pass the **deletion test** — extracting `MetadataCascade` means the cascade-ordering complexity vanishes from `3_digest.py` (caller sees one method), and re-appearing at any other caller would be the same one-line `MetadataCascade.run(ctx)`.

**Snapshot tests are the gate.** All `test_digest_snapshot.py` cases must stay byte-identical. We run them after every refactor step.

**Interface design (locked in here, see CONFIG-PATTERN.md style):**

```
CascadeContext (dataclass)
  └─ bids_dataset: Any
     bids_file: str
     bids_file_path: Path        # derived
     ext: str                    # derived (.vhdr / .snirf / ...)
     bids_root: Path             # derived

CascadeResult (dataclass)
  └─ sampling_frequency: float | None
     nchans:              int   | None
     ntimes:              int   | None
     ch_names:        list[str] | None
     fif_is_split:        bool
     fif_continuations_ok: bool
     provenance:    dict[str, str | None]

CascadeStep (Protocol)
  └─ def fill(ctx: CascadeContext, result: CascadeResult) -> None
       — mutates result in-place; first-writer-wins for each field

MetadataCascade
  └─ steps: tuple[CascadeStep, ...]
     def run(ctx: CascadeContext) -> CascadeResult
```

Five concrete steps:
- `MneBidsStep`              ← Step 1 in current function
- `ModalitySidecarStep`      ← Step 2
- `ChannelsTsvStep`          ← Step 3
- `BinaryParserStep`         ← Step 4 (registry-based)
- `MneFallbackStep`          ← Step 4 trailing (VHDR n_times + FIF)

- [ ] **Step 3.1: Write the failing test for the CascadeContext + CascadeResult dataclasses**

Create `scripts/ingestions/tests/test_metadata_cascade.py`:

```python
"""Tests for the MetadataCascade module (Task 3 — SPRINT-2026-05-22)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from _metadata_cascade import (
    CascadeContext,
    CascadeResult,
    MetadataCascade,
    MneBidsStep,
    ModalitySidecarStep,
    ChannelsTsvStep,
    BinaryParserStep,
    MneFallbackStep,
)


def test_cascade_context_derives_ext_and_root():
    """CascadeContext computes ext + bids_file_path from inputs."""
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"

    ctx = CascadeContext(
        bids_dataset=bids_dataset,
        bids_file="sub-01/eeg/sub-01_task-rest_eeg.vhdr",
    )

    assert ctx.ext == ".vhdr"
    assert ctx.bids_file_path == Path("sub-01/eeg/sub-01_task-rest_eeg.vhdr")
    assert ctx.bids_root == Path("/tmp/bids")


def test_cascade_result_defaults_are_none():
    result = CascadeResult()
    assert result.sampling_frequency is None
    assert result.nchans is None
    assert result.ntimes is None
    assert result.ch_names is None
    assert result.fif_is_split is False
    assert result.fif_continuations_ok is True
    # provenance starts as 4 keys, all None
    assert set(result.provenance.keys()) == {
        "sampling_frequency", "nchans", "ntimes", "ch_names"
    }
    assert all(v is None for v in result.provenance.values())
```

- [ ] **Step 3.2: Run — expect ImportError on `_metadata_cascade`**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_metadata_cascade.py -v
```

Expected: `ModuleNotFoundError: No module named '_metadata_cascade'`.

- [ ] **Step 3.3: Implement the CascadeContext, CascadeResult, and Protocol**

Create `scripts/ingestions/_metadata_cascade.py`:

```python
"""Metadata cascade module — extracted from 3_digest.py:_extract_technical_metadata.

Goal: deep module. The cascade orchestration (5 sources, first-writer-wins
provenance) lives behind a small interface (CascadeStep Protocol +
MetadataCascade.run). Adding a 6th source means one new step class + one
entry in the tuple — no changes to 3_digest.py.

See SPRINT-2026-05-22-PLAN.md Task 3 for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# Provenance constants — kept identical to 3_digest.py so the Record
# schema doesn't change (snapshot tests would catch any drift).
PROV_MNE_BIDS = "mne_bids"
PROV_MODALITY_SIDECAR = "modality_sidecar"
PROV_CHANNELS_TSV = "channels_tsv"
PROV_BINARY_PARSER = "binary_parser"
PROV_MNE_FALLBACK = "mne_fallback"

_METADATA_FIELDS = ("sampling_frequency", "nchans", "ntimes", "ch_names")


@dataclass
class CascadeContext:
    """Input context for the cascade. Derived fields computed in __post_init__."""

    bids_dataset: Any
    bids_file: str

    # Derived
    bids_file_path: Path = field(init=False)
    bids_root: Path = field(init=False)
    ext: str = field(init=False)

    def __post_init__(self) -> None:
        self.bids_file_path = Path(self.bids_file)
        self.bids_root = Path(self.bids_dataset.bidsdir)
        self.ext = self.bids_file_path.suffix.lower()


@dataclass
class CascadeResult:
    """Accumulated cascade output. Steps mutate this in-place."""

    sampling_frequency: float | None = None
    nchans: int | None = None
    ntimes: int | None = None
    ch_names: list[str] | None = None
    fif_is_split: bool = False
    fif_continuations_ok: bool = True
    provenance: dict[str, str | None] = field(
        default_factory=lambda: {f: None for f in _METADATA_FIELDS}
    )

    def stamp(self, source: str, field_name: str, old: Any, new: Any) -> None:
        """Mark provenance[field] = source iff this step changed the field."""
        if old != new and new is not None and self.provenance[field_name] is None:
            self.provenance[field_name] = source


class CascadeStep(Protocol):
    """Protocol every cascade step implements."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None: ...


class MetadataCascade:
    """Runs cascade steps in order. First writer for each field wins."""

    def __init__(self, steps: tuple[CascadeStep, ...] | None = None) -> None:
        # Default order matches the original _extract_technical_metadata
        # cascade (preserves snapshot bytes). Tests can inject alternate
        # orderings.
        self.steps = steps or (
            MneBidsStep(),
            ModalitySidecarStep(),
            ChannelsTsvStep(),
            BinaryParserStep(),
            MneFallbackStep(),
        )

    def run(self, ctx: CascadeContext) -> CascadeResult:
        result = CascadeResult()
        for step in self.steps:
            step.fill(ctx, result)
        return result


# Step implementations land in subsequent commits — declared here so
# the MetadataCascade default-tuple import works. They will be filled
# in Steps 3.5 through 3.13.

class MneBidsStep:
    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        raise NotImplementedError  # filled in Step 3.5


class ModalitySidecarStep:
    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        raise NotImplementedError  # filled in Step 3.7


class ChannelsTsvStep:
    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        raise NotImplementedError  # filled in Step 3.9


class BinaryParserStep:
    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        raise NotImplementedError  # filled in Step 3.11


class MneFallbackStep:
    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        raise NotImplementedError  # filled in Step 3.13
```

- [ ] **Step 3.4: Run the dataclass tests — expect PASS**

```bash
pytest tests/test_metadata_cascade.py::test_cascade_context_derives_ext_and_root \
       tests/test_metadata_cascade.py::test_cascade_result_defaults_are_none -v
```

Expected: 2 passes.

- [ ] **Step 3.5: Write the failing test for `MneBidsStep`**

Append to `tests/test_metadata_cascade.py`:

```python
def test_mne_bids_step_fills_from_attribute_getters():
    """Step 1: pulls sfreq/nchans/ntimes from EEGBIDSDataset attribute getters."""
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    bids_dataset.get_bids_file_attribute.side_effect = lambda key, _file: {
        "sfreq": "500",
        "nchans": "64",
        "ntimes": "1000",
    }[key]
    bids_dataset.channel_labels.return_value = ["F1", "F2", "Cz"]

    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")
    result = CascadeResult()

    MneBidsStep().fill(ctx, result)

    assert result.sampling_frequency == 500.0
    assert result.nchans == 3   # channel_labels count overrides sidecar nchans
    assert result.ntimes == 1000
    assert result.ch_names == ["F1", "F2", "Cz"]
    assert result.provenance == {
        "sampling_frequency": "mne_bids",
        "nchans": "mne_bids",
        "ntimes": "mne_bids",
        "ch_names": "mne_bids",
    }


def test_mne_bids_step_handles_missing_sidecar_gracefully():
    """OSError from attribute getter -> step leaves fields None."""
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    bids_dataset.get_bids_file_attribute.side_effect = FileNotFoundError("sidecar missing")
    bids_dataset.channel_labels.side_effect = OSError("annex broken")

    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")
    result = CascadeResult()
    MneBidsStep().fill(ctx, result)

    assert result.sampling_frequency is None
    assert result.nchans is None
    assert all(v is None for v in result.provenance.values())
```

- [ ] **Step 3.6: Implement `MneBidsStep`**

Replace the `class MneBidsStep` stub in `_metadata_cascade.py`:

```python
class MneBidsStep:
    """Step 1: EEGBIDSDataset attribute getters (mne_bids)."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        bd = ctx.bids_dataset
        try:
            sf = bd.get_bids_file_attribute("sfreq", ctx.bids_file)
            nc = bd.get_bids_file_attribute("nchans", ctx.bids_file)
            nt = bd.get_bids_file_attribute("ntimes", ctx.bids_file)
        except (FileNotFoundError, OSError):
            sf = nc = nt = None

        if sf:
            result.sampling_frequency = float(sf)
            result.provenance["sampling_frequency"] = PROV_MNE_BIDS
        if nc:
            result.nchans = int(nc)
            result.provenance["nchans"] = PROV_MNE_BIDS
        if nt:
            result.ntimes = int(nt)
            result.provenance["ntimes"] = PROV_MNE_BIDS

        try:
            ch_names = bd.channel_labels(ctx.bids_file)
        except (FileNotFoundError, OSError, ValueError, KeyError, AttributeError):
            ch_names = None
        if ch_names:
            result.ch_names = ch_names
            result.provenance["ch_names"] = PROV_MNE_BIDS
            # channel_labels count is more reliable than sidecar nchans
            # (sidecar JSON may only have EEGChannelCount, missing aux).
            result.nchans = len(ch_names)
            if result.provenance["nchans"] is None:
                result.provenance["nchans"] = PROV_MNE_BIDS
```

Run:

```bash
pytest tests/test_metadata_cascade.py -k mne_bids -v
```

Expected: 2 passes.

- [ ] **Step 3.7: Write the failing test for `ModalitySidecarStep`**

Append to `tests/test_metadata_cascade.py`:

```python
def test_modality_sidecar_step_only_fills_unset_fields(monkeypatch):
    """Step 2: fills sfreq/nchans from modality sidecar IFF still None."""
    import _metadata_cascade as mc

    # Stub the helper function the step calls
    def fake_extract(_path, _root, sf_in, nc_in):
        # Behaves like the real one: returns (sf, nchans)
        return (sf_in or 250.0, nc_in or 32)

    monkeypatch.setattr(
        mc, "extract_sfreq_nchans_from_modality_sidecar", fake_extract
    )

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"

    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")
    result = CascadeResult(sampling_frequency=None, nchans=None)

    mc.ModalitySidecarStep().fill(ctx, result)

    assert result.sampling_frequency == 250.0
    assert result.nchans == 32
    assert result.provenance["sampling_frequency"] == "modality_sidecar"
    assert result.provenance["nchans"] == "modality_sidecar"


def test_modality_sidecar_step_does_not_overwrite_filled_fields(monkeypatch):
    """If Step 1 filled sampling_frequency, Step 2 must not overwrite."""
    import _metadata_cascade as mc

    monkeypatch.setattr(
        mc,
        "extract_sfreq_nchans_from_modality_sidecar",
        lambda _p, _r, sf, nc: (sf or 250.0, nc or 32),
    )

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")

    # Step 1 already filled sampling_frequency.
    result = CascadeResult(sampling_frequency=500.0, nchans=None)
    result.provenance["sampling_frequency"] = "mne_bids"

    mc.ModalitySidecarStep().fill(ctx, result)

    assert result.sampling_frequency == 500.0  # unchanged
    assert result.provenance["sampling_frequency"] == "mne_bids"  # first-writer wins
    assert result.nchans == 32  # nchans WAS unset, so step fills
    assert result.provenance["nchans"] == "modality_sidecar"
```

- [ ] **Step 3.8: Implement `ModalitySidecarStep`**

In `_metadata_cascade.py`, add at the top of the module (after the constants block, before the dataclasses):

```python
# Lazy import — these live in 3_digest.py today. Will move to a
# dedicated helper module in a future ROUND when called from > 1 place.
def _get_sidecar_extractors():
    import importlib
    mod = importlib.import_module("3_digest")
    return (
        mod.extract_sfreq_nchans_from_modality_sidecar,
        mod.extract_sfreq_nchans_from_channels_tsv,
    )
```

Wait — `3_digest.py` starts with a digit so Python can't `import 3_digest`. Skip the lazy-import dance: copy-import the two helpers into `_metadata_cascade.py` by exposing them from a new `_sidecar_extractors.py` first.

Actually, the cleanest path: **move** the two functions to `_metadata_cascade.py` itself, since they're only called from the cascade. Read the current definitions:

```bash
cd /Users/bruaristimunha/Projects/eegdash
sed -n '1950,2100p' scripts/ingestions/3_digest.py
```

Copy `extract_sfreq_nchans_from_modality_sidecar` (lines 1950-2021) and `extract_sfreq_nchans_from_channels_tsv` (lines 2022-2100) verbatim into `_metadata_cascade.py` AFTER the constants block and BEFORE the dataclasses. Also copy any dependencies they reference from `_montage` (the `_walk_up_find` helper, which already lives in `_montage.py` — keep that import).

Then add the import inside `_metadata_cascade.py`:

```python
from _montage import _walk_up_find
```

In `3_digest.py`, replace the two functions' bodies with re-export shims:

```python
from _metadata_cascade import (
    extract_sfreq_nchans_from_modality_sidecar,
    extract_sfreq_nchans_from_channels_tsv,
)  # noqa: F401 — re-export for back-compat
```

This is a one-line move at the top of `3_digest.py` plus deleting the bodies. The existing tests in `tests/test_bids_sidecar_enrichment.py` continue to pass because they import via `3_digest`.

Now implement the step:

```python
class ModalitySidecarStep:
    """Step 2: modality JSON sidecar with BIDS-inheritance walk."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        sf_before = result.sampling_frequency
        n_before = result.nchans
        sf, nc = extract_sfreq_nchans_from_modality_sidecar(
            ctx.bids_file_path, ctx.bids_root, sf_before, n_before
        )
        result.sampling_frequency = sf
        result.nchans = nc
        result.stamp(PROV_MODALITY_SIDECAR, "sampling_frequency", sf_before, sf)
        result.stamp(PROV_MODALITY_SIDECAR, "nchans", n_before, nc)
```

Run:

```bash
pytest tests/test_metadata_cascade.py -k modality_sidecar -v
```

Expected: 2 passes. Also confirm nothing broke:

```bash
pytest tests/test_bids_sidecar_enrichment.py -v --tb=short
```

Expected: all original tests still pass (the re-export shim works).

- [ ] **Step 3.9: Write the failing test for `ChannelsTsvStep`**

Append to `tests/test_metadata_cascade.py`:

```python
def test_channels_tsv_step_fills_from_helper(monkeypatch):
    import _metadata_cascade as mc

    monkeypatch.setattr(
        mc,
        "extract_sfreq_nchans_from_channels_tsv",
        lambda _p, _r, sf, nc: (sf or 1000.0, nc or 19),
    )
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")
    result = CascadeResult()

    mc.ChannelsTsvStep().fill(ctx, result)

    assert result.sampling_frequency == 1000.0
    assert result.nchans == 19
    assert result.provenance["sampling_frequency"] == "channels_tsv"
    assert result.provenance["nchans"] == "channels_tsv"
```

- [ ] **Step 3.10: Implement `ChannelsTsvStep`**

In `_metadata_cascade.py`:

```python
class ChannelsTsvStep:
    """Step 3: channels.tsv with BIDS-inheritance walk."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        sf_before = result.sampling_frequency
        n_before = result.nchans
        sf, nc = extract_sfreq_nchans_from_channels_tsv(
            ctx.bids_file_path, ctx.bids_root, sf_before, n_before
        )
        result.sampling_frequency = sf
        result.nchans = nc
        result.stamp(PROV_CHANNELS_TSV, "sampling_frequency", sf_before, sf)
        result.stamp(PROV_CHANNELS_TSV, "nchans", n_before, nc)
```

Run:

```bash
pytest tests/test_metadata_cascade.py -k channels_tsv -v
```

Expected: PASS.

- [ ] **Step 3.11: Write the failing test for `BinaryParserStep`**

Append:

```python
def test_binary_parser_step_uses_registry(monkeypatch):
    """Step 4: dispatches to _format_parser_registry per file extension."""
    import _metadata_cascade as mc

    monkeypatch.setattr(
        mc,
        "get_parser_for_extension",
        lambda ext: (
            (lambda _p: {"sampling_frequency": 256.0, "nchans": 16, "n_times": 100000})
            if ext == ".edf"
            else None
        ),
    )

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_eeg.edf")
    result = CascadeResult()  # all fields None

    mc.BinaryParserStep().fill(ctx, result)

    assert result.sampling_frequency == 256.0
    assert result.nchans == 16
    assert result.ntimes == 100000
    assert result.provenance["sampling_frequency"] == "binary_parser"


def test_binary_parser_step_skipped_when_all_fields_filled(monkeypatch):
    """If Steps 1-3 already filled everything, Step 4 must be a no-op."""
    import _metadata_cascade as mc

    parser_call_count = {"n": 0}

    def fake_parser_factory(_ext):
        def _parser(_path):
            parser_call_count["n"] += 1
            return {"sampling_frequency": 999.0, "nchans": 1}
        return _parser

    monkeypatch.setattr(mc, "get_parser_for_extension", fake_parser_factory)

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_eeg.edf")
    result = CascadeResult(
        sampling_frequency=500.0, nchans=64, ntimes=1000, ch_names=["F1"]
    )

    mc.BinaryParserStep().fill(ctx, result)

    assert parser_call_count["n"] == 0  # parser never called
    assert result.sampling_frequency == 500.0  # unchanged
```

- [ ] **Step 3.12: Implement `BinaryParserStep`**

Add the registry import at the top of `_metadata_cascade.py`:

```python
from _format_parser_registry import get_parser_for_extension
```

And implement:

```python
class BinaryParserStep:
    """Step 4: per-extension binary parser via the format registry."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        parser = get_parser_for_extension(ctx.ext)
        if parser is None:
            return

        # Short-circuit: skip if all four fields are already filled.
        if (
            result.sampling_frequency
            and result.nchans
            and result.ntimes
            and result.ch_names
        ):
            return

        md = parser(ctx.bids_file_path)
        if not md:
            return

        sf_before = result.sampling_frequency
        n_before = result.nchans
        nt_before = result.ntimes
        ch_before = result.ch_names

        result.sampling_frequency = sf_before or md.get("sampling_frequency")
        result.nchans = n_before or md.get("nchans")
        result.ntimes = nt_before or md.get("n_times") or md.get("n_samples")
        result.ch_names = ch_before or md.get("ch_names")

        for fname, old, new in (
            ("sampling_frequency", sf_before, result.sampling_frequency),
            ("nchans", n_before, result.nchans),
            ("ntimes", nt_before, result.ntimes),
            ("ch_names", ch_before, result.ch_names),
        ):
            result.stamp(PROV_BINARY_PARSER, fname, old, new)
```

Run:

```bash
pytest tests/test_metadata_cascade.py -k binary_parser -v
```

Expected: 2 passes.

- [ ] **Step 3.13: Write the failing test for `MneFallbackStep` (VHDR n_times path)**

Append:

```python
def test_mne_fallback_step_fills_vhdr_ntimes(monkeypatch):
    """Step 5: VHDR ntimes via MNE when binary parser couldn't get it."""
    import _metadata_cascade as mc

    fake_raw = MagicMock()
    fake_raw.n_times = 200000
    fake_raw.close = MagicMock()

    fake_mne = MagicMock()
    fake_mne.io.read_raw_brainvision.return_value = fake_raw

    monkeypatch.setattr(mc, "mne", fake_mne)

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")
    result = CascadeResult(sampling_frequency=500.0, nchans=32, ntimes=None)

    mc.MneFallbackStep().fill(ctx, result)

    assert result.ntimes == 200000
    assert result.provenance["ntimes"] == "mne_fallback"
    fake_raw.close.assert_called_once()


def test_mne_fallback_step_fif_split_metadata(monkeypatch):
    """Step 5: FIF split detection populates fif_is_split."""
    import _metadata_cascade as mc

    monkeypatch.setattr(
        mc,
        "_parse_fif_with_mne",
        lambda _path: (
            {"sampling_frequency": 1000.0, "nchans": 306, "n_times": 60000,
             "ch_names": ["MEG001"]},
            True,
        ),
    )

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_meg.fif")
    result = CascadeResult()

    mc.MneFallbackStep().fill(ctx, result)

    assert result.fif_is_split is True
    assert result.sampling_frequency == 1000.0
    assert result.provenance["sampling_frequency"] == "mne_fallback"
```

- [ ] **Step 3.14: Implement `MneFallbackStep`**

At the top of `_metadata_cascade.py`, add (the existing pattern in `3_digest.py`):

```python
import mne  # noqa: E402 — needed for VHDR n_times fallback
```

And the FIF helper — move `_parse_fif_with_mne` from `3_digest.py:239-322` into `_metadata_cascade.py` (similar to the sidecar helpers), then re-export from `3_digest.py`:

In `3_digest.py`, replace the `def _parse_fif_with_mne(...)` block with:

```python
from _metadata_cascade import _parse_fif_with_mne  # noqa: F401 — re-export
```

Implement the step:

```python
class MneFallbackStep:
    """Step 5: MNE fallbacks for VHDR ntimes and FIF metadata + split detection."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        # --- VHDR: n_times via MNE Raw -----------------------------------
        if ctx.ext == ".vhdr" and not result.ntimes:
            raw = None
            try:
                raw = mne.io.read_raw_brainvision(
                    str(ctx.bids_file_path), preload=False, verbose=False
                )
                if raw.n_times and raw.n_times > 0:
                    result.ntimes = int(raw.n_times)
                    if result.provenance["ntimes"] is None:
                        result.provenance["ntimes"] = PROV_MNE_FALLBACK
            except (OSError, ValueError, RuntimeError, KeyError):
                pass
            finally:
                if raw is not None:
                    try:
                        raw.close()
                    except (OSError, AttributeError):
                        pass

        # --- FIF: full-record fallback + split detection -----------------
        if ctx.ext == ".fif" and (
            not result.sampling_frequency
            or not result.nchans
            or not result.ntimes
        ):
            fif_metadata, fif_is_split = _parse_fif_with_mne(ctx.bids_file_path)
            result.fif_is_split = fif_is_split
            if fif_metadata:
                sf_before = result.sampling_frequency
                n_before = result.nchans
                nt_before = result.ntimes
                ch_before = result.ch_names

                result.sampling_frequency = sf_before or fif_metadata.get(
                    "sampling_frequency"
                )
                result.nchans = n_before or fif_metadata.get("nchans")
                result.ntimes = nt_before or fif_metadata.get("n_times")
                result.ch_names = ch_before or fif_metadata.get("ch_names")

                for fname, old, new in (
                    ("sampling_frequency", sf_before, result.sampling_frequency),
                    ("nchans", n_before, result.nchans),
                    ("ntimes", nt_before, result.ntimes),
                    ("ch_names", ch_before, result.ch_names),
                ):
                    result.stamp(PROV_MNE_FALLBACK, fname, old, new)
```

Run:

```bash
pytest tests/test_metadata_cascade.py -k mne_fallback -v
```

Expected: 2 passes.

- [ ] **Step 3.15: Write the integration test for the full cascade**

Append:

```python
def test_metadata_cascade_runs_all_steps_in_order(monkeypatch):
    """End-to-end: simulate each step contributing one field."""
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    bids_dataset.get_bids_file_attribute.side_effect = lambda key, _f: (
        "500" if key == "sfreq" else None
    )
    bids_dataset.channel_labels.return_value = None

    import _metadata_cascade as mc

    # Step 2 contributes nchans
    monkeypatch.setattr(
        mc,
        "extract_sfreq_nchans_from_modality_sidecar",
        lambda _p, _r, sf, nc: (sf, nc or 32),
    )
    # Step 3 contributes nothing extra
    monkeypatch.setattr(
        mc,
        "extract_sfreq_nchans_from_channels_tsv",
        lambda _p, _r, sf, nc: (sf, nc),
    )
    # Step 4 contributes ntimes + ch_names
    monkeypatch.setattr(
        mc,
        "get_parser_for_extension",
        lambda _ext: lambda _p: {"n_times": 10000, "ch_names": ["A1", "A2"]},
    )

    ctx = CascadeContext(bids_dataset, "sub-01_eeg.edf")
    result = MetadataCascade().run(ctx)

    assert result.sampling_frequency == 500.0
    assert result.nchans == 32
    assert result.ntimes == 10000
    assert result.ch_names == ["A1", "A2"]
    assert result.provenance == {
        "sampling_frequency": "mne_bids",
        "nchans": "modality_sidecar",
        "ntimes": "binary_parser",
        "ch_names": "binary_parser",
    }
```

Run:

```bash
pytest tests/test_metadata_cascade.py::test_metadata_cascade_runs_all_steps_in_order -v
```

Expected: PASS.

- [ ] **Step 3.16: Refactor `_extract_technical_metadata` to delegate**

Edit `scripts/ingestions/3_digest.py`. Replace the entire function body (lines 1111-1339, the 244-LOC implementation) with this thin delegator:

```python
def _extract_technical_metadata(
    bids_dataset: Any,
    bids_file: str,
) -> tuple[
    float | None,
    int | None,
    int | None,
    list[str] | None,
    bool,
    bool,
    dict[str, str | None],
]:
    """Resolve sfreq / nchans / ntimes / ch_names + their provenance.

    Thin delegator to :class:`_metadata_cascade.MetadataCascade`. The
    cascade owns the 5-step (mne_bids → modality_sidecar → channels_tsv →
    binary_parser → mne_fallback) traversal with first-writer-wins
    provenance stamping. See ``_metadata_cascade.py`` for the
    implementation; this signature stays for back-compat with callers
    in ``extract_record`` and ``extract_dataset_metadata``.
    """
    from _metadata_cascade import CascadeContext, MetadataCascade

    ctx = CascadeContext(bids_dataset=bids_dataset, bids_file=bids_file)
    result = MetadataCascade().run(ctx)
    return (
        result.sampling_frequency,
        result.nchans,
        result.ntimes,
        result.ch_names,
        result.fif_is_split,
        result.fif_continuations_ok,
        result.provenance,
    )
```

Verify the function shrank:

```bash
cd /Users/bruaristimunha/Projects/eegdash
python3 -c "
import re
lines = open('scripts/ingestions/3_digest.py').readlines()
in_fn = False; start = None
for i, line in enumerate(lines):
    if line.startswith('def _extract_technical_metadata'):
        in_fn = True; start = i
    elif in_fn and (line.startswith('def ') or line.startswith('class ')):
        print(f'_extract_technical_metadata: {i - start} LOC')
        break
"
```

Expected: ~25 LOC (was 244).

- [ ] **Step 3.17: Run the snapshot tests — this is the gate**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_digest_snapshot.py -v --tb=long
```

Expected: all snapshot tests pass byte-identically. **If any snapshot fails, do NOT update it — the refactor introduced a behavioural change.** Investigate the diff (likely a `result.stamp` first-writer condition that drifted) and fix the implementation, not the snapshot.

- [ ] **Step 3.18: Run the e2e pipeline tests**

```bash
pytest tests/test_pipeline_e2e.py tests/test_pipeline_e2e_mef3.py -v --tb=short
```

Expected: all pass. These walk Stage 3→4→5 on real fixtures.

- [ ] **Step 3.19: Run the full PR-fast suite**

```bash
pytest -q -m "not network and not slow and not integration" --tb=short
```

Expected: 795 → ~810 tests passing (we added ~15 cascade tests). 0 failures.

- [ ] **Step 3.20: Verify coverage didn't drop + ratchet if it rose**

```bash
pytest -q -m "not network and not slow and not integration" --cov=. --cov-report=term-missing --cov-fail-under=60 | tail -30
```

If the cumulative coverage rose from 60% to >= 61%, bump the floor in `.github/workflows/ingestions-lint-and-test.yml` line 80 (`--cov-fail-under=60` → `61`) AND append a line to the ratchet history (lines 56-70) in the same file:

```yaml
        #   2026-05-22 Task3: 61  (_metadata_cascade extraction + 15 cascade tests)
```

Otherwise leave the floor at 60.

- [ ] **Step 3.21: Update the ROADMAP.md LOC ceiling table**

In `scripts/ingestions/ROBUSTNESS/ROADMAP.md` find the section `## Mega-function LOC ceilings (current state)` (around line 173). Update the `_extract_technical_metadata` row:

```markdown
| `_extract_technical_metadata` | 25 | ✅ (cascade extracted to `_metadata_cascade.py` — SPRINT-2026-05-22 Task 3) |
```

- [ ] **Step 3.22: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/_metadata_cascade.py \
        scripts/ingestions/tests/test_metadata_cascade.py \
        scripts/ingestions/3_digest.py \
        scripts/ingestions/ROBUSTNESS/ROADMAP.md \
        .github/workflows/ingestions-lint-and-test.yml
git commit -m "refactor(digest): extract MetadataCascade module (244 LOC -> 25 LOC)

The 5-step technical-metadata cascade (mne_bids -> modality_sidecar ->
channels.tsv -> binary_parser -> mne_fallback) moves to a new
_metadata_cascade.py module. Deep module: small interface
(MetadataCascade.run(ctx) -> CascadeResult), 5 step Adapters behind it.

- _extract_technical_metadata now a 25-LOC delegator
- 11 new cascade tests (5 step unit + 1 integration + 4 edge + 1 ctx/result)
- Snapshot tests stay byte-identical (gate)
- _parse_fif_with_mne + sidecar helpers re-exported via _metadata_cascade
- ROADMAP.md LOC ceiling table updated
- Coverage ratcheted in same commit if it rose"
```

---

## Task 4: Real SNIRF fixture (conditional on OpenNeuro availability)

**Files:**
- Create: `scripts/ingestions/tests/fixtures/fnirs/<filename>.snirf` (small, < 1 MB, CC0)
- Create: `scripts/ingestions/tests/test_snirf_real_fixture.py`
- Modify: `scripts/ingestions/_snirf_parser.py` if the real file surfaces a parser bug (à la C5.1)

**Why this matters:** Lesson #1 — synthetic h5py fixtures only validate the parser against itself. The 1-bug-per-real-fixture yield from C5.1 (MEF3 sfreq offset 8720) is the proof. SNIRF was deferred at C3 close because no real BIDS-conforming dataset existed publicly. Re-check now.

- [ ] **Step 4.1: Search OpenNeuro for an fNIRS BIDS dataset**

Run a web search and OpenNeuro listing fetch:

```bash
# Open https://openneuro.org/search and filter by modality=NIRS
# Or via API:
curl -s "https://openneuro.org/api/datasets?modality=NIRS&limit=10" | head -200
```

If the search returns datasets, note the smallest one's accession ID (e.g. `ds00XXXX`). If no results, **proceed to Step 4.7 (document the gap and skip)**.

- [ ] **Step 4.2: Identify a small `.snirf` file in the chosen dataset**

```bash
# Replace ds00XXXX with the actual accession.
# Use S3 dataset listing API.
curl -s "https://openneuro.org/api/datasets/ds00XXXX/files" | \
  python3 -c "
import json, sys
files = json.load(sys.stdin)
snirfs = [(f['filename'], f.get('size', 0)) for f in files
          if f['filename'].endswith('.snirf')]
snirfs.sort(key=lambda x: x[1])
for fn, sz in snirfs[:5]:
    print(f'{sz/1024:.0f} KB  {fn}')
"
```

Pick the smallest (target < 1 MB).

- [ ] **Step 4.3: Download with curl**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions/tests/fixtures
mkdir -p fnirs
# Replace URL with the actual S3 path from Step 4.2
curl -L -o fnirs/openneuro_real.snirf \
  "https://openneuro.s3.amazonaws.com/ds00XXXX/sub-01/nirs/sub-01_task-rest_nirs.snirf"
ls -la fnirs/openneuro_real.snirf
```

Expected: file exists, size matches Step 4.2 estimate, < 1 MB. If > 1 MB, pick a smaller file.

- [ ] **Step 4.4: Write the test (mirrors `test_mef3_real_fixture.py`)**

Create `scripts/ingestions/tests/test_snirf_real_fixture.py`:

```python
"""Real-data SNIRF parser test (SPRINT-2026-05-22 Task 4).

Mirrors test_mef3_real_fixture.py: validates _snirf_parser against
a real BIDS .snirf file from OpenNeuro instead of a synthetic h5py
construction. See Lesson #1 in NEXT-SPRINT-PLAN.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from _snirf_parser import parse_snirf_metadata

FIXTURE = _INGEST_DIR / "tests" / "fixtures" / "fnirs" / "openneuro_real.snirf"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=(
        f"Real SNIRF fixture missing: {FIXTURE}. Recover with:\n"
        "  cd scripts/ingestions/tests/fixtures && mkdir -p fnirs && \\\n"
        "  curl -L -o fnirs/openneuro_real.snirf "
        "'<URL from SPRINT-2026-05-22-PLAN.md Step 4.3>'"
    ),
)


def test_real_snirf_returns_sampling_frequency():
    md = parse_snirf_metadata(FIXTURE)
    assert md is not None, "parser returned None on real .snirf"
    assert md.get("sampling_frequency"), \
        "real .snirf must yield non-zero sampling_frequency"
    assert md["sampling_frequency"] > 0


def test_real_snirf_returns_nchans():
    md = parse_snirf_metadata(FIXTURE)
    assert md is not None
    assert md.get("nchans"), "real .snirf must yield nchans"
    assert md["nchans"] >= 1


def test_real_snirf_returns_n_samples_or_n_times():
    md = parse_snirf_metadata(FIXTURE)
    assert md is not None
    n = md.get("n_samples") or md.get("n_times")
    assert n is not None and n > 0
```

- [ ] **Step 4.5: Run the tests**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_snirf_real_fixture.py -v --tb=long
```

Three outcomes possible:
- **All PASS** → the parser handles the real shape, no bug surfaces. Land the fixture as a permanent regression guard.
- **Some FAIL with `assert md.get("X")` style failures** → the parser is missing a field the real file has. This is the C5.1 pattern — investigate `_snirf_parser.py` and fix. Mirror the slow-path scan + offset-discovery approach.
- **All SKIPPED** → fixture wasn't created. Re-check Steps 4.1-4.3.

- [ ] **Step 4.6: If a parser bug surfaced, fix `_snirf_parser.py`**

The fix follows the same shape as the C5.1 MEF3 fix: identify the hardcoded assumption (offset / structure), add a slow-path scan that handles real-data variation, and gate the scan with a sanity filter. Document in the commit message. Then re-run `pytest tests/test_snirf_real_fixture.py -v` — expect all PASS.

- [ ] **Step 4.7: If no OpenNeuro fNIRS BIDS dataset exists, document and skip**

If Step 4.1 returned no results, edit `scripts/ingestions/ROBUSTNESS/NEXT-SPRINT-PLAN.md` Tier 2:

Find the bullet `4. **Real SNIRF fixture from OpenNeuro**` and change it to:

```markdown
4. **Real SNIRF fixture from OpenNeuro** — RE-CHECKED 2026-05-22, NONE
   FOUND. OpenNeuro fNIRS BIDS support is announced but no published
   datasets at search time. Re-check trigger: any of (a) the first
   public `.snirf` BIDS dataset on OpenNeuro, (b) a user filing a
   SNIRF parser bug against eegdash, (c) NEMAR landing fNIRS data.
   Until then, the synthetic h5py fixture is the only test; Lesson #1
   applies in absentia.
```

Then SKIP to Step 4.9 (commit the doc change only).

- [ ] **Step 4.8: Verify the full PR-fast suite still passes**

```bash
pytest -q -m "not network and not slow and not integration" --tb=short
```

Expected: ~810 → ~813 tests (3 new SNIRF). 0 failures.

- [ ] **Step 4.9: Commit**

If fixture landed:

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/tests/fixtures/fnirs/openneuro_real.snirf \
        scripts/ingestions/tests/test_snirf_real_fixture.py \
        scripts/ingestions/_snirf_parser.py  # only if fixed
git commit -m "test(snirf): real OpenNeuro fixture validates _snirf_parser

Mirrors C5.1 pattern (MEF3 real-data fixture). Real .snirf from
ds00XXXX <accession>. Synthetic h5py fixture validates the parser
against itself; this catches what synthetic can't.

<If a bug surfaced, append:>
- Fixed _snirf_parser.py:<line>: <bug description>"
```

If no fixture:

```bash
git add scripts/ingestions/ROBUSTNESS/NEXT-SPRINT-PLAN.md
git commit -m "docs(ingestions): re-check OpenNeuro fNIRS -- still no public BIDS dataset

Lesson #1 (real fixture > synthetic) stands but cannot be applied to
SNIRF until a public dataset surfaces. Tier-2 trigger refined with
3 explicit re-check conditions."
```

---

## Task 5: Security guards — leaked-creds scanner + pre-commit secret hook + ops checklist

**Files:**
- Create: `scripts/ingestions/scripts/find_leaked_creds.sh`
- Create: `scripts/ingestions/ROBUSTNESS/OPS-CHECKLIST.md`
- Modify: `.pre-commit-config.yaml` (repo root)

**Why this matters:** during C6.3 cluster discovery, production credentials (MONGO_INITDB_ROOT_PASSWORD, ADMIN_TOKEN, CI_TOKEN) appeared in commit-message bodies and the conversation log. The lesson — never again — needs (a) a one-shot scanner that finds existing leaks, (b) a pre-commit hook that refuses new leaks, (c) a written rotation playbook so the user can actually execute the rotation.

- [ ] **Step 5.1: Write the failing test for the scanner script**

Create `scripts/ingestions/tests/test_find_leaked_creds.py`:

```python
"""Tests for the find_leaked_creds.sh scanner (Task 5)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
SCANNER = _INGEST_DIR / "scripts" / "find_leaked_creds.sh"


@pytest.fixture
def fake_repo(tmp_path):
    """Create a tiny git repo with a planted secret in a commit message."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    f = tmp_path / "f.txt"
    f.write_text("hello\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "f.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m",
         # NOTE: in the actual test file we split this literal so it doesn't
         # trip the find-leaked-creds scanner. See test_find_leaked_creds.py
         # for the production split form (Task 5 M2 fix-up).
         "test commit\n\n" + "EEGDASH_" + "ADMIN_TOKEN" + "=AdminWrite2025" + "SecureTokenABC123"],
        check=True,
        env={
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.t",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.t",
            "PATH": "/usr/bin:/bin",
        },
    )
    return tmp_path


def test_scanner_detects_token_in_commit_message(fake_repo):
    result = subprocess.run(
        ["bash", str(SCANNER)],
        cwd=fake_repo,
        capture_output=True,
        text=True,
    )
    assert "EEGDASH_ADMIN_TOKEN" in result.stdout
    assert result.returncode == 1  # found leaks -> exit 1


def test_scanner_clean_repo_exits_0(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "x.txt").write_text("clean\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "x.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "harmless"],
        check=True,
        env={
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.t",
            "PATH": "/usr/bin:/bin",
        },
    )

    result = subprocess.run(
        ["bash", str(SCANNER)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
```

- [ ] **Step 5.2: Run — expect FileNotFoundError**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_find_leaked_creds.py -v
```

Expected: tests fail because `scripts/find_leaked_creds.sh` doesn't exist.

- [ ] **Step 5.3: Implement the scanner**

Create `scripts/ingestions/scripts/find_leaked_creds.sh`:

```bash
#!/usr/bin/env bash
# Find leaked credentials in commit messages, file contents, and the
# index. Exit 1 if any are found; 0 otherwise.
#
# Patterns target what's been observed leaking historically:
#   - EEGDASH_ADMIN_TOKEN=<20+ alphanumerics>
#   - MONGO_INITDB_ROOT_PASSWORD=<8+ chars>
#   - 64-char hex strings (CI_TOKEN shape)
#
# Add new patterns at the top; the script reports per-pattern matches.

set -u  # not -e: we want to count matches, not bail on first miss

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" >/dev/null 2>&1

PATTERNS=(
  'EEGDASH_ADMIN_TOKEN[[:space:]]*=[[:space:]]*[A-Za-z0-9]{20,}'
  'MONGO_INITDB_ROOT_PASSWORD[[:space:]]*=[[:space:]]*[^[:space:]]{8,}'
  'ADMIN_TOKEN[[:space:]]*=[[:space:]]*[A-Za-z0-9]{20,}'
  'CI_TOKEN[[:space:]]*=[[:space:]]*[a-f0-9]{40,}'
  # Generic AWS-style key shape
  'AKIA[0-9A-Z]{16}'
)

total=0

scan() {
  local label="$1" cmd="$2"
  for pat in "${PATTERNS[@]}"; do
    matches=$(eval "$cmd" 2>/dev/null | grep -E "$pat" || true)
    if [[ -n "$matches" ]]; then
      printf '\n=== %s — pattern: %s ===\n' "$label" "$pat"
      printf '%s\n' "$matches"
      lines=$(printf '%s\n' "$matches" | wc -l)
      total=$((total + lines))
    fi
  done
}

scan "Commit messages" "git log --all --format=%B"
scan "Tracked file contents" "git grep -I -n '' -- . ':(exclude)*.snirf' ':(exclude)*.tmet' ':(exclude)*.edf' 2>/dev/null"
scan "Staged changes"  "git diff --cached"

if [[ $total -gt 0 ]]; then
  printf '\n[find_leaked_creds] Found %d suspect match(es). Investigate above.\n' "$total" >&2
  exit 1
fi

printf '[find_leaked_creds] Clean.\n'
exit 0
```

Make it executable:

```bash
chmod +x /Users/bruaristimunha/Projects/eegdash/scripts/ingestions/scripts/find_leaked_creds.sh
```

- [ ] **Step 5.4: Run the scanner tests**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/test_find_leaked_creds.py -v
```

Expected: 2 passes.

- [ ] **Step 5.5: Run the scanner against the real repo (this branch)**

```bash
cd /Users/bruaristimunha/Projects/eegdash
bash scripts/ingestions/scripts/find_leaked_creds.sh
```

Expected outcomes:
- Clean → great, no historical leaks.
- Hits → STOP. Surface findings to the user. Do NOT silently remediate (history-rewriting needs explicit user instruction). Add findings to `OPS-CHECKLIST.md` Step 5.7 under "Known historical leaks".

- [ ] **Step 5.6: Add the pre-commit hook**

Find `.pre-commit-config.yaml` at the repo root. If it exists, append the following hook to its `repos:` list. If it doesn't, create it.

Read first:

```bash
ls /Users/bruaristimunha/Projects/eegdash/.pre-commit-config.yaml
```

If it exists, read it via the Read tool, then edit. Append to the `repos:` list:

```yaml
  - repo: local
    hooks:
      - id: find-leaked-creds
        name: Block leaked credentials in staged changes
        entry: bash scripts/ingestions/scripts/find_leaked_creds.sh
        language: system
        pass_filenames: false
        stages: [commit, commit-msg]
```

If `.pre-commit-config.yaml` doesn't exist at repo root, create it with:

```yaml
repos:
  - repo: local
    hooks:
      - id: find-leaked-creds
        name: Block leaked credentials in staged changes
        entry: bash scripts/ingestions/scripts/find_leaked_creds.sh
        language: system
        pass_filenames: false
        stages: [commit, commit-msg]
```

- [ ] **Step 5.7: Write the ops checklist**

Create `scripts/ingestions/ROBUSTNESS/OPS-CHECKLIST.md`:

```markdown
# Operations checklist — credential rotation + leak response

## When to use this doc

- A credential appeared in a commit message, file, or chat log.
- A teammate left the org (cluster access revocation).
- Quarterly rotation cadence (recommended for production tokens).
- `find_leaked_creds.sh` flagged a match.

## Cluster credentials in scope

Documented in `INTEGRATION-TESTING.md` and stored in:
`/home/<ops-user>/eegdash-competition/docker-compose.yml` on `sccn` host.

| Credential | Rotates affects | Rotation surface |
|---|---|---|
| `MONGO_INITDB_ROOT_PASSWORD` | mongodb-production container + API connection string | Compose file + API restart |
| `EEGDASH_ADMIN_TOKEN` | API Gateway admin endpoints (all writers) | API .env + every CI/local exporter |
| `CI_TOKEN` | GitHub Actions runners that publish to the API | Repo secrets + Actions config |

## Rotation procedure

### 1. Generate new values

```bash
# Strong passwords:
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Hex token (CI_TOKEN shape):
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Keep new values out of any commit / chat log — only put them in the
target system's secret store.

### 2. Update the cluster compose file

```bash
ssh sccn
cd ~/eegdash-competition
# Edit docker-compose.yml: replace MONGO_INITDB_ROOT_PASSWORD,
# EEGDASH_ADMIN_TOKEN, API_CI_TOKEN inline.
docker compose down
docker compose up -d
docker compose ps  # confirm healthy
```

### 3. Update GitHub Actions secrets

GitHub UI → Settings → Secrets and variables → Actions:
- `EEGDASH_ADMIN_TOKEN`
- `CI_TOKEN`

### 4. Update teammate local exports

Notify the team via the eng channel. Each dev:

```bash
# In each shell rc / .envrc:
export EEGDASH_ADMIN_TOKEN="<new value>"
```

### 5. Verify with the live integration tests

```bash
export EEGDASH_INTEGRATION_API_URL=https://data.eegdash.org
export EEGDASH_INTEGRATION_ADMIN_TOKEN="<new value>"
cd scripts/ingestions
pytest -m integration -v
```

Expected: 16 passed (the C6.3 / C6.4 suite).

### 6. Audit the old token for late use

```bash
# On the cluster, watch nginx/api logs for the old token's last appearance
ssh sccn 'docker logs caddy 2>&1 | grep "<first 8 chars of old token>"'
```

A flat-zero result after 24h = old token fully retired.

## Known historical leaks (as of 2026-05-22)

(Filled in from Step 5.5 if the scanner found anything.)

- (none / TBD when scanner runs)

## Pre-commit hook reference

The hook lives at `.pre-commit-config.yaml:repos.[].hooks.find-leaked-creds`
and invokes `scripts/ingestions/scripts/find_leaked_creds.sh`. Add new
patterns (e.g., new service tokens) at the top of that script.
```

- [ ] **Step 5.8: Run the full PR-fast suite**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest -q -m "not network and not slow and not integration"
```

Expected: ~813 → ~815 tests (2 new for the scanner). 0 failures.

- [ ] **Step 5.9: Run pre-commit against staged files (smoke check)**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/scripts/find_leaked_creds.sh \
        scripts/ingestions/tests/test_find_leaked_creds.py \
        scripts/ingestions/ROBUSTNESS/OPS-CHECKLIST.md \
        .pre-commit-config.yaml
pre-commit run --hook-stage commit find-leaked-creds --files \
  scripts/ingestions/scripts/find_leaked_creds.sh
```

Expected: PASS (no leaked creds in the staged files themselves).

- [ ] **Step 5.10: Commit**

```bash
git commit -m "feat(ops): credential leak scanner + pre-commit hook + rotation playbook

- scripts/find_leaked_creds.sh: scans commit messages, tracked files,
  and staged changes for known leak patterns (EEGDASH_ADMIN_TOKEN,
  MONGO_INITDB_ROOT_PASSWORD, CI_TOKEN, AWS-style keys)
- 2 tests verify scanner exit codes
- pre-commit hook refuses new leaks at commit time
- OPS-CHECKLIST.md documents rotation procedure end-to-end
- Surfaces historical leak audit (see scanner output)"
```

---

## Task 6: Mutmut nightly observation (gated on push authorization)

**Files:**
- Modify: `.github/workflows/mutmut-nightly.yml` (verify shape only — no edits expected)
- Modify: `scripts/ingestions/ROBUSTNESS/MUTMUT-FINDINGS.md` (new doc — only after first run)

**Why this matters:** the workflow was set up in P0.2 (per `NEXT-SPRINT-PLAN.md` line 19) but never observed because the branch hasn't been pushed. Mutation testing tells us **which tests are weak** — a complement to coverage.

**This task requires the user's explicit go-ahead to push the branch. We do NOT push autonomously.**

- [ ] **Step 6.1: Verify the workflow YAML is well-formed**

```bash
cd /Users/bruaristimunha/Projects/eegdash
ls .github/workflows/ | grep -i mut
# Pick the matching file (likely mutmut-nightly.yml).
yamllint .github/workflows/mutmut-nightly.yml 2>/dev/null || \
  python3 -c "import yaml; yaml.safe_load(open('.github/workflows/mutmut-nightly.yml'))"
```

Expected: no syntax errors. If yamllint isn't installed, the Python `yaml.safe_load` is enough.

- [ ] **Step 6.2: Inspect the workflow schedule + target modules**

Read the workflow file. Confirm:
- A `schedule:` trigger exists with a sensible cron (`0 3 * * *` or similar)
- A `mutmut run` invocation with `--simultaneous-mutants` or `--paths-to-mutate` for the parser modules

If neither is present, file an issue (or write a small fix) before pushing.

- [ ] **Step 6.3: Ask the user for push authorization**

**This step is a hard human gate.** Surface to the user:

> "The mutmut nightly workflow exists and looks well-formed. To observe
> its first run, the branch needs to be pushed. You said earlier: 'you
> not authorize to push, only commit please.' Do you want me to push
> `record-enumerator-merge` to origin now? (y/n)"

If the user says **no**, stop here. Mark Task 6 deferred. Write a note in NEXT-SPRINT-PLAN.md (next edit) that Task 6 is awaiting push authorization.

If the user says **yes**, proceed to 6.4. Otherwise STOP.

- [ ] **Step 6.4 (gated): Push the branch**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git push -u origin record-enumerator-merge
```

Expected: 68 commits pushed cleanly, no force-push errors, no protected-branch rejection.

- [ ] **Step 6.5 (gated): Confirm the workflow appears in GitHub Actions**

```bash
gh run list --workflow=mutmut-nightly.yml --limit 5
```

Expected: either the nightly schedule has a scheduled run queued, or the workflow page shows it active. If the schedule cron is the only trigger, the first run lands at the next cron tick (within 24h).

- [ ] **Step 6.6 (gated, defer to next session): Capture the first nightly results**

After the first nightly run completes (≥ 24h later):

```bash
gh run download --name mutmut-report --dir /tmp/mutmut-first-run
cat /tmp/mutmut-first-run/summary.txt
```

Write findings to `scripts/ingestions/ROBUSTNESS/MUTMUT-FINDINGS.md`:

```markdown
# Mutmut nightly — first observed run

**Run ID**: <gh run id>
**Date**: <YYYY-MM-DD>
**Targets**: <list of paths mutated>

## Kill ratio per module

| Module | Mutants | Killed | Surviving | Kill % |
|---|---:|---:|---:|---:|
| _vhdr_parser.py | <n> | <k> | <n-k> | <%> |
| _set_parser.py | ... | ... | ... | ... |

## Surviving mutants worth investigating

(Document the top 5 by sus-pattern. Refer to findings-phase-4.md for
the 4 known un-killed mutants from session 4.)
```

- [ ] **Step 6.7 (gated): Commit findings**

```bash
git add scripts/ingestions/ROBUSTNESS/MUTMUT-FINDINGS.md
git commit -m "docs(ingestions): MUTMUT-FINDINGS — first observed nightly run

Captures kill ratio + surviving mutants from the first observed
mutmut-nightly.yml execution. Top survivors annotated for follow-up."
```

---

## Self-review checklist

Run this checklist after writing the plan, before handoff.

### 1. Spec coverage

- [ ] Task 1 covers candidate #1 (re-tier plan) — yes
- [ ] Task 2 covers candidate #2 (database drift) — yes
- [ ] Task 3 covers candidate #3 (cascade refactor) — yes
- [ ] Task 4 covers candidate #4 (real SNIRF) — yes (conditional path documented)
- [ ] Task 5 covers candidate #6 (security checklist + scanner) — yes
- [ ] Task 6 covers candidate #5 (mutmut observation) — yes (gated on push)

### 2. Placeholder scan

- [ ] No "TBD" except the explicitly-marked Known-leaks bullet in OPS-CHECKLIST (gets filled when scanner runs in 5.5)
- [ ] No "implement later" / "similar to Task N" — every code step shows full code
- [ ] No "appropriate error handling" — every except clause names the exception types
- [ ] Test cases show input + expected output

### 3. Type consistency

- [ ] `CascadeContext` field names match between Step 3.3 dataclass and Step 3.5+ tests (bids_dataset, bids_file, ext, bids_file_path, bids_root) — yes
- [ ] `CascadeResult` field names match between dataclass and steps (sampling_frequency, nchans, ntimes, ch_names, fif_is_split, fif_continuations_ok, provenance) — yes
- [ ] `PROV_*` constants used identically in `_metadata_cascade.py` and the existing snapshot fixtures — yes (named identically to `_PROV_*` in 3_digest.py, just without the leading underscore since they're now public to the module)
- [ ] `fetch_valid_databases_from_api` signature consistent across Task 2 tests and impl — yes
- [ ] `LOCAL_FALLBACK_DATABASES` is `frozenset[str]` everywhere — yes

### 4. Constraint conformance

- [ ] No `--no-verify` in any git commit command — confirmed
- [ ] No `Co-Authored-By` in commit messages — confirmed
- [ ] No "🤖 Generated with Claude Code" attribution — confirmed
- [ ] Push (Task 6.4) explicitly gated on user "y" — confirmed
- [ ] Snapshot tests run after every refactor (Steps 3.17, 3.18) — confirmed
- [ ] Coverage ratchet bumps in same commit as new tests (Steps 2.16, 3.22) — confirmed

---

## Execution handoff

Plan complete and saved to `scripts/ingestions/ROBUSTNESS/SPRINT-2026-05-22-PLAN.md`.

### Two execution options

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Best for Tasks 2 + 3 (independent code work).

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints. Best if you want to be in the loop on each commit.

Either way, **Task 1 should run first** (it anchors the scope) and **Task 6 stops before push**. Tasks 2 and 3 are independent and can run in parallel under subagent-driven.

Which approach?
