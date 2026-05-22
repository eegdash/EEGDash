# Acceptance Tests — Implementation Plan (Layers 2–4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock the pipeline output behind three orthogonal acceptance layers — Pydantic schema validation (Layer 2), idempotency (Layer 3), and API-side contract validation (Layer 4) — so future refactors can't silently change Record/Dataset/Montage shape, introduce non-determinism, or drift away from the cluster's accepted document layout.

**Architecture:** A new `tests/acceptance/` directory with a per-layer test module. Layer 2 loads every snapshot fixture's output and validates each Record / Dataset against the producer-side Pydantic models in `eegdash/schemas.py` (which already exist as `RecordModel`, `DatasetModel`). Layer 3 runs `digest_dataset` twice on the same input and asserts byte-identical JSON output (uses the existing snapshot fixtures, plus the real MEF3 and SNIRF fixtures). Layer 4 imports the API's Beanie `Record` Document from `mongodb-eegdash-server/api/main.py` via `sys.path` injection (same pattern the rest of the test suite already uses for cross-package imports) and validates digest output against the cluster's effective contract.

**Tech Stack:** Python 3.11+, Pydantic v2 (`eegdash.schemas`), pytest, pytest-mock, existing snapshot fixtures (`tests/fixtures/digest_snapshots/`), existing real-data fixtures (`tests/fixtures/ieeg/`, `tests/fixtures/fnirs/`).

**Branch + constraints:**
- Branch `record-enumerator-merge` (last commit `4aa3162fa` — post-perf-sprint).
- 851 PR-fast tests pass today; this sprint adds ~12-18 new tests (per-layer detail below).
- Snapshot tests must continue passing byte-identical (these new tests are additive).
- Never `Co-Authored-By`. Never robot attribution. Never `--no-verify`. Do NOT push.
- Coverage floor 61%; not expected to drop.

---

## File structure (new files)

```
scripts/ingestions/tests/acceptance/
├─ __init__.py
├─ conftest.py                ← shared fixtures (load snapshot outputs)
├─ test_schema_acceptance.py  ← Layer 2: producer Pydantic models
├─ test_idempotency.py        ← Layer 3: digest twice → byte-identical
└─ test_api_contract.py       ← Layer 4: vendor + validate against API Document
```

Total: 5 new files, ~250-300 LOC, organised by acceptance layer (one responsibility per file).

---

## Task ordering

```
Task 0  Scaffolding: tests/acceptance/ + conftest.py
Task 1  Layer 2 — Pydantic schema acceptance
Task 2  Layer 3 — Idempotency
Task 3  Layer 4 — API contract
```

Each task lands one focused commit. Tasks 1-3 are independent (no inter-task dependency beyond Task 0's scaffolding).

---

## Task 0: Scaffolding — `tests/acceptance/` + `conftest.py`

**Files:**
- Create: `scripts/ingestions/tests/acceptance/__init__.py`
- Create: `scripts/ingestions/tests/acceptance/conftest.py`

**Why this first:** Layers 2-4 all need the same loader helpers (read a snapshot fixture's `_dataset.json` / `_records.json` / `_montages.json` / `_summary.json`). DRY them into `conftest.py` once.

- [ ] **Step 0.1: Create the package directory**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
mkdir -p tests/acceptance
touch tests/acceptance/__init__.py
```

- [ ] **Step 0.2: Write the failing test (drives the conftest fixtures)**

Create `tests/acceptance/test_scaffold_smoke.py` (TEMPORARY — deleted at end of Task 0 so it doesn't survive the sprint):

```python
"""Smoke test that the acceptance conftest fixtures load fixture output."""

from __future__ import annotations


def test_snapshot_outputs_loadable(snapshot_outputs):
    """conftest.snapshot_outputs returns a dict keyed by fixture name with the
    four JSON files (dataset, records, montages, summary) loaded."""
    assert "ds_snapshot_vhdr" in snapshot_outputs
    fixture = snapshot_outputs["ds_snapshot_vhdr"]
    assert "dataset" in fixture
    assert "records" in fixture
    assert "montages" in fixture
    assert "summary" in fixture
    assert fixture["dataset"]["dataset_id"] == "ds_snapshot_vhdr"
    assert isinstance(fixture["records"], list)
    assert len(fixture["records"]) > 0


def test_manifest_fixture_loadable(snapshot_outputs):
    """The manifest-only snapshot has the same shape."""
    assert "ds_snapshot_manifest" in snapshot_outputs
    fixture = snapshot_outputs["ds_snapshot_manifest"]
    assert fixture["dataset"]["dataset_id"] == "ds_snapshot_manifest"
```

- [ ] **Step 0.3: Run — expect fixture-not-found error**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/acceptance/test_scaffold_smoke.py -v --tb=short 2>&1 | tail -10
```

Expected: pytest reports the `snapshot_outputs` fixture is undefined.

- [ ] **Step 0.4: Implement the conftest**

Create `scripts/ingestions/tests/acceptance/conftest.py`:

```python
"""Shared fixtures for acceptance tests.

The acceptance suite loads outputs from the snapshot fixtures
(``tests/fixtures/digest_snapshots/outputs/``) and validates them
against three layers:

* Layer 2 — producer-side Pydantic models (``eegdash.schemas``).
* Layer 3 — re-running ``digest_dataset`` against the input twice
  yields byte-identical output.
* Layer 4 — API-side Beanie document
  (``mongodb-eegdash-server/api/main.py``).

This conftest exposes the loader fixtures every layer reuses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Path layout:
#   scripts/ingestions/tests/acceptance/conftest.py  ← this file
#   scripts/ingestions/                              ← INGEST_DIR
#   scripts/ingestions/tests/fixtures/digest_snapshots/outputs/
_INGEST_DIR = Path(__file__).resolve().parent.parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

_SNAPSHOT_OUTPUTS = (
    _INGEST_DIR / "tests" / "fixtures" / "digest_snapshots" / "outputs"
)


def _load_dataset_outputs(dataset_dir: Path) -> dict[str, Any] | None:
    """Load the four JSON files emitted by ``write_dataset_outputs``.

    Returns ``None`` if any of the four files are missing — the caller
    can then skip the fixture (e.g. if a future snapshot only emits
    three files because the dataset has no montages).
    """
    dataset_id = dataset_dir.name
    out: dict[str, Any] = {}
    for kind in ("dataset", "records", "montages", "summary"):
        path = dataset_dir / f"{dataset_id}_{kind}.json"
        if not path.exists():
            return None
        out[kind] = json.loads(path.read_text())
    return out


@pytest.fixture(scope="session")
def snapshot_outputs() -> dict[str, dict[str, Any]]:
    """Load every snapshot fixture's outputs.

    Returned dict is keyed by fixture name (e.g. ``"ds_snapshot_vhdr"``);
    each value has keys ``dataset`` / ``records`` / ``montages`` / ``summary``.

    The Layer-3 idempotency test produces its own outputs in a tmp
    dir; the snapshot outputs here are the *committed* baselines that
    Layers 2 + 4 validate.
    """
    if not _SNAPSHOT_OUTPUTS.exists():
        pytest.skip(
            f"Snapshot output dir missing: {_SNAPSHOT_OUTPUTS}. "
            f"Run `pytest tests/test_digest_snapshot.py` to generate, "
            f"or recover via tests/fixtures/digest_snapshots/README.md."
        )
    out: dict[str, dict[str, Any]] = {}
    for child in _SNAPSHOT_OUTPUTS.iterdir():
        if not child.is_dir():
            continue
        loaded = _load_dataset_outputs(child)
        if loaded is not None:
            out[child.name] = loaded
    if not out:
        pytest.skip(
            f"No complete snapshot outputs found under {_SNAPSHOT_OUTPUTS}"
        )
    return out


@pytest.fixture(scope="session")
def ingest_dir() -> Path:
    """Absolute path to ``scripts/ingestions/``."""
    return _INGEST_DIR
```

- [ ] **Step 0.5: Run the smoke test — expect PASS**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/acceptance/test_scaffold_smoke.py -v --tb=short
```

Expected: 2 passed.

- [ ] **Step 0.6: Delete the smoke test (its purpose was to drive the conftest)**

```bash
rm tests/acceptance/test_scaffold_smoke.py
```

- [ ] **Step 0.7: Verify the conftest doesn't break existing collection**

```bash
pytest -q -m "not network and not slow and not integration" --collect-only 2>&1 | tail -3
```

Expected: collection count rises by 0 net (smoke test deleted). No errors.

- [ ] **Step 0.8: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/tests/acceptance/
git commit -m "test(acceptance): scaffold tests/acceptance/ package + shared conftest

Foundation for three acceptance-test layers (Pydantic schemas,
idempotency, API contract) that follow in subsequent commits. This
commit lands the package skeleton + the shared snapshot-output
loader fixture used by all three layers.

- tests/acceptance/__init__.py
- tests/acceptance/conftest.py with snapshot_outputs and ingest_dir
  session-scoped fixtures"
```

---

## Task 1: Layer 2 — Pydantic schema acceptance

**Files:**
- Create: `scripts/ingestions/tests/acceptance/test_schema_acceptance.py`

**Why this matters:** `eegdash/schemas.py` defines `RecordModel`, `DatasetModel`, and `StorageModel` Pydantic models. They are NOT currently exercised against pipeline output — the producer factory (`create_record`, `create_dataset`) emits `TypedDict`-shaped dicts that are never round-tripped through the matching Pydantic model. A refactor that renames `bids_relpath` to `bids_path` in the factory would not break any existing test. This layer closes that gap.

**Models in scope** (from `eegdash/schemas.py`):
- `StorageModel` (line 103)
- `EntitiesModel` (line 126)
- `RecordModel` (line 138)
- `DatasetModel` (line 158)
- `ManifestFileModel` (line 179)
- `ManifestModel` (line 192)

Records and Datasets are the load-bearing two. Montages don't have a corresponding Pydantic model today (TypedDict only); we'll add a JSON-shape assertion instead.

- [ ] **Step 1.1: Write the failing tests**

Create `scripts/ingestions/tests/acceptance/test_schema_acceptance.py`:

```python
"""Layer 2 — Pydantic schema acceptance.

For every Record / Dataset / Montage in every snapshot fixture,
validate it against the producer-side Pydantic model in
``eegdash.schemas``. If a refactor renames a field or changes a
type, this layer catches it without needing a byte-snapshot refresh.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from eegdash.schemas import (
    DatasetModel,
    EntitiesModel,
    RecordModel,
    StorageModel,
)


# ─── Records ──────────────────────────────────────────────────────────────


def test_every_record_validates_against_RecordModel(snapshot_outputs):
    """Every Record in every snapshot fixture must validate.

    Catches: field renames, type changes (str → int), missing required
    fields, extra fields that the model doesn't expect (RecordModel
    uses ``extra="allow"`` so this last one is a soft check).
    """
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]):
            try:
                RecordModel.model_validate(record)
            except ValidationError as exc:
                failures.append(
                    f"{fixture_name}[{idx}] ({record.get('bids_relpath', '?')}): {exc}"
                )
    assert not failures, "RecordModel rejected:\n" + "\n".join(failures)


def test_every_record_storage_validates_against_StorageModel(snapshot_outputs):
    """The nested ``storage`` block has its own Pydantic model."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]):
            storage = record.get("storage")
            if storage is None:
                failures.append(f"{fixture_name}[{idx}]: missing storage block")
                continue
            try:
                StorageModel.model_validate(storage)
            except ValidationError as exc:
                failures.append(f"{fixture_name}[{idx}].storage: {exc}")
    assert not failures, "StorageModel rejected:\n" + "\n".join(failures)


def test_every_record_entities_validates_against_EntitiesModel(snapshot_outputs):
    """The ``entities`` block (subject / session / task / run / acq) is
    optional but, when present, must satisfy ``EntitiesModel``."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]):
            entities = record.get("entities")
            if entities is None:
                continue  # not all records carry entities (manifest-only path)
            try:
                EntitiesModel.model_validate(entities)
            except ValidationError as exc:
                failures.append(f"{fixture_name}[{idx}].entities: {exc}")
    assert not failures, "EntitiesModel rejected:\n" + "\n".join(failures)


# ─── Datasets ─────────────────────────────────────────────────────────────


def test_every_dataset_validates_against_DatasetModel(snapshot_outputs):
    """Every Dataset doc must validate against DatasetModel."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        try:
            DatasetModel.model_validate(payload["dataset"])
        except ValidationError as exc:
            failures.append(f"{fixture_name}: {exc}")
    assert not failures, "DatasetModel rejected:\n" + "\n".join(failures)


# ─── Montages (no Pydantic model — JSON-shape acceptance) ─────────────────


_REQUIRED_MONTAGE_FIELDS: tuple[str, ...] = (
    "montage_hash",
    "n_channels",
    "channels",
    "first_seen",
    "representative_dataset",
)


def test_every_montage_has_required_fields(snapshot_outputs):
    """Montages don't yet have a Pydantic model; assert the JSON keys
    we know the API consumer indexes on are present."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        montages: dict[str, Any] = payload["montages"]
        for hash_key, doc in montages.items():
            for field in _REQUIRED_MONTAGE_FIELDS:
                if field not in doc:
                    failures.append(
                        f"{fixture_name}.montages[{hash_key}]: missing {field}"
                    )
    assert not failures, "Montage shape check failed:\n" + "\n".join(failures)


def test_every_montage_hash_keys_its_own_doc(snapshot_outputs):
    """Dict key (the hash) must equal the embedded ``montage_hash`` field."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        montages: dict[str, Any] = payload["montages"]
        for hash_key, doc in montages.items():
            embedded = doc.get("montage_hash")
            if embedded != hash_key:
                failures.append(
                    f"{fixture_name}.montages[{hash_key}]: "
                    f"embedded hash={embedded!r} mismatches key"
                )
    assert not failures, "Montage hash-key drift:\n" + "\n".join(failures)
```

- [ ] **Step 1.2: Run — expect PASS or surfaced drift**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/acceptance/test_schema_acceptance.py -v --tb=long 2>&1 | tail -30
```

There are two acceptable outcomes:

- **All 6 pass:** the producer factories already emit shapes that round-trip through the Pydantic models. Land the test as the new floor.
- **One or more fail:** a real drift exists between producer factory output and the Pydantic model. Investigate the failure message. If the producer is wrong, fix `create_record` / `create_dataset` in `eegdash/schemas.py`. If the model is wrong, fix the model. **Do NOT update the snapshot fixtures** — the failure is what the test is for.

- [ ] **Step 1.3: Verify the full PR-fast suite is still green**

```bash
pytest -q -m "not network and not slow and not integration" --tb=short 2>&1 | tail -5
```

Expected: 851 → 857 (+6 new tests). 0 failures.

- [ ] **Step 1.4: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/tests/acceptance/test_schema_acceptance.py
git commit -m "test(acceptance): Layer 2 — Pydantic schema validation on snapshot outputs

Every Record / Dataset / nested Storage + Entities block in every
snapshot fixture validates against the producer-side Pydantic model
in eegdash.schemas. Closes the gap where the factory emits dicts
that were never round-tripped through the corresponding Pydantic
model (a rename like bids_relpath -> bids_path would have passed
every existing test).

Montages don't yet have a Pydantic model; this layer adds JSON-key
acceptance checks instead (required-fields, hash-key-vs-embedded
self-consistency)."
```

---

## Task 2: Layer 3 — Idempotency

**Files:**
- Create: `scripts/ingestions/tests/acceptance/test_idempotency.py`

**Why this matters:** `digest_dataset` is supposed to be deterministic — running it twice on the same input must produce byte-identical output. The existing snapshot tests pin a single run; they don't catch non-determinism that only surfaces across runs (e.g., `set()` iteration order, dict insertion order on Python 3.6 vs 3.7+, datetime field that subtly leaks `now()` instead of using the captured `digested_at`). Layer 3 runs the digester twice and diffs.

- [ ] **Step 2.1: Write the failing tests**

Create `scripts/ingestions/tests/acceptance/test_idempotency.py`:

```python
"""Layer 3 — Idempotency acceptance.

``digest_dataset`` is deterministic: same input + same ``digested_at``
must produce byte-identical output. Catches:

* Hidden non-determinism (set iteration, dict ordering on old Python).
* Leaked timestamps (e.g. ``datetime.now()`` in a field that should
  use the captured ``digested_at`` parameter).
* Floating-point reductions whose order changes with the worker pool
  (relevant if Stage 3 ever moves to per-record concurrency).

The Layer-1 snapshot tests run ``digest_dataset`` once and assert
byte-identical with a committed baseline. This layer runs it TWICE
in the same process and diffs.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_digest_module(ingest_dir: Path):
    """Load ``3_digest.py`` via importlib (digit prefix forbids regular import)."""
    spec = importlib.util.spec_from_file_location(
        "digest_under_test_idempotency", ingest_dir / "3_digest.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_digest_against(
    fixture_input: Path,
    output_dir: Path,
    digest_mod,
    digested_at: str,
) -> dict[str, Any]:
    """Run ``digest_dataset`` and return the 4 JSON outputs as a dict.

    We pin ``digested_at`` so the deterministic test isn't fooled by
    wall-clock drift between two consecutive runs.
    """
    # Pin time so the test asserts byte-identical output across the
    # two runs — without this, fields like dataset.digested_at and
    # every record's digested_at would naturally differ.
    import datetime as _dt

    real_datetime = _dt.datetime

    class _FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003
            return real_datetime.fromisoformat(digested_at.replace("Z", "+00:00"))

    _dt.datetime = _FixedDatetime  # type: ignore[misc]
    try:
        digest_mod.digest_dataset(
            fixture_input.name, fixture_input.parent, output_dir
        )
    finally:
        _dt.datetime = real_datetime  # type: ignore[misc]

    out_dir = output_dir / fixture_input.name
    result: dict[str, Any] = {}
    for kind in ("dataset", "records", "montages", "summary"):
        path = out_dir / f"{fixture_input.name}_{kind}.json"
        result[kind] = json.loads(path.read_text())
    return result


@pytest.mark.parametrize(
    "fixture_name",
    ["ds_snapshot_vhdr", "ds_snapshot_manifest"],
)
def test_digest_dataset_is_byte_idempotent(
    fixture_name: str, ingest_dir: Path, tmp_path: Path
):
    """Run ``digest_dataset`` twice on the same input. Byte-identical."""
    fixture_input = (
        ingest_dir / "tests" / "fixtures" / "digest_snapshots" / "inputs" / fixture_name
    )
    if not fixture_input.exists():
        pytest.skip(f"Fixture missing: {fixture_input}")

    digest_mod = _load_digest_module(ingest_dir)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    out_a.mkdir()
    out_b.mkdir()

    digested_at = "2026-05-22T12:00:00+00:00"

    result_a = _run_digest_against(fixture_input, out_a, digest_mod, digested_at)
    result_b = _run_digest_against(fixture_input, out_b, digest_mod, digested_at)

    # Byte-identical at the JSON level (key order, value formatting).
    for kind in ("dataset", "records", "montages", "summary"):
        text_a = json.dumps(result_a[kind], sort_keys=True, indent=2)
        text_b = json.dumps(result_b[kind], sort_keys=True, indent=2)
        assert text_a == text_b, (
            f"{fixture_name}.{kind}.json differed across two runs "
            f"with the same digested_at. Non-deterministic output."
        )


@pytest.mark.parametrize(
    "fixture_name",
    ["ds_snapshot_vhdr", "ds_snapshot_manifest"],
)
def test_record_count_is_stable_across_runs(
    fixture_name: str, ingest_dir: Path, tmp_path: Path
):
    """Lighter-weight idempotency: record count alone.

    Catches missing files (records dropped between runs) or duplicates
    (records added between runs) even if some field varies."""
    fixture_input = (
        ingest_dir / "tests" / "fixtures" / "digest_snapshots" / "inputs" / fixture_name
    )
    if not fixture_input.exists():
        pytest.skip(f"Fixture missing: {fixture_input}")

    digest_mod = _load_digest_module(ingest_dir)
    digested_at = "2026-05-22T12:00:00+00:00"

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    out_a.mkdir()
    out_b.mkdir()

    result_a = _run_digest_against(fixture_input, out_a, digest_mod, digested_at)
    result_b = _run_digest_against(fixture_input, out_b, digest_mod, digested_at)

    assert len(result_a["records"]) == len(result_b["records"]), (
        f"{fixture_name}: record count differs between runs "
        f"(A={len(result_a['records'])}, B={len(result_b['records'])})"
    )


def test_ingestion_fingerprint_is_stable_across_runs(
    ingest_dir: Path, tmp_path: Path
):
    """The dataset's ``ingestion_fingerprint`` is content-derived and
    must NOT change across two runs of the same input."""
    fixture_input = (
        ingest_dir / "tests" / "fixtures" / "digest_snapshots" / "inputs"
        / "ds_snapshot_vhdr"
    )
    if not fixture_input.exists():
        pytest.skip(f"Fixture missing: {fixture_input}")

    digest_mod = _load_digest_module(ingest_dir)
    digested_at = "2026-05-22T12:00:00+00:00"

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    out_a.mkdir()
    out_b.mkdir()

    result_a = _run_digest_against(fixture_input, out_a, digest_mod, digested_at)
    result_b = _run_digest_against(fixture_input, out_b, digest_mod, digested_at)

    fp_a = result_a["dataset"].get("ingestion_fingerprint")
    fp_b = result_b["dataset"].get("ingestion_fingerprint")
    assert fp_a is not None, "ingestion_fingerprint missing from dataset"
    assert fp_a == fp_b, (
        f"ingestion_fingerprint changed across runs: {fp_a!r} → {fp_b!r}. "
        f"The fingerprint is supposed to be content-derived; this is a "
        f"non-determinism bug."
    )
```

- [ ] **Step 2.2: Run — expect PASS (or surfaced non-determinism)**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/acceptance/test_idempotency.py -v --tb=long 2>&1 | tail -25
```

Two acceptable outcomes:

- **All 5 pass** (2 fixtures × byte-test + 2 fixtures × count-test + 1 fingerprint-test = 5 tests): digest is deterministic. Land the test as the floor.
- **One or more fail:** a real non-determinism exists. Read the failure message — likely candidates: a `set()` of strings, a `dict()` of strings, a `datetime.now()` leak. Fix in `3_digest.py` (or wherever the source is). **Do NOT make the test more permissive** — the failure is the entire reason this layer exists.

- [ ] **Step 2.3: Run the snapshot tests + full PR-fast suite — gate**

```bash
pytest tests/test_digest_snapshot.py -v --tb=short 2>&1 | tail -5
pytest -q -m "not network and not slow and not integration" --tb=short 2>&1 | tail -5
```

Expected: 21/21 snapshot byte-identical (Layer 3 didn't touch production code unless a fix was needed in Step 2.2); 857 → 862 PR-fast (+5 new tests).

- [ ] **Step 2.4: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/tests/acceptance/test_idempotency.py
git commit -m "test(acceptance): Layer 3 — digest_dataset idempotency

Run digest_dataset twice in the same process on the same input
with a pinned digested_at, and assert byte-identical output for
the four JSON files plus the ingestion_fingerprint.

Catches non-determinism that single-run snapshot tests can't see:
set iteration order, dict insertion order on old Python, a
datetime.now() that should have been digested_at, floating-point
reductions whose order changes with worker count."
```

---

## Task 3: Layer 4 — API contract

**Files:**
- Create: `scripts/ingestions/tests/acceptance/test_api_contract.py`

**Why this matters:** The cluster's API in `mongodb-eegdash-server/api/main.py` defines the consumer's effective contract — a Beanie `Record(Document)` at line 142 with `extra="allow"`, indexed on `data_name` / `dataset` / `(dataset, subject)` / `(dataset, bids_relpath)`. If a producer refactor accidentally drops `dataset` or `bids_relpath` from a Record, the bulk-upsert path breaks (indexes can't be built). The producer-side `RecordModel` (Layer 2) is stricter, but the API-side document is the actual cluster gate.

This layer also imports the API's response models (`RecordModel`, `DatasetModel` declared at `api/main.py:138, 158`) — which ARE Pydantic v2 models, distinct from the producer-side models with the same name. **Different repos have models with the same name; tests must validate against BOTH.**

- [ ] **Step 3.1: Confirm the API module is importable**

```bash
cd /Users/bruaristimunha/Projects/eegdash
ls mongodb-eegdash-server/api/main.py
python3 -c "
import sys; sys.path.insert(0, 'mongodb-eegdash-server/api')
# main.py expects MOTOR / Beanie / etc. — heavy deps. Import only the
# Pydantic models we need by patching motor at import time.
" 2>&1 | tail -5
```

The API module's imports (`motor`, `beanie`, `pymongo`) are likely available in the dev environment but may pull in connection-attempt side effects at import time. If `main.py` only DEFINES the models without doing top-level connection work, plain import is fine. If it has top-level side effects, we'll need to import just the model definitions.

Quick check:

```bash
cd /Users/bruaristimunha/Projects/eegdash
grep -n '^client\s*=\|^db\s*=\|^app\s*=\|connect()' mongodb-eegdash-server/api/main.py | head -10
```

If `app = FastAPI(...)` or similar lives at module top, the module imports cleanly but creates the app object. Acceptable for test use.

- [ ] **Step 3.2: Write the failing tests**

Create `scripts/ingestions/tests/acceptance/test_api_contract.py`:

```python
"""Layer 4 — API contract acceptance.

The cluster's API in ``mongodb-eegdash-server/api/main.py`` defines
the consumer's effective contract. This layer imports the API's
Pydantic models from that file and validates every snapshot
fixture's records / datasets against them.

Distinct from Layer 2 (producer-side models in ``eegdash.schemas``)
because the two repos have independently-evolving models with the
same class names. Layer 2 catches producer drift; Layer 4 catches
producer/consumer drift.

The API module also defines a Beanie ``Record(Document)`` with
indexes on specific fields — we don't instantiate Beanie (needs a
MongoDB connection) but we DO assert the indexed fields exist on
every Record.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError


# Path to the API module, sibling of the eegdash repo.
_API_DIR = Path(__file__).resolve().parents[4] / "mongodb-eegdash-server" / "api"


@pytest.fixture(scope="session")
def api_module():
    """Lazy-load ``mongodb-eegdash-server/api/main.py`` for its Pydantic models.

    Skips the layer entirely when the API repo isn't checked out as
    a sibling — CI without it shouldn't fail this gate.
    """
    if not _API_DIR.exists():
        pytest.skip(
            f"API repo not found at {_API_DIR}. "
            f"Clone mongodb-eegdash-server as a sibling of this repo "
            f"to enable Layer 4 acceptance."
        )

    if str(_API_DIR) not in sys.path:
        sys.path.insert(0, str(_API_DIR))

    spec = importlib.util.spec_from_file_location(
        "api_main_under_test", _API_DIR / "main.py"
    )
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not load {_API_DIR / 'main.py'}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        pytest.skip(f"API module has unmet imports (likely beanie/motor): {exc}")
    return mod


# ─── Pydantic models on the API side ──────────────────────────────────────


def test_every_record_validates_against_api_RecordModel(
    snapshot_outputs, api_module
):
    """The API's ``RecordModel`` (api/main.py:138) is the response-shape
    Pydantic model. Producer output must round-trip through it."""
    RecordModel = api_module.RecordModel
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]):
            try:
                RecordModel.model_validate(record)
            except ValidationError as exc:
                failures.append(
                    f"{fixture_name}[{idx}] ({record.get('bids_relpath', '?')}): {exc}"
                )
    assert not failures, (
        "API RecordModel rejected producer output:\n" + "\n".join(failures)
    )


def test_every_dataset_validates_against_api_DatasetModel(
    snapshot_outputs, api_module
):
    """The API's ``DatasetModel`` (api/main.py:158) is the response-shape
    Pydantic model for Dataset queries. Producer output must validate."""
    DatasetModel = api_module.DatasetModel
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        try:
            DatasetModel.model_validate(payload["dataset"])
        except ValidationError as exc:
            failures.append(f"{fixture_name}: {exc}")
    assert not failures, (
        "API DatasetModel rejected producer output:\n" + "\n".join(failures)
    )


# ─── Beanie Document index acceptance ─────────────────────────────────────


# The API's Beanie Document (api/main.py:142) declares indexes on:
#   - data_name
#   - dataset
#   - (dataset, subject) compound
#   - (dataset, bids_relpath) compound — the canonical upsert key
# If any of these fields go missing from a Record, the upsert path
# breaks. Encode them here.
_API_INDEXED_RECORD_FIELDS: tuple[str, ...] = (
    "data_name",
    "dataset",
    "subject",
    "bids_relpath",
)


def test_every_record_has_api_indexed_fields(snapshot_outputs):
    """Every indexed field on the API's Beanie Record document must be
    present (even if value is None) so the bulk-upsert key works."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]):
            for field in _API_INDEXED_RECORD_FIELDS:
                if field not in record:
                    failures.append(
                        f"{fixture_name}[{idx}] ({record.get('bids_relpath', '?')}): "
                        f"missing API-indexed field {field!r}"
                    )
    assert not failures, "API index drift:\n" + "\n".join(failures)


def test_record_dataset_field_matches_outer_dataset_id(snapshot_outputs):
    """The Record's ``dataset`` field MUST equal the parent Dataset's
    ``dataset_id``. This is the compound-index canonical join key."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        ds_id = payload["dataset"].get("dataset_id")
        for idx, record in enumerate(payload["records"]):
            if record.get("dataset") != ds_id:
                failures.append(
                    f"{fixture_name}[{idx}]: record.dataset={record.get('dataset')!r} "
                    f"vs dataset.dataset_id={ds_id!r}"
                )
    assert not failures, "Record/Dataset join key drift:\n" + "\n".join(failures)


def test_bids_relpath_is_unique_within_dataset(snapshot_outputs):
    """``(dataset, bids_relpath)`` is the bulk-upsert canonical key.
    Duplicate ``bids_relpath`` within a dataset would silently overwrite
    on inject — surface it here."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        seen: set[str] = set()
        for idx, record in enumerate(payload["records"]):
            rel = record.get("bids_relpath")
            if rel is None:
                continue
            if rel in seen:
                failures.append(
                    f"{fixture_name}[{idx}]: duplicate bids_relpath={rel!r}"
                )
            seen.add(rel)
    assert not failures, "Bulk-upsert key collision:\n" + "\n".join(failures)
```

- [ ] **Step 3.3: Run — expect PASS or skip**

```bash
cd /Users/bruaristimunha/Projects/eegdash/scripts/ingestions
pytest tests/acceptance/test_api_contract.py -v --tb=long 2>&1 | tail -25
```

Three acceptable outcomes:

- **All 5 pass:** the producer and API contracts agree on the snapshot fixtures.
- **Some skipped:** the `mongodb-eegdash-server` sibling repo isn't checked out, OR its imports (beanie / motor) aren't installed. The skip is informative — Layer 4 is opportunistic, not required.
- **Some fail:** real producer/consumer drift exists (the very thing Task 2 of the previous sprint added the consumer-side guard for, now caught at the test level). Read the failure; either fix the producer or update the API contract — your call. **Do NOT silently update the test.**

- [ ] **Step 3.4: Verify full suite**

```bash
pytest -q -m "not network and not slow and not integration" --tb=short 2>&1 | tail -5
```

Expected: 862 → 867 PR-fast (+5 new tests, some possibly skipped).

- [ ] **Step 3.5: Commit**

```bash
cd /Users/bruaristimunha/Projects/eegdash
git add scripts/ingestions/tests/acceptance/test_api_contract.py
git commit -m "test(acceptance): Layer 4 — API contract validation

Import the API's Pydantic models from mongodb-eegdash-server/api/
main.py and validate every snapshot fixture's records and datasets
against them. Distinct from Layer 2 (producer-side eegdash.schemas
models) because the two repos have independently-evolving models
with the same class names.

Also asserts the API's Beanie Document index fields are present
(data_name, dataset, subject, bids_relpath) on every Record — the
(dataset, bids_relpath) compound index is the bulk-upsert canonical
key, and a missing field there breaks the inject path silently.

Test skips cleanly when mongodb-eegdash-server isn't a sibling
clone (CI without the API repo doesn't fail this gate)."
```

---

## Self-review checklist

### 1. Spec coverage

- [x] Layer 2 — `test_schema_acceptance.py` covers RecordModel + StorageModel + EntitiesModel + DatasetModel + montage JSON shape.
- [x] Layer 3 — `test_idempotency.py` covers byte-identical output + record count + ingestion_fingerprint stability across two runs.
- [x] Layer 4 — `test_api_contract.py` covers API RecordModel + DatasetModel + Beanie indexed fields + join-key consistency + bulk-upsert key uniqueness.

### 2. Placeholder scan

- No "TBD" / "implement later" / "similar to Task N".
- Every test body is complete code.
- Every command has expected output documented.

### 3. Type consistency

- `_INGEST_DIR` resolution is consistent across all three test files (3 levels up from each test file → `scripts/ingestions/`).
- `_API_DIR` in Layer 4 is `parents[4] / "mongodb-eegdash-server" / "api"` — counting from `scripts/ingestions/tests/acceptance/test_api_contract.py`: parents[0]=acceptance, [1]=tests, [2]=ingestions, [3]=scripts, [4]=eegdash repo root → then sibling `mongodb-eegdash-server/api`. **Verify in Step 3.1.** Fix locally if the path is wrong (the implementer's `Step 3.1` is the verification gate).
- `snapshot_outputs` fixture name + return shape (`dict[str, dict[str, Any]]` with `dataset` / `records` / `montages` / `summary` keys) consistent across Layer 2, 3, 4.

### 4. Constraint conformance

- No `--no-verify` in any commit step.
- No `Co-Authored-By`, no robot attribution.
- Snapshot tests checked after every layer (steps 1.3, 2.3 mention the snapshot gate).
- Coverage gate not regressed (no production code changes unless a failure surfaces in Step 1.2 / 2.2 / 3.3).

---

## Execution handoff

Plan saved to `scripts/ingestions/ROBUSTNESS/SPRINT-2026-05-22-ACCEPTANCE.md`.

### Two execution options

**1. Subagent-Driven (recommended)** — fresh subagent per task, spec + quality two-stage review, same flow as the prior 3 sprints (post-/code-review fixes, MEG perf, sweep follow-ups).

**2. Inline Execution** — execute tasks here with checkpoints if you want every commit reviewed in-session.

Each task is small (~50-100 LOC). Total expected: 4 commits adding ~15-18 tests. If any layer surfaces a real drift, the implementer fixes the drift before landing — that's the point of acceptance tests, and the explicit instruction at Step 1.2 / 2.2 / 3.3.
