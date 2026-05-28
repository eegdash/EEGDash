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

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import pytest
from _helpers import load_digest

from eegdash.testing import data_file

# Module-level "pinned now" used by ``_FixedDatetime`` below. Set by
# ``_run_digest_against`` before patching the digest module; the class
# is lifted to module scope so the no-nested-functions repo lint passes.
_PINNED_NOW_ISO: str = "2026-05-22T12:00:00+00:00"


class _FixedDatetime(_dt.datetime):
    """A ``datetime`` subclass whose ``now`` is pinned to the module-level
    ``_PINNED_NOW_ISO`` value.

    ``3_digest.py`` does ``from datetime import datetime, timezone`` so
    the call site (``datetime.now(timezone.utc).isoformat()`` in
    ``digest_dataset``) binds the *class* into the digest module's own
    namespace. Patching ``datetime.datetime`` on the stdlib module would
    NOT reach that local binding — we patch the ``datetime`` attribute on
    the loaded digest module itself with this class.
    """

    @classmethod
    def now(cls, tz=None):
        return _dt.datetime.fromisoformat(_PINNED_NOW_ISO.replace("Z", "+00:00"))


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
    global _PINNED_NOW_ISO
    real_datetime = digest_mod.datetime  # the class, not the module
    prev_pin = _PINNED_NOW_ISO
    _PINNED_NOW_ISO = digested_at
    digest_mod.datetime = _FixedDatetime  # type: ignore[misc]
    try:
        digest_mod.digest_dataset(fixture_input.name, fixture_input.parent, output_dir)
    finally:
        digest_mod.datetime = real_datetime  # type: ignore[misc]
        _PINNED_NOW_ISO = prev_pin

    out_dir = output_dir / fixture_input.name
    result: dict[str, Any] = {}
    for kind in ("dataset", "records", "montages", "summary"):
        path = out_dir / f"{fixture_input.name}_{kind}.json"
        result[kind] = json.loads(path.read_text())
    return result


# Fields in ``summary.json`` that legitimately vary by output directory
# (absolute path to each emitted file). Idempotency is a property of
# the *content* the digester produces — pointing at run_a vs run_b is
# expected and isn't what this layer is testing.
#
# Producer (``record_enumerator.write_dataset_outputs``) currently emits
# only the first 3 keys; ``summary_file`` is included defensively in
# case the producer ever surfaces its own location, so the strip stays
# correct without a follow-up patch.
_SUMMARY_LOCATION_KEYS: tuple[str, ...] = (
    "dataset_file",
    "records_file",
    "montages_file",
    "summary_file",  # defensive: not currently emitted, future-proof
)


def _strip_summary_location(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``summary`` without output-path keys."""
    return {k: v for k, v in summary.items() if k not in _SUMMARY_LOCATION_KEYS}


@pytest.mark.parametrize(
    "fixture_name",
    ["ds_snapshot_vhdr", "ds_snapshot_manifest"],
)
def test_digest_dataset_is_byte_idempotent(
    fixture_name: str, ingest_dir: Path, tmp_path: Path
):
    """Run ``digest_dataset`` twice on the same input. Byte-identical."""
    fixture_input = data_file("digest_snapshots/inputs") / fixture_name
    if not fixture_input.exists():
        pytest.skip(f"Fixture missing: {fixture_input}")

    digest_mod = load_digest()

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    out_a.mkdir()
    out_b.mkdir()

    digested_at = "2026-05-22T12:00:00+00:00"

    result_a = _run_digest_against(fixture_input, out_a, digest_mod, digested_at)
    result_b = _run_digest_against(fixture_input, out_b, digest_mod, digested_at)

    # Byte-identical at the JSON level (key order, value formatting).
    # ``summary.json`` embeds absolute paths to the four output files;
    # those legitimately differ between two distinct output dirs and
    # have nothing to do with non-determinism in the digester. Strip
    # them before comparing — every other key (record_count,
    # montage_count, source, status, ...) must match byte-for-byte.
    for kind in ("dataset", "records", "montages", "summary"):
        payload_a = (
            _strip_summary_location(result_a[kind])
            if kind == "summary"
            else result_a[kind]
        )
        payload_b = (
            _strip_summary_location(result_b[kind])
            if kind == "summary"
            else result_b[kind]
        )
        text_a = json.dumps(payload_a, sort_keys=True, indent=2)
        text_b = json.dumps(payload_b, sort_keys=True, indent=2)
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
    fixture_input = data_file("digest_snapshots/inputs") / fixture_name
    if not fixture_input.exists():
        pytest.skip(f"Fixture missing: {fixture_input}")

    digest_mod = load_digest()
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


def test_ingestion_fingerprint_is_stable_across_runs(ingest_dir: Path, tmp_path: Path):
    """The dataset's ``ingestion_fingerprint`` is content-derived and
    must NOT change across two runs of the same input."""
    fixture_input = data_file("digest_snapshots/inputs/ds_snapshot_vhdr")
    if not fixture_input.exists():
        pytest.skip(f"Fixture missing: {fixture_input}")

    digest_mod = load_digest()
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
