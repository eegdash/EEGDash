"""Structural tests for the extracted BIDS-filesystem digest Seam (``_bids_digest.py``).

The headline assertions pin the de-circularization: importing ``_bids_digest`` must
not pull in the ``3_digest`` CLI, and importing ``record_enumerator`` must not eagerly
load ``_bids_digest`` (its ``EnumerationResult`` import would otherwise form a cycle).
With both this and the manifest seam decoupled, ``record_enumerator`` no longer
importlib-loads ``3_digest`` at all.
"""

from __future__ import annotations

import subprocess
import sys

import _bids_digest
import record_enumerator
from tests._helpers import INGEST_DIR


def test_bids_digest_does_not_import_3_digest():
    code = (
        "import sys, _bids_digest; "
        "bad = [m for m, mod in sys.modules.items() "
        "if getattr(mod, '__file__', None) and mod.__file__.endswith('3_digest.py')]; "
        "print('LOADED_3DIGEST:' + ','.join(bad) if bad else 'CLEAN')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(INGEST_DIR),
        timeout=120,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "CLEAN" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"


def test_importing_record_enumerator_does_not_eagerly_load_bids_digest():
    """A future hoist of the lazy ``from _bids_digest import ...`` to module top would
    re-introduce a partial-init cycle; this fails in CI instead of at runtime."""
    code = (
        "import sys, record_enumerator; "
        "print('EAGER' if '_bids_digest' in sys.modules else 'LAZY')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(INGEST_DIR),
        timeout=120,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "LAZY" in proc.stdout, (
        f"record_enumerator eagerly imported _bids_digest: {proc.stdout}"
    )


def test_record_enumerator_no_longer_has_load_digest_module():
    """The importlib-of-3_digest hack is fully removed once both seams are extracted."""
    assert not hasattr(record_enumerator, "_load_digest_module")


def test_enumerate_via_bids_is_owned_by_the_seam():
    assert _bids_digest._enumerate_via_bids.__module__ == "_bids_digest"
