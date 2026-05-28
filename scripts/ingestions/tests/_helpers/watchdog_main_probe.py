"""Run-as-``__main__`` probe for the watchdog's production pickle path.

When ``python 3_digest.py`` runs, ``digest_dataset`` lives in the ``__main__``
module and is injected as ``digest_fn``; under the ``spawn`` start method that
reference is pickled and resolved in the child by re-importing ``__main__``.
This script reproduces that exact shape (a ``__main__``-level ``digest_fn``) and
exits 0 only if the watchdog's success/error accounting is correct. It is invoked
via ``subprocess`` from ``test_digest_runner.py`` so that THIS file is ``__main__``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the ingestions dir importable so the child can resolve `_digest_runner`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _digest_runner import process_datasets_with_watchdog


def digest_dataset(dataset_id: str, input_dir: Path, output_dir: Path) -> dict:
    """A ``__main__``-level digest callable — picklable by reference under spawn."""
    if dataset_id.startswith("err"):
        raise RuntimeError("boom")
    return {"status": "success", "dataset_id": dataset_id}


def main() -> int:
    _results, stats = process_datasets_with_watchdog(
        ["ok-1", "err-1"],
        Path("."),
        Path("."),
        workers=2,
        dataset_timeout=60.0,
        digest_fn=digest_dataset,
    )
    ok = stats.get("success") == 1 and stats.get("error") == 1
    print("PROBE_RESULT:", "OK" if ok else f"BAD {stats}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
