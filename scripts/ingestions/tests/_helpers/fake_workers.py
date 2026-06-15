"""Module-level fake ``digest_fn`` callables for the watchdog's real-subprocess tests.

These MUST live in an importable, non-digit-prefixed module so they are picklable
by reference and resolvable in a spawned child (macOS default start method is
``spawn``, which re-imports the target by qualified name). Closures / lambdas /
``load_digest()``-aliased functions are NOT spawn-safe and must not be used here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def digest_dispatch(dataset_id: str, input_dir: Path, output_dir: Path) -> Any:
    """Branch on the dataset id prefix to exercise every watchdog outcome.

    - ``ok*``      -> success result dict
    - ``err*``     -> raises (worker boundary must convert to an error result)
    - ``slow*``    -> sleeps past any sane test timeout (must be killed)
    - ``nondict*`` -> returns a non-dict (must be coerced to an error result)
    - anything else-> ``empty`` status
    """
    if dataset_id.startswith("ok"):
        return {"status": "success", "dataset_id": dataset_id, "marker": "from-fake"}
    if dataset_id.startswith("err"):
        raise RuntimeError(f"boom-{dataset_id}")
    if dataset_id.startswith("slow"):
        time.sleep(120)
        return {"status": "success", "dataset_id": dataset_id}
    if dataset_id.startswith("nondict"):
        return ["not", "a", "dict"]
    return {"status": "empty", "dataset_id": dataset_id}
