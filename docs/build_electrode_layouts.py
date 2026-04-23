#!/usr/bin/env python3
"""Generate ``source/_static/dataset_generated/electrode-layouts.json``.

Sphinx reads this manifest from ``conf.py::_format_electrodes_section``
to decide which montage (if any) to embed on each dataset page. Until
now the file was hand-curated; this script replaces that with a
registry-backed build step.

Pipeline:

1. ``GET /api/{db}/datasets/summary`` → list of all dataset_ids.
2. ``GET /api/{db}/datasets/{id}/montages`` → distinct montages used by
   that dataset, sorted by descending subject count. We pick the top
   one (the cap most subjects actually wore).
3. Emit a layout entry ``{label, n_channels, modality, montage_id}``
   keyed by dataset_id. ``montage_id`` is the 16-char registry hash;
   the viewer resolves it via ``?montage=<hash>`` once the client-side
   fetcher lands (PLAN step 3).

Datasets with no scalp electrodes (MEG-only, depth iEEG, etc.) are
omitted — the Sphinx section falls back to a "no layout indexed"
placeholder in that case.

Run from the repo's ``docs/`` directory::

    python build_electrode_layouts.py
    python build_electrode_layouts.py --database eegdash_dev
    python build_electrode_layouts.py --api-url http://localhost:8000

Idempotent: if the registry hasn't changed, the JSON output is
byte-identical so Sphinx's cache stays warm.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = (
    DOCS_DIR / "source" / "_static" / "dataset_generated" / "electrode-layouts.json"
)

DEFAULT_API_URL = "https://data.eegdash.org/api"
DEFAULT_DATABASE = "eegdash"
DEFAULT_TIMEOUT = 30.0
DEFAULT_WORKERS = 8

# Pretty modality label for the iframe caption. Kept small — the viewer
# itself already shows the full metadata.
_MODALITY_LABEL = {
    "eeg": "EEG",
    "ieeg": "iEEG",
    "meg": "MEG",
    "nirs": "fNIRS",
    "emg": "EMG",
}


def _get_json(url: str, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """Return the JSON body of a GET, or ``None`` on any fetch error.

    We swallow transport errors instead of raising because the docs
    build should never break on an unreachable or partially-populated
    registry — missing datasets just get the Sphinx placeholder.
    """
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "eegdash-docs/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _fetch_dataset_ids(api_url: str, database: str, limit: int) -> list[str]:
    url = f"{api_url}/{database}/datasets/summary?limit={limit}"
    body = _get_json(url)
    if not body or not body.get("success"):
        return []
    ids = [
        str(ds.get("dataset_id") or "").strip().lower()
        for ds in body.get("data", [])
        if ds.get("dataset_id")
    ]
    return sorted(set(filter(None, ids)))


def _fetch_top_montage(api_url: str, database: str, dataset_id: str) -> dict | None:
    """Return the most-used montage doc for this dataset, or ``None``."""
    url = f"{api_url}/{database}/datasets/{dataset_id}/montages"
    body = _get_json(url)
    if not body:
        return None
    rows = body.get("data") or []
    if not rows:
        return None
    # Endpoint already sorts by subject_count DESC; be defensive anyway.
    return max(rows, key=lambda r: int(r.get("subject_count") or 0))


def _layout_entry(montage: dict) -> dict:
    """Map a registry montage doc → the shape conf.py reads."""
    modality = str(montage.get("modality") or "").strip().lower()
    n = int(montage.get("n_sensors") or 0)
    mod_label = _MODALITY_LABEL.get(modality, modality.upper() or "Sensors")
    label = f"{mod_label} · {n} sensors" if n else mod_label
    return {
        "label": label,
        "n_channels": n,
        "modality": modality or None,
        "montage_id": str(montage.get("hash") or "").strip(),
    }


def _fetch_dataset_montage(
    api_url: str, database: str, dataset_id: str
) -> tuple[str, dict | None]:
    """Worker for the thread pool — thin wrapper around ``_fetch_top_montage``."""
    return dataset_id, _fetch_top_montage(api_url, database, dataset_id)


def build_manifest(
    api_url: str,
    database: str,
    limit: int = 1000,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    dataset_ids = _fetch_dataset_ids(api_url, database, limit)
    if not dataset_ids:
        print(
            f"[electrode-layouts] no datasets returned from {api_url}/{database}/datasets/summary",
            file=sys.stderr,
        )

    layouts: dict[str, dict] = {}
    skipped: list[str] = []

    if dataset_ids:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_dataset_montage, api_url, database, ds_id): ds_id
                for ds_id in dataset_ids
            }
            for i, fut in enumerate(as_completed(futures), 1):
                ds_id, mont = fut.result()
                if mont is None or not mont.get("hash"):
                    skipped.append(ds_id)
                    continue
                layouts[ds_id] = _layout_entry(mont)
                if i % 50 == 0:
                    print(
                        f"  [{i}/{len(dataset_ids)}] {len(layouts)} mapped, {len(skipped)} empty",
                        file=sys.stderr,
                    )

    return {
        "_comment": (
            "Auto-generated by docs/build_electrode_layouts.py from the "
            "eegdash montage registry. Do not edit by hand."
        ),
        "_schema_version": 2,
        "_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_source": {
            "api_url": api_url,
            "database": database,
            "datasets_scanned": len(dataset_ids),
            "layouts_mapped": len(layouts),
            "datasets_without_montage": len(skipped),
        },
        "layouts": dict(sorted(layouts.items())),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--api-url", default=DEFAULT_API_URL)
    p.add_argument("--database", default=DEFAULT_DATABASE)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    p.add_argument(
        "--keep-existing-on-empty",
        action="store_true",
        help=(
            "If the registry returns zero layouts, leave the existing "
            "manifest file intact. Useful in CI to avoid wiping a "
            "previously-good build when the API is briefly unreachable."
        ),
    )
    args = p.parse_args()

    manifest = build_manifest(
        api_url=args.api_url,
        database=args.database,
        limit=args.limit,
        workers=args.workers,
    )

    n_layouts = len(manifest["layouts"])
    if n_layouts == 0 and args.keep_existing_on_empty and args.output.exists():
        print(
            f"[electrode-layouts] zero layouts from registry; keeping existing {args.output}",
            file=sys.stderr,
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(
        f"Wrote {args.output} — "
        f"{n_layouts} layouts "
        f"({manifest['_source']['datasets_without_montage']} datasets without a registered montage)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
