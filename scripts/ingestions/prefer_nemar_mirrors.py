#!/usr/bin/env python3
"""Prefer NEMAR mirrors: retire OpenNeuro ``dsNNNNNN`` twins of ``onNNNNNN`` datasets.

NEMAR re-hosts OpenNeuro datasets under an ``on`` prefix (e.g. OpenNeuro's
``ds005506`` is mirrored as ``on005506``). When EEGDash consumes only NEMAR,
both IDs can coexist in MongoDB and serve duplicate content. This script
enforces the "prefer NEMAR" policy:

1. List every dataset ID in the target database.
2. For each ``onNNNNNN`` dataset, check whether its OpenNeuro twin
   ``dsNNNNNN`` also exists.
3. Verify the twin really is an OpenNeuro dataset (``source == "openneuro"``).
4. Carry curated, non-ingested fields (LLM ``tags`` and ``tagger_meta``) from
   the twin onto the ``on`` dataset when it has none of its own. Ingestion
   never produces these, so deleting the twin without this step silently
   drops them (it emptied ~540 catalog tags in Aug 2026).
5. Delete the twin's records and dataset document.

Dry run by default; pass ``--apply`` to perform deletions.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

from _source_id import _openneuro_twin_of

DEFAULT_API_URL = "https://data.eegdash.org"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Delete OpenNeuro dsNNNNNN twins of NEMAR onNNNNNN datasets "
            "(dry run unless --apply)."
        )
    )
    p.add_argument("--api-url", default=DEFAULT_API_URL)
    p.add_argument("--database", default="eegdash")
    p.add_argument(
        "--token",
        default=None,
        help="Admin token (env fallback: EEGDASH_ADMIN_TOKEN).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete twins. Without this flag only a preview is printed.",
    )
    p.add_argument(
        "--timeout", type=float, default=30.0, help="Per-request timeout (s)."
    )
    return p.parse_args(argv)


def _request_with_retry(
    client: httpx.Client, method: str, url: str, *, retries: int = 3, **kwargs
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.request(method, url, **kwargs)
            if resp.status_code == 429:
                wait = min(60.0, 5.0 * attempt)
                print(f"  [rate-limited] {url}; retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            return resp
        except httpx.HTTPError as exc:  # pragma: no cover - network guard
            last_exc = exc
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"{method} {url} failed after {retries} attempts: {last_exc}")


def list_dataset_ids(client: httpx.Client, api_url: str, database: str) -> list[str]:
    resp = _request_with_retry(
        client, "GET", f"{api_url}/api/{database}/datasets/names"
    )
    resp.raise_for_status()
    payload = resp.json()
    return sorted(payload.get("data") or [])


# Curated fields that only exist on the dataset document (ingestion never
# writes them), so they must be carried across before a twin is deleted.
CURATED_FIELDS = ("tags", "tagger_meta")
_TAG_AXES = ("pathology", "modality", "type")


def fetch_dataset(
    client: httpx.Client, api_url: str, database: str, dataset_id: str
) -> dict | None:
    resp = _request_with_retry(
        client, "GET", f"{api_url}/api/{database}/datasets/{dataset_id}"
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json().get("data") or {}


def dataset_is_openneuro(doc: dict | None) -> bool:
    return bool(doc) and (doc.get("source") or "").lower() == "openneuro"


def has_real_tags(doc: dict | None) -> bool:
    """True when any tag axis holds a value other than ``Unknown``."""
    tags = (doc or {}).get("tags") or {}
    for axis in _TAG_AXES:
        value = tags.get(axis)
        values = [value] if isinstance(value, str) else (value or [])
        if any(v and v != "Unknown" for v in values):
            return True
    return False


def curated_carry_over(ds_doc: dict | None, on_doc: dict | None) -> dict:
    """``$set`` payload moving the twin's curated fields onto the ``on`` doc.

    Empty when the twin has nothing worth keeping or the ``on`` doc already
    has its own tags (never overwrite a newer curation).
    """
    if not has_real_tags(ds_doc) or has_real_tags(on_doc):
        return {}
    return {f: ds_doc[f] for f in CURATED_FIELDS if ds_doc.get(f) is not None}


def carry_over(
    client: httpx.Client,
    api_url: str,
    database: str,
    headers: dict,
    on_id: str,
    update: dict,
) -> None:
    resp = _request_with_retry(
        client,
        "PATCH",
        f"{api_url}/admin/{database}/datasets/{on_id}",
        json={"update": update},
        headers=headers,
    )
    resp.raise_for_status()


def delete_twin(
    client: httpx.Client,
    api_url: str,
    database: str,
    headers: dict,
    dataset_id: str,
) -> tuple[int, int]:
    """Delete a twin's records then its dataset document; return (records, datasets) counts."""
    params = {"filter": f'{{"dataset": "{dataset_id}"}}'}
    resp = _request_with_retry(
        client,
        "DELETE",
        f"{api_url}/admin/{database}/records",
        params=params,
        headers=headers,
    )
    resp.raise_for_status()
    n_records = int(resp.json().get("deleted_count") or 0)

    resp = _request_with_retry(
        client,
        "DELETE",
        f"{api_url}/admin/{database}/datasets/{dataset_id}",
        headers=headers,
    )
    if resp.status_code == 404:
        n_datasets = 0
    else:
        resp.raise_for_status()
        n_datasets = int(resp.json().get("deleted_count") or 0)
    return n_records, n_datasets


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    token = args.token or os.environ.get("EEGDASH_ADMIN_TOKEN")
    if args.apply and not token:
        print(
            "Error: admin token required for --apply "
            "(set EEGDASH_ADMIN_TOKEN or use --token)",
            file=sys.stderr,
        )
        return 1

    mode = "APPLY" if args.apply else "DRY RUN"
    print(
        f"[{mode}] prefer-NEMAR mirror retirement on "
        f"{args.api_url} database={args.database}"
    )

    try:
        with httpx.Client(timeout=args.timeout) as client:
            ids = list_dataset_ids(client, args.api_url, args.database)
            nemar_ids = [i for i in ids if _openneuro_twin_of(i)]
            id_set = set(ids)

            twins: list[tuple[str, str, dict]] = []
            missing_twins = 0
            non_openneuro_twins: list[str] = []
            for on_id in nemar_ids:
                ds_id = _openneuro_twin_of(on_id)
                if ds_id not in id_set:
                    missing_twins += 1
                    continue
                ds_doc = fetch_dataset(client, args.api_url, args.database, ds_id)
                if dataset_is_openneuro(ds_doc):
                    on_doc = fetch_dataset(client, args.api_url, args.database, on_id)
                    twins.append((on_id, ds_id, curated_carry_over(ds_doc, on_doc)))
                else:
                    non_openneuro_twins.append(ds_id)

            print("\nMirror audit")
            print(f"  NEMAR on* datasets in DB : {len(nemar_ids)}")
            print(f"  without ds* twin         : {missing_twins}")
            if non_openneuro_twins:
                print(
                    f"  twin not openneuro source: {len(non_openneuro_twins)} "
                    f"(skipped: {', '.join(non_openneuro_twins[:10])})"
                )
            print(f"  OpenNeuro twins to retire: {len(twins)}")
            n_carry = sum(1 for *_, update in twins if update)
            print(f"  curated tags to carry    : {n_carry}")

            if not twins:
                print("\nNothing to do.")
                return 0

            total_records = 0
            total_datasets = 0
            errors = 0
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            for i, (on_id, ds_id, update) in enumerate(twins, start=1):
                if not args.apply:
                    carry = f"; carry {sorted(update)} to {on_id}" if update else ""
                    print(
                        f"  [preview] would delete {ds_id} (mirror of {on_id}){carry}"
                    )
                    continue
                try:
                    # Carry first: if it fails the twin is kept, so nothing
                    # curated is ever lost.
                    if update:
                        carry_over(
                            client, args.api_url, args.database, headers, on_id, update
                        )
                    n_records, n_datasets = delete_twin(
                        client, args.api_url, args.database, headers, ds_id
                    )
                    total_records += n_records
                    total_datasets += n_datasets
                    print(
                        f"  [{i}/{len(twins)}] deleted {ds_id}: "
                        f"{n_records} records, {n_datasets} dataset docs"
                    )
                except (RuntimeError, httpx.HTTPError) as exc:
                    errors += 1
                    print(f"  [{i}/{len(twins)}] FAILED deleting {ds_id}: {exc}")
    except RuntimeError as exc:
        print(
            f"Error: could not reach {args.api_url}: {exc}",
            file=sys.stderr,
        )
        return 1

    print("\nSummary")
    print(f"  Mode           : {mode}")
    print(f"  Twins targeted : {len(twins)}")
    if args.apply:
        print(f"  Records deleted: {total_records}")
        print(f"  Datasets deleted: {total_datasets}")
        print(f"  Errors          : {errors}")
        return 0 if errors == 0 else 1
    print("  (dry run: no changes written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
