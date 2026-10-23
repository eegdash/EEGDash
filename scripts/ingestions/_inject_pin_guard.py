"""Fail closed before Stage 5 writes can replace curated historical identities."""

from __future__ import annotations

import json
import time

import httpx

from _http import request_json

_PAGE_SIZE = 100
_READ_ATTEMPTS = 3
_SCIENTIFIC_FIELDS = (
    "ntimes",
    "nchans",
    "sampling_frequency",
    "duration_seconds",
    "recording_duration",
    "ch_names",
    "montage_hash",
)


class PinConflictError(ValueError):
    """The target's explicit pin cannot safely be replaced by this digest."""


def _read(client, url, params, *, count=False):
    """Read an uncached, validated envelope with bounded transport/JSON retries."""
    for attempt in range(_READ_ATTEMPTS):
        try:
            payload, response = request_json(
                "get",
                url,
                params=params,
                client=client,
                retries=1,
                headers={"Cache-Control": "no-cache"},
                timeout=60,
                raise_for_status=True,
                raise_for_request=True,
            )
            if (
                response is None
                or response.status_code != 200
                or not isinstance(payload, dict)
            ):
                raise ValueError("missing or malformed response")
            value = payload.get("count" if count else "data")
            if count:
                if type(value) is not int or value < 0:
                    raise ValueError("invalid count")
            elif not isinstance(value, list) or any(
                not isinstance(r, dict) for r in value
            ):
                raise ValueError("invalid record page")
            return value
        except (httpx.HTTPError, ValueError) as exc:
            if attempt + 1 == _READ_ATTEMPTS:
                # No request/auth values in this actionable diagnostic.
                raise PinConflictError(
                    "Live pin preflight could not read a complete target response"
                ) from exc
            time.sleep(2**attempt)


def _pins(record):
    storage = record.get("storage") or {}
    if not isinstance(storage, dict):
        raise PinConflictError("Malformed live/incoming pin storage")
    pin = storage.get("version_id")
    pins = storage.get("version_ids") or {}
    if not isinstance(pins, dict):
        raise PinConflictError("Malformed live/incoming pin storage")
    if pin is not None and (not isinstance(pin, str) or not pin):
        raise PinConflictError("Malformed raw version_id")
    if any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v
        for k, v in pins.items()
    ):
        raise PinConflictError("Malformed dependency version_ids")
    keys = storage.get("annex_keys") or {}
    deps = storage.get("dep_keys") or []
    if (
        not isinstance(keys, dict)
        or not isinstance(deps, list)
        or any(not isinstance(p, str) for p in deps)
    ):
        raise PinConflictError(
            "Malformed annex key/dependency mapping for pinned record"
        )
    return pin, pins


def _live_pinned_records(client, api_url, database, dataset_id):
    """Read only pin-bearing rows from the exact target, without twin aliases."""
    query = {
        "dataset": {"$eq": dataset_id},
        "$or": [
            {"storage.version_id": {"$exists": True}},
            {"storage.version_ids": {"$exists": True}},
        ],
    }
    params = {"filter": json.dumps(query)}
    base = f"{api_url.rstrip('/')}/api/{database}"
    expected = _read(client, f"{base}/count", params, count=True)
    records = []
    seen = set()
    # Exact cardinalities detect early EOF, capped/partial pages and repeats.
    for skip in range(0, expected, _PAGE_SIZE):
        limit = min(_PAGE_SIZE, expected - skip)
        page = _read(
            client, f"{base}/records", {**params, "skip": skip, "limit": limit}
        )
        if len(page) != limit:
            raise PinConflictError(
                f"{dataset_id}: incomplete live pin page ({len(page)} of {limit})"
            )
        for record in page:
            path = record.get("bids_relpath")
            storage = record.get("storage")
            if (
                record.get("dataset") != dataset_id
                or not isinstance(path, str)
                or not path
                or not isinstance(storage, dict)
                or not ("version_id" in storage or "version_ids" in storage)
                or path in seen
            ):
                raise PinConflictError(
                    f"{dataset_id}: ambiguous live pin record identity"
                )
            seen.add(path)
            records.append(record)
    tail = _read(client, f"{base}/records", {**params, "skip": expected, "limit": 1})
    after = _read(client, f"{base}/count", params, count=True)
    if tail or after != expected:
        raise PinConflictError(
            f"{dataset_id}: live pin set changed or pagination was incomplete"
        )
    return records


def _identity(storage, path):
    """Literal effective object identity, not latest-object metadata or bytes."""
    backend, base = storage.get("backend"), storage.get("base")
    if (
        not isinstance(backend, str)
        or not backend
        or not isinstance(base, str)
        or not base
    ):
        raise PinConflictError("Pinned object has ambiguous backend/base")
    return (
        backend,
        f"{base.rstrip('/')}/{path}",
        (storage.get("annex_keys") or {}).get(path),
    )


def _check_record(live, incoming, cutoff):
    dataset, path = live["dataset"], live["bids_relpath"]
    old_pin, old_pins = _pins(live)
    if not old_pin and not old_pins:
        return
    label = f"{dataset}/{path}"
    if len(incoming) != 1:
        raise PinConflictError(
            f"{label}: live pinned record is absent or duplicated in incoming digest"
        )
    new = incoming[0]
    # Gateway $set retains omitted fields. Prove the resulting timestamp is not
    # selected by the real pruner's string $lt cutoff, not just that P exists.
    stamp = new.get("digested_at", live.get("digested_at"))
    if cutoff is not None and isinstance(stamp, str) and stamp < cutoff:
        raise PinConflictError(
            f"{label}: post-upsert digested_at would be deleted by stale-record prune"
        )
    new_pin, new_pins = _pins(new)
    old_st, new_st = live["storage"], new.get("storage") or {}
    if (old_pin, old_pins) != (new_pin, new_pins):
        raise PinConflictError(
            f"{label}: explicit version_id/version_ids would change or disappear"
        )
    raw = old_st.get("raw_key")
    if not isinstance(raw, str) or not raw or new_st.get("raw_key") != raw:
        raise PinConflictError(
            f"{label}: pinned recording's raw_key would change or is ambiguous"
        )
    if _identity(old_st, raw) != _identity(new_st, raw):
        raise PinConflictError(
            f"{label}: pinned recording's raw URI/backend/annex key would change"
        )
    for dep in old_pins:
        if dep not in (old_st.get("dep_keys") or []) or dep not in (
            new_st.get("dep_keys") or []
        ):
            raise PinConflictError(
                f"{label}: pinned dependency {dep} would disappear or is undeclared"
            )
        if _identity(old_st, dep) != _identity(new_st, dep):
            raise PinConflictError(
                f"{label}: pinned dependency {dep} URI/backend/annex key would change"
            )
    for field in _SCIENTIFIC_FIELDS:
        if live.get(field) is not None and live[field] != new.get(field):
            raise PinConflictError(
                f"{label}: pinned recording scientific field {field} would change or disappear"
            )


def guard_live_pins(records, dataset_ids, api_url, database):
    """Refuse the entire plan on a pin conflict, before any collection writer.

    This is a preflight, not a server-side transaction against concurrent admin
    writers. It neither curates records nor bypasses changed pins under --force.
    Dry-run and collection-only invocations retain their existing semantics.
    """
    by_dataset = {}
    cutoffs = {}
    for record in records:
        # Match gateway /records/upsert normalization exactly. Both canonical
        # and legacy rows may address one row; neither may bypass admission.
        dataset = record.get("dataset")
        path = record.get("bids_relpath") or record.get("bidspath")
        if (
            not isinstance(dataset, str)
            or not dataset
            or not isinstance(path, str)
            or not path
        ):
            raise PinConflictError(
                "Incoming record has no unambiguous effective writer identity"
            )
        by_dataset.setdefault(dataset, {}).setdefault(path, []).append(record)
        stamp = record.get("digested_at")
        if isinstance(stamp, str) and stamp:
            cutoffs[dataset] = min(stamp, cutoffs.get(dataset, stamp))
    # Deliberately not _http.get_client(): its optional cache can hide live pins.
    with httpx.Client(timeout=60, transport=httpx.HTTPTransport(retries=0)) as client:
        for dataset_id in sorted(set(dataset_ids)):
            try:
                live = _live_pinned_records(client, api_url, database, dataset_id)
                incoming = by_dataset.get(dataset_id, {})
                for record in live:
                    _check_record(
                        record,
                        incoming.get(record["bids_relpath"], []),
                        cutoffs.get(dataset_id),
                    )
            except PinConflictError as exc:
                raise PinConflictError(
                    f"{dataset_id}: pin preflight refused: {exc}. "
                    "Reconcile the explicit historical source identity through reviewed manual curation; no injection writes were started."
                ) from exc
