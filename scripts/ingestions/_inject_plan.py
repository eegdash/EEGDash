"""Build the Stage 5 Injection Plan from a Digest Corpus."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from _constants import EXCLUDED_DATASETS
from _fingerprint import fingerprint_from_records
from _http import get_client, request_json


@dataclass
class InjectionPlan:
    """Structured write plan produced from a Digest Corpus."""

    dataset_dirs: list[Path] = field(default_factory=list)
    datasets: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    montages: list[dict] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)
    changed_ids: list[str] = field(default_factory=list)
    duplicate_montage_sightings: int = 0


def find_digested_datasets(
    input_dir: Path, datasets: list[str] | None = None
) -> list[Path]:
    """Find all dataset directories in the Digest Corpus."""
    dataset_dirs = []

    for dataset_dir in sorted(input_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_id = dataset_dir.name

        if (
            dataset_id.startswith(".")
            or dataset_id.startswith("_")
            or dataset_id in EXCLUDED_DATASETS
        ):
            continue

        if datasets and dataset_id not in datasets:
            continue

        dataset_file = dataset_dir / f"{dataset_id}_dataset.json"
        records_file = dataset_dir / f"{dataset_id}_records.json"

        if dataset_file.exists() or records_file.exists():
            dataset_dirs.append(dataset_dir)

    return dataset_dirs


def load_dataset(dataset_dir: Path) -> dict | None:
    """Load a Dataset document from a directory."""
    dataset_id = dataset_dir.name
    dataset_file = dataset_dir / f"{dataset_id}_dataset.json"

    if not dataset_file.exists():
        return None

    with open(dataset_file) as f:
        return json.load(f)


def load_records(dataset_dir: Path) -> list[dict]:
    """Load Records from a directory.

    Supports both new schema (``_records.json``) and legacy formats.
    Flattens entities to top-level fields for compatibility with EEGDash API.

    Stamps ``record['dataset']`` from the directory name when missing so
    downstream grouping in build_injection_plan cannot lose records.
    """
    dataset_id = dataset_dir.name

    def _prepare(records: list) -> list[dict]:
        out = []
        for r in records:
            r = _flatten_entities(r)
            r.setdefault("dataset", dataset_id)
            out.append(r)
        return out

    records_file = dataset_dir / f"{dataset_id}_records.json"
    if records_file.exists():
        with open(records_file) as f:
            data = json.load(f)
        if isinstance(data, dict) and "records" in data:
            records = data["records"]
        elif isinstance(data, list):
            records = data
        else:
            records = []
        return _prepare(records)

    for legacy_name in [f"{dataset_id}_core.json", f"{dataset_id}_minimal.json"]:
        legacy_file = dataset_dir / legacy_name
        if legacy_file.exists():
            with open(legacy_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "records" in data:
                records = data["records"]
            else:
                records = []
            return _prepare(records)

    return []


def load_montages(dataset_dir: Path) -> list[dict]:
    """Load Montage documents from a dataset's digest directory."""
    dataset_id = dataset_dir.name
    montages_file = dataset_dir / f"{dataset_id}_montages.json"
    try:
        with open(montages_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        return []

    if isinstance(data, dict) and "montages" in data:
        return data.get("montages") or []
    if isinstance(data, list):
        return data
    return []


def _flatten_entities(record: dict) -> dict:
    """Flatten BIDS entities to top-level fields for Gateway compatibility."""
    result = record.copy()
    entities = result.pop("entities", {})
    if entities:
        for key in ("subject", "task", "session", "run"):
            if key in entities and key not in result:
                result[key] = entities[key]
    return result


def fetch_existing_dataset(
    api_url: str,
    database: str,
    dataset_id: str,
):
    """Fetch existing dataset metadata from the API, if present.

    Returns the dataset dict on success, or None when the dataset is
    missing/ambiguous. A 200 OK with an empty body is treated as
    ambiguous (not 'new') so we don't force a redundant reinjection
    on transient API hiccups.
    """
    url = f"{api_url}/api/{database}/datasets/{dataset_id}"
    data, response = request_json("get", url, timeout=30, client=get_client())
    if response is None:
        return None
    if response.status_code == 404:
        return None
    if response.status_code != 200 or data is None:
        return None
    payload = data.get("data")
    if not payload:
        # 200 with missing/empty data is ambiguous; don't claim 'new'.
        return None
    return payload


def _ensure_fingerprint(dataset_id: str, dataset: dict | None, records: list[dict]):
    """Ensure ingestion_fingerprint is set on dataset or derived from records.

    Honour any fingerprint the digest stage already stamped onto the
    dataset doc — required for --only-datasets runs where records aren't
    loaded; without this the change-detection short-circuit would always
    declare every dataset 'changed'.
    """
    if dataset is None:
        dataset = {"dataset_id": dataset_id}
    if dataset.get("ingestion_fingerprint"):
        return dataset
    if records:
        dataset["ingestion_fingerprint"] = fingerprint_from_records(
            dataset_id,
            dataset.get("source", "unknown"),
            records,
        )
    return dataset


def filter_changed_datasets(
    dataset_ids: list[str],
    datasets_by_id: dict[str, dict],
    records_by_id: dict[str, list[dict]],
    api_url: str,
    database: str,
):
    """Return dataset IDs that are new or updated, plus skipped IDs."""
    changed_ids: list[str] = []
    skipped_ids: list[str] = []

    for dataset_id in dataset_ids:
        dataset = datasets_by_id.get(dataset_id)
        records = records_by_id.get(dataset_id, [])
        dataset = _ensure_fingerprint(dataset_id, dataset, records)
        datasets_by_id[dataset_id] = dataset

        existing = fetch_existing_dataset(api_url, database, dataset_id)
        existing_fp = (existing or {}).get("ingestion_fingerprint")
        current_fp = dataset.get("ingestion_fingerprint")

        if not existing:
            changed_ids.append(dataset_id)
            continue
        if existing_fp and current_fp and existing_fp == current_fp:
            skipped_ids.append(dataset_id)
            continue
        changed_ids.append(dataset_id)

    return changed_ids, skipped_ids


def build_injection_plan(
    dataset_dirs: list[Path],
    *,
    want_datasets: bool,
    want_records: bool,
    want_montages: bool,
    force: bool,
    only_montages: bool,
    api_url: str,
    database: str,
    progress: Callable[[Iterable[Path]], Iterable[Path]] | None = None,
) -> InjectionPlan:
    """Load a Digest Corpus and decide which documents Stage 5 should write."""
    iter_dirs = progress(dataset_dirs) if progress else dataset_dirs

    all_datasets = []
    all_records = []
    all_montages_by_hash: dict[str, dict] = {}
    montage_dataset_sources: dict[str, set[str]] = {}
    dataset_docs: dict[str, dict] = {}
    errors: list[dict[str, str]] = []

    for dataset_dir in iter_dirs:
        dataset_id = dataset_dir.name

        try:
            records = load_records(dataset_dir)
            if want_records and records:
                all_records.extend(records)

            # NB: never skip the dataset doc just because records is empty
            # — a metadata-only seed (or an --only-datasets run that pulls
            # no records by design) should still emit the Dataset document.
            dataset = load_dataset(dataset_dir)
            if dataset:
                dataset_docs[dataset_id] = dataset
                if want_datasets:
                    all_datasets.append(dataset)

            if want_montages:
                for montage in load_montages(dataset_dir):
                    montage_hash = montage.get("hash")
                    if not montage_hash:
                        continue
                    if montage_hash not in all_montages_by_hash:
                        all_montages_by_hash[montage_hash] = montage
                    montage_dataset_sources.setdefault(montage_hash, set()).add(
                        dataset_id
                    )

        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            errors.append({"dataset": dataset_id, "error": str(e)})
            print(f"  Error loading {dataset_id}: {e}", file=sys.stderr)

    all_montages = list(all_montages_by_hash.values())
    duplicate_sightings = sum(
        max(0, len(sources) - 1) for sources in montage_dataset_sources.values()
    )

    datasets_by_id = {ds_id: ds for ds_id, ds in dataset_docs.items() if ds_id and ds}
    records_by_id: dict[str, list[dict]] = {}
    for record in all_records:
        dataset_id = record.get("dataset")
        if not dataset_id:
            continue
        records_by_id.setdefault(dataset_id, []).append(record)

    dataset_ids = sorted(set(datasets_by_id) | set(records_by_id))

    for dataset_id in dataset_ids:
        dataset = datasets_by_id.get(dataset_id)
        records = records_by_id.get(dataset_id, [])
        datasets_by_id[dataset_id] = _ensure_fingerprint(dataset_id, dataset, records)

    changed_ids = dataset_ids
    skipped_ids: list[str] = []
    if not force and not only_montages:
        changed_ids, skipped_ids = filter_changed_datasets(
            dataset_ids,
            datasets_by_id,
            records_by_id,
            api_url,
            database,
        )
        changed_set = set(changed_ids)
        all_datasets = [
            datasets_by_id[ds_id] for ds_id in changed_ids if ds_id in datasets_by_id
        ]
        all_records = [r for r in all_records if r.get("dataset") in changed_set]

    return InjectionPlan(
        dataset_dirs=dataset_dirs,
        datasets=all_datasets,
        records=all_records,
        montages=all_montages,
        errors=errors,
        skipped_ids=skipped_ids,
        changed_ids=changed_ids,
        duplicate_montage_sightings=duplicate_sightings,
    )
