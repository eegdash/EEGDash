"""Manifest digest Seam: synthesize records from a manifest.json (API-only sources).

Extracted from ``3_digest.py``. Builds an ``EnumerationResult`` for datasets whose
files are not on local disk (OSF, Figshare, Zenodo, NEMAR, …) by interpreting a
manifest's file list — covering ZIP-extracted, subject-ZIP, BIDS-data-ZIP, regular,
standalone-ZIP-content and CTF ``.ds`` record shapes.

Depends only on leaf helpers (``_bids_path``, ``_source_id``, ``_parser_utils``,
``_fingerprint``, ``_constants``) plus ``eegdash`` and ``record_enumerator``'s
``EnumerationResult`` — never on ``3_digest`` — so ``record_enumerator`` can call it
directly instead of importlib-loading the digit-prefixed CLI.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

import httpx

from _bids_path import (
    detect_modality_from_path,
    is_neuro_data_file,
    normalize_modality,
    parse_bids_entities_from_path,
)
from _constants import NEURO_MODALITIES
from _fingerprint import fingerprint_from_manifest
from _parser_utils import _http_client
from _source_id import _reconcile_source
from eegdash.dataset._source_inference import DEFAULT_STORAGE_CONFIG, STORAGE_CONFIGS
from eegdash.schemas import create_dataset, create_record
from record_enumerator import EnumerationResult

__all__ = ["_enumerate_via_manifest"]


def _determine_manifest_storage_base(
    source: str,
    dataset_id: str,
    manifest: dict,
) -> str:
    """Resolve the canonical storage.base for a manifest-only ingest; rejects cross-source misrouting."""
    storage_base = manifest.get("storage_base")

    if not storage_base:
        base = STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)["base"]

        if source == "figshare":
            # Figshare: use source_url from external_links if available.
            source_url = manifest.get("external_links", {}).get("source_url", "")
            return source_url if source_url else f"{base}/{dataset_id}"
        if source == "zenodo":
            zenodo_id = manifest.get("zenodo_id", dataset_id)
            return f"{base}/{zenodo_id}"
        if source == "osf":
            osf_id = manifest.get("osf_id", dataset_id)
            return f"{base}/{osf_id}"
        if source == "gin":
            org = manifest.get("organization", "EEGManyLabs")
            return f"{base}/{org}/{dataset_id}"
        return f"{base}/{dataset_id}"

    expected_prefix = STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG).get(
        "base", ""
    )
    if expected_prefix and not str(storage_base).startswith(expected_prefix):
        print(
            f"WARNING [digest]: {dataset_id} manifest storage_base="
            f"{storage_base!r} does not start with {expected_prefix!r} "
            f"for source={source!r}; rebuilding from source config.",
            file=sys.stderr,
        )
        return f"{expected_prefix}/{dataset_id}"

    return storage_base


def _collect_bids_entities_from_paths(
    files: list, zip_contents: list
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Parse BIDS entities from all file paths (including ZIP contents) into (subjects, sessions, tasks, modalities)."""
    subjects: set[str] = set()
    sessions: set[str] = set()
    tasks: set[str] = set()
    modalities: set[str] = set()

    all_paths: list[str] = []
    for f in files:
        filepath = f.get("path", "") if isinstance(f, dict) else f
        all_paths.append(filepath)
        # Check if this file has extracted ZIP contents
        if isinstance(f, dict) and f.get("_zip_contents"):
            for zf in f["_zip_contents"]:
                zpath = zf.get("path", "") if isinstance(zf, dict) else zf
                all_paths.append(zpath)

    for zpath in zip_contents:
        all_paths.append(zpath.get("path", "") if isinstance(zpath, dict) else zpath)

    for filepath in all_paths:
        entities = parse_bids_entities_from_path(filepath)
        if entities.get("modality"):
            modalities.add(entities["modality"])
        if entities.get("modality") in NEURO_MODALITIES:
            if entities.get("subject"):
                subjects.add(entities["subject"])
            if entities.get("session"):
                sessions.add(entities["session"])
            if entities.get("task"):
                tasks.add(entities["task"])

    return subjects, sessions, tasks, modalities


def _fetch_subject_count_via_http(files: list, fallback: int) -> int:
    """Last-resort HTTP fetch of dataset_description.json or participants.tsv for subject count."""
    desc_url = None
    participants_url = None
    for f in files:
        if isinstance(f, dict):
            path = f.get("path", "").lower()
            url = f.get("download_url")
            if path.endswith("dataset_description.json"):
                desc_url = url
            elif path.endswith("participants.tsv"):
                participants_url = url

    subjects_count = fallback
    if desc_url:
        try:
            resp = _http_client().get(desc_url, timeout=10)
            resp.raise_for_status()
            desc_data = json.loads(resp.content.decode("utf-8"))
            if "Subjects" in desc_data:  # heuristic
                subjects_count = int(desc_data["Subjects"])
        except (
            httpx.HTTPError,
            RuntimeError,
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
            KeyError,
        ) as e:
            print(f"Failed to fetch/parse dataset_description.json: {e}")

    if subjects_count == 0 and participants_url:
        try:
            resp = _http_client().get(participants_url, timeout=10)
            resp.raise_for_status()
            content = resp.content.decode("utf-8")
            lines = [line for line in content.splitlines() if line.strip()]
            if len(lines) > 1:  # subtract the header
                subjects_count = len(lines) - 1
        except (
            httpx.HTTPError,
            RuntimeError,
            OSError,
            UnicodeDecodeError,
            ValueError,
        ) as e:
            print(f"Failed to fetch/parse participants.tsv: {e}")

    return subjects_count


def _build_zip_extracted_records(
    file_info: dict,
    dataset_id: str,
    storage_base: str,
    digested_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthesize Records for neuro files inside a peeked ZIP (_zip_contents); stamps container_url."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    zip_name = file_info.get("name", "")
    zip_download_url = file_info.get("download_url", "")

    for zf in file_info.get("_zip_contents", []):
        zf_path = zf.get("path", "") if isinstance(zf, dict) else zf
        zf_size = zf.get("size", 0) if isinstance(zf, dict) else 0

        if not is_neuro_data_file(zf_path):
            continue
        try:
            entities = parse_bids_entities_from_path(zf_path)
            detected_modality = detect_modality_from_path(zf_path)
            record = create_record(
                dataset=dataset_id,
                storage_base=storage_base,
                bids_relpath=zf_path,
                subject=entities.get("subject"),
                session=entities.get("session"),
                task=entities.get("task"),
                run=entities.get("run"),
                acquisition=entities.get("acquisition"),
                dep_keys=[],
                datatype=entities.get("datatype", detected_modality),
                suffix=entities.get("suffix", detected_modality),
                storage_backend="https",
                recording_modality=[detected_modality],
                digested_at=digested_at,
            )
            if zip_download_url:
                record["container_url"] = zip_download_url
                record["container_type"] = "zip"
                record["container_name"] = zip_name
            if zf_size:
                record["file_size"] = zf_size
            records.append(dict(record))
        except (KeyError, ValueError, TypeError) as e:
            errors.append({"file": zf_path, "error": str(e)})

    return records, errors


def _build_subject_zip_record(
    file_info: dict,
    dataset_id: str,
    storage_base: str,
    primary_mod: str,
    recording_modality_val: list[str],
    digested_at: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Synthesize a placeholder Record for a sub-<id>.zip archive."""
    filepath = file_info.get("path", "")
    download_url = file_info.get("download_url")
    file_size = file_info.get("size", 0)

    subject_match = re.match(r"^(sub-[a-zA-Z0-9]+)\.zip$", filepath, re.IGNORECASE)
    if not subject_match:
        return None, []
    subject_id = subject_match.group(1)
    try:
        record = create_record(
            dataset=dataset_id,
            storage_base=storage_base,
            bids_relpath=(f"{subject_id}/{primary_mod}/{subject_id}_{primary_mod}.set"),
            subject=subject_id.replace("sub-", ""),
            session=None,
            task=None,
            run=None,
            dep_keys=[],
            datatype=primary_mod,
            suffix=primary_mod,
            storage_backend="https",
            recording_modality=recording_modality_val,
            digested_at=digested_at,
        )
        if download_url:
            record["container_url"] = download_url
            record["container_type"] = "zip"
            record["container_name"] = filepath
            record["zip_contains_bids"] = True
        if file_size:
            record["container_size"] = file_size
        return dict(record), []
    except (KeyError, ValueError, TypeError) as e:
        return None, [{"file": filepath, "error": str(e)}]


_BIDS_DATA_ZIP_PATTERNS: tuple[str, ...] = (
    r".*bids.*\.zip$",
    r".*_eeg\.zip$",
    r".*_meg\.zip$",
    r".*_ieeg\.zip$",
    r".*dataset.*\.zip$",
    r".*rawdata.*\.zip$",
    r".*data\.zip$",
    r".*eeg.*\.zip$",
    r".*meg.*\.zip$",
    r".*nirs.*\.zip$",
    r".*fnirs.*\.zip$",
)


def _is_bids_data_zip(filepath: str) -> bool:
    """Return True if the filename matches a known BIDS-bundle ZIP pattern (e.g. ``*_eeg.zip``, ``data.zip``)."""
    fp_lower = filepath.lower()
    return any(re.match(p, fp_lower) for p in _BIDS_DATA_ZIP_PATTERNS)


def _build_bids_data_zip_records(
    file_info: dict,
    manifest: dict,
    dataset_id: str,
    storage_base: str,
    source: str,
    primary_mod: str,
    recording_modality_val: list[str],
    digested_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthesize Records for a BIDS-bundled ZIP (one per inferred subject, capped at 200, or a single placeholder)."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    filepath = file_info.get("path", "")
    download_url = file_info.get("download_url")
    file_size = file_info.get("size", 0)

    demographics = manifest.get("demographics", {})
    inferred_subjects = demographics.get("subjects_count", 0)

    if inferred_subjects and inferred_subjects > 0:
        for sub_idx in range(1, min(inferred_subjects + 1, 201)):
            sub_id = f"{sub_idx:02d}" if inferred_subjects < 100 else f"{sub_idx:03d}"
            try:
                record = create_record(
                    dataset=dataset_id,
                    storage_base=storage_base,
                    bids_relpath=(
                        f"sub-{sub_id}/{primary_mod}/sub-{sub_id}_{primary_mod}.set"
                    ),
                    subject=sub_id,
                    session=None,
                    task=manifest.get("tasks", [None])[0]
                    if manifest.get("tasks")
                    else None,
                    run=None,
                    dep_keys=[],
                    datatype=primary_mod,
                    suffix=primary_mod,
                    storage_backend="https",
                    recording_modality=recording_modality_val,
                    digested_at=digested_at,
                )
                if download_url:
                    record["container_url"] = download_url
                    record["container_type"] = "zip"
                    record["container_name"] = filepath
                    record["needs_extraction"] = True
                    record["inferred_from_metadata"] = True
                if file_size:
                    record["container_size"] = file_size
                records.append(dict(record))
            except (KeyError, ValueError, TypeError) as e:
                errors.append({"file": f"sub-{sub_id}", "error": str(e)})
        return records, errors

    try:
        record = create_record(
            dataset=dataset_id,
            storage_base=storage_base,
            bids_relpath=f"__ZIP__/{filepath}",
            subject=None,
            session=None,
            task=None,
            run=None,
            dep_keys=[],
            datatype=primary_mod,
            suffix=primary_mod,
            storage_backend=STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)[
                "backend"
            ],
            recording_modality=recording_modality_val,
            digested_at=digested_at,
        )
        if download_url:
            record["container_url"] = download_url
            record["container_type"] = "zip"
            record["container_name"] = filepath
            record["needs_extraction"] = True
        if file_size:
            record["container_size"] = file_size
        records.append(dict(record))
    except (KeyError, ValueError, TypeError) as e:
        errors.append({"file": filepath, "error": str(e)})

    return records, errors


def _build_regular_manifest_record(
    file_info: dict | str,
    dataset_id: str,
    storage_base: str,
    source: str,
    digested_at: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Synthesize a Record for a regular (non-ZIP) neuro file from the manifest."""
    if isinstance(file_info, dict):
        filepath = file_info.get("path", "")
        download_url = file_info.get("download_url")
        file_size = file_info.get("size", 0)
    else:
        filepath = file_info
        download_url = None
        file_size = 0

    if not is_neuro_data_file(filepath):
        return None, []

    try:
        entities = parse_bids_entities_from_path(filepath)
        detected_modality = detect_modality_from_path(filepath)
        record = create_record(
            dataset=dataset_id,
            storage_base=storage_base,
            bids_relpath=filepath,
            subject=entities.get("subject"),
            session=entities.get("session"),
            task=entities.get("task"),
            run=entities.get("run"),
            acquisition=entities.get("acquisition"),
            dep_keys=[],
            datatype=entities.get("datatype", detected_modality),
            suffix=entities.get("suffix", detected_modality),
            storage_backend=STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)[
                "backend"
            ],
            recording_modality=[detected_modality],
            digested_at=digested_at,
        )
        if download_url:
            record["download_url"] = download_url
        if file_size:
            record["file_size"] = file_size
        return dict(record), []
    except (KeyError, ValueError, TypeError) as e:
        return None, [{"file": filepath, "error": str(e)}]


def _build_standalone_zip_content_records(
    zip_contents: list,
    dataset_id: str,
    storage_base: str,
    digested_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthesize Records from the manifest's top-level zip_contents array."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for zpath in zip_contents:
        if isinstance(zpath, dict):
            filepath = zpath.get("path", "")
            file_size = zpath.get("size", 0)
        else:
            filepath = zpath
            file_size = 0

        if not is_neuro_data_file(filepath):
            continue
        try:
            entities = parse_bids_entities_from_path(filepath)
            detected_modality = detect_modality_from_path(filepath)
            record = create_record(
                dataset=dataset_id,
                storage_base=storage_base,
                bids_relpath=filepath,
                subject=entities.get("subject"),
                session=entities.get("session"),
                task=entities.get("task"),
                run=entities.get("run"),
                acquisition=entities.get("acquisition"),
                dep_keys=[],
                datatype=entities.get("datatype", detected_modality),
                suffix=entities.get("suffix", detected_modality),
                storage_backend="https",
                recording_modality=[detected_modality],
                digested_at=digested_at,
            )
            if file_size:
                record["file_size"] = file_size
            records.append(dict(record))
        except (KeyError, ValueError, TypeError) as e:
            errors.append({"file": filepath, "error": str(e)})
    return records, errors


def _build_ctf_ds_records(
    files: list,
    dataset_id: str,
    storage_base: str,
    source: str,
    digested_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate CTF .ds file entries to one Record per .ds directory."""
    ctf_ds_dirs: set[str] = set()
    for file_info in files:
        filepath = (
            file_info.get("path", "") if isinstance(file_info, dict) else file_info
        )
        filepath_lower = filepath.lower()
        if ".ds/" in filepath_lower:
            # Extract the .ds directory path (everything up to and including ".ds")
            ds_idx = filepath_lower.index(".ds/") + 3  # +3 to include ".ds"
            ctf_ds_dirs.add(filepath[:ds_idx])  # use original case

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for ds_path in ctf_ds_dirs:
        try:
            entities = parse_bids_entities_from_path(ds_path)
            detected_modality = detect_modality_from_path(ds_path)
            record = create_record(
                dataset=dataset_id,
                storage_base=storage_base,
                bids_relpath=ds_path,
                subject=entities.get("subject"),
                session=entities.get("session"),
                task=entities.get("task"),
                run=entities.get("run"),
                acquisition=entities.get("acquisition"),
                dep_keys=[],
                datatype=entities.get("datatype", detected_modality),
                suffix=entities.get("suffix", detected_modality),
                storage_backend=STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)[
                    "backend"
                ],
                recording_modality=[detected_modality],
                digested_at=digested_at,
            )
            records.append(dict(record))
        except (KeyError, ValueError, TypeError) as e:
            errors.append({"file": ds_path, "error": str(e)})

    return records, errors


def _enumerate_via_manifest(
    dataset_id: str,
    manifest: dict,
    digested_at: str,
) -> tuple[EnumerationResult, int]:
    """Build an EnumerationResult from a parsed manifest.json and return (result, total_files)."""
    source = _reconcile_source(
        manifest.get("source"), dataset_id, context="digest_from_manifest"
    )

    files = manifest.get("files", [])
    if not files:
        return (
            EnumerationResult(
                dataset_meta={"dataset_id": dataset_id, "source": source},
                digest_method="manifest_only",
            ),
            0,
        )

    zip_contents = manifest.get("zip_contents", [])

    storage_base = _determine_manifest_storage_base(source, dataset_id, manifest)

    subjects, sessions, tasks, modalities = _collect_bids_entities_from_paths(
        files, zip_contents
    )

    demographics = manifest.get("demographics", {})
    # Prefer subject count from validated neuro files over manifest demographics.
    subjects_count = (
        len(subjects)
        if subjects
        else (
            demographics.get("subjects_count", 0)
            or manifest.get("bids_subject_count", 0)
        )
    )

    if subjects_count == 0 or not tasks:
        subjects_count = _fetch_subject_count_via_http(files, fallback=subjects_count)
    ages = demographics.get("ages", [])

    dataset_doi = manifest.get("dataset_doi")
    if not dataset_doi:
        identifiers = manifest.get("identifiers", {})
        dataset_doi = identifiers.get("doi")

    source_url = manifest.get("external_links", {}).get("source_url")
    if not source_url:
        source_url = manifest.get("external_links", {}).get("osf_url")

    fingerprint = fingerprint_from_manifest(dataset_id, source, manifest)

    recording_modality_val = manifest.get("recording_modality")
    if isinstance(recording_modality_val, str):
        recording_modality_val = [
            m.strip()
            for m in recording_modality_val.replace("+", ",").split(",")
            if m.strip()
        ]

    if recording_modality_val:
        recording_modality_val = [normalize_modality(m) for m in recording_modality_val]
        recording_modality_val = sorted(
            list({m for m in recording_modality_val if m in NEURO_MODALITIES})
        )
    if not recording_modality_val:
        recording_modality_val = sorted(
            list({m for m in modalities if m in NEURO_MODALITIES})
        ) or ["eeg"]

    primary_mod = recording_modality_val[0] if recording_modality_val else "eeg"

    dataset_doc = create_dataset(
        dataset_id=dataset_id,
        name=manifest.get("name"),
        source=source,
        readme=manifest.get("readme"),
        recording_modality=recording_modality_val,
        datatypes=sorted(manifest.get("modalities", list(modalities))),
        bids_version=None,  # Not available from API
        license=manifest.get("license"),
        authors=manifest.get("authors", []),
        funding=manifest.get("funding", []),
        dataset_doi=dataset_doi,
        tasks=sorted(manifest.get("tasks") or sorted(list(tasks))),
        sessions=sorted(manifest.get("sessions") or sorted(list(sessions))),
        total_files=len(files),
        subjects_count=subjects_count,
        ages=ages,
        age_mean=sum(ages) / len(ages) if ages else None,
        study_domain=manifest.get("study_domain"),
        source_url=source_url,
        digested_at=digested_at,
    )
    dataset_doc["ingestion_fingerprint"] = fingerprint

    records = []
    errors = []

    ctf_records, ctf_errors = _build_ctf_ds_records(
        files, dataset_id, storage_base, source, digested_at
    )
    records.extend(ctf_records)
    errors.extend(ctf_errors)

    for file_info in files:
        if isinstance(file_info, dict) and file_info.get("_zip_contents"):
            sub_records, sub_errors = _build_zip_extracted_records(
                file_info, dataset_id, storage_base, digested_at
            )
            records.extend(sub_records)
            errors.extend(sub_errors)
            continue

        filepath = (
            file_info.get("path", "") if isinstance(file_info, dict) else file_info
        )

        if filepath.lower().endswith(".zip") and isinstance(file_info, dict):
            if re.match(r"^(sub-[a-zA-Z0-9]+)\.zip$", filepath, re.IGNORECASE):
                rec, errs = _build_subject_zip_record(
                    file_info,
                    dataset_id,
                    storage_base,
                    primary_mod,
                    recording_modality_val,
                    digested_at,
                )
                if rec is not None:
                    records.append(rec)
                errors.extend(errs)
                continue

            if _is_bids_data_zip(filepath):
                sub_records, sub_errors = _build_bids_data_zip_records(
                    file_info,
                    manifest,
                    dataset_id,
                    storage_base,
                    source,
                    primary_mod,
                    recording_modality_val,
                    digested_at,
                )
                records.extend(sub_records)
                errors.extend(sub_errors)
                continue

        rec, errs = _build_regular_manifest_record(
            file_info, dataset_id, storage_base, source, digested_at
        )
        if rec is not None:
            records.append(rec)
        errors.extend(errs)

    extra_records, extra_errors = _build_standalone_zip_content_records(
        zip_contents, dataset_id, storage_base, digested_at
    )
    records.extend(extra_records)
    errors.extend(extra_errors)

    return (
        EnumerationResult(
            dataset_meta=dict(dataset_doc),
            records=records,
            errors=errors,
            montages={},  # manifest path produces no montages
            digest_method="manifest_only",
        ),
        len(files),
    )
