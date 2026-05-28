"""Per-record BIDS extraction Seam.

Extracted from ``3_digest.py``. Builds one record document for a single BIDS data
file: companion-file validation, sidecar/channel/technical metadata, dependency
keys, value clamping, and final assembly via ``create_record``. Pure leaf logic —
depends only on ``_bids_path``, ``_metadata_cascade``, ``_montage``, ``source_adapter``,
``_constants`` and ``eegdash`` — never on ``3_digest``.
"""

from __future__ import annotations

import csv as _csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from _bids_path import normalize_modality, strip_dataset_prefix
from _constants import NEURO_MODALITIES
from _metadata_cascade import CascadeContext, MetadataCascade
from _montage import _walk_up_find as _walk
from eegdash.dataset._source_inference import DEFAULT_STORAGE_CONFIG, STORAGE_CONFIGS
from eegdash.dataset.bids_dataset import _COMPANION_FILES
from eegdash.schemas import create_record
from source_adapter import SourceAdapter, get_source_adapter

logger = logging.getLogger(__name__)

__all__ = ["extract_record"]


# Companion files required for different formats
COMPANION_FILE_REQUIREMENTS = {
    ".vhdr": {
        "required": [".eeg", ".dat"],  # Need at least one of these
        "optional": [".vmrk"],
        "mode": "any",  # any = at least one required file must exist
    },
    ".set": {
        "required": [".fdt"],
        "optional": [],
        "mode": "optional",  # optional = may or may not have .fdt (data can be in .set)
    },
}


def _file_exists_or_symlink(path: Path, allow_symlinks: bool = True) -> bool:
    """Return True if path exists or is a broken git-annex symlink."""
    if path.exists():
        return True
    if allow_symlinks and path.is_symlink():
        return True
    return False


def validate_companion_files(
    file_path: Path, allow_symlinks: bool = True
) -> dict[str, Any]:
    """Check that required companion files exist for a data file (e.g. .eeg for .vhdr)."""
    result = {
        "valid": True,
        "missing_required": [],
        "missing_optional": [],
        "found": [],
        "warnings": [],
        "errors": [],
    }

    ext = file_path.suffix.lower()
    requirements = COMPANION_FILE_REQUIREMENTS.get(ext)

    if not requirements:
        # No companion file requirements for this format
        return result

    parent_dir = file_path.parent
    stem = file_path.stem

    # Check required companions
    required_exts = requirements.get("required", [])
    mode = requirements.get("mode", "all")

    found_required = []
    for req_ext in required_exts:
        companion_path = parent_dir / f"{stem}{req_ext}"
        if _file_exists_or_symlink(companion_path, allow_symlinks):
            found_required.append(req_ext)
            result["found"].append(str(companion_path.name))

    # Validate based on mode
    if mode == "any":
        # At least one required file must exist
        if required_exts and not found_required:
            result["valid"] = False
            result["missing_required"] = required_exts
            result["errors"].append(f"Missing data file: need one of {required_exts}")
    elif mode == "all":
        # All required files must exist
        missing = [e for e in required_exts if e not in found_required]
        if missing:
            result["valid"] = False
            result["missing_required"] = missing
            result["errors"].append(f"Missing required files: {missing}")
    elif mode == "optional":
        # Required files are not strictly required (e.g., .set can contain data)
        missing = [e for e in required_exts if e not in found_required]
        if missing:
            result["warnings"].append(f"Optional companion files not found: {missing}")
            result["missing_optional"].extend(missing)

    # Check optional companions
    optional_exts = requirements.get("optional", [])
    for opt_ext in optional_exts:
        companion_path = parent_dir / f"{stem}{opt_ext}"
        if _file_exists_or_symlink(companion_path, allow_symlinks):
            result["found"].append(str(companion_path.name))
        else:
            result["missing_optional"].append(opt_ext)

    # Special case for BrainVision: try to read VHDR to check referenced files
    if ext == ".vhdr" and not result["valid"]:
        # Try to get more information from the VHDR file
        try:
            from _vhdr_parser import extract_vhdr_references  # noqa: PLC0415

            refs = extract_vhdr_references(file_path)
            if refs.get("datafile"):
                data_file = refs["datafile"]
                result["errors"].append(
                    f"VHDR references missing data file: {data_file}"
                )
        except (OSError, ValueError, UnicodeDecodeError, KeyError):
            pass

    return result


# camelCase sidecar key → snake_case Record field (highest-leverage BIDS fields only).
_BIDS_SIDECAR_RECORD_FIELDS: dict[str, str] = {
    "PowerLineFrequency": "power_line_frequency",
    "EEGReference": "eeg_reference",
    "iEEGReference": "ieeg_reference",
    "SoftwareFilters": "software_filters",
    "HardwareFilters": "hardware_filters",
    "Manufacturer": "manufacturer",
    "ManufacturersModelName": "manufacturers_model_name",
    "EEGPlacementScheme": "eeg_placement_scheme",
    "CapManufacturer": "cap_manufacturer",
    "CapManufacturersModelName": "cap_manufacturers_model_name",
    "InstitutionName": "institution_name",
    "RecordingType": "recording_type",
    "RecordingDuration": "recording_duration",
    "EEGGround": "eeg_ground",
}


def _extract_bids_sidecar_fields(bids_dataset: Any, bids_file: str) -> dict[str, Any]:
    """Extract BIDS sidecar fields (PowerLineFrequency, EEGReference, …) into structured Record fields."""
    out: dict[str, Any] = {}

    try:
        bids_root = Path(bids_dataset.bidsdir)
        data_file = Path(bids_file)
        modality = bids_dataset.get_bids_file_attribute("modality", bids_file) or "eeg"
    except (AttributeError, TypeError):
        return {}

    json_pattern = f"*_{modality}.json"
    sidecar = _walk(data_file, bids_root, json_pattern)
    if sidecar is None or not sidecar.exists():
        return {}

    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    for sidecar_key, record_key in _BIDS_SIDECAR_RECORD_FIELDS.items():
        val = data.get(sidecar_key)
        if val is None or val == "" or val == [] or val == {}:
            continue
        out[record_key] = val
    return out


def _extract_channel_status_counts(bids_dataset: Any, bids_file: str) -> dict[str, Any]:
    """Return bad channel names and count from channels.tsv, or {} if absent/unreadable."""
    try:
        bids_root = Path(bids_dataset.bidsdir)
        data_file = Path(bids_file)
    except (AttributeError, TypeError):
        return {}

    tsv = _walk(data_file, bids_root, "*_channels.tsv")
    if tsv is None:
        tsv = _walk(data_file, bids_root, "channels.tsv")
    if tsv is None or not tsv.exists():
        return {}

    bad_channels: list[str] = []
    try:
        with open(tsv, encoding="utf-8") as f:
            reader = _csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                return {}
            status_col = next(
                (c for c in reader.fieldnames if c.lower() == "status"), None
            )
            name_col = next((c for c in reader.fieldnames if c.lower() == "name"), None)
            if status_col is None or name_col is None:
                return {}
            for row in reader:
                status = str(row.get(status_col, "")).strip().lower()
                if status == "bad":
                    name = str(row.get(name_col, "")).strip()
                    if name:
                        bad_channels.append(name)
    except (OSError, _csv.Error, UnicodeDecodeError):
        return {}

    if not bad_channels:
        return {"bad_channels": [], "bad_channels_count": 0}
    return {"bad_channels": bad_channels, "bad_channels_count": len(bad_channels)}


def _extract_technical_metadata(
    bids_dataset: Any, bids_file: str
) -> tuple[
    float | None,
    int | None,
    int | None,
    list[str] | None,
    bool,
    bool,
    dict[str, str | None],
]:
    """Delegate to MetadataCascade; returns (sfreq, nchans, ntimes, ch_names, fif_is_split, fif_continuations_ok, provenance)."""
    ctx = CascadeContext(bids_dataset=bids_dataset, bids_file=bids_file)
    result = MetadataCascade().run(ctx)
    return (
        result.sampling_frequency,
        result.nchans,
        result.ntimes,
        result.ch_names,
        result.fif_is_split,
        result.fif_continuations_ok,
        result.provenance,
    )


_DEP_SUFFIXES: tuple[str, ...] = (
    "_channels.tsv",
    "_events.tsv",
    "_events.json",
    "_electrodes.tsv",
    "_coordsystem.json",
    "_eeg.json",
    # NIRS-specific sidecars
    "_optodes.tsv",
    "_optodes.json",
    "_nirs.json",
)


def _build_dep_keys(
    bids_file_path: Path,
    bids_root: Path,
    fif_is_split: bool,
    fif_continuations_ok: bool,
) -> tuple[list[str], bool, bool]:
    """Return (dep_keys, fif_is_split, fif_continuations_ok) for one BIDS file."""
    dep_keys: list[str] = []
    parent_dir = bids_file_path.parent
    base_name = bids_file_path.stem.rsplit("_", 1)[0]

    search_dirs = [parent_dir]
    if parent_dir.name in NEURO_MODALITIES or parent_dir.name in {
        "eeg",
        "meg",
        "ieeg",
        "beh",
        "nirs",
    }:
        search_dirs.append(parent_dir.parent)

    base_names_to_search = [base_name]
    session_base = re.sub(r"_task-[^_]+", "", base_name)
    session_base = re.sub(r"_run-[^_]+", "", session_base)
    session_base = re.sub(r"_acq-[^_]+", "", session_base)
    if session_base != base_name:
        base_names_to_search.append(session_base)

    for search_dir in search_dirs:
        for dep_suffix in _DEP_SUFFIXES:
            for search_base in base_names_to_search:
                dep_file = search_dir / f"{search_base}{dep_suffix}"
                if dep_file.exists() or dep_file.is_symlink():
                    try:
                        dep_keys.append(str(dep_file.relative_to(bids_root)))
                    except ValueError:
                        pass

    ext = bids_file_path.suffix.lower()
    for comp_ext in _COMPANION_FILES.get(ext, []):
        comp_file = bids_file_path.with_suffix(comp_ext)
        try:
            dep_keys.append(str(comp_file.relative_to(bids_root)))
        except ValueError:
            pass

    if ext == ".fif":
        if not fif_is_split:
            cont_check = bids_file_path.parent / f"{bids_file_path.stem}-1{ext}"
            if cont_check.exists() or cont_check.is_symlink():
                fif_is_split = True
        if fif_is_split:
            fif_continuations_ok = True
            for i in range(1, 100):
                cont = bids_file_path.parent / f"{bids_file_path.stem}-{i}{ext}"
                if not cont.exists() and not cont.is_symlink():
                    break
                if cont.is_symlink() and not cont.resolve().exists():
                    fif_continuations_ok = False
                try:
                    dep_keys.append(str(cont.relative_to(bids_root)))
                except ValueError:
                    pass

    return dep_keys, fif_is_split, fif_continuations_ok


def _clamp_metadata_extremes(
    sampling_frequency: float | None,
    nchans: int | None,
    ch_names: list[str] | None,
    bids_relpath: str,
    provenance: dict[str, str | None] | None = None,
) -> tuple[float | None, int | None]:
    """Zero out impossible sfreq/nchans values and warn on suspicious extremes (>1 MHz, >10000 channels)."""
    if sampling_frequency is not None:
        if sampling_frequency <= 0:
            logging.warning(
                "Invalid sampling_frequency <= 0 for %s: %s",
                bids_relpath,
                sampling_frequency,
            )
            sampling_frequency = None
            if provenance is not None:
                provenance["sampling_frequency"] = None
        elif sampling_frequency > 1_000_000:
            logging.warning(
                "Suspicious sampling_frequency > 1MHz for %s: %s",
                bids_relpath,
                sampling_frequency,
            )

    if nchans is not None:
        if nchans <= 0:
            logging.warning("Invalid nchans <= 0 for %s: %s", bids_relpath, nchans)
            nchans = None
            if provenance is not None:
                provenance["nchans"] = None
        elif nchans > 10000:
            logging.warning(
                "Suspicious nchans > 10000 for %s: %s", bids_relpath, nchans
            )

    if ch_names and nchans and len(ch_names) != nchans:
        logging.debug(
            "ch_names count (%d) != nchans (%d) for %s",
            len(ch_names),
            nchans,
            bids_relpath,
        )

    return sampling_frequency, nchans


def extract_record(
    bids_dataset,
    bids_file: str,
    dataset_id: str,
    source: str,
    digested_at: str,
    apex_sidecar_inline: dict[str, str] | None = None,
    source_adapter: SourceAdapter | None = None,
) -> dict[str, Any]:
    """Extract Record metadata for a single BIDS file."""
    subject = bids_dataset.get_bids_file_attribute("subject", bids_file)
    session = bids_dataset.get_bids_file_attribute("session", bids_file)
    task = bids_dataset.get_bids_file_attribute("task", bids_file)
    run = bids_dataset.get_bids_file_attribute("run", bids_file)
    acquisition = bids_dataset.get_bids_file_attribute("acquisition", bids_file)
    modality = bids_dataset.get_bids_file_attribute("modality", bids_file) or "eeg"
    mod_canon = normalize_modality(modality) or "eeg"

    bids_relpath = strip_dataset_prefix(
        str(bids_dataset.get_relative_bidspath(bids_file)), dataset_id
    )

    datatype = mod_canon
    suffix = mod_canon

    cfg = STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)
    storage_base = f"{cfg['base']}/{dataset_id}"
    storage_backend = cfg["backend"]

    (
        sampling_frequency,
        nchans,
        ntimes,
        ch_names,
        fif_is_split,
        fif_continuations_ok,
        metadata_provenance,
    ) = _extract_technical_metadata(bids_dataset, bids_file)
    bids_file_path = Path(bids_file)

    dep_keys, fif_is_split, fif_continuations_ok = _build_dep_keys(
        bids_file_path,
        Path(bids_dataset.bidsdir),
        fif_is_split,
        fif_continuations_ok,
    )
    ext = bids_file_path.suffix.lower()

    companion_validation = validate_companion_files(bids_file_path, allow_symlinks=True)
    data_integrity_issues = []

    if not companion_validation["valid"]:
        for error in companion_validation["errors"]:
            logging.warning("Data integrity issue for %s: %s", bids_relpath, error)
            data_integrity_issues.append(error)

    for warning in companion_validation.get("warnings", []):
        logging.info("Companion file note for %s: %s", bids_relpath, warning)

    if ext == ".fif" and fif_is_split and not fif_continuations_ok:
        data_integrity_issues.append(
            "Split FIF: continuation files not available in source"
        )

    sampling_frequency, nchans = _clamp_metadata_extremes(
        sampling_frequency,
        nchans,
        ch_names,
        bids_relpath,
        provenance=metadata_provenance,
    )

    # TODO(scale): apex sidecars (dataset_description.json, README, etc.) are duplicated
    # across every record. Move into a per-dataset side-collection when >100 MB inline
    # payload or >500 KB participants.tsv is encountered.
    if source_adapter is None:
        source_adapter = get_source_adapter(source, dataset_id, bids_dataset.bidsdir)
    bids_root_path = bids_dataset.bidsdir
    dep_paths = [bids_root_path / dep for dep in dep_keys]
    annex_keys, sidecar_inline = source_adapter.resolve_storage_extensions(
        Path(bids_file), dep_paths
    )
    if apex_sidecar_inline:
        for k, v in apex_sidecar_inline.items():
            sidecar_inline.setdefault(k, v)

    record = create_record(
        dataset=dataset_id,
        storage_base=storage_base,
        bids_relpath=bids_relpath,
        subject=subject,
        session=session,
        task=task,
        run=str(run) if run is not None else None,
        acquisition=acquisition,
        dep_keys=dep_keys,
        datatype=datatype,
        suffix=suffix,
        storage_backend=storage_backend,
        recording_modality=[mod_canon],
        ch_names=ch_names,
        sampling_frequency=sampling_frequency,
        nchans=nchans,
        ntimes=ntimes,
        digested_at=digested_at,
        annex_keys=annex_keys or None,
        sidecar_inline=sidecar_inline or None,
    )

    participant_tsv = bids_dataset.subject_participant_tsv(bids_file)
    if participant_tsv:
        has_real_data = any(v not in (None, "n/a") for v in participant_tsv.values())
        if not has_real_data:
            logging.debug(
                "No participant match for %s, storing column skeleton", bids_relpath
            )
        for k, v in participant_tsv.items():
            if k == "participant_id":
                continue
            if isinstance(v, str):
                try:
                    if v.strip():
                        participant_tsv[k] = float(v)
                except (ValueError, TypeError):
                    pass
        record["participant_tsv"] = participant_tsv

    if data_integrity_issues:
        record["_data_integrity_issues"] = data_integrity_issues
        record["_has_missing_files"] = True
    else:
        record["_has_missing_files"] = False

    if any(v is not None for v in metadata_provenance.values()):
        record["_metadata_provenance"] = metadata_provenance

    sidecar_extras = _extract_bids_sidecar_fields(bids_dataset, bids_file)
    record.update(sidecar_extras)

    channel_status = _extract_channel_status_counts(bids_dataset, bids_file)
    record.update(channel_status)

    return dict(record)
