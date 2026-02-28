"""Local BIDS discovery helpers.

These utilities support offline workflows (no DB/S3) by discovering BIDS
recordings on the filesystem and returning EEGDash v2 records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mne_bids import find_matching_paths
from mne_bids.config import ALLOWED_DATATYPE_EXTENSIONS

from .const import MODALITY_ALIASES
from .schemas import create_record

# Keys added to each record when EEGBIDSDataset is available
_FILE_METADATA_KEYS = ("sampling_frequency", "nchans", "ntimes", "ch_names")


def _get_file_metadata(ds_helper: Any, file_path: str) -> dict[str, Any]:
    """Get BIDS file metadata (sfreq, nchans, ntimes, ch_names) when possible."""
    empty = {k: None for k in _FILE_METADATA_KEYS}
    if ds_helper is None:
        return empty
    try:
        result = {
            "sampling_frequency": ds_helper.get_bids_file_attribute("sfreq", file_path),
            "nchans": ds_helper.get_bids_file_attribute("nchans", file_path),
            "ntimes": ds_helper.get_bids_file_attribute("ntimes", file_path),
        }
        try:
            result["ch_names"] = ds_helper.channel_labels(file_path)
        except Exception:
            result["ch_names"] = None
        return result
    except Exception:
        return empty


def _normalize_modalities(modality_filter: Any) -> list[str]:
    """Normalize modality filter to a list of strings.

    If None, returns a default set of physiological modalities (excluding motion).
    """
    if modality_filter is None:
        return ["eeg", "meg", "ieeg", "nirs", "fnirs", "emg"]

    if isinstance(modality_filter, (list, tuple, set)):
        modalities = [str(m).strip().lower() for m in modality_filter if m]
    else:
        modalities = [str(modality_filter).strip().lower()]

    modalities = [MODALITY_ALIASES.get(m, m) for m in modalities if m]
    return modalities or ["eeg", "meg", "ieeg", "nirs", "fnirs", "emg"]


def discover_local_bids_records(
    dataset_root: str | Path, filters: dict[str, Any]
) -> list[dict[str, Any]]:
    """Discover local BIDS recordings and build EEGDash v2 records.

    Parameters
    ----------
    dataset_root : str | Path
        Local dataset directory (e.g., ``/path/to/ds005509``).
    filters : dict
        Filters dict. Must include ``'dataset'`` and may include BIDS entities
        like ``'subject'``, ``'session'``, ``'task'``, ``'run'``, plus
        ``'modality'`` (default: ``'eeg'``).

    Returns
    -------
    list[dict[str, Any]]
        A list of v2 records, one for each matched recording file.

    Notes
    -----
    Matching is performed via :func:`mne_bids.find_matching_paths` using
    datatypes/suffixes derived from the ``'modality'`` filter. The returned
    records use ``storage.backend='local'`` and point ``storage.base`` at
    ``dataset_root``.

    """
    dataset_id = filters["dataset"]
    modalities = _normalize_modalities(filters.get("modality"))

    arg_map = {
        "subjects": "subject",
        "sessions": "session",
        "tasks": "task",
        "runs": "run",
    }
    matching_args: dict[str, list[str]] = {}
    for finder_key, entity_key in arg_map.items():
        entity_val = filters.get(entity_key)
        if entity_val is None:
            continue
        if isinstance(entity_val, (list, tuple, set)):
            entity_vals = [str(v) for v in entity_val if v is not None]
            if not entity_vals:
                continue
            matching_args[finder_key] = entity_vals
        else:
            matching_args[finder_key] = [str(entity_val)]

    matched_paths = find_matching_paths(
        root=str(dataset_root),
        datatypes=modalities,
        suffixes=modalities,
        ignore_json=True,
        **matching_args,
    )

    dataset_root_path = Path(dataset_root)
    records_out: list[dict[str, Any]] = []

    valid_raw_extensions = {
        ext
        for m in modalities
        for ext in ALLOWED_DATATYPE_EXTENSIONS.get(MODALITY_ALIASES.get(m, m), [])
    }

    # create a dataset object to extract more metadata if possible
    try:
        # local import to avoid circular dependency
        from .dataset.bids_dataset import EEGBIDSDataset

        ds_helper = EEGBIDSDataset(
            data_dir=dataset_root, dataset=dataset_id, allow_symlinks=True
        )
    except Exception:
        ds_helper = None

    for bids_path in matched_paths:
        file_path = Path(bids_path.fpath)

        # Filter out sidecars based on extension
        # find_matching_paths with suffix='eeg' returns .json too.
        # We only want strictly the raw data file.
        # Note: BIDSPath.extension might be None for some directories (like .ds), check fpath
        _ = "".join(
            file_path.suffixes
        )  # handle .eeg.json if needed, but we check final
        final_ext = file_path.suffix.lower()

        if final_ext not in valid_raw_extensions:
            continue

        # Skip files without a task entity — these are typically auxiliary
        # BIDS files (calibration, crosstalk, empty-room) that are not
        # actual recordings and will fail in read_raw_bids.
        if not bids_path.task:
            continue

        try:
            # IMPORTANT: keep the BIDS symlink path, do NOT resolve to annex
            # objects. Resolving would turn a BIDS path like:
            #   sub-02/ses-001/eeg/sub-02_ses-001_task-main_run-001_eeg.vhdr
            # into:
            #   .git/annex/objects/.../MD5E-s...vhdr/MD5E-s...vhdr
            # which breaks downstream helpers that expect BIDS-relative paths.
            bids_relpath = file_path.relative_to(dataset_root_path.resolve())
        except ValueError:
            bids_relpath = Path(file_path.name)

        datatype = (bids_path.datatype or modalities[0] or "eeg").lower()
        suffix = (bids_path.suffix or datatype).lower()

        rec = create_record(
            dataset=dataset_id,
            storage_base=str(dataset_root_path),
            bids_relpath=bids_relpath.as_posix(),
            subject=bids_path.subject or None,
            session=bids_path.session or None,
            task=bids_path.task or None,
            run=bids_path.run or None,
            acquisition=bids_path.acquisition or None,
            dep_keys=[],
            datatype=datatype,
            suffix=suffix,
            storage_backend="local",
        )

        # Enrich with file metadata when available (sfreq, nchans, ntimes, ch_names)
        rec.update(_get_file_metadata(ds_helper, str(bids_path.fpath)))

        records_out.append(rec)

    return records_out
