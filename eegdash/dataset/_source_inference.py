"""Source / storage inference shared by runtime and ingestion.

Records and dataset documents both carry a ``source`` field and a
``storage.base`` URI. They were sometimes written with the wrong values
(e.g. NEMAR datasets digested as ``source="openneuro"`` and
``storage.base="s3://openneuro.org/<id>"`` before PR #327). This module
centralises the pattern-based recovery rules so:

* the runtime client can self-heal already-ingested records, and
* the digestion pipeline can refuse to commit a source that disagrees
  with the dataset_id pattern.

Keep ``STORAGE_CONFIGS`` aligned with ``scripts/ingestions/3_digest.py``.
"""

from __future__ import annotations

# Source -> remote storage prefix. Mirrors STORAGE_CONFIGS in
# scripts/ingestions/3_digest.py and the patterns in
# scripts/ingestions/_validate.py::SOURCE_STORAGE_PATTERNS.
STORAGE_CONFIGS: dict[str, dict[str, str]] = {
    "openneuro": {"backend": "s3", "base": "s3://openneuro.org"},
    "nemar": {"backend": "s3", "base": "s3://nemar"},
    "gin": {"backend": "https", "base": "https://gin.g-node.org"},
    "figshare": {"backend": "https", "base": "https://figshare.com/ndownloader/files"},
    "zenodo": {"backend": "https", "base": "https://zenodo.org/records"},
    "osf": {"backend": "https", "base": "https://files.osf.io"},
    "scidb": {"backend": "https", "base": "https://www.scidb.cn"},
    "datarn": {"backend": "webdav", "base": "https://webdav.data.ru.nl"},
}


def infer_source_from_dataset_id(dataset_id: str) -> str | None:
    """Return the source implied by ``dataset_id``, or ``None`` if unknown.

    OpenNeuro IDs are ``dsNNNNNN``; NEMAR IDs are ``nmNNNNNN``. The
    ``EEGManyLabs`` projects live on GIN, and the HBN ``EEG2025*``
    releases are mirrored on NEMAR.
    """
    if not dataset_id:
        return None
    if dataset_id.startswith("ds") and dataset_id[2:].isdigit():
        return "openneuro"
    if dataset_id.startswith("nm") and dataset_id[2:].isdigit():
        return "nemar"
    if "EEGManyLabs" in dataset_id:
        return "gin"
    if dataset_id.startswith("EEG2025"):
        return "nemar"
    return None


def expected_storage_base(dataset_id: str) -> str | None:
    """Return the canonical ``storage.base`` for ``dataset_id``.

    Only returns a value for dataset IDs whose pattern unambiguously
    determines the source (OpenNeuro and NEMAR). Returns ``None`` for
    sources that need extra metadata to build the URL (figshare/zenodo/
    osf use per-record IDs that aren't derivable from ``dataset_id``).
    """
    source = infer_source_from_dataset_id(dataset_id)
    if source not in {"openneuro", "nemar"}:
        return None
    return f"{STORAGE_CONFIGS[source]['base']}/{dataset_id}"


def expected_backend(dataset_id: str) -> str | None:
    """Return the canonical ``storage.backend`` for ``dataset_id``."""
    source = infer_source_from_dataset_id(dataset_id)
    if source is None:
        return None
    return STORAGE_CONFIGS[source]["backend"]


def correct_storage_inplace(
    record: dict, dataset_id: str | None = None
) -> tuple[bool, str | None]:
    """Rewrite a record's ``storage.base``/``backend`` if it is misrouted.

    A record is "misrouted" when its dataset_id pattern (e.g. ``nm*``)
    implies one source but ``storage.base`` points at a different source's
    bucket (e.g. ``s3://openneuro.org/...``). This is the residual fallout
    from the pre-PR-#327 ingestion bug.

    Returns
    -------
    (corrected, old_base)
        ``corrected`` is ``True`` when ``storage.base`` was rewritten.
        ``old_base`` is the previous value (only when corrected).
    """
    dsid = dataset_id or record.get("dataset")
    expected = expected_storage_base(dsid) if dsid else None
    if expected is None:
        return False, None

    storage = record.setdefault("storage", {})
    current_base = (storage.get("base") or "").rstrip("/")
    if not current_base or current_base == expected:
        return False, None

    # Only auto-correct when the current base matches a *known* foreign
    # bucket — never silently rewrite arbitrary user-supplied URIs.
    foreign_prefixes = {
        cfg["base"] for src, cfg in STORAGE_CONFIGS.items() if src != "local"
    }
    if not any(current_base.startswith(p + "/") for p in foreign_prefixes):
        return False, None

    storage["base"] = expected
    storage["backend"] = expected_backend(dsid) or storage.get("backend") or "s3"
    return True, current_base
