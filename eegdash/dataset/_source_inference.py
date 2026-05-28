"""Source / storage inference shared by runtime and ingestion.

Records and dataset documents both carry a ``source`` field and a
``storage.base`` URI. They were sometimes written with the wrong values
(e.g. NEMAR datasets digested as ``source="openneuro"`` and
``storage.base="s3://openneuro.org/<id>"`` before PR #327). This module
centralises the pattern-based recovery rules so:

* the runtime client can self-heal already-ingested records, and
* the digestion pipeline can refuse to commit a source that disagrees
  with the dataset_id pattern.

The ``"nemar"`` backend identifier is *logical only* — NEMAR's S3 bucket
has ``s3:ListBucket`` and ``s3:GetObject`` closed by design and filenames
are SHA-resolved by git-annex, so the URI ``s3://nemar/<id>/<file>`` is
never directly fetchable. EEGDash treats ``backend="nemar"`` as a marker
that means "do not attempt a public S3 fetch; surface an actionable
``StorageAccessError`` if the file is not already in cache."

**Single source of truth** for ``STORAGE_CONFIGS``. The ingestion
pipeline (``scripts/ingestions/3_digest.py``) imports from here rather
than maintaining its own copy. This was promoted in Phase 8 / S1.thick
(2026-05) — see ``scripts/ingestions/source_adapter.py`` for the
ingest-side SourceAdapter Module that consumes this table.
"""

from __future__ import annotations

# Source -> remote storage prefix. Canonical home; ingestion imports
# from here (was previously duplicated in 3_digest.py).
STORAGE_CONFIGS: dict[str, dict[str, str]] = {
    "openneuro": {"backend": "s3", "base": "s3://openneuro.org"},
    # NEMAR uses a dedicated, non-fetchable backend tag. The ``base`` is
    # kept for traceability/error messages; the actual data must be
    # obtained via git-annex / nemar CLI / NEMAR API.
    "nemar": {"backend": "nemar", "base": "s3://nemar"},
    "gin": {"backend": "https", "base": "https://gin.g-node.org"},
    "figshare": {"backend": "https", "base": "https://figshare.com/ndownloader/files"},
    "zenodo": {"backend": "https", "base": "https://zenodo.org/records"},
    "osf": {"backend": "https", "base": "https://files.osf.io"},
    "scidb": {"backend": "https", "base": "https://www.scidb.cn"},
    "datarn": {"backend": "webdav", "base": "https://webdav.data.ru.nl"},
}

# Hot-path: a record is "misrouted" only when its base sits under one of
# these known foreign buckets. Frozen at module load; consulted in
# ``correct_storage_inplace`` on every record.
_FOREIGN_PREFIXES: frozenset[str] = frozenset(
    cfg["base"] for cfg in STORAGE_CONFIGS.values()
)


def infer_source_from_dataset_id(dataset_id: str) -> str | None:
    """Return the source implied by ``dataset_id``, or ``None`` if unknown.

    OpenNeuro IDs are ``dsNNNNNN``; NEMAR IDs are ``nmNNNNNN``;
    OpenNeuro-imported NEMAR mirrors are ``onNNNNNN`` (produced by
    ``nemar-cli``'s ``mapDatasetId``: ``dsNNNNNN -> onNNNNNN``). The
    ``EEGManyLabs`` projects live on GIN, and the HBN ``EEG2025*``
    releases are mirrored on NEMAR.
    """
    if not dataset_id:
        return None
    if dataset_id.startswith("ds") and dataset_id[2:].isdigit():
        return "openneuro"
    if dataset_id.startswith("nm") and dataset_id[2:].isdigit():
        return "nemar"
    if dataset_id.startswith("on") and dataset_id[2:].isdigit():
        return "nemar"
    if "EEGManyLabs" in dataset_id:
        return "gin"
    if dataset_id.startswith("EEG2025"):
        return "nemar"
    return None


def expected_storage_base(dataset_id: str) -> str | None:
    """Return the canonical ``storage.base`` for ``dataset_id``.

    Only returns a value for IDs whose pattern unambiguously determines
    the source (OpenNeuro and NEMAR). Returns ``None`` for sources that
    need extra metadata to build the URL (figshare/zenodo/osf use per-
    record IDs that aren't derivable from ``dataset_id``).
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


# Fallback when ``source`` isn't in STORAGE_CONFIGS — kept tiny so call
# sites can spell ``STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)``
# inline without a wrapper.
DEFAULT_STORAGE_CONFIG: dict[str, str] = {"backend": "https", "base": "https://unknown"}


def correct_storage_inplace(
    record: dict, dataset_id: str | None = None
) -> tuple[bool, str | None]:
    """Rewrite a record's ``storage.base`` / ``storage.backend`` if misrouted.

    Two distinct corrections happen here:

    1. **Base mismatch** — when the dataset_id pattern (e.g. ``nm*``)
       implies one source but ``storage.base`` points at a different
       known foreign bucket (e.g. ``s3://openneuro.org/...``). Residual
       fallout from the pre-PR-#327 ingestion bug.
    2. **Backend mismatch** — when the dataset_id pattern implies a
       backend (e.g. ``"nemar"``) that differs from what was stored
       (e.g. ``"s3"``). Records ingested between the source fix
       (PR #327) and the nemar-backend split still claim ``s3``.

    The two corrections are independent: a record with the right base
    but the wrong backend (very common for already-digested NEMAR
    records) still gets fixed.

    Returns
    -------
    (corrected, old_base)
        ``corrected`` is ``True`` when *either* field was rewritten.
        ``old_base`` is the previous ``storage.base`` value when it was
        rewritten, otherwise ``None`` (even if backend was rewritten).

    """
    dsid = dataset_id or record.get("dataset")
    if not dsid:
        return False, None

    expected_base = expected_storage_base(dsid)
    expected_backend_value = expected_backend(dsid)
    if expected_base is None:
        # Source can't be derived unambiguously from the dataset_id —
        # be conservative and leave both fields alone.
        return False, None

    storage = record.setdefault("storage", {})
    current_base = (storage.get("base") or "").rstrip("/")
    current_backend = storage.get("backend")

    if not current_base:
        # No base to reason about — don't flip backend in isolation.
        return False, None

    # Hot-path early-out for the dominant case (canonical base + correct
    # backend). Avoids the foreign-prefix scan on every healthy record.
    if current_base == expected_base and current_backend == expected_backend_value:
        return False, None

    # We rewrite only when the base is canonical (legacy backend may
    # still be stale) or a known foreign bucket (mis-routed by the
    # pre-PR-#327 ingestion bug). User-supplied bases are left alone.
    is_canonical = current_base == expected_base
    is_known_foreign = not is_canonical and any(
        current_base.startswith(p + "/") for p in _FOREIGN_PREFIXES
    )

    if not is_canonical and not is_known_foreign:
        return False, None

    old_base: str | None = None
    if is_known_foreign:
        storage["base"] = expected_base
        old_base = current_base

    if expected_backend_value is not None and current_backend != expected_backend_value:
        storage["backend"] = expected_backend_value

    return True, old_base
