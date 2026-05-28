"""Per-Source ingest behaviour — SourceAdapter Seam.

One Adapter class per Source, instantiated per Dataset so per-Dataset state
(apex sidecar cache, annex-key cache) lives as instance attributes. OpenNeuro
and NEMAR have dedicated subclasses; everything else falls through to
:class:`DefaultAdapter`. The factory :func:`get_source_adapter` is the sole
entry point.

The four divergences between OpenNeuro and NEMAR — storage backend tag, base
URL, per-file addressing (NEMAR requires git-annex SHA resolution), and apex
sidecar prefetch — are encapsulated here rather than scattered as
``if source == "nemar"`` ladders across the pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from _file_utils import get_annex_file_key, read_inline_sidecar
from eegdash.dataset._source_inference import (
    get_storage_backend,
    get_storage_base,
)
from eegdash.schemas import NEMAR_ROOT_METADATA_FILES

logger = logging.getLogger(__name__)


class SourceAdapter(ABC):
    """Per-Source ingest behaviour for one Dataset.

    Subclasses override only the methods that differ from the default
    (straight S3/HTTPS addressing, no prefetch, no annex resolution).
    ``bids_root`` is required for any Adapter that does on-disk reads
    (NEMAR apex prefetch, annex-key resolution).
    """

    def __init__(self, dataset_id: str, bids_root: Path | None = None) -> None:
        self.dataset_id = dataset_id
        self.bids_root = bids_root

    # ─── Source identity ─────────────────────────────────────────────

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Source identifier as written into Records (``"openneuro"`` etc.)."""

    @property
    def storage_backend(self) -> str:
        """The ``Record.storage.backend`` marker for this Source."""
        return get_storage_backend(self.source_name)

    @property
    def storage_base(self) -> str:
        """The ``Record.storage.base`` URI prefix for this Dataset."""
        return get_storage_base(self.dataset_id, self.source_name)

    # ─── Behaviour hooks ─────────────────────────────────────────────

    def address_file(self, bids_relpath: str) -> str:
        """Return the storage address for a BIDS file (``<base>/<relpath>``)."""
        return f"{self.storage_base}/{bids_relpath}"

    def dataset_url(self) -> str | None:
        """User-facing landing-page URL. Returns ``None`` for secondary Sources."""
        return None

    def resolve_storage_extensions(
        self,
        record_path: Path,
        dep_paths: list[Path],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Return ``(annex_keys, sidecar_inline)`` for a single Record.

        Default returns two empty dicts. NEMAR overrides to resolve
        git-annex SHA keys and inline apex sidecar content.
        """
        return {}, {}


class OpenNeuroAdapter(SourceAdapter):
    """OpenNeuro Source — direct S3 addressing.

    All file URIs follow ``s3://openneuro.org/<dataset_id>/<rel_path>``.
    No git-annex resolution required; no sidecar prefetch needed.
    """

    @property
    def source_name(self) -> str:
        return "openneuro"

    def dataset_url(self) -> str | None:
        """Public OpenNeuro dataset landing page."""
        return f"https://openneuro.org/datasets/{self.dataset_id}"


class NEMARAdapter(SourceAdapter):
    """NEMAR Source — git-annex SHA addressing + apex sidecar prefetch.

    NEMAR's S3 bucket has ``s3:ListBucket`` closed by design; all file
    addresses are SHA-resolved via git-annex on the cloned filesystem.
    The ``backend="nemar"`` marker signals downstream consumers not to
    attempt a public S3 fetch.  :data:`NEMAR_ROOT_METADATA_FILES` are
    read once (lazily, into ``_apex_cache``) and reused across every
    Record to avoid repeated filesystem round-trips.
    """

    def __init__(self, dataset_id: str, bids_root: Path | None = None) -> None:
        super().__init__(dataset_id, bids_root)
        self._apex_cache: dict[str, str] | None = None

    @property
    def source_name(self) -> str:
        return "nemar"

    def dataset_url(self) -> str | None:
        """NEMAR DataExplorer landing page."""
        return f"https://nemar.org/dataexplorer/detail/{self.dataset_id}"

    def _ensure_apex_cache(self) -> dict[str, str]:
        """Lazy-build the apex sidecar inline cache (once per Adapter)."""
        if self._apex_cache is not None:
            return self._apex_cache
        cache: dict[str, str] = {}
        if self.bids_root is None:
            self._apex_cache = cache
            return cache
        for root_name in NEMAR_ROOT_METADATA_FILES:
            inline = read_inline_sidecar(self.bids_root / root_name)
            if inline is not None:
                cache[root_name] = inline
        self._apex_cache = cache
        return cache

    def resolve_storage_extensions(
        self,
        record_path: Path,
        dep_paths: list[Path],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Resolve annex keys and inline sidecars for one Record."""
        annex_keys: dict[str, str] = {}
        sidecar_inline: dict[str, str] = dict(self._ensure_apex_cache())

        # The recording itself
        raw_key = get_annex_file_key(record_path)
        if raw_key is not None:
            # NEMAR annex keys are stored against the BIDS relative path
            # — preserve the convention 3_digest used before this refactor.
            if self.bids_root is not None:
                try:
                    rel = str(record_path.relative_to(self.bids_root))
                except ValueError:
                    rel = str(record_path)
            else:
                rel = str(record_path)
            annex_keys[rel] = raw_key

        # Each dep file: annex key OR inlined text
        for dep_path in dep_paths:
            if self.bids_root is not None:
                try:
                    dep_relpath = str(dep_path.relative_to(self.bids_root))
                except ValueError:
                    dep_relpath = str(dep_path)
            else:
                dep_relpath = str(dep_path)

            dep_key = get_annex_file_key(dep_path)
            if dep_key is not None:
                annex_keys[dep_relpath] = dep_key
                continue
            if dep_relpath in sidecar_inline:
                continue  # already covered by apex cache
            inline = read_inline_sidecar(dep_path)
            if inline is not None:
                sidecar_inline[dep_relpath] = inline

        return annex_keys, sidecar_inline


class DefaultAdapter(SourceAdapter):
    """Fallback Adapter for GIN and secondary Sources (table-driven, no overrides)."""

    def __init__(
        self,
        dataset_id: str,
        source_name: str,
        bids_root: Path | None = None,
    ) -> None:
        super().__init__(dataset_id, bids_root)
        self._source_name = source_name

    @property
    def source_name(self) -> str:
        return self._source_name


# ─── Factory ──────────────────────────────────────────────────────────


_ADAPTER_CLASSES: dict[str, type[SourceAdapter]] = {
    "openneuro": OpenNeuroAdapter,
    "nemar": NEMARAdapter,
}


def get_source_adapter(
    source: str,
    dataset_id: str,
    bids_root: Path | None = None,
) -> SourceAdapter:
    """Return the :class:`SourceAdapter` for a ``(source, dataset_id)`` pair.

    Unknown sources fall through to :class:`DefaultAdapter`.
    ``bids_root`` is required for NEMAR (apex prefetch + annex resolution).
    """
    adapter_cls = _ADAPTER_CLASSES.get(source)
    if adapter_cls is None:
        return DefaultAdapter(dataset_id, source, bids_root)
    return adapter_cls(dataset_id, bids_root)


__all__ = [
    "DefaultAdapter",
    "NEMARAdapter",
    "OpenNeuroAdapter",
    "SourceAdapter",
    "get_source_adapter",
]
