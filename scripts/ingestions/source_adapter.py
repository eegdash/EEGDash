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

from _file_utils import get_annex_file_key
from eegdash.dataset._source_inference import DEFAULT_STORAGE_CONFIG, STORAGE_CONFIGS

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
        return STORAGE_CONFIGS.get(self.source_name, DEFAULT_STORAGE_CONFIG)["backend"]

    @property
    def storage_base(self) -> str:
        """The ``Record.storage.base`` URI prefix for this Dataset."""
        base = STORAGE_CONFIGS.get(self.source_name, DEFAULT_STORAGE_CONFIG)["base"]
        return f"{base}/{self.dataset_id}"

    # ─── Behaviour hooks ─────────────────────────────────────────────

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
    """NEMAR Source — git-annex SHA addressing.

    NEMAR's S3 bucket has ``s3:ListBucket`` closed by design; all file
    addresses are SHA-resolved via git-annex on the cloned filesystem.
    The ``backend="nemar"`` marker signals downstream consumers not to
    attempt a public S3 fetch.

    Sidecar *contents* are intentionally NOT inlined into records. A
    digest-time inline copy bloats every record (a single typing
    ``events.tsv`` is ~230 KB, and dataset-level apex files duplicate
    across every record) and goes stale as the dataset evolves. The
    runtime fetches sidecars on demand from the NEMAR GitHub mirror
    (``HEAD``, always current) instead, so only annex SHA keys — the
    binary recordings on S3 — are resolved here.
    """

    @property
    def source_name(self) -> str:
        return "nemar"

    def dataset_url(self) -> str | None:
        """NEMAR DataExplorer landing page."""
        return f"https://nemar.org/dataexplorer/detail/{self.dataset_id}"

    def _annex_relpath(self, path: Path) -> str:
        """BIDS-relative key for *path* (annex keys are stored by relpath)."""
        if self.bids_root is not None:
            try:
                return str(path.relative_to(self.bids_root))
            except ValueError:
                pass
        return str(path)

    def resolve_storage_extensions(
        self,
        record_path: Path,
        dep_paths: list[Path],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Resolve git-annex SHA keys for one Record (no sidecar inlining)."""
        annex_keys: dict[str, str] = {}

        raw_key = get_annex_file_key(record_path)
        if raw_key is not None:
            annex_keys[self._annex_relpath(record_path)] = raw_key

        for dep_path in dep_paths:
            dep_key = get_annex_file_key(dep_path)
            if dep_key is not None:
                annex_keys[self._annex_relpath(dep_path)] = dep_key

        return annex_keys, {}


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
