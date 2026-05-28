"""Per-Source ingest behaviour — SourceAdapter Seam.

Background
----------

The ingestion pipeline distinguishes between several *Sources* —
OpenNeuro, NEMAR, GIN/EEGManyLabs, and a handful of "secondary"
providers (Figshare / Zenodo / OSF / SciDB / DataRN) that are
best-effort per ADRs/0001-secondary-source-deferral.md``.

In production, only **OpenNeuro** and **NEMAR** are exercised on every
CI run. Their ingest behaviour diverges in four places:

1. **Storage backend tag.** ``openneuro`` -> ``"s3"``, ``nemar`` ->
   ``"nemar"`` (a marker; the bucket is not directly fetchable, see
   :mod:`eegdash.dataset._source_inference`).
2. **Storage base URL.** ``s3://openneuro.org`` vs ``s3://nemar``.
3. **Per-file addressing.** OpenNeuro: ``<base>/<dataset>/<bids_relpath>``
   straight. NEMAR: BIDS path must be resolved to a git-annex SHA key
   before download is possible.
4. **Apex sidecar inline-prefetch.** NEMAR datasets carry per-dataset
   metadata files (``participants.tsv``, ``dataset_description.json``,
   ...) that are read once per Dataset and inlined into every Record so
   the runtime never has to re-resolve them.
5. **User-facing landing URL.** ``https://openneuro.org/datasets/<id>``
   vs ``https://nemar.org/dataexplorer/detail/<id>``.

Before this Module these five divergences lived as
``if source == "nemar"`` ladders in four different places inside
``3_digest.py``. Each new Source-specific tweak meant editing all four.

Design
------

One Adapter class per Source. Instances are created per Dataset (in
``digest_dataset``) so per-Dataset state — the apex sidecar inline
cache, the annex-key cache — lives as instance attributes rather than
parameters threaded through ``extract_record``'s signature.

The base class :class:`SourceAdapter` implements the dominant case
(straight S3 / HTTPS addressing, no prefetch, no annex resolution).
:class:`OpenNeuroAdapter` adds only the public landing URL.
:class:`NEMARAdapter` adds the annex-key + apex-prefetch overrides.
:class:`DefaultAdapter` is the fallback for GIN and the five secondary
sources — table-driven, no overrides.

The factory :func:`get_source_adapter` is the only entry point.

See Also
--------
- ``eegdash.dataset._source_inference`` — the shared table
  (``STORAGE_CONFIGS``) + the runtime self-heal counterpart
  (``correct_storage_inplace``).
  secondary sources don't get their own Adapter classes.
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

    Parameters
    ----------
    dataset_id : str
        Dataset accession (e.g. ``"ds002893"``).
    bids_root : Path, optional
        Filesystem root of the BIDS layout for this dataset. Required
        for any Adapter that does on-disk reads (NEMAR apex prefetch,
        annex-key resolution). Can be ``None`` for callers that only
        need URL / backend information.

    Attributes
    ----------
    dataset_id : str
    bids_root : Path or None
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
        """Return the storage address for a specific BIDS file.

        Default: ``<storage_base>/<bids_relpath>``. NEMAR overrides
        because BIDS paths must be resolved through git-annex first.
        """
        return f"{self.storage_base}/{bids_relpath}"

    def dataset_url(self) -> str | None:
        """User-facing landing-page URL for this Dataset.

        Default ``None`` — most secondary Sources don't have a public
        landing page that's discoverable from the ``dataset_id`` alone.
        OpenNeuro and NEMAR override.
        """
        return None

    def resolve_storage_extensions(
        self,
        record_path: Path,
        dep_paths: list[Path],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Return ``(annex_keys, sidecar_inline)`` for a single Record.

        The default returns two empty dicts — most Sources don't use
        git-annex addressing or per-Record sidecar inlining. NEMAR
        overrides to populate both.

        Parameters
        ----------
        record_path : Path
            Absolute path to the recording's data file.
        dep_paths : list[Path]
            Absolute paths to every dependency sidecar (channels.tsv,
            electrodes.tsv, etc.) that the Record's ``dep_keys`` lists.

        Returns
        -------
        annex_keys : dict[str, str]
            Mapping of ``bids_relpath -> annex SHA key`` for files
            tracked by git-annex. Empty for OpenNeuro.
        sidecar_inline : dict[str, str]
            Mapping of ``bids_relpath -> file content (str)`` for the
            sidecars that are inlined into the Record's storage entry
            so the runtime doesn't need to re-resolve them.
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

    NEMAR's S3 bucket has ``s3:ListBucket`` closed by design; filenames
    are SHA-resolved by git-annex on the cloned filesystem. The
    ``backend="nemar"`` marker tells downstream consumers "do not
    attempt a public S3 fetch".

    The apex inline-prefetch optimisation reads
    :data:`NEMAR_ROOT_METADATA_FILES` once at Adapter init and reuses
    the decoded content across every Record (Python string immutability
    means the values are shared by reference). Avoids
    ``N x len(NEMAR_ROOT_METADATA_FILES)`` filesystem reads per Dataset.

    State (per Adapter / per Dataset):

    - ``_apex_cache`` — populated lazily on first
      :meth:`resolve_storage_extensions` call, then reused.
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
        """Resolve annex keys + inline sidecars for one Record (NEMAR-specific).

        - The recording's own annex key (if it's an annex-tracked file).
        - Each dependency's annex key, OR its inlined content if it's
          a small enough sidecar (see :func:`read_inline_sidecar`).
        - The apex cache (built once per Dataset) is merged into the
          returned ``sidecar_inline`` dict.
        """
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
    """Fallback Adapter for GIN + the 5 secondary Sources.

    Table-driven addressing only; no per-Source overrides. Used for
    sources that exist in :data:`STORAGE_CONFIGS` but don't have a
    dedicated Adapter class (because they have no special behaviour to
    encapsulate — yet).

    See  for the
    rationale (1 production Adapter + 6 secondary = no leverage for a
    7-Adapter shared Protocol today).
    """

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

    Parameters
    ----------
    source : str
        Source identifier (``"openneuro"``, ``"nemar"``, ``"gin"``,
        ``"zenodo"``, etc.). Must be a key in :data:`STORAGE_CONFIGS`
        or a :class:`DefaultAdapter` is returned that uses
        :data:`DEFAULT_STORAGE_CONFIG`.
    dataset_id : str
        Dataset accession.
    bids_root : Path, optional
        Filesystem root of the BIDS layout. Required for NEMAR (apex
        prefetch + annex resolution); optional for the others.

    Returns
    -------
    SourceAdapter
        An :class:`OpenNeuroAdapter`, :class:`NEMARAdapter`, or
        :class:`DefaultAdapter` instance bound to the given dataset.
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
