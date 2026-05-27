"""eegdash ingestion pipeline.

This package contains the scripts and helpers that ingest BIDS EEG/MEG/iEEG
datasets from OpenNeuro, NEMAR, Figshare, Zenodo, OSF, SciDB, DataRN, and
EEGManyLabs into the eegdash MongoDB.

Pipeline stages (top-level scripts):

1. ``1_fetch_sources/`` — fetch dataset listings from each source.
2. ``2_clone.py``       — clone or mirror selected datasets to local disk.
3. ``3_digest.py``      — walk BIDS hierarchies, parse headers, emit Records.
4. ``4_validate_output.py`` — Pydantic-validate the emitted JSON corpus.
5. ``5_inject.py``      — write validated Records into MongoDB.

Shared helpers live in the underscore-prefixed modules:

- ``_bids``           BIDS sidecar inheritance walk
- ``_constants``      shared magic numbers and enums
- ``_file_utils``     filesystem + zip helpers
- ``_fingerprint``    content-hash helpers for record dedup
- ``_github``         GitHub-API helpers (datalad mirrors)
- ``_http``           shared ``httpx.Client`` with ``tenacity`` retry
- ``_keywords``       BIDS keyword normalisation
- ``_mef3_parser``    Multiscale Electrophysiology Format v3 metadata
- ``_montage``        electrode coordinate / montage extraction
- ``_parser_utils``   shared helpers for the per-format parsers
- ``_serialize``      JSON (de)serialisation helpers
- ``_set_parser``     EEGLAB ``.set`` header parser
- ``_snirf_parser``   fNIRS SNIRF metadata
- ``_validate``       Pydantic validation re-exports
- ``_vhdr_parser``    BrainVision ``.vhdr`` parser

See  for the on-going test-coverage / robustness programme.
"""

from __future__ import annotations

__version__ = "0.1.0"
