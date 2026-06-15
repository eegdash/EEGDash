"""Ingestion domain glossary — the ubiquitous language + one import surface.

This module defines **no new types**. It pins the canonical vocabulary for the
ingestion pipeline (so code, tests, and docs share one language) and re-exports
the existing artifact types. The Pydantic models and the ``create_dataset`` /
``create_record`` factories remain the single source of truth in
``eegdash.schemas`` — do not duplicate them here.

Stage vocabulary (the pipeline, left to right):

    SourceListing     stage 1  files discovered at a remote source (Zenodo / OSF /
                               OpenNeuro / Figshare / SciDB / DataRN / NEMAR)
    CloneManifest     stage 2  ``manifest.json`` describing a clonable/materialized
                               dataset (validated by ``ManifestModel``)
    LocalDataset      stage 2  a dataset present on disk (id + dir + source)
    DigestBundle      stage 3  one dataset's dataset-doc + records + montages
                               (currently produced as ``EnumerationResult``)
    ValidationReport  stage 4  structured errors / warnings / stats
                               (currently ``ValidationResult``)
    InjectionPlan     stage 5  the create / update / skip decision for the API

Naming convergence (see docs/adr/0001): the canonical names are ``DigestBundle``
and ``ValidationReport``; ``EnumerationResult`` / ``ValidationResult`` are the
current implementations and will gain back-compat aliases in a later phase. The
single database write boundary is the *EegdashApi* (today: the inline calls in
``5_inject.py`` and ``_inject_plan.fetch_existing_dataset``).
"""

from __future__ import annotations

# Stage artifacts owned by the ingestion package.
from _inject_plan import InjectionPlan
from _validate import ValidationResult

# Core entity contracts — single source of truth lives in eegdash.schemas.
from eegdash.schemas import (
    Dataset,
    DatasetModel,
    ManifestModel,
    Montage,
    Record,
    RecordModel,
    create_dataset,
    create_record,
)
from record_enumerator import EnumerationResult

__all__ = [
    # entities / contracts (from eegdash.schemas)
    "Dataset",
    "DatasetModel",
    # stage artifacts
    "EnumerationResult",  # DigestBundle (stage 3)
    "InjectionPlan",  # stage 5
    "ManifestModel",
    "Montage",
    "Record",
    "RecordModel",
    "ValidationResult",  # ValidationReport (stage 4)
    "create_dataset",
    "create_record",
]
