# Authors: The EEGDash contributors.
# License: BSD-3-Clause

"""Pydantic models for EEGDash documents.

These models are intended for validation of documents exchanged with the EEGDash
REST API and ingestion pipeline outputs. They are kept permissive (extra fields
allowed) so the schema can evolve without breaking older clients.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StorageModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = Field(min_length=1)
    base: str = Field(min_length=1)
    raw_key: str = Field(min_length=1)
    dep_keys: list[str] = Field(default_factory=list)


class EntitiesModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    subject: str | None = None
    session: str | None = None
    task: str | None = None
    run: str | None = None


class RecordModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset: str = Field(min_length=1)
    bids_relpath: str = Field(min_length=1)
    storage: StorageModel
    recording_modality: list[str] = Field(min_length=1)

    datatype: str | None = None
    suffix: str | None = None
    extension: str | None = None
    entities: EntitiesModel | dict[str, Any] | None = None


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    recording_modality: list[str] = Field(min_length=1)
    ingestion_fingerprint: str | None = None


class ManifestFileModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str | None = None
    name: str | None = None

    def path_or_name(self) -> str:
        return (self.path or self.name or "").strip()


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str | None = None
    files: list[str | ManifestFileModel]
