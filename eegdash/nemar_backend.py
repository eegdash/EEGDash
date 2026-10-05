# Authors: The EEGDash contributors.
# License: BSD-3-Clause
"""Experimental, metadata-only NEMAR backend (not a recording client).

Uses the same neuroschema field mappings as the ingestion adapter, but deliberately
retains unknown modalities/species/ages instead of its ingestion defaults. The
records.json fast path and nemar-py remain responsible for signal metadata and
retrieval; neither implies parity with EEGDash's primary-file record contract.
"""

import json
import re
import time

import requests


class NemarContractError(RuntimeError):
    """Upstream metadata cannot satisfy the supported contract safely."""


class NemarUnsupportedOperation(NotImplementedError):
    """An operation needs EEGDash semantics not supplied by this backend."""


class NemarMetadataClient:
    """Public discovery only; explicit canonical IDs, no automatic failover.

    Requests have 5s connect/15s read timeouts, a 30s streaming deadline,
    a 2 MiB decoded body cap and no retries or redirects. Listings allow
    at most 1000 documents (five 200-document pages). They are live views,
    not snapshot-consistent exports. Errors never return a partial list.
    """

    API_URL = "https://api.nemar.org"
    DATA_URL = "https://data.nemar.org"
    MAX_BYTES = 2 * 1024 * 1024

    def __init__(self):
        # Do not forward EEGDash credentials/admin headers to another provider.
        self._session = requests.Session()
        self._session.trust_env = False

    def _get(self, url, *, params=None, missing_ok=False):
        deadline = time.monotonic() + 30
        with self._session.get(
            url, params=params, timeout=(5, 15), stream=True, allow_redirects=False
        ) as response:
            if missing_ok and response.status_code == 404:
                return None
            if 300 <= response.status_code < 400:
                raise NemarContractError("Unexpected NEMAR redirect")
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_content(chunk_size=16384):
                body.extend(chunk)
                if len(body) > self.MAX_BYTES or time.monotonic() > deadline:
                    raise NemarContractError("NEMAR metadata exceeds response budget")
            try:
                value = json.loads(body)
            except (ValueError, UnicodeError) as exc:
                raise NemarContractError("Invalid NEMAR JSON") from exc
            if not isinstance(value, dict):
                raise NemarContractError("Expected NEMAR JSON object")
            if value.get("fallback") or value.get("degraded") or value.get("error"):
                raise NemarContractError("NEMAR returned degraded/error metadata")
            return value

    @staticmethod
    def _id(value):
        if not isinstance(value, str) or not re.fullmatch(r"(?:nm|on)\d{6}", value):
            raise NemarUnsupportedOperation(
                "Use a canonical nmNNNNNN/onNNNNNN ID; source-ID aliases are unsupported"
            )
        return value

    @staticmethod
    def _version(value):
        if value is not None and (
            not isinstance(value, str) or not re.fullmatch(r"v\d+\.\d+\.\d+", value)
        ):
            raise NemarContractError("Invalid NEMAR version")
        return value

    def get_dataset(self, dataset_id):
        """Read current rich metadata, preserving its reported source snapshot.

        The metadata endpoint is mutable; version DOIs do not make the document
        an immutable historical snapshot. No historical metadata claim is made.
        """
        dataset_id = self._id(dataset_id)
        url = f"{self.DATA_URL}/{dataset_id}/metadata.json"
        doc = self._get(url, missing_ok=True)
        if doc is None:
            return None
        if doc.get("dataset_id") != dataset_id or doc.get("doc_type") != "dataset":
            raise NemarContractError("Unexpected NEMAR dataset identity/type")
        if not str(doc.get("schema_version", "")).startswith("0.4."):
            raise NemarContractError("Unsupported NEMAR neuroschema version")
        try:
            prov = doc.get("provenance") or {}
            ext = doc.get("external_links") or {}
            demo = doc.get("demographics") or {}
            version = self._version(prov.get("latest_snapshot"))
            versions = (
                (doc.get("extensions") or {}).get("nemar", {}).get("versions", [])
            )
            matching = [v for v in versions if v.get("version") == version]
            if len(matching) > 1:
                raise NemarContractError("Duplicate NEMAR version identity")
            version_doi = matching[0].get("doi") if matching else None
            # Same source fields as scripts/ingestions/1_fetch_sources/nemar.py.
            # Keep the original structured authors, citations and provenance too.
            return {
                **doc,
                "requested_dataset_id": dataset_id,
                "provider": "nemar",
                "modality": doc.get("recording_modality"),
                "task": doc.get("tasks"),
                "n_subjects": demo.get("subjects_count"),
                "subjects_count": demo.get("subjects_count"),
                "dataset_doi": ext.get("dataset_doi"),
                "version": version,
                "version_doi": version_doi,
                "citation": doc.get("citation"),
                "metadata_url": url,
                "metadata_scope": "current",
            }
        except (AttributeError, TypeError) as exc:
            raise NemarContractError("Malformed NEMAR rich metadata") from exc

    def find_datasets(self, query=None, limit=1000):
        """List up to limit datasets, or fetch one exact canonical dataset ID.

        Only {} / None / {'dataset_id': 'nmNNNNNN' or 'onNNNNNN'} is supported.
        No Mongo filters, source aliases, fuzzy search, record counts or writes.
        """
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("NEMAR dataset limit must be an integer from 1 to 1000")
        if query is not None and not isinstance(query, dict):
            raise NemarUnsupportedOperation("NEMAR query must be a dictionary")
        if query:
            if set(query) != {"dataset_id"}:
                raise NemarUnsupportedOperation("Only exact dataset_id is supported")
            doc = self.get_dataset(query["dataset_id"])
            return [] if doc is None else [doc]
        result, seen = [], set()
        total = None
        while len(result) < limit:
            size = min(200, limit - len(result))
            page = self._get(
                f"{self.API_URL}/datasets",
                params={"limit": size, "offset": len(result)},
            )
            rows = page.get("datasets")
            count = page.get("total_count")
            if (
                not isinstance(rows, list)
                or type(count) is not int
                or count < 0
                or page.get("offset") != len(result)
                or page.get("limit") != size
                or page.get("count") != len(rows)
                or len(rows) != min(size, max(0, count - len(result)))
                or (total is not None and count != total)
            ):
                raise NemarContractError("Inconsistent NEMAR catalog pagination")
            total = count
            for row in rows:
                if not isinstance(row, dict):
                    raise NemarContractError("Malformed NEMAR catalog row")
                try:
                    dataset_id = self._id(row.get("dataset_id"))
                except NemarUnsupportedOperation as exc:
                    raise NemarContractError("Invalid catalog dataset ID") from exc
                if dataset_id in seen:
                    raise NemarContractError("NEMAR catalog changed during pagination")
                seen.add(dataset_id)
                result.append(
                    {
                        "dataset_id": dataset_id,
                        "name": row.get("name"),
                        "source": row.get("source"),
                        "source_id": row.get("source_id"),
                        "provider": "nemar",
                        "license": row.get("license"),
                        "dataset_doi": row.get("concept_doi"),
                        "version": self._version(row.get("latest_version")),
                        "n_subjects": row.get("subject_count"),
                        "subjects_count": row.get("subject_count"),
                        "modality": row.get("modalities"),
                        "task": row.get("tasks"),
                        "citation": row.get("citation"),
                        "metadata_scope": "catalog",
                        "nemar_catalog": row,
                    }
                )
            if len(result) >= count:
                break
        return result

    def _unsupported(self, *args, **kwargs):
        raise NemarUnsupportedOperation(
            "NEMAR metadata backend supports only find_datasets/get_dataset; "
            "record queries/counts, participants, loading and writes require EEGDash"
        )

    find = find_one = count_documents = _unsupported
    insert_one = insert_many = update_many = update_dataset = upsert_many = _unsupported
