# Authors: The EEGDash contributors.
# License: BSD-3-Clause

"""HTTP API client for EEGDash REST API."""

import json
import os
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .dataset._source_inference import nemar_twin_of

DEFAULT_API_URL = "https://data.eegdash.org"
# Retry on 429 (Too Many Requests) and standard server errors
_RETRY = Retry(total=5, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1.0)


def _nemar_twin(dataset_id: Any) -> str | None:
    """Return the NEMAR twin id for an OpenNeuro dataset id, else ``None``."""
    return nemar_twin_of(dataset_id) if isinstance(dataset_id, str) else None


def _twin_value(value: Any) -> Any | None:
    """Return ``value`` with OpenNeuro ids swapped for NEMAR twins, else ``None``.

    Handles a plain id and an ``{"$in": [...]}`` list, since selecting several
    datasets by id is a common query shape. Any other operator form (a regex, a
    negated list) is left alone.
    """
    twin = _nemar_twin(value)
    if twin is not None:
        return twin
    if isinstance(value, dict) and isinstance(value.get("$in"), list):
        members = value["$in"]
        swapped = [_nemar_twin(m) or m for m in members]
        if swapped != members:
            return {**value, "$in": swapped}
    return None


def _aliased(query: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    """Copy of ``query`` with ``query[key]`` swapped for its NEMAR twin(s).

    Returns ``None`` when there is nothing to alias.
    """
    if not isinstance(query, dict):
        return None
    swapped = _twin_value(query.get(key))
    return None if swapped is None else {**query, key: swapped}


def _make_session(auth_token: str | None = None) -> requests.Session:
    """Create session with retry strategy."""
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=_RETRY))
    session.mount("http://", HTTPAdapter(max_retries=_RETRY))
    if auth_token:
        session.headers.update({"Authorization": f"Bearer {auth_token}"})

    # Inject Admin Token for Rate Limit Bypass in tests/dev (without full auth)
    admin_token_env = os.environ.get("EEGDASH_ADMIN_TOKEN")
    if admin_token_env:
        session.headers.update({"X-EEGDASH-TOKEN": admin_token_env})

    return session


class EEGDashAPIClient:
    """HTTP client for EEGDash API.

    Parameters
    ----------
    api_url : str, optional
        Base API URL. Default: https://data.eegdash.org
    database : str, default "eegdash"
        Database name ("eegdash", "eegdash_staging", or "eegdash_v1").
    auth_token : str, optional
        Auth token for admin write operations.

    """

    def __init__(
        self,
        api_url: str | None = None,
        database: str = "eegdash",
        auth_token: str | None = None,
    ):
        self.api_url = (
            api_url or os.getenv("EEGDASH_API_URL", DEFAULT_API_URL)
        ).rstrip("/")
        self.database = database
        self._session = _make_session(auth_token or os.getenv("EEGDASH_API_TOKEN"))

    def find(
        self,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
        skip: int | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Query records. Auto-paginates if no limit specified.

        A ``dataset`` filter naming an OpenNeuro id that returns nothing is
        retried once against its NEMAR twin, so code written against a
        ``ds`` id keeps working after that id is retired in favour of the
        NEMAR re-host. Ids that still resolve are never rewritten.
        """
        results = self._find_records(query, limit=limit, skip=skip)
        if not results:
            alias = _aliased(query, "dataset")
            if alias is not None:
                results = self._find_records(alias, limit=limit, skip=skip)
        return results

    def _find_records(
        self,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ) -> list[dict[str, Any]]:
        """Single-shot record query with no twin fallback."""
        params: dict[str, Any] = {}
        if query:
            params["filter"] = json.dumps(query)
        if skip:
            params["skip"] = skip

        url = f"{self.api_url}/api/{self.database}/records"

        if limit is not None:
            params["limit"] = limit
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("data", [])

        # Auto-paginate
        all_records: list[dict] = []
        page_skip = skip or 0
        while True:
            resp = self._session.get(
                url, params={**params, "limit": 1000, "skip": page_skip}, timeout=60
            )
            resp.raise_for_status()
            records = resp.json().get("data", [])
            if not records:
                break
            all_records.extend(records)
            if len(records) < 1000:
                break
            page_skip += 1000
        return all_records

    def find_one(
        self, query: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any] | None:
        """Find a single record."""
        results = self.find(query, limit=1)
        return results[0] if results else None

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        """Fetch a dataset document by ID.

        A retired OpenNeuro id falls back to its NEMAR twin (see
        :func:`_nemar_twin`) so existing references keep resolving.
        """
        doc = self._get_dataset_doc(dataset_id)
        if doc is None:
            twin = _nemar_twin(dataset_id)
            if twin is not None:
                doc = self._get_dataset_doc(twin)
        return doc

    def _get_dataset_doc(self, dataset_id: str) -> dict[str, Any] | None:
        """Single-shot dataset fetch with no twin fallback."""
        resp = self._session.get(
            f"{self.api_url}/api/{self.database}/datasets/{dataset_id}", timeout=30
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("data")

    def find_datasets(
        self, query: dict[str, Any] | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Find datasets matching query.

        A ``dataset_id`` filter that returns nothing is retried against its
        NEMAR twin, mirroring :meth:`find`.
        """
        results = self._find_dataset_docs(query, limit=limit)
        if not results:
            alias = _aliased(query, "dataset_id")
            if alias is not None:
                results = self._find_dataset_docs(alias, limit=limit)
        return results

    def _find_dataset_docs(
        self, query: dict[str, Any] | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Single-shot dataset query with no twin fallback."""
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["filter"] = json.dumps(query)

        resp = self._session.get(
            f"{self.api_url}/api/{self.database}/datasets", params=params, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get("data", [])
        return data

    def count_documents(self, query: dict[str, Any] | None = None, **kwargs) -> int:
        """Count documents matching query.

        A zero count for an OpenNeuro ``dataset`` id is retried against its
        NEMAR twin, mirroring :meth:`find`.
        """
        count = self._count_documents(query)
        if count == 0:
            alias = _aliased(query, "dataset")
            if alias is not None:
                count = self._count_documents(alias)
        return count

    def _count_documents(self, query: dict[str, Any] | None = None) -> int:
        """Single-shot count with no twin fallback."""
        params = {"filter": json.dumps(query)} if query else {}
        resp = self._session.get(
            f"{self.api_url}/api/{self.database}/count", params=params, timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("count", 0)

    def insert_one(self, record: dict[str, Any]) -> str:
        """Insert single record (requires auth)."""
        resp = self._session.post(
            f"{self.api_url}/admin/{self.database}/records", json=record, timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("insertedId", "")

    def insert_many(self, records: list[dict[str, Any]]) -> int:
        """Insert multiple records (requires auth)."""
        resp = self._session.post(
            f"{self.api_url}/admin/{self.database}/records/bulk",
            json=records,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("insertedCount", 0)

    def update_many(
        self, query: dict[str, Any], update: dict[str, Any]
    ) -> tuple[int, int]:
        """Update records matching query (requires auth).

        Parameters
        ----------
        query : dict
            Filter query to match records.
        update : dict
            Fields to set (wrapped in $set automatically).

        Returns
        -------
        tuple of (matched_count, modified_count)

        """
        resp = self._session.patch(
            f"{self.api_url}/admin/{self.database}/records",
            json={"filter": query, "update": update},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("matched_count", 0), data.get("modified_count", 0)

    def update_dataset(self, dataset_id: str, update: dict[str, Any]) -> int:
        """Update dataset metadata (requires auth).

        Parameters
        ----------
        dataset_id : str
            The dataset identifier.
        update : dict
            Fields to update (will be wrapped in $set automatically).

        Returns
        -------
        int
            Modified count (1 or 0).

        """
        resp = self._session.patch(
            f"{self.api_url}/admin/{self.database}/datasets/{dataset_id}",
            json={"update": update},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("modified_count", 0)

    def upsert_many(self, records: list[dict[str, Any]]) -> dict[str, int]:
        """Upsert multiple records (requires auth).

        New endpoint that uses bulk upsert based on dataset+bidspath.
        """
        resp = self._session.post(
            f"{self.api_url}/admin/{self.database}/records/upsert",
            json=records,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "inserted_count": data.get("inserted_count", 0),
            "updated_count": data.get("updated_count", 0),
        }


def get_client(
    api_url: str | None = None, database: str = "eegdash", auth_token: str | None = None
) -> EEGDashAPIClient:
    """Get an API client instance."""
    return EEGDashAPIClient(api_url=api_url, database=database, auth_token=auth_token)


__all__ = ["EEGDashAPIClient", "get_client"]
