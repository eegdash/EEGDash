# Authors: The EEGDash contributors.
# License: BSD-3-Clause

"""HTTP API client for EEGDash REST API."""

import json
import os
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_API_URL = "https://data.eegdash.org"
_RETRY = Retry(total=3, status_forcelist=[500, 502, 503, 504], backoff_factor=1)


def _make_session(auth_token: str | None = None) -> requests.Session:
    """Create session with retry strategy."""
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=_RETRY))
    session.mount("http://", HTTPAdapter(max_retries=_RETRY))
    if auth_token:
        session.headers["Authorization"] = f"Bearer {auth_token}"
    return session


class EEGDashAPIClient:
    """HTTP client for EEGDash API.

    Parameters
    ----------
    api_url : str, optional
        Base API URL. Default: https://data.eegdash.org
    database : str, default "eegdash"
        Database name ("eegdash" or "eegdashstaging").
    auth_token : str, optional
        Auth token for admin write operations.
    """

    def __init__(
        self,
        api_url: str | None = None,
        database: str = "eegdash",
        auth_token: str | None = None,
    ):
        self.api_url = (api_url or os.getenv("EEGDASH_API_URL", DEFAULT_API_URL)).rstrip("/")
        self.database = database
        self._session = _make_session(auth_token or os.getenv("EEGDASH_API_TOKEN"))

    def find(
        self,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
        skip: int | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Query records. Auto-paginates if no limit specified."""
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
            resp = self._session.get(url, params={**params, "limit": 1000, "skip": page_skip}, timeout=60)
            resp.raise_for_status()
            records = resp.json().get("data", [])
            if not records:
                break
            all_records.extend(records)
            if len(records) < 1000:
                break
            page_skip += 1000
        return all_records

    def find_one(self, query: dict[str, Any] | None = None, **kwargs) -> dict[str, Any] | None:
        """Find a single record."""
        results = self.find(query, limit=1)
        return results[0] if results else None

    def count_documents(self, query: dict[str, Any] | None = None, **kwargs) -> int:
        """Count documents matching query."""
        params = {"filter": json.dumps(query)} if query else {}
        resp = self._session.get(f"{self.api_url}/api/{self.database}/count", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("count", 0)

    def insert_one(self, record: dict[str, Any]) -> str:
        """Insert single record (requires auth)."""
        resp = self._session.post(f"{self.api_url}/admin/{self.database}/records", json=record, timeout=30)
        resp.raise_for_status()
        return resp.json().get("insertedId", "")

    def insert_many(self, records: list[dict[str, Any]]) -> int:
        """Insert multiple records (requires auth)."""
        resp = self._session.post(f"{self.api_url}/admin/{self.database}/records/bulk", json=records, timeout=60)
        resp.raise_for_status()
        return resp.json().get("insertedCount", 0)


def get_client(api_url: str | None = None, database: str = "eegdash", auth_token: str | None = None) -> EEGDashAPIClient:
    """Get an API client instance."""
    return EEGDashAPIClient(api_url=api_url, database=database, auth_token=auth_token)


__all__ = ["EEGDashAPIClient", "get_client"]
