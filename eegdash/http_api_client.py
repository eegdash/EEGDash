# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""HTTP API client for EEGDash REST API.

This module provides a client that communicates with the EEGDash REST API gateway
(https://data.eegdash.org) instead of connecting directly to MongoDB. It maintains
a similar interface to MongoConnectionManager for backward compatibility.

API Features
------------
The EEGDash REST API (v2.1.0+) provides:

- **Rate Limiting**: Public endpoints limited to 100 requests/minute per IP
- **Request Tracing**: Responses include ``X-Request-ID`` for debugging
- **Response Timing**: ``X-Response-Time`` header shows processing time in ms
- **Health Checks**: Service status available at ``/health``

Configuration
-------------
The API URL can be configured via environment variables:

- ``EEGDASH_API_URL``: Override the default API URL (default: https://data.eegdash.org)
- ``EEGDASH_API_TOKEN``: Admin token for write operations

Example
-------
>>> from eegdash.http_api_client import HTTPAPIConnectionManager
>>> conn = HTTPAPIConnectionManager()
>>> collection = conn.get_collection("eegdash", "records")
>>> count = collection.count_documents({})
>>> print(f"Total records: {count}")
"""

import json
import threading
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HTTPAPICollection:
    """A collection-like interface for the EEGDash HTTP API.

    This class mimics PyMongo's Collection interface but uses REST API calls
    under the hood instead of direct MongoDB connections.

    Parameters
    ----------
    api_url : str
        Base URL of the API (e.g., "https://data.eegdash.org").
    database : str
        Name of the database (e.g., "eegdash" or "eegdashstaging").
    collection : str
        Name of the collection (typically "records").
    auth_token : str, optional
        Authentication token for API access. Not required for public read access.
    is_admin : bool, default False
        If True, use admin endpoints with write access (requires auth_token).

    """

    def __init__(
        self,
        api_url: str,
        database: str,
        collection: str,
        auth_token: str = None,
        is_admin: bool = False,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.database = database
        self.collection = collection
        self.auth_token = auth_token
        self.is_admin = is_admin

        # Create a session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Only add auth header if token is provided (for admin operations)
        if auth_token:
            self.session.headers.update({"Authorization": f"Bearer {auth_token}"})

    def find(self, query: dict[str, Any] = None, **kwargs) -> list[dict[str, Any]]:
        """Query records from the API.

        Parameters
        ----------
        query : dict, optional
            MongoDB-style query filter.
        **kwargs
            Additional parameters like projection, limit, skip.

        Returns
        -------
        list of dict
            List of matching records.

        """
        if query is None:
            query = {}

        # Prepare request parameters
        params = {}
        if query:
            params["filter"] = json.dumps(query)

        # Handle limit and skip from kwargs
        if "limit" in kwargs:
            params["limit"] = kwargs["limit"]
        if "skip" in kwargs:
            params["skip"] = kwargs["skip"]

        # Build URL
        if self.is_admin:
            url = f"{self.api_url}/admin/{self.database}/records"
        else:
            url = f"{self.api_url}/api/{self.database}/records"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                return data.get("data", [])
            else:
                raise RuntimeError(f"API request failed: {data}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to query API: {e}") from e

    def find_one(
        self, query: dict[str, Any] = None, projection: dict[str, Any] = None
    ) -> Optional[dict[str, Any]]:
        """Find a single record matching the query.

        Parameters
        ----------
        query : dict, optional
            MongoDB-style query filter.
        projection : dict, optional
            Field projection (not fully supported by REST API).

        Returns
        -------
        dict or None
            The first matching record, or None if no match.

        """
        results = self.find(query, limit=1)
        return results[0] if results else None

    def count_documents(self, query: dict[str, Any] = None, **kwargs) -> int:
        """Count documents matching the query.

        Parameters
        ----------
        query : dict, optional
            MongoDB-style query filter.
        **kwargs
            Additional parameters (e.g., maxTimeMS).

        Returns
        -------
        int
            Number of matching documents.

        """
        if query is None:
            query = {}

        # Build URL
        url = f"{self.api_url}/api/{self.database}/count"

        # Prepare parameters
        params = {}
        if query:
            params["filter"] = json.dumps(query)

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                return data.get("count", 0)
            else:
                raise RuntimeError(f"API count request failed: {data}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to count via API: {e}") from e

    def estimated_document_count(self) -> int:
        """Get estimated document count (calls count_documents with empty filter).

        Returns
        -------
        int
            Estimated number of documents in the collection.

        """
        return self.count_documents({})

    def insert_one(self, record: dict[str, Any]) -> Any:
        """Insert a single record (admin only).

        Parameters
        ----------
        record : dict
            The record to insert.

        Returns
        -------
        InsertOneResult-like object
            Contains the insertedId.

        Raises
        ------
        PermissionError
            If no auth token is configured for admin operations.

        """
        if not self.auth_token:
            raise PermissionError(
                "Insert operations require admin authentication. "
                "Provide auth_token when creating the client."
            )

        url = f"{self.api_url}/admin/{self.database}/records"

        try:
            response = self.session.post(url, json=record, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                # Return a simple object with insertedId attribute
                class InsertResult:
                    def __init__(self, inserted_id):
                        self.inserted_id = inserted_id

                return InsertResult(data.get("insertedId"))
            else:
                raise RuntimeError(f"Insert failed: {data}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to insert via API: {e}") from e

    def insert_many(self, records: list[dict[str, Any]]) -> Any:
        """Insert multiple records (admin only).

        Parameters
        ----------
        records : list of dict
            The records to insert.

        Returns
        -------
        InsertManyResult-like object
            Contains insertedIds.

        Raises
        ------
        PermissionError
            If no auth token is configured for admin operations.

        """
        if not self.auth_token:
            raise PermissionError(
                "Insert operations require admin authentication. "
                "Provide auth_token when creating the client."
            )

        url = f"{self.api_url}/admin/{self.database}/records/bulk"

        try:
            response = self.session.post(url, json=records, timeout=60)
            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                # Return a simple object with insertedIds attribute
                class InsertManyResult:
                    def __init__(self, inserted_count, inserted_ids=None):
                        self.inserted_count = inserted_count
                        self.inserted_ids = inserted_ids or []

                return InsertManyResult(
                    data.get("insertedCount", 0), data.get("insertedIds", [])
                )
            else:
                raise RuntimeError(f"Bulk insert failed: {data}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to bulk insert via API: {e}") from e


class HTTPAPIDatabase:
    """A database-like interface for the EEGDash HTTP API.

    This class mimics PyMongo's Database interface.

    Parameters
    ----------
    api_url : str
        Base URL of the API.
    database : str
        Name of the database.
    auth_token : str, optional
        Authentication token for API access. Not required for public reads.

    """

    def __init__(self, api_url: str, database: str, auth_token: str = None) -> None:
        self.api_url = api_url
        self.database = database
        self.auth_token = auth_token
        self._collections = {}

    def __getitem__(self, collection_name: str) -> HTTPAPICollection:
        """Get a collection by name.

        Parameters
        ----------
        collection_name : str
            Name of the collection.

        Returns
        -------
        HTTPAPICollection
            Collection interface.

        """
        if collection_name not in self._collections:
            self._collections[collection_name] = HTTPAPICollection(
                self.api_url,
                self.database,
                collection_name,
                self.auth_token,
            )
        return self._collections[collection_name]


class HTTPAPIClient:
    """A client-like interface for the EEGDash HTTP API.

    This class mimics PyMongo's MongoClient interface but communicates
    with the REST API gateway instead of connecting directly to MongoDB.

    Parameters
    ----------
    api_url : str
        Base URL of the API (e.g., "https://data.eegdash.org").
    auth_token : str, optional
        Authentication token for admin operations. Not required for public reads.

    """

    def __init__(self, api_url: str, auth_token: str = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.auth_token = auth_token
        self._databases = {}

    def __getitem__(self, database_name: str) -> HTTPAPIDatabase:
        """Get a database by name.

        Parameters
        ----------
        database_name : str
            Name of the database.

        Returns
        -------
        HTTPAPIDatabase
            Database interface.

        """
        if database_name not in self._databases:
            self._databases[database_name] = HTTPAPIDatabase(
                self.api_url, database_name, self.auth_token
            )
        return self._databases[database_name]

    def close(self) -> None:
        """Close the client (no-op for HTTP client)."""
        pass

    @property
    def topology(self):
        """Mock topology property for compatibility checks."""

        class MockTopology:
            def isConnected(self):
                # Could add actual health check here
                return True

        return MockTopology()


class HTTPAPIConnectionManager:
    """A thread-safe singleton to manage HTTP API client connections.

    This class mirrors the interface of MongoConnectionManager but uses
    HTTP API clients instead of direct MongoDB connections.

    Attributes
    ----------
    _instances : dict
        A dictionary to store singleton instances.
    _lock : threading.Lock
        A lock to ensure thread-safe instantiation.

    """

    _instances: dict[
        tuple[str, bool], tuple[HTTPAPIClient, HTTPAPIDatabase, HTTPAPICollection]
    ] = {}
    _lock = threading.Lock()

    @classmethod
    def get_client(
        cls, api_url: str, is_staging: bool = False, auth_token: str = None
    ) -> tuple[HTTPAPIClient, HTTPAPIDatabase, HTTPAPICollection]:
        """Get or create an HTTP API client for the given parameters.

        Parameters
        ----------
        api_url : str
            Base URL of the API (e.g., "https://data.eegdash.org").
        is_staging : bool, default False
            If True, connect to the staging database ("eegdashstaging").
            Otherwise, connect to the production database ("eegdash").
        auth_token : str, optional
            Authentication token for admin operations. Not required for reads.

        Returns
        -------
        tuple[HTTPAPIClient, HTTPAPIDatabase, HTTPAPICollection]
            A tuple containing the HTTP client, database interface, and
            collection interface for the "records" collection.

        """
        # Create a unique key based on API URL and staging flag
        key = (api_url, is_staging)

        if key not in cls._instances:
            with cls._lock:
                # Double-check pattern to avoid race conditions
                if key not in cls._instances:
                    client = HTTPAPIClient(api_url, auth_token)
                    db_name = "eegdashstaging" if is_staging else "eegdash"
                    db = client[db_name]
                    collection = db["records"]
                    cls._instances[key] = (client, db, collection)

        return cls._instances[key]

    @classmethod
    def close_all(cls) -> None:
        """Close all managed HTTP API clients.

        This method iterates through all cached client instances and closes
        their connections. It also clears the instance cache.
        """
        with cls._lock:
            for client, _, _ in cls._instances.values():
                try:
                    client.close()
                except Exception:
                    pass
            cls._instances.clear()


__all__ = [
    "HTTPAPIClient",
    "HTTPAPIDatabase",
    "HTTPAPICollection",
    "HTTPAPIConnectionManager",
]
