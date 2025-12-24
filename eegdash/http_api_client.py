# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""HTTP API client for EEGDash REST API.

This module provides a client that communicates with the EEGDash REST API gateway
(https://data.eegdash.org) instead of connecting directly to MongoDB. It maintains
a similar interface to MongoConnectionManager for backward compatibility.

Configuration
-------------
The API URL can be configured via environment variables:

- ``EEGDASH_API_URL``: Override the default API URL (default: https://data.eegdash.org)
- ``EEGDASH_API_TOKEN``: Admin token for write operations

Example:
-------
>>> from eegdash.http_api_client import HTTPAPIConnectionManager
>>> conn = HTTPAPIConnectionManager()
>>> collection = conn.get_collection("eegdash", "records")
>>> count = collection.count_documents({})
>>> print(f"Total records: {count}")

Error Handling
--------------
This module uses standard ``requests`` exceptions. Handle errors like this:

>>> import requests
>>> try:
...     records = collection.find({})
... except requests.HTTPError as e:
...     if e.response.status_code == 429:
...         print("Rate limited! Try again later.")
...     else:
...         print(f"HTTP error: {e}")
... except requests.RequestException as e:
...     print(f"Network error: {e}")

"""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class InsertOneResult:
    """Result of an insert_one operation."""

    inserted_id: str


@dataclass
class InsertManyResult:
    """Result of an insert_many operation."""

    inserted_count: int
    inserted_ids: list[str] = field(default_factory=list)


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

    Raises
    ------
    requests.HTTPError
        For HTTP errors (including rate limiting with status 429).
    requests.RequestException
        For network/connection errors.

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
            status_forcelist=[500, 502, 503, 504],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        if auth_token:
            self.session.headers.update({"Authorization": f"Bearer {auth_token}"})

    def find(self, query: dict[str, Any] = None, **kwargs) -> list[dict[str, Any]]:
        """Query records from the API.

        Parameters
        ----------
        query : dict, optional
            MongoDB-style query filter.
        **kwargs
            Additional parameters like limit, skip.

        Returns
        -------
        list of dict
            List of matching records.

        Notes
        -----
        If no limit is specified, this method will automatically paginate
        through all results (the server has a max of 1000 per request).

        """
        prefix = "admin" if self.is_admin else "api"
        url = f"{self.api_url}/{prefix}/{self.database}/records"

        params = {}
        if query:
            params["filter"] = json.dumps(query)

        # If limit is specified, make a single request
        if "limit" in kwargs:
            params["limit"] = kwargs["limit"]
            if "skip" in kwargs:
                params["skip"] = kwargs["skip"]
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json().get("data", [])

        # No limit specified: paginate through all results
        all_records = []
        skip = kwargs.get("skip", 0)
        page_size = 1000  # Server max

        while True:
            params["limit"] = page_size
            params["skip"] = skip
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            records = response.json().get("data", [])

            if not records:
                break

            all_records.extend(records)

            if len(records) < page_size:
                break  # Last page

            skip += page_size

        return all_records

    def find_one(
        self, query: dict[str, Any] = None, projection: dict[str, Any] = None
    ) -> Optional[dict[str, Any]]:
        """Find a single record matching the query."""
        results = self.find(query, limit=1)
        return results[0] if results else None

    def count_documents(self, query: dict[str, Any] = None, **kwargs) -> int:
        """Count documents matching the query."""
        params = {}
        if query:
            params["filter"] = json.dumps(query)

        url = f"{self.api_url}/api/{self.database}/count"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("count", 0)

    def estimated_document_count(self) -> int:
        """Get estimated document count."""
        return self.count_documents({})

    def insert_one(self, record: dict[str, Any]) -> InsertOneResult:
        """Insert a single record (admin only)."""
        if not self.auth_token:
            raise PermissionError("Insert requires auth_token.")

        url = f"{self.api_url}/admin/{self.database}/records"
        response = self.session.post(url, json=record, timeout=30)
        response.raise_for_status()
        return InsertOneResult(inserted_id=response.json().get("insertedId", ""))

    def insert_many(self, records: list[dict[str, Any]]) -> InsertManyResult:
        """Insert multiple records (admin only)."""
        if not self.auth_token:
            raise PermissionError("Insert requires auth_token.")

        url = f"{self.api_url}/admin/{self.database}/records/bulk"
        response = self.session.post(url, json=records, timeout=60)
        response.raise_for_status()
        data = response.json()
        return InsertManyResult(
            inserted_count=data.get("insertedCount", 0),
            inserted_ids=data.get("insertedIds", []),
        )


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
    "HTTPAPICollection",
    "HTTPAPIConnectionManager",
    "HTTPAPIDatabase",
    "InsertManyResult",
    "InsertOneResult",
]
