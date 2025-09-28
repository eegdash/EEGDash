# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""MongoDB connection and operations management.

This module provides a thread-safe singleton manager for MongoDB connections,
ensuring that connections to the database are handled efficiently and consistently
across the application.
"""

import threading

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


class MongoConnectionManager:
    """A thread-safe singleton to manage MongoDB client connections.

    This class ensures that only one connection instance is created for each
    unique combination of a connection string and staging flag. It provides
    class methods to get a client and to close all active connections.

    Attributes
    ----------
    _instances : dict
        A dictionary to store singleton instances, mapping a
        (connection_string, is_staging) tuple to a (client, db, collection)
        tuple.
    _lock : threading.Lock
        A lock to ensure thread-safe instantiation of clients.
    """

    _instances: dict[tuple[str, bool], tuple[MongoClient, Database, Collection]] = {}
    _lock = threading.Lock()

    @classmethod
    def get_client(
        cls, connection_string: str, is_staging: bool = False
    ) -> tuple[MongoClient, Database, Collection]:
        """Get or create a MongoDB client for the given connection parameters.

        This method returns a cached client if one already exists for the given
        connection string and staging flag. Otherwise, it creates a new client,
        connects to the appropriate database ("eegdash" or "eegdashstaging"),
        and returns the client, database, and "records" collection.

        Parameters
        ----------
        connection_string : str
            The MongoDB connection string.
        is_staging : bool, default False
            If True, connect to the staging database ("eegdashstaging").
            Otherwise, connect to the production database ("eegdash").

        Returns
        -------
        tuple[MongoClient, Database, Collection]
            A tuple containing the connected MongoClient instance, the Database
            object, and the Collection object for the "records" collection.
        """
        # Create a unique key based on connection string and staging flag
        key = (connection_string, is_staging)

        if key not in cls._instances:
            with cls._lock:
                # Double-check pattern to avoid race conditions
                if key not in cls._instances:
                    client = MongoClient(connection_string)
                    db_name = "eegdashstaging" if is_staging else "eegdash"
                    db = client[db_name]
                    collection = db["records"]
                    cls._instances[key] = (client, db, collection)

        return cls._instances[key]

    @classmethod
    def close_all(cls) -> None:
        """Close all managed MongoDB client connections.

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


__all__ = ["MongoConnectionManager"]