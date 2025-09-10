"""
This module provides a singleton class for managing MongoDB client connections.

The MongoConnectionManager class ensures that only one connection is established
per unique connection string and staging flag, preventing resource leakage and
improving performance.
"""
import threading

from pymongo import MongoClient


class MongoConnectionManager:
    """Singleton class to manage MongoDB client connections.

    This class uses a singleton pattern to ensure that only one instance of the
    MongoDB client is created for each unique connection string and staging flag.
    This helps to prevent resource leakage and improve performance by reusing
    existing connections.
    """

    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get_client(cls, connection_string: str, is_staging: bool = False):
        """Get or create a MongoDB client for the given connection string and staging flag.

        This method returns a tuple containing the client, database, and collection
        for the specified connection parameters. If a client for the given
        parameters already exists, it is returned; otherwise, a new one is created.

        Parameters
        ----------
        connection_string : str
            The MongoDB connection string.
        is_staging : bool, default False
            Whether to use the staging database.

        Returns
        -------
        tuple
            A tuple of (MongoClient, Database, Collection).
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
    def close_all(cls):
        """Close all managed MongoDB client connections.

        This method iterates through all managed client instances and closes them.
        It is useful for cleaning up resources when the application is shutting down.
        """
        with cls._lock:
            for client, _, _ in cls._instances.values():
                try:
                    client.close()
                except Exception:
                    pass
            cls._instances.clear()
