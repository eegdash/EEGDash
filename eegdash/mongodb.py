import threading

from pymongo import MongoClient

# MongoDB Operations
# These methods provide a high-level interface to interact with the MongoDB
# collection, allowing users to find, add, and update EEG data records.
# - find:
# - exist:
# - add_request:
# - add:
# - update_request:
# - remove_field:
# - remove_field_from_db:
# - close: Close the MongoDB connection.
# - __del__: Destructor to close the MongoDB connection.


class MongoConnectionManager:
    """Singleton class to manage MongoDB client connections."""

    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get_client(cls, connection_string: str, is_staging: bool = False):
        """Get or create a MongoDB client for the given connection string and staging flag.

        Parameters
        ----------
        connection_string : str
            The MongoDB connection string
        is_staging : bool
            Whether to use staging database

        Returns
        -------
        tuple
            A tuple of (client, database, collection)

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
        """Close all MongoDB client connections."""
        with cls._lock:
            for client, _, _ in cls._instances.values():
                try:
                    client.close()
                except:
                    pass
            cls._instances.clear()
