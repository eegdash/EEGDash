"""
This module provides utility functions for the eegdash package.

It includes helper functions for initializing the MongoDB client and other
miscellaneous tasks that are used across the application.
"""
from mne.utils import get_config, set_config, use_log_level


def _init_mongo_client():
    """Initialize the MongoDB client configuration.

    This function checks if the EEGDASH_DB_URI is set in the MNE config.
    If it is not set, it sets a default URI for the public database.
    This ensures that the application can connect to the database without
    requiring manual configuration for public access.
    """
    with use_log_level("ERROR"):
        if get_config("EEGDASH_DB_URI") is None:
            set_config(
                "EEGDASH_DB_URI",
                "mongodb+srv://eegdash-user:mdzoMjQcHWTVnKDq@cluster0.vz35p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
                set_env=True,
            )
