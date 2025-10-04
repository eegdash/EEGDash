# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""General utility functions for EEGDash.

This module contains miscellaneous utility functions used across the EEGDash package,
including MongoDB client initialization and configuration helpers.
"""

from mne.utils import get_config, set_config, use_log_level


def _init_mongo_client() -> None:
    """Initialize the default MongoDB connection URI in the MNE config.

    This function checks if the ``EEGDASH_DB_URI`` is already set in the
    MNE-Python configuration. If it is not set, this function sets it to the
    default public EEGDash MongoDB Atlas cluster URI.

    The operation is performed with MNE's logging level temporarily set to
    "ERROR" to suppress verbose output.

    Notes
    -----
    This is an internal helper function and is not intended for direct use
    by end-users.

    """
    with use_log_level("ERROR"):
        if get_config("EEGDASH_DB_URI") is None:
            set_config(
                "EEGDASH_DB_URI",
                "mongodb+srv://eegdash-user:mdzoMjQcHWTVnKDq@cluster0.vz35p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
                set_env=True,
            )


__all__ = ["_init_mongo_client"]
