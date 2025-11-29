# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""General utility functions for EEGDash.

This module contains miscellaneous utility functions used across the EEGDash package,
including API client initialization and configuration helpers.
"""

from mne.utils import get_config, set_config, use_log_level


def _init_mongo_client() -> None:
    """Initialize the default EEGDash API configuration in the MNE config.

    This function checks if the ``EEGDASH_API_URL`` and ``EEGDASH_API_TOKEN``
    are already set in the MNE-Python configuration. If they are not set, this
    function sets them to the default public EEGDash REST API gateway values.

    The operation is performed with MNE's logging level temporarily set to
    "ERROR" to suppress verbose output.

    Notes
    -----
    This is an internal helper function and is not intended for direct use
    by end-users.

    The default configuration uses the production REST API gateway with
    read-only access token.

    """
    with use_log_level("ERROR"):
        # Set API URL if not configured (using direct IP since DNS not configured)
        if get_config("EEGDASH_API_URL") is None:
            set_config(
                "EEGDASH_API_URL",
                "http://137.110.244.65:3000",
                set_env=True,
            )
        
        # Set read-only API token if not configured
        if get_config("EEGDASH_API_TOKEN") is None:
            set_config(
                "EEGDASH_API_TOKEN",
                "Competition2025AccessToken",
                set_env=True,
            )


__all__ = ["_init_mongo_client"]
