# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Logging configuration for EEGDash.

This module sets up centralized logging for the EEGDash package using Rich for enhanced
console output formatting. It provides a consistent logging interface across all modules.
"""

import logging

from rich.logging import RichHandler

# Get the root logger
root_logger = logging.getLogger()

# --- This is the key part ---
# 1. Remove any handlers that may have been added by default
root_logger.handlers = []

# 2. Add your RichHandler
root_logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
# ---------------------------

# 3. Set the level for the root logger
root_logger.setLevel(logging.INFO)

# Now, get your package-specific logger. It will inherit the
# configuration from the root logger we just set up.
logger = logging.getLogger("eegdash")

logger.setLevel(logging.INFO)

__all__ = ["logger"]
