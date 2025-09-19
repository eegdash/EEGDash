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
