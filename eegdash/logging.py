import logging

from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",  # Set the default level for all loggers
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("eegdash").setLevel(logging.INFO)

logger = logging.getLogger("eegdash")
