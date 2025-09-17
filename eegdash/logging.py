import logging

from rich.logging import RichHandler

logger = logging.getLogger(__name__)

logger.addHandler(RichHandler(show_path=False, rich_tracebacks=True))
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)
