import logging

from rich.logging import RichHandler

logger = logging.getLogger("eegdash")
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(show_path=False, rich_tracebacks=True))
logger.addHandler(logging.NullHandler())
