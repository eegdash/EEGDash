"""eegdash ingestion pipeline.

Five-stage pipeline (``1_fetch_sources`` / ``2_clone`` / ``3_digest`` /
``4_validate_output`` / ``5_inject``) ingesting BIDS EEG/MEG/iEEG datasets
into the eegdash MongoDB. Shared helpers live in the underscore-prefixed
sibling modules.
"""

from __future__ import annotations

__version__ = "0.1.0"
