"""DOI validation and resolution utilities.

Provides helpers for validating DOI format, resolving DOIs via the
doi.org content-negotiation API, caching results, and comparing
author lists. Inspired by MOABB's ``test_doi_validation.py``.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOI_REGEX = re.compile(r"^10\.\d{4,}/\S+$")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CACHE_PATH = _REPO_ROOT / "tests" / "doi_cache.json"

# Delay between consecutive doi.org requests (seconds) to respect rate limits.
RESOLVE_DELAY: float = 0.15
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_doi(raw: str | None) -> str | None:
    """Strip URL prefixes and whitespace from a DOI string.

    Handles forms like ``"doi:10.1234/x"``, ``"https://doi.org/10.1234/x"``,
    or bare ``"10.1234/x"``.

    Returns *None* when the input is falsy or cannot be cleaned.
    """
    if not raw:
        return None
    doi = str(raw).strip()
    # Remove common prefixes
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
            break
    doi = doi.strip()
    return doi if doi else None


def is_valid_doi(doi: str | None) -> bool:
    """Return *True* if *doi* matches the canonical DOI pattern ``10.XXXX/...``."""
    if doi is None:
        return False
    return bool(DOI_REGEX.match(normalize_doi(doi) or ""))


# ---------------------------------------------------------------------------
# Resolution via doi.org
# ---------------------------------------------------------------------------


def resolve_doi(
    doi: str,
    *,
    timeout: float = 15,
) -> dict[str, Any] | None:
    """Resolve a single DOI via doi.org content-negotiation (CSL-JSON).

    Parameters
    ----------
    doi : str
        A normalised DOI (e.g. ``"10.1234/foo"``).
    timeout : float
        HTTP timeout in seconds.

    Returns
    -------
    dict or None
        The CSL-JSON metadata dict, or *None* on failure.

    """
    clean = normalize_doi(doi)
    if not clean:
        return None

    url = f"https://doi.org/{clean}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.citationstyles.csl+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        socket.timeout,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        logger.warning(
            "Failed to resolve DOI %s (%s): %s", clean, type(exc).__name__, exc
        )
        return None


def resolve_doi_cached(
    doi: str,
    cache: dict[str, dict[str, Any]],
    *,
    timeout: float = 15,
    delay: float = RESOLVE_DELAY,
) -> dict[str, Any] | None:
    """Resolve a DOI, checking *cache* first.

    On a cache miss the result is fetched from doi.org and inserted into
    *cache* (in-place) so callers can persist it afterwards.
    """
    clean = normalize_doi(doi)
    if not clean:
        return None

    if clean in cache:
        return cache[clean]

    time.sleep(delay)
    result = resolve_doi(clean, timeout=timeout)
    if result is not None:
        # Store a compact subset to keep the cache small.
        cache[clean] = _compact_metadata(result)
    return cache.get(clean)


def _compact_metadata(csl: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields we need for validation."""
    authors = []
    for a in csl.get("author", []):
        name = {}
        if "family" in a:
            name["family"] = a["family"]
        if "given" in a:
            name["given"] = a["given"]
        if name:
            authors.append(name)

    issued = csl.get("issued", {})
    date_parts = issued.get("date-parts", [[]])
    year = date_parts[0][0] if date_parts and date_parts[0] else None

    return {
        "title": csl.get("title", ""),
        "authors": authors,
        "year": year,
        "type": csl.get("type", ""),
    }


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def load_cache(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Load the persistent DOI cache from *path* (JSON)."""
    path = Path(path) if path else _DEFAULT_CACHE_PATH
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_cache(
    cache: dict[str, dict[str, Any]],
    path: Path | str | None = None,
) -> None:
    """Persist *cache* to *path* as pretty-printed JSON."""
    path = Path(path) if path else _DEFAULT_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Author helpers
# ---------------------------------------------------------------------------


def extract_surnames(authors: list[dict[str, str]]) -> list[str]:
    """Return lowercased family names from a CSL-JSON author list."""
    out: list[str] = []
    for a in authors:
        family = a.get("family", "").strip()
        if family:
            out.append(family.lower())
    return out


def surnames_overlap(
    authors_a: list[dict[str, str]],
    authors_b: list[dict[str, str]],
) -> set[str]:
    """Return the set of shared family names between two author lists."""
    return set(extract_surnames(authors_a)) & set(extract_surnames(authors_b))


# ---------------------------------------------------------------------------
# Bulk helpers for dataset CSV
# ---------------------------------------------------------------------------


def load_dataset_dois(csv_path: Path | str | None = None) -> list[tuple[str, str]]:
    """Load ``(dataset_id, doi)`` pairs from ``dataset_summary.csv``.

    Returns only rows where the DOI column is non-empty.
    """
    import csv

    if csv_path is None:
        csv_path = _REPO_ROOT / "eegdash" / "dataset" / "dataset_summary.csv"
    csv_path = Path(csv_path)
    pairs: list[tuple[str, str]] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get("doi", "").strip()
            if raw:
                did = row.get("dataset", "").strip()
                pairs.append((did, raw))
    return pairs
