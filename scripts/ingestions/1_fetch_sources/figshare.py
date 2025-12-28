"""Fetch EEG BIDS datasets from Figshare.

This script searches Figshare for datasets containing both "EEG" and "BIDS" keywords
using the Figshare API v2. It retrieves comprehensive metadata including DOIs,
descriptions, files, authors, and download URLs.

BIDS validation is performed by checking for:
- Required BIDS files (dataset_description.json)
- Optional BIDS files (participants.tsv, README, etc.)
- BIDS-like subject folder patterns (sub-XX or sub-XX.zip)

Output: consolidated/figshare_datasets.json

Authentication:
- Uses API key from .env.figshare file (FIGSHARE_API_KEY)
- Higher rate limits with authentication (vs 1 req/sec for anonymous)
"""

import argparse
import os
import re
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Add ingestion paths before importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import (
    PROJECT_ROOT,
    generate_dataset_id,
    save_datasets_deterministically,
    setup_paths,
)

setup_paths()
from eegdash.records import create_dataset

# Load API key from .env.figshare
_env_path = PROJECT_ROOT / ".env.figshare"
load_dotenv(_env_path)
FIGSHARE_API_KEY = os.getenv("FIGSHARE_API_KEY", "")
if FIGSHARE_API_KEY:
    print(f"✓ Figshare API key loaded from {_env_path}")
else:
    print(f"⚠ No Figshare API key found in {_env_path} (using anonymous access)")

# BIDS indicator files
BIDS_REQUIRED_FILES = ["dataset_description.json"]
BIDS_OPTIONAL_FILES = [
    "participants.tsv",
    "participants.json",
    "readme",
    "readme.md",
    "readme.txt",
    "changes",
]

# Patterns for BIDS subject folders/zips
BIDS_SUBJECT_PATTERN = re.compile(r"^sub-[a-zA-Z0-9]+(?:\.zip)?$", re.IGNORECASE)


def validate_bids_structure(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate BIDS structure from file list.

    Checks for:
    - Required BIDS files (dataset_description.json)
    - Optional BIDS files (participants.tsv, README, etc.)
    - Subject-level files/zips (sub-XX or sub-XX.zip)

    Args:
        files: List of file dictionaries from Figshare API

    Returns:
        Dictionary with validation results:
        - is_bids: Whether dataset appears to be BIDS compliant
        - bids_files_found: List of BIDS files found
        - subject_count: Number of subject folders/zips detected
        - has_subject_zips: Whether subjects are in ZIP format

    """
    file_names = [f.get("name", "").lower() for f in files]
    file_names_original = [f.get("name", "") for f in files]

    # Check for required BIDS files
    found_required = []
    for bf in BIDS_REQUIRED_FILES:
        if bf.lower() in file_names:
            found_required.append(bf)

    # Check for optional BIDS files
    found_optional = []
    for bf in BIDS_OPTIONAL_FILES:
        if bf.lower() in file_names:
            found_optional.append(bf)

    # Check for subject folders/zips
    subject_files = []
    for fn in file_names_original:
        if BIDS_SUBJECT_PATTERN.match(fn):
            subject_files.append(fn)

    # Determine if it's BIDS
    # Accept if: has dataset_description.json OR has subject folders/zips with BIDS naming
    is_bids = len(found_required) > 0 or len(subject_files) >= 2

    # Check if subjects are zipped
    has_subject_zips = any(fn.lower().endswith(".zip") for fn in subject_files)

    return {
        "is_bids": is_bids,
        "bids_files_found": found_required + found_optional,
        "subject_count": len(subject_files),
        "has_subject_zips": has_subject_zips,
        "subject_files": subject_files[:10],  # Store first 10 for reference
    }


def search_figshare(
    query: str,
    size: int = 100,
    page_size: int = 1000,  # API max is 1000 - use it for efficiency
    item_type: int = 3,  # 3 = dataset
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    """Search Figshare for datasets matching the query.

    Args:
        query: Search query string
        size: Maximum number of results to fetch (0 = unlimited)
        page_size: Number of results per page (max 1000)
        item_type: Figshare item type (3=dataset, 1=figure, 2=media, etc.)
        max_retries: Maximum number of retries on rate limit errors

    Returns:
        List of Figshare article dictionaries

    """
    global _figshare_request_count, _figshare_last_request_time

    base_url = "https://api.figshare.com/v2/articles/search"

    print(f"Searching Figshare with query: {query}")
    print(f"Item type: {item_type} (dataset)")
    print(f"Max results to fetch: {size if size > 0 else 'unlimited'}")
    print(f"Results per page: {min(page_size, 1000)}")
    print(
        f"Rate limit delay: {FIGSHARE_RATE_LIMIT_DELAY}s (auth: {bool(FIGSHARE_API_KEY)})"
    )

    all_articles = []
    page = 1
    actual_page_size = min(page_size, 1000)  # Figshare max is 1000

    while True:
        print(f"\nFetching page {page}...", end=" ", flush=True)

        # Build request payload
        payload = {
            "search_for": query,
            "item_type": item_type,
            "page": page,
            "page_size": actual_page_size,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "EEGDash/1.0 (https://github.com/eegdash/EEGDash)",
        }
        # Add API key if available for higher rate limits
        if FIGSHARE_API_KEY:
            headers["Authorization"] = f"token {FIGSHARE_API_KEY}"

        # Retry loop for this page with aggressive backoff
        articles = None
        for attempt in range(max_retries):
            # Rate limiting
            elapsed = time.time() - _figshare_last_request_time
            if elapsed < FIGSHARE_RATE_LIMIT_DELAY:
                time.sleep(FIGSHARE_RATE_LIMIT_DELAY - elapsed)

            try:
                _figshare_last_request_time = time.time()
                _figshare_request_count += 1

                response = requests.post(
                    base_url,
                    json=payload,
                    headers=headers,
                    timeout=30,
                )

                if response.status_code == 200:
                    articles = response.json()
                    break
                elif response.status_code in (403, 429):
                    # Rate limited - aggressive exponential backoff
                    wait_time = (2**attempt) * 30  # 30, 60, 120 seconds
                    print(
                        f"\\n  Rate limited ({response.status_code}), "
                        f"waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...",
                        flush=True,
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"\\nError {response.status_code}", file=sys.stderr)
                    break

            except requests.RequestException as e:
                print(f"\\nRequest error: {e}", file=sys.stderr)
                if attempt < max_retries - 1:
                    time.sleep(2**attempt * 10)
                    continue
                break

        if articles is None:
            print("Failed to fetch page, stopping", file=sys.stderr)
            break

        if not articles:
            print("No more results")
            break

        print(f"Got {len(articles)} articles")
        all_articles.extend(articles)

        # Check if we've reached the requested size (only if size > 0)
        if size > 0 and len(all_articles) >= size:
            print(f"Reached limit ({len(all_articles)} articles)")
            break

        # If we got fewer results than page_size, we've reached the end
        if len(articles) < actual_page_size:
            print("Reached end of results")
            break

        page += 1

    print(f"\\nTotal articles fetched: {len(all_articles)}")
    # Return all articles if size=0, otherwise trim to exact size
    return all_articles if size <= 0 else all_articles[:size]


# Rate limiting for Figshare API
# Figshare recommends max 1 request/second for all users
# WAF-level rate limiting may kick in before API limits
_figshare_request_count = 0
_figshare_last_request_time = 0.0
# Very conservative rate limit - Figshare has aggressive WAF-level blocking
# Use longer delays to avoid being blocked
FIGSHARE_RATE_LIMIT_DELAY = float(os.getenv("FIGSHARE_RATE_LIMIT", "3.0"))


def _get_headers() -> dict[str, str]:
    """Get standard headers for Figshare API requests."""
    headers = {
        "User-Agent": "EEGDash/1.0 (https://github.com/eegdash/EEGDash)",
        "Accept": "application/json",
    }
    if FIGSHARE_API_KEY:
        headers["Authorization"] = f"token {FIGSHARE_API_KEY}"
    return headers


def peek_zip_contents(
    download_url: str,
    timeout: int = 30,
    max_bytes: int = 65536,
) -> list[dict[str, Any]] | None:
    """Peek at ZIP file contents by reading only the central directory.

    Uses HTTP Range requests to download only the last ~64KB of the file,
    which contains the ZIP central directory with file listings.

    Args:
        download_url: URL to the ZIP file
        timeout: Request timeout in seconds
        max_bytes: Maximum bytes to download from end of file

    Returns:
        List of file info dicts with 'path' and 'size', or None if failed

    """
    global _figshare_last_request_time

    try:
        # Rate limiting
        elapsed = time.time() - _figshare_last_request_time
        if elapsed < FIGSHARE_RATE_LIMIT_DELAY:
            time.sleep(FIGSHARE_RATE_LIMIT_DELAY - elapsed)
        _figshare_last_request_time = time.time()

        # Get file size with HEAD request - include API key for authenticated access
        headers = {
            "User-Agent": "EEGDash/1.0 (https://github.com/eegdash/EEGDash)",
            "Accept": "application/octet-stream",
        }
        if FIGSHARE_API_KEY:
            headers["Authorization"] = f"token {FIGSHARE_API_KEY}"

        head = requests.head(
            download_url, headers=headers, timeout=timeout, allow_redirects=True
        )

        # Figshare may return 302 redirect - follow it
        if head.status_code in (301, 302, 307, 308):
            redirect_url = head.headers.get("Location", download_url)
            head = requests.head(
                redirect_url, headers=headers, timeout=timeout, allow_redirects=True
            )

        if head.status_code == 403:
            # Try without auth header (some downloads are public)
            headers_no_auth = {
                "User-Agent": headers["User-Agent"],
                "Accept": headers["Accept"],
            }
            head = requests.head(
                download_url,
                headers=headers_no_auth,
                timeout=timeout,
                allow_redirects=True,
            )
            if head.status_code == 403:
                return None
            headers = headers_no_auth

        if head.status_code != 200:
            return None

        file_size = int(head.headers.get("Content-Length", 0))
        if file_size == 0:
            return None

        # Check if server supports range requests
        accept_ranges = head.headers.get("Accept-Ranges", "")
        if accept_ranges.lower() == "none":
            # Server explicitly doesn't support ranges - try anyway
            pass

        # Rate limit before range request
        time.sleep(FIGSHARE_RATE_LIMIT_DELAY)
        _figshare_last_request_time = time.time()

        # Download last 64KB to get central directory
        start_byte = max(0, file_size - max_bytes)
        range_headers = {
            **headers,
            "Range": f"bytes={start_byte}-{file_size - 1}",
        }

        response = requests.get(
            download_url, headers=range_headers, timeout=timeout, allow_redirects=True
        )

        if response.status_code not in (200, 206):
            return None

        content = response.content
        # If server returned full file (200) instead of range (206), take last portion
        if response.status_code == 200 and len(content) > max_bytes:
            content = content[-max_bytes:]

        # Try to read ZIP central directory
        try:
            with zipfile.ZipFile(BytesIO(content)) as zf:
                files = []
                for info in zf.infolist():
                    if not info.is_dir():
                        files.append(
                            {
                                "path": info.filename,
                                "size": info.file_size,
                            }
                        )
                return files
        except zipfile.BadZipFile:
            # Central directory not in last 64KB - try larger chunk
            if max_bytes < 262144:  # 256KB
                time.sleep(FIGSHARE_RATE_LIMIT_DELAY)
                return peek_zip_contents(download_url, timeout, max_bytes=262144)
            return None

    except requests.RequestException:
        return None
    except Exception:
        return None


def download_and_extract_zip(
    download_url: str,
    zip_name: str,
    download_dir: Path,
    timeout: int = 300,
) -> list[dict[str, Any]] | None:
    """Download a ZIP file and extract its contents listing.

    This is a fallback for when peek_zip_contents fails (e.g., due to rate limits).
    Downloads the full ZIP file, extracts the file listing, then deletes the ZIP.

    Args:
        download_url: URL to download the ZIP file
        zip_name: Name of the ZIP file (for display and temp file)
        download_dir: Directory to store temporary downloads
        timeout: Request timeout in seconds

    Returns:
        List of file info dicts with 'path' and 'size', or None if failed

    """
    global _figshare_last_request_time

    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / zip_name

    try:
        # Rate limiting
        elapsed = time.time() - _figshare_last_request_time
        if elapsed < FIGSHARE_RATE_LIMIT_DELAY * 2:  # Extra delay for downloads
            time.sleep(FIGSHARE_RATE_LIMIT_DELAY * 2 - elapsed)
        _figshare_last_request_time = time.time()

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        }

        print(f"      Downloading {zip_name}...", end=" ", flush=True)
        response = requests.get(
            download_url, headers=headers, timeout=timeout, stream=True
        )

        if response.status_code == 403:
            print("blocked (403)", flush=True)
            return None

        if response.status_code != 200:
            print(f"failed ({response.status_code})", flush=True)
            return None

        # Stream to file to handle large files
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        size_mb = downloaded / (1024 * 1024)
        print(f"{size_mb:.1f} MB downloaded", flush=True)

        # Extract file listing
        files = []
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for info in zf.infolist():
                    if not info.is_dir():
                        files.append(
                            {
                                "path": info.filename,
                                "size": info.file_size,
                            }
                        )
            print(f"        Extracted listing: {len(files)} files", flush=True)
        except zipfile.BadZipFile:
            print("        Invalid ZIP file", flush=True)
            return None
        finally:
            # Clean up the downloaded file
            if zip_path.exists():
                zip_path.unlink()

        return files

    except requests.RequestException as e:
        print(f"download error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"error: {e}", flush=True)
        return None
    finally:
        # Ensure cleanup
        if zip_path.exists():
            try:
                zip_path.unlink()
            except Exception:
                pass


def get_article_files(article_id: int, max_retries: int = 3) -> list[dict[str, Any]]:
    """Fetch files list for a specific article using dedicated endpoint.

    This is more efficient than fetching full article details when we only need files.
    Uses /articles/{id}/files endpoint.

    Args:
        article_id: Figshare article ID
        max_retries: Maximum number of retries on rate limit

    Returns:
        List of file dictionaries with name, size, download_url

    """
    global _figshare_request_count, _figshare_last_request_time

    url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    headers = _get_headers()

    for attempt in range(max_retries):
        # Rate limiting
        elapsed = time.time() - _figshare_last_request_time
        if elapsed < FIGSHARE_RATE_LIMIT_DELAY:
            time.sleep(FIGSHARE_RATE_LIMIT_DELAY - elapsed)

        try:
            _figshare_last_request_time = time.time()
            _figshare_request_count += 1

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.json()
            elif response.status_code in (403, 429):
                wait_time = (2**attempt) * 3
                print(
                    f"\n  Rate limited for files {article_id}, waiting {wait_time}s...",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                continue
            else:
                return []

        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return []

    return []


def get_article_details(article_id: int, max_retries: int = 3) -> dict[str, Any]:
    """Fetch detailed information for a specific article with rate limiting.

    Args:
        article_id: Figshare article ID
        max_retries: Maximum number of retries on rate limit

    Returns:
        Detailed article dictionary

    """
    global _figshare_request_count, _figshare_last_request_time

    url = f"https://api.figshare.com/v2/articles/{article_id}"
    headers = _get_headers()

    for attempt in range(max_retries):
        # Rate limiting (reduced with API key)
        elapsed = time.time() - _figshare_last_request_time
        if elapsed < FIGSHARE_RATE_LIMIT_DELAY:
            time.sleep(FIGSHARE_RATE_LIMIT_DELAY - elapsed)

        try:
            _figshare_last_request_time = time.time()
            _figshare_request_count += 1

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                # Rate limited - exponential backoff
                wait_time = (2**attempt) * 5  # 5, 10, 20 seconds
                print(
                    f"\n  Rate limited (403) for article {article_id}, "
                    f"waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                continue
            elif response.status_code == 429:
                # Too Many Requests - longer backoff
                wait_time = (2**attempt) * 10  # 10, 20, 40 seconds
                print(
                    f"\n  Too many requests (429) for article {article_id}, "
                    f"waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                continue
            else:
                print(
                    f"\n  Warning: Error {response.status_code} for article {article_id}",
                    file=sys.stderr,
                )
                return {}

        except requests.RequestException as e:
            print(
                f"\n  Warning: Request error for article {article_id}: {e}",
                file=sys.stderr,
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return {}

    print(
        f"\n  Warning: Max retries exceeded for article {article_id}",
        file=sys.stderr,
    )
    return {}


def detect_modalities(article: dict) -> list[str]:
    """Detect modalities from article metadata.

    Args:
        article: Figshare article dictionary

    Returns:
        List of detected modalities

    """
    modalities = []

    # Search in title, description, tags, categories
    search_text = " ".join(
        [
            article.get("title", ""),
            article.get("description", ""),
            " ".join(article.get("tags", [])),
            " ".join([c.get("title", "") for c in article.get("categories", [])]),
        ]
    ).lower()

    # Map keywords to modalities
    modality_keywords = {
        "eeg": ["eeg"],
        "meg": ["meg"],
        "ieeg": ["ieeg", "intracranial"],
        "emg": ["emg"],
        "fnirs": ["fnirs", "fNIRS", "nirs"],
        "lfp": ["lfp", "local field potential"],
    }

    for modality, keywords in modality_keywords.items():
        if any(kw in search_text for kw in keywords):
            modalities.append(modality)

    # Default to EEG if nothing detected (since we search for "EEG BIDS")
    if not modalities:
        modalities = ["eeg"]

    return modalities


def extract_dataset_info(
    article: dict,
    fetch_details: bool = False,
    digested_at: str | None = None,
    download_zips: bool = False,
    download_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract relevant information from a Figshare article and normalize to Dataset schema.

    Args:
        article: Figshare article dictionary
        fetch_details: Whether to fetch full details (slower but more complete)
        digested_at: ISO 8601 timestamp for digested_at field
        download_zips: Whether to download and extract ZIP contents (fallback when peek fails)
        download_dir: Directory for temporary ZIP downloads

    Returns:
        Dataset schema document

    """
    # Get basic info from search results
    article_id = article.get("id", "")
    title = article.get("title", "")
    doi = article.get("doi", "")

    # Clean HTML tags from title (search results may have highlighting)
    title = re.sub(r"<[^>]+>", "", title)

    # Get files - either from article or fetch separately
    files = article.get("files", [])

    # Fetch full details if requested
    if fetch_details and article_id:
        details = get_article_details(article_id)
        if details:
            article = details
            files = details.get("files", files)
            # Re-clean title from details (might also have HTML)
            title = re.sub(r"<[^>]+>", "", article.get("title", title))
        elif not files:
            # Fallback: use dedicated files endpoint (more efficient than full details)
            files = get_article_files(article_id)

    # Extract URLs
    url_public_html = article.get("url_public_html", "")

    # Extract authors - normalize to simple list of names
    author_names = []
    for author in article.get("authors", []):
        name = author.get("full_name", "")
        if name:
            author_names.append(name)

    # Extract license
    license_info = article.get("license", {})
    license_name = license_info.get("name", "") if license_info else None

    # Calculate total size (files already set above)
    total_size_bytes = sum(f.get("size", 0) for f in files)

    # Validate BIDS structure
    bids_validation = validate_bids_structure(files)

    # Detect modalities from content
    modalities = detect_modalities(article)
    recording_modality = modalities[0] if modalities else "eeg"

    # Extract modified date for dataset_modified_at
    modified_date = article.get("modified_date")

    # Generate SurnameYEAR dataset_id
    dataset_id = generate_dataset_id(
        source="figshare",
        authors=author_names,
        date=modified_date or article.get("created_date"),
        fallback_id=str(article_id),
    )

    # Create Dataset document using the schema
    dataset = create_dataset(
        dataset_id=dataset_id,
        name=title,
        source="figshare",
        recording_modality=recording_modality,
        modalities=modalities,
        license=license_name,
        authors=author_names,
        dataset_doi=doi,
        source_url=url_public_html,
        total_files=len(files),
        size_bytes=total_size_bytes if total_size_bytes > 0 else None,
        dataset_modified_at=modified_date,
        digested_at=digested_at,
    )

    # Store original Figshare ID for reference
    dataset["figshare_id"] = str(article_id)

    # Add BIDS validation results
    dataset["bids_validated"] = bids_validation["is_bids"]
    if bids_validation["bids_files_found"]:
        dataset["bids_files_found"] = bids_validation["bids_files_found"]
    if bids_validation["subject_count"] > 0:
        dataset["bids_subject_count"] = bids_validation["subject_count"]
        dataset["bids_has_subject_zips"] = bids_validation["has_subject_zips"]

    # Store files list for use by clone/manifest script (avoids extra API calls)
    if files:
        dataset["_files"] = [
            {
                "name": f.get("name", ""),
                "size": f.get("size", 0),
                "download_url": f.get("download_url", ""),
            }
            for f in files
        ]

        # Extract ZIP contents for datasets with subject zips (BIDS data inside)
        if bids_validation.get("has_subject_zips") and fetch_details:
            zip_contents = []
            zip_files = [f for f in files if f.get("name", "").lower().endswith(".zip")]

            for zf in zip_files[:5]:  # Limit to first 5 ZIPs to avoid rate limits
                download_url = zf.get("download_url", "")
                if download_url:
                    print(f"    Peeking ZIP: {zf.get('name')}...", end=" ", flush=True)
                    contents = peek_zip_contents(download_url)

                    if contents:
                        zip_contents.append(
                            {
                                "zip_name": zf.get("name"),
                                "zip_size": zf.get("size", 0),
                                "files": contents,
                            }
                        )
                        print(f"{len(contents)} files", flush=True)
                    elif download_zips and download_dir:
                        # Fallback: download and extract
                        print("peek failed, trying download...", flush=True)
                        contents = download_and_extract_zip(
                            download_url,
                            zf.get("name", ""),
                            download_dir,
                        )
                        if contents:
                            zip_contents.append(
                                {
                                    "zip_name": zf.get("name"),
                                    "zip_size": zf.get("size", 0),
                                    "files": contents,
                                }
                            )
                    else:
                        print("failed (rate limit or unsupported)", flush=True)

            if zip_contents:
                dataset["_zip_contents"] = zip_contents
                # Update BIDS files found from ZIP contents
                for zc in zip_contents:
                    for f in zc.get("files", []):
                        path = f.get("path", "")
                        if any(
                            bids_file in path.lower()
                            for bids_file in BIDS_REQUIRED_FILES + BIDS_OPTIONAL_FILES
                        ):
                            if path not in dataset.get("bids_files_found", []):
                                dataset.setdefault("bids_files_found", []).append(path)

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EEG BIDS datasets from Figshare."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="EEG BIDS",
        help='Search query (default: "EEG BIDS").',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/figshare_datasets.json"),
        help="Output JSON file path (default: consolidated/figshare_datasets.json).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Maximum number of results to fetch (default: 1000).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of results per page (max 1000, default: 100).",
    )
    parser.add_argument(
        "--fetch-details",
        action="store_true",
        help="Fetch full details for each article (slower but more complete).",
    )
    parser.add_argument(
        "--item-type",
        type=int,
        default=3,
        help="Figshare item type: 3=dataset, 1=figure, 2=media, etc. (default: 3).",
    )
    parser.add_argument(
        "--digested-at",
        type=str,
        default=None,
        help="ISO 8601 timestamp for digested_at field (for deterministic output).",
    )
    parser.add_argument(
        "--article-id",
        type=int,
        default=None,
        help="Fetch a specific article by ID (bypasses search).",
    )
    parser.add_argument(
        "--download-zips",
        action="store_true",
        help="Download and extract ZIP contents (for datasets with subject ZIPs).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("/tmp/figshare_downloads"),
        help="Directory for temporary ZIP downloads (default: /tmp/figshare_downloads).",
    )
    parser.add_argument(
        "--multi-query",
        action="store_true",
        help="Search with multiple queries to find more datasets.",
    )

    args = parser.parse_args()

    # Handle single article fetch
    if args.article_id:
        print(f"Fetching single article: {args.article_id}")
        details = get_article_details(args.article_id)
        if not details:
            print(f"Failed to fetch article {args.article_id}", file=sys.stderr)
            sys.exit(1)
        articles = [details]
    elif args.multi_query:
        # Multiple search queries to find more EEG/neural recording datasets
        queries = [
            "EEG BIDS",
            "electroencephalography BIDS",
            "EEG dataset",
            "ERP EEG",
            "MEG BIDS",
            "magnetoencephalography",
            "iEEG BIDS",
            "intracranial EEG",
            "ECoG dataset",
            "EMG dataset",
            "electromyography",
            "fNIRS dataset",
            "brain signals dataset",
            "neural recording dataset",
            "BCI dataset",
            "brain-computer interface",
            "sleep EEG",
            "epilepsy EEG",
        ]

        all_articles = {}  # Use dict to dedupe by article ID
        for q in queries:
            print(f"\n{'=' * 60}")
            print(f"Searching: {q}")
            print("=" * 60)

            results = search_figshare(
                query=q,
                size=args.size // len(queries) + 50,  # Distribute quota
                page_size=args.page_size,
                item_type=args.item_type,
            )

            new_count = 0
            for article in results:
                aid = article.get("id")
                if aid and aid not in all_articles:
                    all_articles[aid] = article
                    new_count += 1

            print(f"  New unique articles: {new_count}")
            print(f"  Total unique so far: {len(all_articles)}")

            # Longer delay between different queries
            time.sleep(10)

        articles = list(all_articles.values())
        print(f"\n{'=' * 60}")
        print(f"Total unique articles across all queries: {len(articles)}")
        print("=" * 60)
    else:
        # Search Figshare
        articles = search_figshare(
            query=args.query,
            size=args.size,
            page_size=args.page_size,
            item_type=args.item_type,
        )

    if not articles:
        print("No articles found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Extract dataset information
    print("\nExtracting dataset information...")
    if args.download_zips:
        print(f"ZIP download enabled, using dir: {args.download_dir}")
        args.download_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for idx, article in enumerate(articles, start=1):
        if args.fetch_details and idx % 10 == 0:
            print(f"Processing {idx}/{len(articles)}...", flush=True)

        try:
            dataset = extract_dataset_info(
                article,
                fetch_details=args.fetch_details,
                digested_at=args.digested_at,
                download_zips=args.download_zips,
                download_dir=args.download_dir if args.download_zips else None,
            )
            datasets.append(dataset)
        except Exception as e:
            print(
                f"Warning: Error extracting info for article {article.get('id')}: {e}",
                file=sys.stderr,
            )
            continue

        # Be nice to the API when fetching details
        if args.fetch_details:
            time.sleep(0.2)

    # Save deterministically
    save_datasets_deterministically(datasets, args.output)

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 60}")
    print("\nDataset Statistics:")

    # BIDS validation summary
    bids_validated = sum(1 for ds in datasets if ds.get("bids_validated"))
    bids_with_subjects = sum(
        1 for ds in datasets if ds.get("bids_subject_count", 0) > 0
    )
    bids_with_zips = sum(1 for ds in datasets if ds.get("bids_has_subject_zips"))

    print("\nBIDS Validation:")
    print(f"  Confirmed BIDS: {bids_validated}/{len(datasets)}")
    print(f"  With subject folders/zips: {bids_with_subjects}/{len(datasets)}")
    print(f"  Using subject-level ZIPs: {bids_with_zips}/{len(datasets)}")

    # Count by modality
    modalities_found = {}
    for ds in datasets:
        for mod in ds.get("modalities", []):
            modalities_found[mod] = modalities_found.get(mod, 0) + 1

    print("\nBy Modality:")
    for mod, count in sorted(
        modalities_found.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {mod}: {count}")

    # Datasets with files
    datasets_with_files = sum(1 for ds in datasets if ds.get("total_files", 0) > 0)
    print(f"\nDatasets with files: {datasets_with_files}/{len(datasets)}")

    # Total size
    total_size_bytes = sum(ds.get("size_bytes", 0) or 0 for ds in datasets)
    total_size_gb = round(total_size_bytes / (1024 * 1024 * 1024), 2)
    print(f"Total Size: {total_size_gb} GB")

    # Datasets with DOI
    datasets_with_doi = sum(1 for ds in datasets if ds.get("dataset_doi", ""))
    print(f"Datasets with DOI: {datasets_with_doi}/{len(datasets)}")

    # Datasets with authors
    datasets_with_authors = sum(1 for ds in datasets if ds.get("authors"))
    print(f"Datasets with authors: {datasets_with_authors}/{len(datasets)}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
