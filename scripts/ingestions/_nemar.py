"""Shared NEMAR CLI helpers for ingestion scripts.

This module provides subprocess wrappers for the NEMAR CLI tool (`nemar-cli`),
which is installed via Bun and provides official NEMAR tooling for dataset
listing and metadata retrieval.

The CLI returns full metadata JSON, making it the preferred method for fetching
NEMAR dataset information.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from typing import Any


def get_nemar_api_key(api_key: str | None = None) -> str | None:
    """Return a NEMAR API key from args or env (NEMAR_API_KEY)."""
    return api_key or os.getenv("NEMAR_API_KEY")


def is_nemar_cli_available() -> bool:
    """Check if the nemar CLI is available in PATH."""
    return shutil.which("nemar") is not None


def run_nemar_cli(
    args: list[str],
    *,
    timeout: float = 120.0,
    api_key: str | None = None,
) -> dict | list | None:
    """Run nemar CLI command and parse JSON output.

    Args:
        args: Command arguments (e.g., ["dataset", "list", "--json"])
        timeout: Command timeout in seconds
        api_key: Optional NEMAR API key for authentication

    Returns:
        Parsed JSON output, or None if command fails

    """
    if not is_nemar_cli_available():
        return None

    cmd = ["nemar", *args]

    # Build environment with optional API key
    env = os.environ.copy()
    api_key = get_nemar_api_key(api_key)
    if api_key:
        env["NEMAR_API_KEY"] = api_key

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else ""
            if stderr:
                print(f"NEMAR CLI error: {stderr}")
            return None

        stdout = result.stdout.strip()
        if not stdout:
            return None

        return json.loads(stdout)

    except subprocess.TimeoutExpired:
        print(f"NEMAR CLI command timed out after {timeout}s")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse NEMAR CLI JSON output: {e}")
        return None
    except FileNotFoundError:
        print("NEMAR CLI not found in PATH")
        return None
    except Exception as e:
        print(f"NEMAR CLI error: {e}")
        return None


def iter_nemar_datasets(
    *,
    timeout: float = 120.0,
    api_key: str | None = None,
) -> Iterator[dict[str, Any]]:
    """List all NEMAR datasets via CLI (nemar dataset list --json).

    Args:
        timeout: Command timeout in seconds
        api_key: Optional NEMAR API key for authentication

    Yields:
        Dataset metadata dicts from CLI output

    """
    result = run_nemar_cli(
        ["dataset", "list", "--json"],
        timeout=timeout,
        api_key=api_key,
    )

    if result is None:
        return

    # Handle both list and dict responses
    if isinstance(result, list):
        yield from result
    elif isinstance(result, dict):
        # Some APIs wrap results in a data key
        datasets = result.get("data") or result.get("datasets") or result.get("items")
        if isinstance(datasets, list):
            yield from datasets
        else:
            # Single dataset response
            yield result


def fetch_dataset_status(
    dataset_id: str,
    *,
    timeout: float = 30.0,
    api_key: str | None = None,
) -> dict | None:
    """Get detailed status for a dataset (nemar dataset status <id> --json).

    Args:
        dataset_id: NEMAR dataset ID (e.g., "nm000103")
        timeout: Command timeout in seconds
        api_key: Optional NEMAR API key for authentication

    Returns:
        Dataset status dict, or None if command fails

    """
    result = run_nemar_cli(
        ["dataset", "status", dataset_id, "--json"],
        timeout=timeout,
        api_key=api_key,
    )

    if isinstance(result, dict):
        return result
    return None


def map_cli_dataset_to_schema(cli_data: dict[str, Any]) -> dict[str, Any]:
    """Map NEMAR CLI dataset fields to our Dataset schema.

    This maps the CLI output fields to the fields expected by create_dataset().

    CLI output fields (from `nemar dataset list --json`):
    - dataset_id: "nm000150"
    - name: "My Dataset"
    - description: null or string
    - status: "active"
    - github_repo: "nemarDatasets/nm000150"
    - concept_doi: null or string
    - created_at: "2026-01-21 05:32:05"
    - updated_at: "2026-01-21 05:32:05"
    - owner_username: "username"

    Args:
        cli_data: Raw dataset dict from NEMAR CLI

    Returns:
        Dict with fields mapped to our schema conventions

    """
    # Extract dataset ID - CLI uses "dataset_id"
    dataset_id = (
        cli_data.get("dataset_id")
        or cli_data.get("id")
        or cli_data.get("datasetId")
        or ""
    )

    # Extract name/title
    name = (
        cli_data.get("name")
        or cli_data.get("title")
        or cli_data.get("Name")
        or dataset_id
    )

    # Extract authors - handle both list and string
    # Note: CLI doesn't return authors directly, need to fetch from BIDS
    authors = cli_data.get("authors") or cli_data.get("Authors") or []
    if isinstance(authors, str):
        # Split if comma-separated string
        authors = [a.strip() for a in authors.split(",") if a.strip()]

    # Extract license - CLI doesn't return license, need to fetch from BIDS
    license_str = cli_data.get("license") or cli_data.get("License")

    # Extract BIDS version - CLI doesn't return this, need to fetch from BIDS
    bids_version = (
        cli_data.get("bidsVersion")
        or cli_data.get("BIDSVersion")
        or cli_data.get("bids_version")
    )

    # Extract DOI - CLI uses "concept_doi"
    dataset_doi = (
        cli_data.get("concept_doi")
        or cli_data.get("doi")
        or cli_data.get("DatasetDOI")
        or cli_data.get("dataset_doi")
    )

    # Extract subjects count - CLI doesn't return this, need to fetch from BIDS
    subjects_count = (
        cli_data.get("subjectsCount")
        or cli_data.get("subjects_count")
        or cli_data.get("numSubjects")
        or cli_data.get("participantCount")
        or 0
    )
    if isinstance(subjects_count, str):
        try:
            subjects_count = int(subjects_count)
        except ValueError:
            subjects_count = 0

    # Extract GitHub URL from github_repo field
    github_repo = cli_data.get("github_repo")
    github_url = (
        cli_data.get("github_url")
        or cli_data.get("githubUrl")
        or cli_data.get("sourceUrl")
        or cli_data.get("html_url")
    )
    if not github_url and github_repo:
        github_url = f"https://github.com/{github_repo}"
    elif not github_url and dataset_id:
        github_url = f"https://github.com/nemardatasets/{dataset_id}"

    # Extract timestamps
    created_at = (
        cli_data.get("created_at")
        or cli_data.get("createdAt")
        or cli_data.get("dateCreated")
    )
    updated_at = (
        cli_data.get("updated_at")
        or cli_data.get("updatedAt")
        or cli_data.get("dateModified")
        or cli_data.get("pushed_at")
    )

    # Extract funding - CLI doesn't return this, need to fetch from BIDS
    funding = cli_data.get("funding") or cli_data.get("Funding") or []
    if isinstance(funding, str):
        funding = [funding] if funding else []

    # Extract description/readme
    # CLI returns "description" but it's often null
    readme = (
        cli_data.get("readme")
        or cli_data.get("description")
        or cli_data.get("Description")
    )

    # Extract ages if available - CLI doesn't return this
    ages = cli_data.get("ages") or []

    return {
        "dataset_id": dataset_id,
        "name": name,
        "authors": authors,
        "license": license_str,
        "bids_version": bids_version,
        "dataset_doi": dataset_doi,
        "subjects_count": subjects_count,
        "source_url": github_url,
        "created_at": created_at,
        "updated_at": updated_at,
        "funding": funding,
        "readme": readme,
        "ages": ages,
        # Pass through raw data for fields we might need later
        "_raw": cli_data,
    }
