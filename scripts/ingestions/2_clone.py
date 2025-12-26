#!/usr/bin/env python3
"""Smart clone/manifest BIDS datasets from multiple sources.

This script intelligently handles different source types:

1. **Git-based sources (OpenNeuro, NEMAR)**:
   - Shallow clone with GIT_LFS_SKIP_SMUDGE (metadata only, no raw data download)
   - Files are symlinks, not actual data

2. **GIN-based sources (EEGManyLabs)**:
   - Use GIN API to fetch file structure without cloning

3. **API-based sources (Figshare, Zenodo, OSF, SciDB, datarn)**:
   - Fetch file listings via API without downloading
   - Handle zip contents via API when available

The goal is to build BIDS path records WITHOUT downloading large data files.

Usage:
    # Process all sources (git clone for openneuro/nemar, API manifest for others)
    python 2_clone.py --input consolidated --output data/cloned

    # Clone only git-based sources
    python 2_clone.py --input consolidated --output data/cloned --sources openneuro nemar

    # Manifest only (no git clones - pure API mode)
    python 2_clone.py --input consolidated --output data/cloned --manifest-only

    # With BIDS validation (skip datasets that don't look like valid BIDS)
    python 2_clone.py --input consolidated --output data/cloned --validate-bids
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import requests
from tqdm import tqdm

# Thread-safe counters
_stats_lock = Lock()
_stats = {"success": 0, "skip": 0, "failed": 0, "timeout": 0, "error": 0, "manifest": 0}

# BIDS required files for validation
BIDS_REQUIRED = ["dataset_description.json"]
BIDS_INDICATORS = ["participants.tsv", "README", "CHANGES"]


# =============================================================================
# Source Detection
# =============================================================================


def detect_source(dataset: dict) -> str:
    """Detect the data source from dataset metadata."""
    if "source" in dataset:
        return dataset["source"]

    ext_links = dataset.get("external_links", {})
    source_url = ext_links.get("source_url", "")
    clone_url = dataset.get("clone_url", "") + dataset.get("ssh_url", "")
    all_urls = source_url + clone_url

    if "gin.g-node.org" in all_urls:
        return "gin"
    if "openneuro.org" in all_urls or "OpenNeuroDatasets" in all_urls:
        return "openneuro"
    if "nemardatasets" in all_urls.lower():
        return "nemar"
    if "figshare" in all_urls:
        return "figshare"
    if "zenodo.org" in all_urls:
        return "zenodo"
    if "osf.io" in all_urls:
        return "osf"
    if "scidb.cn" in all_urls:
        return "scidb"
    if "data.ru.nl" in all_urls:
        return "datarn"

    return "unknown"


def get_source_handler(source: str):
    """Get the appropriate handler for a source type."""
    handlers = {
        "openneuro": clone_git_shallow,
        "nemar": clone_git_shallow,
        "gin": clone_git_shallow,  # GIN also uses git, shallow clone is best
        "figshare": fetch_figshare_manifest,
        "zenodo": fetch_zenodo_manifest,
        "osf": fetch_osf_manifest,
        "scidb": fetch_scidb_manifest,
        "datarn": fetch_datarn_manifest,
    }
    return handlers.get(source, fetch_generic_manifest)


# =============================================================================
# Git-based Sources (OpenNeuro, NEMAR) - Shallow Clone
# =============================================================================


def get_git_clone_url(dataset: dict, source: str) -> str:
    """Generate Git clone URL based on source type."""
    dataset_id = dataset["dataset_id"]

    if source == "openneuro":
        return f"https://github.com/OpenNeuroDatasets/{dataset_id}"
    elif source == "nemar":
        return (
            dataset.get("clone_url") or f"https://github.com/nemardatasets/{dataset_id}"
        )
    elif source == "gin":
        # GIN repos - use source_url from external_links
        ext_links = dataset.get("external_links", {})
        source_url = ext_links.get("source_url", "")
        if source_url:
            return source_url
        return dataset.get("clone_url", "")
    else:
        ext_links = dataset.get("external_links", {})
        return ext_links.get("source_url", "")


def clone_git_shallow(
    dataset: dict,
    output_dir: Path,
    timeout: int = 300,
    validate_bids: bool = False,
) -> dict[str, Any]:
    """Clone a git repository in shallow mode without downloading annexed/LFS data.

    Uses:
    - --depth 1: Only latest commit (shallow clone)
    - GIT_LFS_SKIP_SMUDGE=1: Skip LFS file download (symlinks only)

    Annexed files appear as symlinks, NOT actual data files.
    """
    dataset_id = dataset["dataset_id"]
    source = detect_source(dataset)
    clone_dir = output_dir / dataset_id

    # Skip if already exists
    if clone_dir.exists():
        # Optionally validate existing clone
        if validate_bids:
            is_valid, reason = validate_bids_structure(clone_dir)
            if not is_valid:
                return {
                    "status": "invalid_bids",
                    "dataset_id": dataset_id,
                    "source": source,
                    "reason": reason,
                }

        # Extract manifest from existing clone
        manifest = extract_local_manifest(clone_dir, dataset_id)
        return {
            "status": "skip",
            "dataset_id": dataset_id,
            "source": source,
            "path": str(clone_dir),
            "file_count": len(manifest.get("files", [])),
            "manifest": manifest,
        }

    # Get clone URL
    url = get_git_clone_url(dataset, source)
    if not url:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": source,
            "error": "No clone URL",
        }

    try:
        # Clone without LFS data (shallow + skip smudge)
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(clone_dir)],
            timeout=timeout,
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            if clone_dir.exists():
                shutil.rmtree(clone_dir, ignore_errors=True)
            return {
                "status": "failed",
                "dataset_id": dataset_id,
                "source": source,
                "error": result.stderr[:300],
            }

        # Validate BIDS structure if requested
        if validate_bids:
            is_valid, reason = validate_bids_structure(clone_dir)
            if not is_valid:
                shutil.rmtree(clone_dir, ignore_errors=True)
                return {
                    "status": "invalid_bids",
                    "dataset_id": dataset_id,
                    "source": source,
                    "reason": reason,
                }

        # Extract file manifest
        manifest = extract_local_manifest(clone_dir, dataset_id)

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "source": source,
            "path": str(clone_dir),
            "file_count": len(manifest.get("files", [])),
            "manifest": manifest,
        }

    except subprocess.TimeoutExpired:
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)
        return {
            "status": "timeout",
            "dataset_id": dataset_id,
            "source": source,
            "timeout": timeout,
        }

    except Exception as e:
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": source,
            "error": str(e),
        }


def extract_local_manifest(clone_dir: Path, dataset_id: str) -> dict:
    """Extract file manifest from a cloned directory."""
    files = []
    bids_files = []

    for f in clone_dir.rglob("*"):
        if f.is_file() or f.is_symlink():
            rel_path = str(f.relative_to(clone_dir))

            # Skip git internals
            if ".git" in rel_path:
                continue

            file_info = {
                "path": rel_path,
                "name": f.name,
                "is_symlink": f.is_symlink(),
            }

            # Identify BIDS data files
            if is_bids_data_file(f.name):
                bids_files.append(rel_path)

            files.append(file_info)

    return {
        "dataset_id": dataset_id,
        "total_files": len(files),
        "bids_files": bids_files,
        "files": files,
    }


# =============================================================================
# GIN (EEGManyLabs) - API-based manifest
# =============================================================================


def fetch_gin_manifest(
    dataset: dict,
    output_dir: Path,
    timeout: int = 60,
    validate_bids: bool = False,
) -> dict[str, Any]:
    """Fetch file manifest from GIN repository via API (no clone)."""
    dataset_id = dataset["dataset_id"]
    ext_links = dataset.get("external_links", {})
    source_url = ext_links.get("source_url", "") or dataset.get("clone_url", "")

    # Parse org/repo from URL
    # https://gin.g-node.org/EEGManyLabs/RepoName -> EEGManyLabs/RepoName
    match = re.search(r"gin\.g-node\.org/([^/]+)/([^/]+)", source_url)
    if not match:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": "gin",
            "error": "Cannot parse GIN URL",
        }

    org, repo = match.groups()
    repo = repo.rstrip(".git")

    try:
        # Fetch repository tree via GIN API
        api_url = f"https://gin.g-node.org/api/v1/repos/{org}/{repo}/git/trees/HEAD?recursive=1"
        response = requests.get(api_url, timeout=timeout)

        if response.status_code != 200:
            # Try master branch
            api_url = f"https://gin.g-node.org/api/v1/repos/{org}/{repo}/git/trees/master?recursive=1"
            response = requests.get(api_url, timeout=timeout)

        if response.status_code != 200:
            return {
                "status": "error",
                "dataset_id": dataset_id,
                "source": "gin",
                "error": f"API error {response.status_code}",
            }

        data = response.json()
        tree = data.get("tree", [])

        # Build manifest
        files = []
        bids_files = []

        for item in tree:
            if item.get("type") == "blob":
                path = item.get("path", "")
                files.append(
                    {
                        "path": path,
                        "name": Path(path).name,
                        "size": item.get("size", 0),
                    }
                )

                if is_bids_data_file(Path(path).name):
                    bids_files.append(path)

        # Validate BIDS if requested
        if validate_bids:
            has_desc = any("dataset_description.json" in f["path"] for f in files)
            if not has_desc:
                return {
                    "status": "invalid_bids",
                    "dataset_id": dataset_id,
                    "source": "gin",
                    "reason": "No dataset_description.json",
                }

        manifest = {
            "dataset_id": dataset_id,
            "total_files": len(files),
            "bids_files": bids_files,
            "files": files,
        }

        # Save manifest
        manifest_dir = output_dir / dataset_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return {
            "status": "manifest",
            "dataset_id": dataset_id,
            "source": "gin",
            "file_count": len(files),
            "bids_file_count": len(bids_files),
            "manifest": manifest,
        }

    except Exception as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": "gin",
            "error": str(e),
        }


# =============================================================================
# Figshare - API-based manifest
# =============================================================================


def fetch_figshare_manifest(
    dataset: dict,
    output_dir: Path,
    timeout: int = 60,
    validate_bids: bool = False,
) -> dict[str, Any]:
    """Fetch file manifest from Figshare via API."""
    dataset_id = dataset["dataset_id"]

    # Extract article ID from dataset_id (figshare_12345)
    article_id = dataset_id.replace("figshare_", "")

    try:
        # Fetch article details including files
        api_url = f"https://api.figshare.com/v2/articles/{article_id}"
        response = requests.get(api_url, timeout=timeout)

        if response.status_code != 200:
            return {
                "status": "error",
                "dataset_id": dataset_id,
                "source": "figshare",
                "error": f"API error {response.status_code}",
            }

        data = response.json()
        files_data = data.get("files", [])

        files = []
        bids_files = []
        zip_contents = []

        for f in files_data:
            file_info = {
                "path": f.get("name", ""),
                "name": f.get("name", ""),
                "size": f.get("size", 0),
                "download_url": f.get("download_url", ""),
            }
            files.append(file_info)

            # Check if it's a zip that might contain BIDS
            if f.get("name", "").endswith(".zip"):
                # Try to read zip contents without downloading full file
                zip_manifest = peek_zip_contents(f.get("download_url"), timeout=timeout)
                if zip_manifest:
                    zip_contents.extend(zip_manifest)
                    for zf in zip_manifest:
                        if is_bids_data_file(Path(zf).name):
                            bids_files.append(zf)
            elif is_bids_data_file(f.get("name", "")):
                bids_files.append(f.get("name", ""))

        manifest = {
            "dataset_id": dataset_id,
            "total_files": len(files),
            "bids_files": bids_files,
            "files": files,
        }
        if zip_contents:
            manifest["zip_contents"] = zip_contents

        # Save manifest
        manifest_dir = output_dir / dataset_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return {
            "status": "manifest",
            "dataset_id": dataset_id,
            "source": "figshare",
            "file_count": len(files),
            "bids_file_count": len(bids_files),
            "manifest": manifest,
        }

    except Exception as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": "figshare",
            "error": str(e),
        }


# =============================================================================
# Zenodo - API-based manifest
# =============================================================================


def fetch_zenodo_manifest(
    dataset: dict,
    output_dir: Path,
    timeout: int = 60,
    validate_bids: bool = False,
) -> dict[str, Any]:
    """Fetch file manifest from Zenodo via API."""
    dataset_id = dataset["dataset_id"]

    # Extract record ID from dataset_id (zenodo_12345) or external_links
    record_id = dataset_id.replace("zenodo_", "")

    try:
        api_url = f"https://zenodo.org/api/records/{record_id}"
        response = requests.get(api_url, timeout=timeout)

        if response.status_code != 200:
            return {
                "status": "error",
                "dataset_id": dataset_id,
                "source": "zenodo",
                "error": f"API error {response.status_code}",
            }

        data = response.json()
        files_data = data.get("files", [])

        files = []
        bids_files = []

        for f in files_data:
            file_info = {
                "path": f.get("key", ""),
                "name": f.get("key", ""),
                "size": f.get("size", 0),
            }
            files.append(file_info)

            if is_bids_data_file(f.get("key", "")):
                bids_files.append(f.get("key", ""))

        manifest = {
            "dataset_id": dataset_id,
            "total_files": len(files),
            "bids_files": bids_files,
            "files": files,
        }

        # Save manifest
        manifest_dir = output_dir / dataset_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return {
            "status": "manifest",
            "dataset_id": dataset_id,
            "source": "zenodo",
            "file_count": len(files),
            "bids_file_count": len(bids_files),
            "manifest": manifest,
        }

    except Exception as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": "zenodo",
            "error": str(e),
        }


# =============================================================================
# OSF - API-based manifest
# =============================================================================


def fetch_osf_manifest(
    dataset: dict,
    output_dir: Path,
    timeout: int = 60,
    validate_bids: bool = False,
) -> dict[str, Any]:
    """Fetch file manifest from OSF via API."""
    dataset_id = dataset["dataset_id"]

    # Extract node ID from dataset_id (osf_xxxxx)
    node_id = dataset_id.replace("osf_", "")

    try:
        # Fetch files from OSF storage (recursively)
        files = []
        bids_files = []

        def fetch_folder(api_url: str, prefix: str = ""):
            """Recursively fetch files from OSF storage."""
            while api_url:
                response = requests.get(api_url, timeout=timeout)

                if response.status_code != 200:
                    break

                data = response.json()

                for item in data.get("data", []):
                    attrs = item.get("attributes", {})
                    kind = attrs.get("kind", "file")
                    name = attrs.get("name", "")
                    path = (prefix + "/" + name).lstrip("/") if prefix else name

                    if kind == "folder":
                        # Get folder contents link
                        folder_link = (
                            item.get("relationships", {})
                            .get("files", {})
                            .get("links", {})
                            .get("related", {})
                            .get("href")
                        )
                        if folder_link:
                            fetch_folder(folder_link, path)
                    else:
                        file_info = {
                            "path": path,
                            "name": name,
                            "size": attrs.get("size", 0),
                            "kind": kind,
                        }
                        files.append(file_info)

                        if is_bids_data_file(name):
                            bids_files.append(path)

                # Pagination
                next_link = data.get("links", {}).get("next")
                api_url = next_link if next_link else None

        # Start fetching from root storage
        root_url = f"https://api.osf.io/v2/nodes/{node_id}/files/osfstorage/"
        fetch_folder(root_url)

        manifest = {
            "dataset_id": dataset_id,
            "total_files": len(files),
            "bids_files": bids_files,
            "files": files,
        }

        # Save manifest
        manifest_dir = output_dir / dataset_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return {
            "status": "manifest",
            "dataset_id": dataset_id,
            "source": "osf",
            "file_count": len(files),
            "bids_file_count": len(bids_files),
            "manifest": manifest,
        }

    except Exception as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": "osf",
            "error": str(e),
        }


# =============================================================================
# SciDB & datarn - Metadata only (limited API)
# =============================================================================


def fetch_scidb_manifest(
    dataset: dict,
    output_dir: Path,
    timeout: int = 60,
    validate_bids: bool = False,
) -> dict[str, Any]:
    """Create manifest stub for ScienceDB (scidb.cn) datasets.

    ScienceDB file listing requires authentication via the detail endpoint.
    Dataset metadata (name, description, license, DOI) is already captured
    by the fetch script via the query-service API.

    The clone script only stores path-related info, which is not available
    without authentication.
    """
    dataset_id = dataset["dataset_id"]

    manifest = {
        "dataset_id": dataset_id,
        "source": "scidb",
        "dataset_doi": dataset.get("dataset_doi"),
        "external_links": dataset.get("external_links", {}),
        "note": "ScienceDB file listing requires authentication - metadata already in consolidated JSON",
        "total_files": 0,
        "bids_files": [],
        "files": [],
    }

    # Save manifest
    manifest_dir = output_dir / dataset_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "status": "manifest",
        "dataset_id": dataset_id,
        "source": "scidb",
        "file_count": 0,
        "note": "File listing requires authentication",
    }


def fetch_datarn_manifest(
    dataset: dict,
    output_dir: Path,
    timeout: int = 60,
    validate_bids: bool = False,
) -> dict[str, Any]:
    """Fetch file manifest from data.ru.nl via WebDAV PROPFIND.

    data.ru.nl provides WebDAV access for public datasets. We can list
    files recursively using PROPFIND without downloading actual data.
    """
    import xml.etree.ElementTree as ET
    from urllib.parse import unquote

    dataset_id = dataset["dataset_id"]
    ext_links = dataset.get("external_links", {})
    source_url = ext_links.get("source_url", "")

    files = []
    bids_files = []
    webdav_url = None

    # Try to get WebDAV URL from page JSON-LD
    if source_url:
        try:
            response = requests.get(source_url, timeout=timeout)
            if response.status_code == 200:
                ld_match = re.search(
                    r'<script type="application/ld\+json">([^<]+)</script>',
                    response.text,
                )
                if ld_match:
                    ld_data = json.loads(ld_match.group(1))
                    dist = ld_data.get("distribution", {})
                    if isinstance(dist, dict):
                        webdav_url = dist.get("contentUrl")
        except Exception:
            pass

    def parse_propfind_response(xml_text: str, current_url: str) -> tuple[list, list]:
        """Parse WebDAV PROPFIND response, return (files, directories)."""
        files_found = []
        dirs_found = []

        # Extract the path from current URL for comparison
        from urllib.parse import urlparse

        current_path = urlparse(current_url).path.rstrip("/")

        try:
            root = ET.fromstring(xml_text)
            ns = {"d": "DAV:"}

            for resp in root.findall(".//d:response", ns):
                href = resp.find("d:href", ns)
                if href is None or not href.text:
                    continue

                path = unquote(href.text).rstrip("/")

                # Skip the directory itself
                if path == current_path:
                    continue

                # Check if collection (directory)
                resourcetype = resp.find(".//d:resourcetype", ns)
                is_collection = (
                    resourcetype is not None
                    and resourcetype.find("d:collection", ns) is not None
                )

                if is_collection:
                    dirs_found.append(path)
                else:
                    # Get file size
                    size_elem = resp.find(".//d:getcontentlength", ns)
                    size = (
                        int(size_elem.text)
                        if size_elem is not None and size_elem.text
                        else 0
                    )
                    files_found.append({"path": path, "size": size})

        except ET.ParseError:
            pass

        return files_found, dirs_found

    def list_webdav_recursive(url: str, base_url: str, max_depth: int = 10) -> list:
        """Recursively list WebDAV directory using depth-1 PROPFIND."""
        all_files = []
        dirs_to_visit = [url]
        visited = set()
        depth = 0

        # Base for constructing absolute URLs from relative paths
        url_base = url.rsplit("/dcc/", 1)[0]  # e.g., https://webdav.data.ru.nl

        while dirs_to_visit and depth < max_depth:
            current_dirs = dirs_to_visit[:]
            dirs_to_visit = []

            for dir_url in current_dirs:
                if dir_url in visited:
                    continue
                visited.add(dir_url)

                try:
                    r = requests.request(
                        "PROPFIND",
                        dir_url,
                        headers={"Depth": "1"},
                        timeout=timeout,
                    )

                    if r.status_code in (200, 207):
                        files_found, subdirs = parse_propfind_response(r.text, dir_url)
                        all_files.extend(files_found)

                        # Queue subdirectories (they come as absolute paths like /dcc/xxx/)
                        for subdir in subdirs:
                            full_url = url_base + subdir
                            if full_url not in visited:
                                dirs_to_visit.append(full_url)

                except Exception:
                    pass

            depth += 1

        return all_files

    if webdav_url:
        try:
            raw_files = list_webdav_recursive(webdav_url, webdav_url)

            # Process file paths
            base_path = webdav_url.split("/")[-1]  # e.g., DSC_2017.00097_354_v1

            for f in raw_files:
                path = f["path"]
                # Extract relative path after the dataset folder
                parts = path.split(base_path + "/", 1)
                rel_path = parts[1] if len(parts) > 1 else Path(path).name

                file_info = {
                    "path": rel_path,
                    "name": Path(rel_path).name,
                    "size": f["size"],
                }
                files.append(file_info)

                if is_bids_data_file(Path(rel_path).name):
                    bids_files.append(rel_path)

        except Exception:
            files = []
            bids_files = []

    manifest = {
        "dataset_id": dataset_id,
        "source": "datarn",
        "webdav_url": webdav_url,
        "external_links": ext_links,
        "dataset_doi": dataset.get("dataset_doi"),
        "total_files": len(files),
        "bids_files": bids_files,
        "files": files,
    }

    if not files and webdav_url:
        manifest["note"] = "WebDAV listing failed"
    elif not webdav_url:
        manifest["note"] = "WebDAV URL not found"

    # Save manifest
    manifest_dir = output_dir / dataset_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "status": "manifest",
        "dataset_id": dataset_id,
        "source": "datarn",
        "file_count": len(files),
        "bids_file_count": len(bids_files),
        "webdav_available": webdav_url is not None,
    }


def fetch_generic_manifest(
    dataset: dict,
    output_dir: Path,
    timeout: int = 60,
    validate_bids: bool = False,
) -> dict[str, Any]:
    """Generic fallback for unknown sources."""
    dataset_id = dataset["dataset_id"]
    source = detect_source(dataset)

    return {
        "status": "skip",
        "dataset_id": dataset_id,
        "source": source,
        "reason": f"Unknown source type: {source}",
    }


# =============================================================================
# Helper Functions
# =============================================================================


def is_bids_data_file(filename: str) -> bool:
    """Check if a filename looks like a BIDS data file."""
    bids_extensions = {
        ".edf",
        ".bdf",
        ".set",
        ".fif",
        ".vhdr",
        ".eeg",
        ".vmrk",
        ".nii",
        ".nii.gz",
        ".meg4",
        ".ds",
        ".json",
        ".tsv",  # Sidecar files
    }

    name_lower = filename.lower()

    # Check extension
    for ext in bids_extensions:
        if name_lower.endswith(ext):
            return True

    # Check BIDS naming pattern (sub-XX, ses-XX, etc.)
    if re.match(r"sub-[a-zA-Z0-9]+", filename):
        return True

    return False


def validate_bids_structure(path: Path) -> tuple[bool, str]:
    """Quick validation of BIDS structure.

    Returns (is_valid, reason)
    """
    # Check for required files
    desc_file = path / "dataset_description.json"
    if not desc_file.exists():
        return False, "Missing dataset_description.json"

    # Try to parse it
    try:
        with open(desc_file) as f:
            desc = json.load(f)
        if "Name" not in desc:
            return False, "dataset_description.json missing 'Name' field"
    except (json.JSONDecodeError, IOError) as e:
        return False, f"Invalid dataset_description.json: {e}"

    # Check for at least one subject folder or data file
    has_subject = any(d.name.startswith("sub-") for d in path.iterdir() if d.is_dir())
    has_data = any(is_bids_data_file(f.name) for f in path.rglob("*") if f.is_file())

    if not has_subject and not has_data:
        return False, "No subject folders or data files found"

    return True, "Valid BIDS structure"


def peek_zip_contents(
    url: str, timeout: int = 30, max_bytes: int = 65536
) -> list[str] | None:
    """Peek at zip file contents by reading only the central directory.

    This downloads only the end of the zip file to read the file listing.
    """
    try:
        # First, get the file size with HEAD request
        head = requests.head(url, timeout=timeout, allow_redirects=True)
        if head.status_code != 200:
            return None

        file_size = int(head.headers.get("Content-Length", 0))
        if file_size == 0:
            return None

        # Download last 64KB to get central directory
        start_byte = max(0, file_size - max_bytes)
        headers = {"Range": f"bytes={start_byte}-{file_size}"}

        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code not in (200, 206):
            return None

        # Try to read as zip
        try:
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                return zf.namelist()
        except zipfile.BadZipFile:
            return None

    except Exception:
        return None


# =============================================================================
# Main Processing
# =============================================================================


def process_dataset(
    dataset: dict,
    output_dir: Path,
    timeout: int,
    validate_bids: bool,
    manifest_only: bool,
) -> dict[str, Any]:
    """Process a single dataset - clone or manifest based on source."""
    source = detect_source(dataset)

    # For git sources, clone unless manifest_only
    if source in ("openneuro", "nemar", "gin") and not manifest_only:
        result = clone_git_shallow(dataset, output_dir, timeout, validate_bids)
    else:
        # Use API-based manifest for all other sources
        handler = get_source_handler(source)
        result = handler(dataset, output_dir, timeout, validate_bids)

    # Update stats
    with _stats_lock:
        status = result.get("status", "error")
        if status in _stats:
            _stats[status] += 1
        elif status == "invalid_bids":
            _stats["failed"] += 1

    return result


def load_datasets(input_path: Path, sources: list[str] | None = None) -> list[dict]:
    """Load datasets from JSON file or directory."""
    datasets = []

    if input_path.is_file():
        with open(input_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                datasets.extend(data)
            elif isinstance(data, dict) and "datasets" in data:
                datasets.extend(data["datasets"])
    elif input_path.is_dir():
        for json_file in sorted(input_path.glob("*_datasets.json")):
            # Filter by source if specified
            if sources:
                source_name = json_file.stem.replace("_datasets", "")
                if source_name not in sources:
                    continue

            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    datasets.extend(data)

    # Filter by sources if specified
    if sources:
        datasets = [d for d in datasets if detect_source(d) in sources]

    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Smart clone/manifest BIDS datasets from multiple sources."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("consolidated"),
        help="Input JSON file or directory (default: consolidated/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cloned"),
        help="Output directory (default: data/cloned/)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        choices=[
            "openneuro",
            "nemar",
            "gin",
            "figshare",
            "zenodo",
            "osf",
            "scidb",
            "datarn",
        ],
        help="Process only specific sources (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for operations (default: 300)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum datasets to process (for testing)",
    )
    parser.add_argument(
        "--validate-bids",
        action="store_true",
        help="Validate BIDS structure (skip invalid datasets)",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only create manifests, no git clones (pure API mode)",
    )

    args = parser.parse_args()

    # Load datasets
    print(f"Loading datasets from: {args.input}")
    if args.sources:
        print(f"Filtering sources: {args.sources}")

    datasets = load_datasets(args.input, args.sources)
    print(f"Found {len(datasets)} datasets")

    if args.limit:
        datasets = datasets[: args.limit]
        print(f"Limited to {len(datasets)} datasets")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process datasets
    print(f"\nProcessing with {args.workers} workers")
    print(f"BIDS validation: {'enabled' if args.validate_bids else 'disabled'}")
    print(
        f"Manifest only: {'yes' if args.manifest_only else 'no (git sources will clone)'}"
    )
    print("=" * 60)

    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_dataset,
                ds,
                args.output,
                args.timeout,
                args.validate_bids,
                args.manifest_only,
            ): ds
            for ds in datasets
        }

        with tqdm(total=len(datasets), desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result.get("status") in (
                    "failed",
                    "timeout",
                    "error",
                    "invalid_bids",
                ):
                    failed.append(futures[future])

                pbar.update(1)
                pbar.set_postfix(_stats)

    # Save results
    results_path = args.output / "clone_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save retry list if needed
    if failed:
        retry_path = args.output / "retry.json"
        with open(retry_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"\nRetry list saved: {retry_path}")

    # Create enriched datasets summary
    enriched = []
    for result in results:
        if result.get("manifest"):
            enriched.append(
                {
                    "dataset_id": result["dataset_id"],
                    "source": result["source"],
                    "file_count": result.get("file_count", 0),
                    "bids_file_count": len(
                        result.get("manifest", {}).get("bids_files", [])
                    ),
                }
            )

    if enriched:
        enriched_path = args.output / "enriched_summary.json"
        with open(enriched_path, "w") as f:
            json.dump(enriched, f, indent=2)
        print(f"\nEnriched summary: {enriched_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Success (cloned):  {_stats['success']}")
    print(f"  Manifest (API):    {_stats['manifest']}")
    print(f"  Skipped:           {_stats['skip']}")
    print(f"  Failed:            {_stats['failed']}")
    print(f"  Timeout:           {_stats['timeout']}")
    print(f"  Error:             {_stats['error']}")
    print(f"\nResults saved: {results_path}")
    print("=" * 60)

    return 0 if _stats["failed"] + _stats["timeout"] + _stats["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
