#!/usr/bin/env python3
"""Clone/manifest BIDS datasets from multiple sources.

This script creates file manifests for datasets from various sources:
- Git-based: OpenNeuro, NEMAR, GIN (shallow clone)
- API-based: Figshare, Zenodo, OSF, SciDB, datarn (file listing via API)

Usage:
    python 2_clone.py --input consolidated --output data/cloned
    python 2_clone.py --sources figshare zenodo --limit 10
    python 2_clone.py --manifest-only  # Skip git clones
"""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import (
    CancelledError,
    ThreadPoolExecutor,
    TimeoutError,
    as_completed,
)
from pathlib import Path
from threading import Lock

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from _file_utils import (
    build_manifest,
    list_datarn_files,
    list_figshare_files,
    list_git_files,
    list_local_bids_files,
    list_osf_files,
    list_scidb_files,
    list_zenodo_files,
    save_manifest,
)

# Load API keys
_env_figshare = Path(__file__).resolve().parents[1] / ".env.figshare"
_env_zenodo = Path(__file__).resolve().parents[1] / ".env.zenodo"
load_dotenv(_env_figshare)
load_dotenv(_env_zenodo)

FIGSHARE_API_KEY = os.getenv("FIGSHARE_API_KEY", "")
ZENODO_API_KEY = os.getenv("ZENODO_API_KEY", "")

# Thread-safe stats
_lock = Lock()
_stats = {"success": 0, "manifest": 0, "skip": 0, "error": 0}


# =============================================================================
# Source Detection
# =============================================================================


def detect_source(dataset: dict) -> str:
    """Detect source from dataset metadata."""
    if src := dataset.get("source"):
        return src

    urls = "".join(
        str(v)
        for v in [
            dataset.get("external_links", {}).get("source_url", ""),
            dataset.get("clone_url", ""),
            dataset.get("ssh_url", ""),
        ]
    ).lower()

    if "gin.g-node.org" in urls:
        return "gin"
    if "openneuro" in urls:
        return "openneuro"
    if "nemar" in urls:
        return "nemar"
    if "figshare" in urls:
        return "figshare"
    if "zenodo" in urls:
        return "zenodo"
    if "osf.io" in urls:
        return "osf"
    if "scidb.cn" in urls:
        return "scidb"
    if "data.ru.nl" in urls:
        return "datarn"

    # Check by dataset_id pattern
    dataset_id = dataset.get("dataset_id", "")
    if dataset_id.startswith("ds") and dataset_id[2:].isdigit():
        return "openneuro"
    if dataset_id.startswith("nm") and dataset_id[2:].isdigit():
        return "nemar"

    return "unknown"


# =============================================================================
# Git Clone Handler
# =============================================================================


def clone_git(dataset: dict, output_dir: Path, timeout: int = 300) -> dict:
    """Shallow clone a git repository and extract manifest."""
    dataset_id = dataset["dataset_id"]
    source = detect_source(dataset)

    # Build clone URL
    if source == "openneuro":
        clone_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"
    elif source == "nemar":
        clone_url = f"https://github.com/NEMARDatasets/{dataset_id}.git"
    elif source == "gin":
        clone_url = dataset.get("clone_url") or dataset.get("ssh_url", "")
        # Fall back to source_url for GIN repos
        if not clone_url:
            source_url = dataset.get("external_links", {}).get("source_url", "")
            if "gin.g-node.org" in source_url:
                clone_url = (
                    source_url + ".git"
                    if not source_url.endswith(".git")
                    else source_url
                )
    else:
        clone_url = dataset.get("clone_url", "")

    if not clone_url:
        return {"status": "error", "dataset_id": dataset_id, "error": "No clone URL"}

    clone_dir = output_dir / dataset_id

    # Skip if already cloned
    if (clone_dir / ".git").exists():
        files = list_git_files(clone_dir)
        manifest = build_manifest(dataset_id, source, files, dataset)
        if source == "openneuro":
            manifest["storage_base"] = f"s3://openneuro.org/{dataset_id}"
        save_manifest(manifest, output_dir)
        return {"status": "skip", "dataset_id": dataset_id, "file_count": len(files)}

    # Clean up partial clone
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    # Shallow clone (no LFS, no data)
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(clone_dir)],
            env=env,
            capture_output=True,
            timeout=timeout,
            check=True,
        )

        files = list_git_files(clone_dir)
        manifest = build_manifest(dataset_id, source, files, dataset)
        if source == "openneuro":
            manifest["storage_base"] = f"s3://openneuro.org/{dataset_id}"
        save_manifest(manifest, output_dir)

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "file_count": len(files),
        }

    except subprocess.TimeoutExpired:
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        return {"status": "error", "dataset_id": dataset_id, "error": "Timeout"}

    except subprocess.CalledProcessError as e:
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        return {"status": "error", "dataset_id": dataset_id, "error": str(e.stderr)}


# =============================================================================
# API Manifest Handlers
# =============================================================================


def fetch_figshare(dataset: dict, output_dir: Path, **_kwargs) -> dict:
    """Fetch file manifest from Figshare."""
    dataset_id = dataset["dataset_id"]
    manifest_path = output_dir / dataset_id / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        return {
            "status": "skip",
            "dataset_id": dataset_id,
            "file_count": manifest.get("total_files", 0),
        }
    figshare_id = dataset.get("figshare_id")

    # Try to get figshare_id from various places
    if not figshare_id:
        ext_links = dataset.get("external_links", {})
        source_url = ext_links.get("source_url", "")
        import re

        match = re.search(r"/articles?/(\d+)", source_url)
        if match:
            figshare_id = match.group(1)

    if not figshare_id:
        return {"status": "error", "dataset_id": dataset_id, "error": "No figshare_id"}

    # Check for pre-fetched files
    if pre_files := dataset.get("_files"):
        files = []
        for f in pre_files:
            file_info = {"name": f.get("name", ""), "size": f.get("size", 0)}
            if zip_contents := f.get("zip_contents"):
                file_info["zip_contents"] = zip_contents
            files.append(file_info)
    else:
        files = list_figshare_files(figshare_id, FIGSHARE_API_KEY)

    if not files:
        return {"status": "error", "dataset_id": dataset_id, "error": "No files found"}

    manifest = build_manifest(dataset_id, "figshare", files, dataset)
    save_manifest(manifest, output_dir)

    return {"status": "manifest", "dataset_id": dataset_id, "file_count": len(files)}


def fetch_zenodo(dataset: dict, output_dir: Path, **_kwargs) -> dict:
    """Fetch file manifest from Zenodo."""
    dataset_id = dataset["dataset_id"]
    manifest_path = output_dir / dataset_id / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        return {
            "status": "skip",
            "dataset_id": dataset_id,
            "file_count": manifest.get("total_files", 0),
        }
    zenodo_id = dataset.get("zenodo_id")

    if not zenodo_id:
        ext_links = dataset.get("external_links", {})
        source_url = ext_links.get("source_url", "")
        import re

        match = re.search(r"/records?/(\d+)", source_url)
        if match:
            zenodo_id = match.group(1)

    if not zenodo_id:
        return {"status": "error", "dataset_id": dataset_id, "error": "No zenodo_id"}

    files = list_zenodo_files(zenodo_id, ZENODO_API_KEY)

    if not files:
        return {"status": "error", "dataset_id": dataset_id, "error": "No files found"}

    manifest = build_manifest(dataset_id, "zenodo", files, dataset)
    save_manifest(manifest, output_dir)

    return {"status": "manifest", "dataset_id": dataset_id, "file_count": len(files)}


def fetch_osf(dataset: dict, output_dir: Path, **_kwargs) -> dict:
    """Fetch file manifest from OSF."""
    dataset_id = dataset["dataset_id"]
    osf_id = dataset.get("osf_id")

    if not osf_id:
        ext_links = dataset.get("external_links", {})
        source_url = ext_links.get("source_url", "")
        import re

        match = re.search(r"osf\.io/([a-z0-9]+)", source_url, re.I)
        if match:
            osf_id = match.group(1)

    if not osf_id:
        return {"status": "error", "dataset_id": dataset_id, "error": "No osf_id"}

    files = list_osf_files(osf_id)

    if not files:
        return {"status": "error", "dataset_id": dataset_id, "error": "No files found"}

    manifest = build_manifest(dataset_id, "osf", files, dataset)
    save_manifest(manifest, output_dir)

    return {"status": "manifest", "dataset_id": dataset_id, "file_count": len(files)}


def fetch_scidb(dataset: dict, output_dir: Path, **_kwargs) -> dict:
    """Fetch file manifest from SciDB."""
    dataset_id = dataset["dataset_id"]
    scidb_dataset_id = dataset.get("scidb_dataset_id")
    version = dataset.get("scidb_version", "V1")

    if not scidb_dataset_id:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": "No scidb_dataset_id",
        }

    files = list_scidb_files(scidb_dataset_id, version, max_depth=8)

    # Even if empty, create manifest with metadata
    manifest = build_manifest(dataset_id, "scidb", files, dataset)
    save_manifest(manifest, output_dir)

    return {"status": "manifest", "dataset_id": dataset_id, "file_count": len(files)}


def fetch_datarn(dataset: dict, output_dir: Path, **_kwargs) -> dict:
    """Fetch file manifest from data.ru.nl."""
    dataset_id = dataset["dataset_id"]
    ext_links = dataset.get("external_links", {})
    source_url = ext_links.get("source_url", "")

    if not source_url:
        return {"status": "error", "dataset_id": dataset_id, "error": "No source_url"}

    files = list_datarn_files(source_url)

    manifest = build_manifest(dataset_id, "datarn", files, dataset)
    save_manifest(manifest, output_dir)

    return {"status": "manifest", "dataset_id": dataset_id, "file_count": len(files)}


def fetch_hbn(dataset: dict, output_dir: Path, **_kwargs) -> dict:
    """Fetch file manifest from local HBN/NeurIPS2025 BIDS directories.

    This handler scans local BIDS directories and creates manifests
    with S3 paths for storage.
    """
    dataset_id = dataset["dataset_id"]
    manifest_path = output_dir / dataset_id / "manifest.json"

    # Skip if manifest already exists
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        return {
            "status": "skip",
            "dataset_id": dataset_id,
            "file_count": manifest.get("total_files", 0),
        }

    # Get local path from dataset metadata
    local_path = dataset.get("local_path")
    if not local_path:
        return {"status": "error", "dataset_id": dataset_id, "error": "No local_path"}

    local_path = Path(local_path)
    if not local_path.exists():
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Path not found: {local_path}",
        }

    # List files from local BIDS directory
    files = list_local_bids_files(local_path)

    if not files:
        return {"status": "error", "dataset_id": dataset_id, "error": "No files found"}

    manifest = build_manifest(dataset_id, dataset.get("source", "hbn"), files, dataset)
    save_manifest(manifest, output_dir)

    return {"status": "manifest", "dataset_id": dataset_id, "file_count": len(files)}


def fetch_generic(dataset: dict, output_dir: Path, **_kwargs) -> dict:
    """Generic handler - just save metadata as manifest."""
    dataset_id = dataset["dataset_id"]
    source = detect_source(dataset)

    manifest = build_manifest(dataset_id, source, [], dataset)
    manifest["note"] = "Generic handler - no file listing available"
    save_manifest(manifest, output_dir)

    return {"status": "manifest", "dataset_id": dataset_id, "file_count": 0}


# =============================================================================
# Unified Processing
# =============================================================================


HANDLERS = {
    "openneuro": clone_git,
    "nemar": clone_git,
    "gin": clone_git,
    "figshare": fetch_figshare,
    "zenodo": fetch_zenodo,
    "osf": fetch_osf,
    "scidb": fetch_scidb,
    "datarn": fetch_datarn,
    "hbn": fetch_hbn,
    "neurips25": fetch_hbn,
    "unknown": fetch_generic,
}


def process_dataset(
    dataset: dict,
    output_dir: Path,
    manifest_only: bool = False,
    timeout: int = 300,
) -> dict:
    """Process a single dataset."""
    source = detect_source(dataset)

    # Use manifest-only handlers for git sources if requested
    if manifest_only and source in ("openneuro", "nemar", "gin"):
        handler = fetch_generic
    elif dataset.get("local_path"):
        # For local datasets (like NeurIPS2025), use the HBN/local handler
        handler = fetch_hbn
    else:
        handler = HANDLERS.get(source, fetch_generic)

    try:
        result = handler(dataset, output_dir, timeout=timeout)

        with _lock:
            status = result.get("status", "error")
            if status in _stats:
                _stats[status] += 1

        return result

    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        FileNotFoundError,
        PermissionError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        TimeoutError,
        ValueError,
        KeyError,
    ) as e:
        # Recoverable failures: network blip, missing file, malformed
        # dataset metadata. Convert to a structured error record so the
        # batch aggregate can continue. Programmer errors (AttributeError,
        # TypeError, etc.) propagate — that's deliberate.
        with _lock:
            _stats["error"] += 1
        return {
            "status": "error",
            "dataset_id": dataset.get("dataset_id", "unknown"),
            "error": str(e),
        }


# =============================================================================
# Main Entry Point
# =============================================================================


def load_datasets(
    input_path: Path,
    sources: list[str] | None = None,
    limit_per_source: int | None = None,
) -> list[dict]:
    """Load datasets from file or directory.

    Args:
        input_path: Path to JSON file or directory containing *_full.json
        sources: Optional list of sources to filter
        limit_per_source: If set, take at most this many from each source

    """
    datasets = []

    if input_path.is_file():
        with open(input_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                datasets = data
            elif "datasets" in data:
                datasets = data["datasets"]
    elif input_path.is_dir():
        for json_file in sorted(input_path.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    datasets.extend(data)
                elif "datasets" in data:
                    datasets.extend(data["datasets"])

    # Filter by source
    if sources:
        datasets = [d for d in datasets if detect_source(d) in sources]

    # Apply per-source limit if specified
    if limit_per_source:
        from collections import defaultdict

        by_source: dict[str, list] = defaultdict(list)
        for d in datasets:
            src = detect_source(d)
            if len(by_source[src]) < limit_per_source:
                by_source[src].append(d)

        datasets = []
        for src_datasets in by_source.values():
            datasets.extend(src_datasets)

    return datasets


def main():
    # CLI + env vars + validation via Pydantic-settings (C8 — final
    # stage in the C6.5 cross-stage pattern roll-out). 9 argparse flags
    # become declarative fields; bounds on workers / timeout / limits
    # prevent the "--workers 99999" misconfig that used to silently
    # sail through argparse. Sources list cross-checked against
    # KNOWN_SOURCES (kept aligned with HANDLERS).
    from pydantic import ValidationError

    from _clone_config import load_clone_config_from_argv

    try:
        cfg = load_clone_config_from_argv()
    except ValidationError as exc:
        print("Config error(s):", file=sys.stderr)
        for err in exc.errors():
            field = ".".join(str(p) for p in err.get("loc", []))
            print(f"  {field}: {err.get('msg')}", file=sys.stderr)
        return 1

    # Shim so the existing args.X access below works unchanged
    args = cfg

    # Load datasets
    print(f"Loading datasets from: {args.input}")
    datasets = load_datasets(args.input, args.sources, args.limit_per_source)

    if args.datasets:
        datasets = [d for d in datasets if d.get("dataset_id") in args.datasets]

    if args.limit:
        datasets = datasets[: args.limit]

    if not datasets:
        print("No datasets to process")
        return

    print(f"Processing {len(datasets)} datasets with {args.workers} workers")
    args.output.mkdir(parents=True, exist_ok=True)

    # Process datasets. Wall-clock budget across all workers protects
    # against a hung worker indefinitely blocking the main loop — even
    # though `args.timeout` already caps each individual handler, the
    # as_completed iterator itself blocks on the GIL waiting for
    # futures, so a stuck thread that never times out internally would
    # still deadlock. See  F3.
    per_dataset_timeout_s = float(args.timeout)
    wall_clock_budget_s = max(
        per_dataset_timeout_s * 2,
        len(datasets) * per_dataset_timeout_s / max(1, args.workers) + 60.0,
    )

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_dataset,
                ds,
                args.output,
                args.manifest_only,
                args.timeout,
            ): ds
            for ds in datasets
        }

        pbar = tqdm(
            as_completed(futures, timeout=wall_clock_budget_s),
            total=len(futures),
            desc="Processing",
            postfix=_stats,
        )
        for future in pbar:
            try:
                # process_dataset itself catches recoverable errors and
                # returns {"status": "error", ...}, so future.result() should
                # NOT normally raise. The outer except below catches only the
                # documented "transport" failures (cancellation, futures-API)
                # plus the network classes that COULD escape a handler that
                # forgot to wrap a particular path. A programmer error like
                # AttributeError or TypeError on a misshapen dataset dict will
                # propagate — that's correct: silent error masking is the
                # bug this fix addresses.
                result = future.result()
                results.append(result)
                pbar.set_postfix(_stats)
            except (
                CancelledError,
                TimeoutError,
                httpx.RequestError,
                httpx.HTTPStatusError,
                OSError,
            ) as e:
                results.append({"status": "error", "error": str(e)})

    # Save results
    results_path = args.output / "clone_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for status, count in _stats.items():
        if count:
            print(f"  {status}: {count}")
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
