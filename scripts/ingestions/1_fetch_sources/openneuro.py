"""Fetch OpenNeuro dataset IDs with metadata using requests library."""

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

# Add ingestion paths before importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _http import request_json
from _serialize import save_datasets_deterministically, setup_paths

setup_paths()
from eegdash.schemas import Dataset, create_dataset

GRAPHQL_URL = "https://openneuro.org/crn/graphql"

# Query to list datasets by modality (for initial discovery)
DATASETS_QUERY = """
query ($modality: String!, $first: Int!, $after: String) {
  datasets(modality: $modality, first: $first, after: $after) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        created
      }
    }
  }
}
"""

# Rich query to get full metadata for a single dataset
DATASET_DETAIL_QUERY = """
query($id: ID!) {
  dataset(id: $id) {
    id
    created
    publishDate
    name
    metadata {
      studyDomain
      studyDesign
      species
      associatedPaperDOI
      modalities
      ages
      trialCount
      grantFunderName
      grantIdentifier
      openneuroPaperDOI
      openneuroPaperDOI
      seniorAuthor
      adminUsers
      affirmedDefaced
      affirmedConsent
    }
    latestSnapshot {
      tag
      created
      size
      readme
      summary {
        modalities
        primaryModality
        secondaryModalities
        sessions
        subjects
        tasks
        totalFiles
        dataProcessed
        size
      }
      description {
        Name
        BIDSVersion
        License
        Authors
        Funding
        DatasetDOI
        DatasetType
        Acknowledgements
        HowToAcknowledge
        ReferencesAndLinks
        EthicsApprovals
      }
    }
  }
}
"""


# Batch query template for fetching details of multiple datasets
def build_batch_detail_query(dataset_ids: list[str]) -> str:
    """Build a batch query to fetch full details for multiple datasets."""
    query_parts = []
    for i, dataset_id in enumerate(dataset_ids):
        query_parts.append(f"""
    ds{i}: dataset(id: "{dataset_id}") {{
      id
      created
      publishDate
      name
      metadata {{
        studyDomain
        studyDesign
        species
        associatedPaperDOI
        modalities
        ages
        trialCount
        grantFunderName
        grantIdentifier
        openneuroPaperDOI
        seniorAuthor
        adminUsers
        affirmedDefaced
        affirmedConsent
      }}
      latestSnapshot {{
        tag
        created
        size
        readme
        summary {{
          modalities
          primaryModality
          secondaryModalities
          sessions
          subjects
          tasks
          totalFiles
          dataProcessed
          size
        }}
        description {{
          Name
          BIDSVersion
          License
          Authors
          Funding
          DatasetDOI
          DatasetType
          Acknowledgements
          HowToAcknowledge
          ReferencesAndLinks
          EthicsApprovals
        }}
      }}
    }}""")

    return f"query {{\n{''.join(query_parts)}\n}}"


def fetch_batch_details(
    dataset_ids: list[str], timeout: float = 30.0
) -> dict[str, dict]:
    """Fetch full details for a batch of datasets in a single query."""
    result = {}

    if not dataset_ids:
        return result

    try:
        query = build_batch_detail_query(dataset_ids)
        data, response = request_json(
            "post",
            GRAPHQL_URL,
            json_body={"query": query},
            timeout=timeout,
        )
        if not response or not isinstance(data, dict):
            print("  Error in batch query: no data")
            return result
        if "errors" in data:
            print(f"  Error in batch query: {data.get('errors')}")
            # If we return here, we lose data? But usually errors means query failed?
            # Partial errors?
            # If partial errors, data might still be present.
            # But let's see.

        response_data = data.get("data")
        if not response_data:
            print(f"  No data field in response. Keys: {data.keys()}")
            return result

        # print(f"DEBUG: Response keys: {response_data.keys()}")

        for i, dataset_id in enumerate(dataset_ids):
            alias = f"ds{i}"
            if alias in response_data and response_data[alias]:
                result[dataset_id] = response_data[alias]
            else:
                print(f"DEBUG: {alias} not found or null for {dataset_id}")
    except Exception as e:
        print(f"  Error fetching batch details: {str(e)[:100]}")

    return result


def extract_dataset_metadata(raw: dict, modality: str) -> Dataset:
    """Extract metadata from GraphQL response and create a Dataset document."""
    snapshot = raw.get("latestSnapshot") or {}
    summary = snapshot.get("summary") or {}
    description = snapshot.get("description") or {}
    metadata = raw.get("metadata") or {}

    # Extract subjects count (exclude "emptyroom")
    subjects_list = summary.get("subjects") or []
    subjects_count = len([s for s in subjects_list if s and s != "emptyroom"])

    # Extract and clean tasks
    tasks_list = summary.get("tasks") or []
    tasks_clean = [t for t in tasks_list if t and not t.startswith("TODO:")]

    # Extract ages (filter None values)
    ages = metadata.get("ages") or []
    ages_int = [int(a) for a in ages if a is not None]

    # Build OpenNeuro URL
    dataset_id = raw.get("id")
    snapshot_tag = snapshot.get("tag")
    openneuro_url = f"https://openneuro.org/datasets/{dataset_id}"
    if snapshot_tag:
        openneuro_url += f"/versions/{snapshot_tag}"

    # Get README from snapshot
    readme = snapshot.get("readme")

    # Get paper DOI - check multiple sources
    paper_doi = metadata.get("associatedPaperDOI") or metadata.get("openneuroPaperDOI")

    # Combine funding sources
    funding_list = description.get("Funding") or []
    grant_funder = metadata.get("grantFunderName")
    grant_id = metadata.get("grantIdentifier")
    if grant_funder and grant_funder not in funding_list:
        grant_info = f"{grant_funder}: {grant_id}" if grant_id else grant_funder
        funding_list = funding_list + [grant_info]

    # Extract senior author
    senior_author = metadata.get("seniorAuthor")

    return create_dataset(
        dataset_id=dataset_id,
        name=description.get("Name") or raw.get("name"),
        source="openneuro",
        readme=readme,
        recording_modality=modality,
        experimental_modalities=summary.get("modalities")
        or metadata.get("modalities")
        or [],
        bids_version=description.get("BIDSVersion"),
        license=description.get("License"),
        authors=description.get("Authors") or [],
        funding=funding_list,
        dataset_doi=description.get("DatasetDOI"),
        associated_paper_doi=paper_doi,
        tasks=tasks_clean,
        sessions=summary.get("sessions") or [],
        total_files=summary.get("totalFiles"),
        size_bytes=snapshot.get("size") or summary.get("size"),
        data_processed=summary.get("dataProcessed"),
        study_domain=metadata.get("studyDomain"),
        study_design=metadata.get("studyDesign"),
        subjects_count=subjects_count,
        ages=ages_int,
        species=metadata.get("species"),
        source_url=openneuro_url,
        dataset_created_at=raw.get("created") or metadata.get("created"),
        dataset_modified_at=snapshot.get("created"),
        senior_author=senior_author,
        contact_info=metadata.get("adminUsers"),
    )


def fetch_dataset_ids(
    page_size: int = 100,
    timeout: float = 30.0,
) -> Iterator[tuple[str, str]]:
    """Fetch all OpenNeuro dataset IDs with their modality.

    Yields tuples of (dataset_id, modality).
    """
    modality_configs = {
        "eeg": {"page_size": page_size, "max_errors": 3},
        "ieeg": {"page_size": 10, "max_errors": 5},  # Smaller for iEEG
        "meg": {"page_size": page_size, "max_errors": 3},
        "nirs": {"page_size": page_size, "max_errors": 3},  # fNIRS
    }

    for modality in ["eeg", "ieeg", "meg", "nirs"]:  # include this later "emg"
        config = modality_configs[modality]
        cursor = None
        consecutive_errors = 0
        current_page_size = config["page_size"]
        print(f"Fetching {modality} datasets (page size: {current_page_size})...")

        while True:
            try:
                payload = {
                    "query": DATASETS_QUERY,
                    "variables": {
                        "modality": modality,
                        "first": current_page_size,
                        "after": cursor,
                    },
                }

                result, response = request_json(
                    "post",
                    GRAPHQL_URL,
                    json_body=payload,
                    timeout=timeout,
                    raise_for_status=True,
                )
                if not response or not isinstance(result, dict):
                    raise Exception("Empty response")

                if "errors" in result:
                    raise Exception(f"GraphQL Error: {result['errors'][0]['message']}")

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)

                if "Not Found" in error_msg or "INTERNAL_SERVER_ERROR" in error_msg:
                    print(
                        f"  [{modality}] Server error (attempt {consecutive_errors}/{config['max_errors']})"
                    )
                    if consecutive_errors < config["max_errors"]:
                        continue
                    print(f"  [{modality}] Reached max errors, moving to next modality")
                    break
                else:
                    print(f"  [{modality}] Error: {error_msg[:100]}")
                    break

            data = result.get("data", {})
            page = data.get("datasets", {})
            edges = page.get("edges", [])

            if not edges:
                break

            for edge in edges:
                node = edge.get("node")
                if node and node.get("id"):
                    yield (node["id"], modality)

            page_info = page.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                print(f"  [{modality}] Completed")
                break
            cursor = page_info.get("endCursor")


def fetch_datasets_with_details(
    page_size: int = 100,
    timeout: float = 30.0,
    batch_size: int = 10,
    limit: int | None = None,
    dataset_ids: list[str] | None = None,
) -> list[dict]:
    """Fetch all OpenNeuro datasets with full metadata."""
    # First, collect all dataset IDs
    print("Phase 1: Discovering datasets...")
    dataset_modalities = {}
    for did, modality in fetch_dataset_ids(page_size=page_size, timeout=timeout):
        # Filter specific IDs if requested
        if dataset_ids and did not in dataset_ids:
            continue

        # Track modality (prefer more specific: ieeg > eeg > meg)
        if did not in dataset_modalities:
            dataset_modalities[did] = modality

    found_ids = list(dataset_modalities.keys())
    if limit:
        found_ids = found_ids[:limit]
    print(f"\nFound {len(found_ids)} unique datasets (limit: {limit})")

    # Then fetch full details in batches
    print(f"\nPhase 2: Fetching full metadata (batch size: {batch_size})...")
    datasets = []

    for batch_start in range(0, len(found_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(found_ids))
        batch_ids = found_ids[batch_start:batch_end]

        details = fetch_batch_details(batch_ids, timeout=timeout)

        for dataset_id in batch_ids:
            modality = dataset_modalities[dataset_id]
            if dataset_id in details:
                metadata = extract_dataset_metadata(details[dataset_id], modality)
                datasets.append(metadata)
            else:
                # Fallback: minimal record if details fetch failed
                datasets.append(
                    {
                        "dataset_id": dataset_id,
                        "recording_modality": modality,
                    }
                )

        if batch_end % 50 == 0 or batch_end == len(found_ids):
            print(f"  Processed {batch_end}/{len(found_ids)} datasets")

    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch all OpenNeuro datasets with metadata."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/openneuro_datasets.json"),
        help="Output JSON file (default: consolidated/openneuro_datasets.json).",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of datasets to fetch details for per batch query (default: 10)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Fetch only dataset IDs and modality (faster, less data)",
    )
    parser.add_argument(
        "--digested-at",
        type=str,
        default=None,
        help="ISO 8601 timestamp for digested_at field (for deterministic output, default: omitted for determinism)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of datasets to fetch (default: all)",
    )
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        help="List of specific dataset IDs to fetch (e.g. ds004504)",
    )
    args = parser.parse_args()

    if args.minimal:
        # Fast mode: just IDs and modality (returns dicts, not Dataset objects)
        datasets = [
            {"dataset_id": did, "recording_modality": mod}
            for did, mod in fetch_dataset_ids(
                page_size=args.page_size, timeout=args.timeout
            )
        ]
        if args.limit:
            datasets = datasets[: args.limit]
    else:
        # Full mode: complete metadata using Dataset schema
        datasets = fetch_datasets_with_details(
            page_size=args.page_size,
            timeout=args.timeout,
            batch_size=args.batch_size,
            limit=args.limit,
            dataset_ids=args.dataset_ids,
        )

        # Add digested_at timestamp if provided
        if args.digested_at:
            for ds in datasets:
                if "timestamps" in ds:
                    ds["timestamps"]["digested_at"] = args.digested_at

    # Save deterministically
    save_datasets_deterministically(datasets, args.output)

    print(f"\nSaved {len(datasets)} dataset entries to {args.output}")

    # Print summary stats
    if datasets and not args.minimal:
        modalities = {}
        total_subjects = 0
        for ds in datasets:
            mod = ds.get("recording_modality", "unknown")
            modalities[mod] = modalities.get(mod, 0) + 1
            # Access nested demographics
            demographics = ds.get("demographics", {})
            total_subjects += demographics.get("subjects_count", 0)

        print("\nSummary:")
        for mod, count in sorted(modalities.items()):
            print(f"  {mod}: {count} datasets")
        print(f"  Total subjects: {total_subjects}")


if __name__ == "__main__":
    main()
