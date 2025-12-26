"""Fetch neural recording BIDS datasets from data.ru.nl (Radboud RDM repository).

This script searches the Radboud University Research Data Management repository
(data.ru.nl) for BIDS-compliant datasets across neural recording modalities. It 
uses the web interface search functionality to retrieve dataset information, 
outputting in the EEGDash Dataset schema format.

Data.ru.nl is the Research Data Management system of Radboud University, hosting
research data from Radboud University and affiliated institutions with support 
for BIDS datasets.

Output: consolidated/datarn_datasets.json (Dataset schema format)

Note: This script uses web scraping since data.ru.nl does not expose a public API.
It may require adjustments if the web interface structure changes.
"""

"""Fetch neural recording BIDS datasets from data.ru.nl (Radboud RDM repository).

This script provides a template and helper functions for working with datasets from
data.ru.nl (Radboud University Research Data Management system).

data.ru.nl is a custom Dataverse-based RDM system with a JavaScript-heavy SPA interface.
Unlike other sources (OpenNeuro, Zenodo, OSF), it does not expose a public REST API for
bulk dataset discovery. This script provides the following approaches:

1. MANUAL DATASET CURATION: Users can manually identify BIDS datasets on data.ru.nl
   and create a JSON manifest (see `create_datarn_manifest_template()`)

2. BROWSER AUTOMATION: For bulk fetching, the script provides Selenium support to
   render the JavaScript frontend and extract dataset information (requires Selenium + Chrome)

3. DIRECT LINK PROCESSING: Users can provide a pre-existing list of data.ru.nl dataset
   DOIs or persistent IDs for batch processing

Current Status:
- No public API available for automated discovery
- Manual curation recommended for now
- Browser automation provided as optional fallback

Output: consolidated/datarn_datasets.json (Dataset schema format)
"""

import argparse
import json
import sys
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import requests
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from eegdash.records import create_dataset

# Add ingestions dir to path for _serialize module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import save_datasets_deterministically

# Base URL
DATARN_BASE_URL = "https://data.ru.nl"
DATARN_SEARCH_URL = f"{DATARN_BASE_URL}/collections/published"

# Helper function to create manifest template
def create_datarn_manifest_template() -> dict[str, Any]:
    """Create a template for manual data.ru.nl dataset manifest.
    
    Returns:
        Template dictionary for user to fill in
    """
    return {
        "metadata": {
            "source": "datarn",
            "source_name": "Radboud University RDM (data.ru.nl)",
            "description": "Manually curated BIDS datasets from data.ru.nl",
            "harvest_method": "manual_curation",
            "harvest_date": datetime.now(timezone.utc).isoformat(),
            "note": "Add datasets manually by filling in the datasets array below"
        },
        "datasets": [
            {
                "dataset_id": "example_dataset_1",
                "name": "Example BIDS Dataset",
                "url": "https://data.ru.nl/dataset.xhtml?persistentId=doi:10.34894/XXXXXX",
                "doi": "10.34894/XXXXXX",
                "modalities": ["eeg"],
                "authors": ["Author Name"],
                "description": "Brief description of dataset"
            }
        ]
    }


def fetch_dataset_metadata_via_selenium(
    dataset_url: str,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Fetch dataset metadata using Selenium (requires Chrome + WebDriver).
    
    Args:
        dataset_url: URL of data.ru.nl dataset page
        timeout: Timeout in seconds
        
    Returns:
        Dataset metadata or None if fetch fails
    """
    if not SELENIUM_AVAILABLE:
        return None
    
    try:
        # Set up Chrome options for headless mode
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"--timeout={int(timeout)}")
        
        # Try to create driver (may fail if Chrome not installed)
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception:
            # Fall back to system Chrome
            return None
        
        try:
            driver.set_page_load_timeout(timeout)
            driver.get(dataset_url)
            
            # Wait for content to load
            WebDriverWait(driver, timeout).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "rdr-detail"))
            )
            
            # Extract data from rendered page
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")
            
            # Try to extract dataset info from rendered HTML
            metadata = {
                "title": None,
                "description": None,
                "authors": [],
                "doi": None,
            }
            
            # Look for title
            title_elem = soup.find(class_=re.compile("dataset.*title|title.*dataset", re.I))
            if title_elem:
                metadata["title"] = title_elem.get_text(strip=True)
            
            # Look for description
            desc_elem = soup.find(class_=re.compile("description|abstract", re.I))
            if desc_elem:
                metadata["description"] = desc_elem.get_text(strip=True)[:500]
            
            # Look for DOI
            doi_text = page_source
            doi_match = re.search(r"10\.\d+/\S+", doi_text)
            if doi_match:
                metadata["doi"] = doi_match.group(0)
            
            return metadata if metadata["title"] else None
            
        finally:
            driver.quit()
            
    except Exception as e:
        return None


def load_manual_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load manually curated dataset manifest.
    
    Args:
        manifest_path: Path to JSON manifest file
        
    Returns:
        List of dataset records
    """
    if not manifest_path.exists():
        print(f"Manifest file not found: {manifest_path}", file=sys.stderr)
        return []
    
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        datasets = manifest.get("datasets", [])
        print(f"Loaded {len(datasets)} datasets from manifest: {manifest_path}")
        return datasets
        
    except json.JSONDecodeError as e:
        print(f"Error parsing manifest: {e}", file=sys.stderr)
        return []


def extract_dataset_info(
    record: dict[str, Any],
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Extract dataset information from a data.ru.nl record.
    
    Args:
        record: Record dictionary (from manifest or API)
        timeout: Request timeout
        
    Returns:
        Dataset schema document or None if extraction fails
    """
    try:
        # Extract basic info
        dataset_id = record.get("dataset_id") or record.get("id", "datarn_unknown")
        name = record.get("name") or record.get("title", dataset_id)
        url = record.get("url") or f"{DATARN_BASE_URL}/dataset.xhtml"
        
        # Extract metadata
        description = record.get("description", "")
        authors = record.get("authors", [])
        doi = record.get("doi")
        modalities = record.get("modalities", [])
        
        # Try to fetch additional metadata from page if not in record
        if SELENIUM_AVAILABLE and not modalities:
            page_metadata = fetch_dataset_metadata_via_selenium(url, timeout)
            if page_metadata:
                if not description:
                    description = page_metadata.get("description", "")
                if not authors:
                    authors = page_metadata.get("authors", [])
                if not doi:
                    doi = page_metadata.get("doi")
        
        # Detect modalities from description and name
        text_to_search = f"{name} {description}".lower()
        if not modalities:
            modality_keywords = {
                "eeg": ["eeg", "electroencephalography", "electroencephalogram"],
                "meg": ["meg", "magnetoencephalography"],
                "emg": ["emg", "electromyography"],
                "fnirs": ["fnirs", "nirs", "near-infrared"],
                "lfp": ["lfp", "local field potential"],
                "ieeg": ["ieeg", "intracranial eeg", "ecog", "electrocorticography"],
            }
            
            for modality, keywords in modality_keywords.items():
                if any(kw in text_to_search for kw in keywords):
                    if modality not in modalities:
                        modalities.append(modality)
        
        # Default to unknown if no modality detected
        primary_modality = modalities[0] if modalities else "unknown"
        if not modalities:
            modalities = ["unknown"]
        
        # Create Dataset document
        dataset = create_dataset(
            dataset_id=dataset_id,
            name=name,
            source="datarn",
            recording_modality=primary_modality,
            modalities=modalities,
            authors=authors,
            dataset_doi=doi,
            source_url=url,
        )
        
        return dataset
        
    except Exception as e:
        print(f"Error extracting dataset {record.get('dataset_id', 'unknown')}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch neural recording BIDS datasets from data.ru.nl (Radboud RDM).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/datarn_datasets.json"),
        help="Output JSON file path (default: consolidated/datarn_datasets.json).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manually curated dataset manifest JSON file.",
    )
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create a template manifest file for manual curation.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--digested-at",
        type=str,
        default=None,
        help="ISO 8601 timestamp for digested_at field (for deterministic output).",
    )

    args = parser.parse_args()

    # Handle template creation
    if args.create_template:
        template = create_datarn_manifest_template()
        template_path = Path("datarn_manifest_template.json")
        with open(template_path, "w") as f:
            json.dump(template, f, indent=2)
        print(f"Template created: {template_path}")
        print("Edit this file to add your manually curated datasets, then run with --manifest")
        sys.exit(0)

    # Determine input source
    if args.manifest:
        print(f"Loading datasets from manifest: {args.manifest}...")
        records = load_manual_manifest(args.manifest)
    else:
        print("\n" + "=" * 70)
        print("data.ru.nl (Radboud RDM) - Manual Dataset Import")
        print("=" * 70)
        print("\nNote: data.ru.nl does not expose a public API for automated discovery.")
        print("Please use one of these approaches:\n")
        print("1. MANUAL CURATION (Recommended):")
        print(f"   {Path(__file__).name} --create-template")
        print("   # Edit datarn_manifest_template.json with your datasets")
        print(f"   {Path(__file__).name} --manifest datarn_manifest_template.json\n")
        print("2. VISIT data.ru.nl DIRECTLY:")
        print(f"   {DATARN_SEARCH_URL}")
        print("   # Search for 'BIDS' and identify relevant datasets\n")
        print("=" * 70)
        sys.exit(1)

    if not records:
        print("No datasets to process. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Extract dataset info
    print(f"\nProcessing {len(records)} records...")
    datasets = []
    for i, record in enumerate(records, 1):
        dataset = extract_dataset_info(record, timeout=args.timeout)
        if dataset:
            datasets.append(dataset)
        
        # Rate limiting
        time.sleep(0.1)

    if not datasets:
        print("No valid datasets extracted. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Add digested_at timestamp if provided
    if args.digested_at:
        for ds in datasets:
            if "timestamps" in ds:
                ds["timestamps"]["digested_at"] = args.digested_at

    # Save deterministically
    save_datasets_deterministically(datasets, args.output)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 70}")

    # Statistics
    modalities_found = {}
    for ds in datasets:
        for mod in ds.get("modalities", []):
            modalities_found[mod] = modalities_found.get(mod, 0) + 1

    if modalities_found:
        print("\nDatasets by modality:")
        for mod in sorted(modalities_found.keys()):
            count = modalities_found[mod]
            print(f"  {mod.upper():12s}: {count:4d}")

    datasets_with_doi = sum(1 for ds in datasets if ds.get("dataset_doi"))
    print(f"\nDatasets with DOI: {datasets_with_doi}/{len(datasets)}")

    datasets_with_authors = sum(1 for ds in datasets if ds.get("authors"))
    print(f"Datasets with authors: {datasets_with_authors}/{len(datasets)}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
