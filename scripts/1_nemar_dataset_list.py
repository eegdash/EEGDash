#!/usr/bin/env python3
"""Script to retrieve and process NEMAR datasets."""

import logging
import os
import sys
from typing import Dict, List, Optional

import requests
import urllib3

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eegdash.dataset

# Disable SSL warnings since we're using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class NemarAPI:
    """Client for interacting with the NEMAR API."""

    def __init__(self, token: Optional[str] = None):
        """Initialize NEMAR API client.

        Args:
            token: NEMAR access token. If not provided, will look for NEMAR_TOKEN env variable.

        Raises:
            ValueError: If no token is provided or found in environment.

        """
        self.base_url = "https://nemar.org/api/dataexplorer/datapipeline"
        self.token = token or os.environ.get("NEMAR_TOKEN")
        if not self.token:
            raise ValueError(
                "NEMAR token must be provided either as argument or NEMAR_TOKEN environment variable"
            )

    def get_datasets(self, start: int = 0, limit: int = 500) -> Optional[Dict]:
        """Get list of datasets from NEMAR.

        Args:
            start: Starting index for pagination.
            limit: Maximum number of datasets to return.

        Returns:
            JSON response containing dataset information or None if request fails.

        """
        payload = {
            "nemar_access_token": self.token,
            "table_name": "dataexplorer_dataset",
            "start": start,
            "limit": limit,
        }

        try:
            response = requests.post(
                f"{self.base_url}/list",
                headers={"Content-Type": "application/json"},
                json=payload,
                verify=False,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Error fetching datasets: %s", e)
            return None

    @staticmethod
    def extract_dataset_info(datasets_response: Dict) -> List[Dict]:
        """Extract relevant information from datasets response.

        Args:
            datasets_response: Response from get_datasets().

        Returns:
            List of dictionaries containing dataset information.

        """
        if not datasets_response or "entries" not in datasets_response:
            return []

        return [
            {
                "id": data["id"],
                "name": data["name"],
                "modalities": data["modalities"],
                "participants": data["participants"],
                "file_size": data["file_size"],
                "file_size_gb": float(data["file_size"]) / (1024 * 1024 * 1024),
                "tasks": data.get("tasks", ""),
                "authors": data.get("Authors", ""),
                "doi": data.get("DatasetDOI", ""),
            }
            for _, data in datasets_response["entries"].items()
        ]


def fetch_all_datasets() -> List[Dict]:
    """Fetch all available datasets from NEMAR.

    Returns:
        List of dataset information dictionaries.

    """
    try:
        nemar = NemarAPI()
    except ValueError as e:
        logger.error("Error: %s", e)
        logger.error(
            "Please set your NEMAR token using: export NEMAR_TOKEN='your_token_here'"
        )
        return []

    all_datasets = []
    start = 0
    batch_size = 500

    logger.info("Fetching datasets...")
    while True:
        datasets = nemar.get_datasets(start=start, limit=batch_size)
        if not datasets or not datasets.get("entries"):
            break

        batch_info = nemar.extract_dataset_info(datasets)
        if not batch_info:
            break

        all_datasets.extend(batch_info)
        logger.info("Retrieved %d datasets so far...", len(all_datasets))

        if len(batch_info) < batch_size:
            break

        start += batch_size

    return all_datasets


def find_undigested_datasets() -> List[Dict]:
    """Find datasets that haven't been digested into eegdash yet.

    Returns:
        List of dataset information dictionaries for undigested datasets.

    """
    # Get all available datasets from NEMAR
    all_datasets = fetch_all_datasets()

    # Get all classes from eegdash.dataset
    eegdash_classes = dir(eegdash.dataset)

    # Filter for undigested datasets
    undigested = []
    for dataset in all_datasets:
        # Convert dataset ID to expected class name format (e.g., ds001785 -> DS001785)
        class_name = dataset["id"].upper()

        # Check if this dataset exists as a class in eegdash.dataset
        if class_name not in eegdash_classes:
            undigested.append(dataset)

    return undigested


def main():
    """Main function to find and output undigested datasets."""
    undigested = find_undigested_datasets()

    # Print just the dataset IDs and names
    print("\nUndigested Datasets:")
    print("-" * 80)
    for dataset in undigested:
        print(f"{dataset['id']}: {dataset['name']}")
    print(f"\nTotal undigested datasets: {len(undigested)}")


if __name__ == "__main__":
    main()
