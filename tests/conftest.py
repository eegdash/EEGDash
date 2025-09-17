from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def download_dir(tmp_path_factory) -> Path:
    """Provide a shared download directory for tests that need S3 assets."""
    return tmp_path_factory.mktemp("downloader")
