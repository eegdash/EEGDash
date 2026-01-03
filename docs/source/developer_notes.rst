.. _developer_notes:

Developer Notes
===============

This guide is for project maintainers and contributors who need to work on the
EEGDash package, manage the data ingestion pipeline, or administer supporting services.

Package Overview
----------------

EEGDash (``eegdash``) provides a unified interface for accessing large-scale EEG datasets
from multiple sources. The package architecture consists of:

**Core Modules**

.. list-table::
   :widths: 25 75

   * - :mod:`eegdash.api`
     - ``EEGDash`` client for querying metadata via REST API and coordinating downloads
   * - :mod:`eegdash.dataset`
     - ``EEGDashDataset``, ``EEGChallengeDataset``, and dynamically registered dataset classes
   * - :mod:`eegdash.schemas`
     - Schema definitions for ``Dataset`` and ``Record`` TypedDicts
   * - :mod:`eegdash.http_api_client`
     - HTTP connection management for the EEGDash API gateway
   * - :mod:`eegdash.downloader`
     - S3 and HTTPS download utilities with progress tracking
   * - :mod:`eegdash.features`
     - Feature extraction utilities for EEG analysis

**Configuration**

Configuration defaults live in :mod:`eegdash.const`. Key environment variables:

- ``EEGDASH_API_URL`` - Override API endpoint (default: ``https://data.eegdash.org``)
- ``EEGDASH_ADMIN_TOKEN`` - Admin token for write operations

Local Development
-----------------

**Setup**

.. code-block:: bash

   # Clone and install in editable mode
   git clone https://github.com/eegdash/EEGDash.git
   cd EEGDash
   pip install -e .[dev,digestion]

   # Verify installation
   python -c "from eegdash import EEGDash; print(EEGDash)"

**Code Quality**

.. code-block:: bash

   pip install pre-commit
   pre-commit install
   pre-commit run -a

The pre-commit suite runs Ruff for linting/formatting and Codespell for spelling.

**Running Tests**

.. code-block:: bash

   pytest tests/ -v

Database Architecture
---------------------

EEGDash uses MongoDB with a **two-level schema** optimized for different query patterns:

**1. Datasets Collection** (discovery & filtering)

One document per dataset containing metadata for browsing and filtering:

.. code-block:: json

   {
     "dataset_id": "ds002718",
     "name": "A multi-subject EEG dataset",
     "source": "openneuro",
     "recording_modality": "eeg",
     "modalities": ["eeg"],
     "bids_version": "1.6.0",
     "license": "CC0",
     "tasks": ["RestingState", "GoNoGo"],
     "sessions": ["01", "02"],
     "demographics": {
       "subjects_count": 32,
       "age_mean": 28.5,
       "sex_distribution": {"m": 16, "f": 16}
     },
     "external_links": {
       "source_url": "https://openneuro.org/datasets/ds002718"
     },
     "timestamps": {
       "digested_at": "2024-01-15T10:30:00Z"
     }
   }

**2. Records Collection** (fast file loading)

One document per EEG file with storage information for direct loading:

.. code-block:: json

   {
     "dataset": "ds002718",
     "data_name": "ds002718_sub-012_task-RestingState_eeg.set",
     "bids_relpath": "sub-012/eeg/sub-012_task-RestingState_eeg.set",
     "datatype": "eeg",
     "suffix": "eeg",
     "extension": ".set",
     "entities": {
       "subject": "012",
       "task": "RestingState",
       "session": "01"
     },
     "entities_mne": {
       "subject": "012",
       "task": "RestingState",
       "session": "01"
     },
     "storage": {
       "backend": "s3",
       "base": "s3://openneuro.org/ds002718",
       "raw_key": "sub-012/eeg/sub-012_task-RestingState_eeg.set",
       "dep_keys": [
         "sub-012/eeg/sub-012_task-RestingState_events.tsv",
         "sub-012/eeg/sub-012_task-RestingState_eeg.fdt"
       ]
     },
     "digested_at": "2024-01-15T10:30:00Z"
   }

**Note on ``dep_keys``**: The digester automatically detects companion files required for loading:

- ``.fdt`` files for EEGLAB ``.set`` format
- ``.vmrk`` and ``.eeg`` files for BrainVision ``.vhdr`` format
- BIDS sidecar files (``_events.tsv``, ``_channels.tsv``, ``_electrodes.tsv``, ``_coordsystem.json``)

Data Ingestion Pipeline
-----------------------

The ingestion pipeline fetches BIDS datasets from 8 sources and transforms them
into MongoDB documents. All scripts are in ``scripts/ingestions/``.

**Supported Sources**

.. list-table::
   :header-rows: 1
   :widths: 15 15 35 35

   * - Source
     - Storage
     - Fetch Method
     - Clone Strategy
   * - OpenNeuro
     - S3
     - GraphQL API
     - Git shallow clone (``GIT_LFS_SKIP_SMUDGE=1``)
   * - NEMAR
     - HTTPS
     - GitHub API
     - Git shallow clone
   * - EEGManyLabs
     - HTTPS
     - GIN API
     - Git shallow clone
   * - Figshare
     - HTTPS
     - REST API
     - API manifest (no clone)
   * - Zenodo
     - HTTPS
     - REST API
     - API manifest (no clone)
   * - OSF
     - HTTPS
     - REST API
     - Recursive folder traversal
   * - ScienceDB
     - HTTPS
     - Query Service API
     - Metadata only (auth required for files)
   * - data.ru.nl
     - HTTPS
     - REST API
     - WebDAV PROPFIND

**Pipeline Scripts**

The pipeline consists of 4 steps:

.. code-block:: text

   1_fetch_sources/     → consolidated/*.json     (dataset listings)
         ↓
   2_clone.py           → data/cloned/*/         (shallow clones / manifests)
         ↓
   3_digest.py          → digestion_output/*/    (Dataset + Records JSON)
         ↓
   validate_output.py   → validation report      (optional but recommended)
         ↓
   4_inject.py          → MongoDB                (datasets + records collections)

**Step 1: Fetch** - Retrieve dataset listings from each source:

.. code-block:: bash

   # Fetch OpenNeuro datasets
   python scripts/ingestions/1_fetch_sources/openneuro.py \
     --output consolidated/openneuro_datasets.json

   # Available scripts: openneuro.py, nemar.py, eegmanylabs.py,
   #                    figshare.py, zenodo.py, osf.py, scidb.py, datarn.py

**Step 2: Clone** - Smart clone without downloading raw data:

.. code-block:: bash

   # Clone all datasets from consolidated files
   python scripts/ingestions/2_clone.py \
     --input consolidated \
     --output data/cloned \
     --workers 4

   # Clone specific sources
   python scripts/ingestions/2_clone.py \
     --input consolidated \
     --output data/cloned \
     --sources openneuro nemar

The clone script uses source-specific strategies:

- **Git sources**: Shallow clone with ``GIT_LFS_SKIP_SMUDGE=1`` (~300KB per dataset)
- **API sources**: REST API manifest fetching (no files downloaded)
- **WebDAV**: PROPFIND recursive directory listing

**Note on Git-Annex**: OpenNeuro and other git sources create **broken symlinks** 
(pointers to ``.git/annex/objects/``) rather than actual files. The digester handles 
these correctly using ``Path.is_symlink()`` to detect files and extract metadata 
without requiring actual file content.

**Step 3: Digest** - Extract BIDS metadata and generate documents:

.. code-block:: bash

   python scripts/ingestions/3_digest.py \
     --input data/cloned \
     --output digestion_output \
     --workers 4

Output structure:

.. code-block:: text

   digestion_output/
   ├── ds001785/
   │   ├── ds001785_dataset.json    # Dataset document
   │   ├── ds001785_records.json    # Records array
   │   └── ds001785_summary.json    # Processing stats
   ├── ds002718/
   │   └── ...
   └── BATCH_SUMMARY.json

**Step 4: Validate** (optional but recommended):

.. code-block:: bash

   python scripts/ingestions/validate_output.py

Checks for missing mandatory fields, invalid storage URLs, empty datasets, and ZIP placeholders.

**Step 5: Inject** - Upload to MongoDB:

.. code-block:: bash

   # Dry run (validate without uploading)
   python scripts/ingestions/4_inject.py \
     --input digestion_output \
     --database eegdash_staging \
     --dry-run

   # Actual injection
   python scripts/ingestions/4_inject.py \
     --input digestion_output \
     --database eegdash

   # Inject only datasets or records
   python scripts/ingestions/4_inject.py \
     --input digestion_output \
     --database eegdash \
     --only-datasets

CI/CD Workflows
---------------

Automated GitHub Actions workflows handle the full pipeline:

**Fetch Workflows** (``1-fetch-*.yml``)

Run weekly on Monday to update dataset listings:

- ``1-fetch-openneuro.yml``, ``1-fetch-nemar.yml``, etc.
- ``1-fetch-all.yml`` - Orchestrates all sources

**Digest Workflows** (``2-digest-*.yml``)

Triggered automatically after fetch completes:

- ``2-digest-openneuro.yml``, ``2-digest-nemar.yml``, etc.
- Uses ``2-clone-digest.yml`` reusable workflow

**Inject Workflow** (``3-inject-all.yml``)

Runs weekly on Tuesday to upload digested data:

- Injects to ``eegdash_staging`` by default (dry run)
- Manual trigger to inject to production ``eegdash``

**Full Pipeline** (``full-pipeline.yml``)

Manual workflow for end-to-end processing:

.. code-block:: yaml

   # Trigger via GitHub Actions UI with options:
   # - sources: all / openneuro / nemar / ...
   # - database: eegdashstaging / eegdash
   # - dry_run: true / false
   # - max_datasets: 0 (all) or limit

Data is stored in the ``eegdash-dataset-listings`` repository:

.. code-block:: text

   eegdash-dataset-listings/
   ├── consolidated/          # Fetched dataset listings
   │   ├── openneuro_datasets.json
   │   ├── nemar_datasets.json
   │   └── ...
   ├── cloned/                # Shallow clones / manifests
   │   ├── ds001785/
   │   └── ...
   └── digested/              # MongoDB-ready documents
       ├── ds001785/
       └── ...

API Server
----------

The API server (``mongodb-eegdash-server/``) is a FastAPI application:

**Environment Configuration**

Create ``.env`` in ``mongodb-eegdash-server/api/``:

.. code-block:: bash

   MONGO_URI=mongodb://user:password@host:27017
   MONGO_DB=eegdash
   ADMIN_TOKEN=your-secure-token

   # Optional
   REDIS_URL=redis://localhost:6379/0
   ENABLE_METRICS=true

**API Endpoints**

.. code-block:: text

   GET  /                                - API info
   GET  /health                          - Health check
   GET  /metrics                         - Prometheus metrics

   GET  /api/{db}/records                - Query records
   GET  /api/{db}/count                  - Count records
   GET  /api/{db}/datasets               - List dataset IDs
   GET  /api/{db}/metadata/{dataset_id}  - Get dataset metadata

   POST /admin/{db}/records              - Insert records (auth required)
   POST /admin/{db}/records/bulk         - Bulk insert (auth required)
   POST /admin/{db}/datasets             - Insert datasets (auth required)

**Rate Limiting**: 100 requests/minute per IP on public endpoints.

Release Process
---------------

1. Update version in ``pyproject.toml``
2. Update ``CHANGELOG.md``
3. Build and upload:

   .. code-block:: bash

      python -m build
      python -m twine upload dist/*

4. Create GitHub release with tag ``v{version}``

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html-noplot  # Fast build (no examples)
   make html         # Full build with examples

Documentation is auto-deployed to https://eegdash.org via GitHub Pages.
