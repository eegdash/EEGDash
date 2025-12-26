.. _developer_notes:

Developer Notes
===============

This page collects the institutional knowledge that lived in ``DevNotes.md`` and
expands it with context about the EEG Dash codebase. It is intended for project
maintainers and contributors who need to work on the package, publish releases,
or administer supporting services.

Library Overview
----------------

EEG Dash ships a Python package named :mod:`eegdash` that provides several
layers of abstraction:

* ``EEGDash`` (:mod:`eegdash.api`) is the primary client for querying EEGDash
  metadata via REST API, coordinating S3 downloads, and performing bulk updates.
  It handles connection management via :class:`~eegdash.http_api_client.HTTPAPIConnectionManager`
  and communicates with the EEGDash API gateway at ``https://data.eegdash.org``.
* ``EEGDashDataset`` (:mod:`eegdash.data_utils`) wraps query results as a
  :class:`braindecode.datasets.BaseConcatDataset`, making it straightforward to
  integrate curated EEG collections into deep-learning pipelines.
* ``EEGChallengeDataset`` (:mod:`eegdash.dataset`) and the dynamically
  registered OpenNeuro dataset classes expose challenge-ready datasets with
  consistent preprocessing and metadata merging, all discoverable from
  ``eegdash.dataset``.
* ``features`` and ``plotting`` modules provide convenience utilities for
  feature extraction, summary reporting, and visualizations that appear in the
  documentation gallery.

Configuration defaults live in :mod:`eegdash.const`. The API URL can be
overridden via the ``EEGDASH_API_URL`` environment variable. For admin write
operations, set the ``EEGDASH_API_TOKEN`` environment variable with a valid
authentication token.

Local Development Workflow
--------------------------

Install editable dependencies and ensure the local import path points at your
workspace:

.. code-block:: bash

   pip install -r requirements.txt
   pip uninstall eegdash -y
   python -m pip install --editable .

.. warning::

   Use the exact ``python -m pip install --editable .`` command above. Running
   ``pip install`` without ``python -m`` may resolve to a different Python
   interpreter and leave the editable install in a broken state.

Smoke-test your environment from a clean shell or a different working directory:

.. code-block:: bash

   python -c "from eegdash import EEGDashDataset; print(EEGDashDataset)"

Code Quality Automation
-----------------------

.. code-block:: bash

   pip install pre-commit
   pre-commit install
   pre-commit run --all-files

The pre-commit suite runs Ruff for linting and import sorting, Black for
documentation snippets, and Codespell for spelling corrections.

Release Checklist
-----------------

1. Update the package version in ``pyproject.toml``.
2. Build distribution artifacts:

   .. code-block:: bash

      python -m build

3. Upload to TestPyPI or PyPI:

   .. code-block:: bash

      python -m twine upload --repository testpypi dist/*
      # or
      python -m twine upload dist/*

4. Retrieve the appropriate API token from the project email inbox (separate
   tokens exist for TestPyPI and PyPI).

Metadata & Database Management
------------------------------

The EEGDash API server uses a modern FastAPI-based architecture with MongoDB for
metadata storage and optional Redis for distributed rate limiting.

**Database Schema (Two-Level Hierarchy)**

EEGDash uses two separate MongoDB collections for efficient querying:

1. **datasets** - Per-dataset metadata for discovery and filtering:

   .. code-block:: json

      {
        "dataset_id": "ds002718",
        "name": "EEG Dataset Name",
        "source": "openneuro",
        "recording_modality": "eeg",
        "modalities": ["eeg"],
        "tasks": ["RestingState"],
        "demographics": {
          "subjects_count": 32,
          "age_mean": 28.5,
          "sex_distribution": {"m": 16, "f": 16}
        },
        "external_links": {
          "source_url": "https://openneuro.org/datasets/ds002718"
        }
      }

2. **records** - Per-file metadata optimized for fast loading:

   .. code-block:: json

      {
        "dataset": "ds002718",
        "data_name": "ds002718_sub-012_task-RestingState_eeg.set",
        "bids_relpath": "sub-012/eeg/sub-012_task-RestingState_eeg.set",
        "entities": {"subject": "012", "task": "RestingState"},
        "storage": {
          "backend": "s3",
          "base": "s3://openneuro.org/ds002718",
          "raw_key": "sub-012/eeg/sub-012_task-RestingState_eeg.set"
        }
      }

**Database Access**

* Sign in to `mongodb.com <https://mongodb.com>`_ using the shared account
  (``sccn3709@gmail.com``; credentials are stored in the team password vault).
* Toggle the target database inside ``scripts/data_ingest.py`` by updating the
  ``eegdash`` or ``eegdashstaging`` reference in ``main.py``.
* Run the ingestion script to populate or refresh records:

  .. code-block:: bash

     python scripts/data_ingest.py

**Server Configuration**

The API server is configured via environment variables. Create a ``.env`` file
in the ``mongodb-eegdash-server/api/`` directory:

.. code-block:: bash

   # Required
   MONGO_URI=mongodb://user:password@host:27017
   MONGO_DB=eegdash
   MONGO_COLLECTION=records
   ADMIN_TOKEN=your-secure-admin-token

   # Optional
   REDIS_URL=redis://localhost:6379/0      # For distributed rate limiting
   API_VERSION=2.1.0
   ENABLE_METRICS=true
   MONGO_MAX_POOL_SIZE=10
   MONGO_MIN_POOL_SIZE=1
   MONGO_CONNECT_TIMEOUT_MS=5000


Dataset Ingestion Pipeline
--------------------------

The ingestion pipeline transforms datasets from 8 sources into MongoDB-ready
documents. It runs as GitHub Actions workflows with data stored in the
``eegdash-dataset-listings`` repository.

**Supported Sources (8 total)**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Source
     - Method
     - Notes
   * - OpenNeuro
     - GraphQL API + Git shallow clone
     - S3 storage backend (``s3://openneuro.org``)
   * - NEMAR
     - GitHub API + Git shallow clone
     - HTTPS storage backend
   * - EEGManyLabs (GIN)
     - GIN API + Git shallow clone
     - HTTPS storage backend
   * - Figshare
     - REST API manifest
     - No git clone needed
   * - Zenodo
     - REST API manifest
     - No git clone needed
   * - OSF
     - REST API recursive traversal
     - No git clone needed
   * - ScienceDB (scidb.cn)
     - Query Service API
     - File listing requires authentication
   * - data.ru.nl
     - WebDAV PROPFIND
     - Recursive directory listing

**Pipeline Steps**

The pipeline consists of 3 scripts in ``scripts/ingestions/``:

1. **Fetch** (``1_fetch_sources/*.py``) - Retrieve dataset listings from each source:

   .. code-block:: bash

      python scripts/ingestions/1_fetch_sources/openneuro.py \
        --output consolidated/openneuro_datasets.json

2. **Clone & Digest** (``2_clone.py`` + ``3_digest.py``) - Smart clone without
   downloading raw data, then extract BIDS metadata:

   .. code-block:: bash

      # Smart clone (shallow git or API manifest)
      python scripts/ingestions/2_clone.py \
        --input consolidated/openneuro_datasets.json \
        --output data/cloned

      # Digest to Dataset + Records JSON
      python scripts/ingestions/3_digest.py \
        --input data/cloned \
        --output digestion_output

3. **Inject** (``4_inject.py``) - Upload to MongoDB collections:

   .. code-block:: bash

      python scripts/ingestions/4_inject.py \
        --input digestion_output \
        --database eegdashstaging \
        --dry-run  # Remove for actual upload

**Smart Clone Strategies**

The clone script uses different strategies per source to avoid downloading
large raw data files:

- **Git sources** (OpenNeuro, NEMAR, GIN): Shallow clone with
  ``GIT_LFS_SKIP_SMUDGE=1`` - files appear as symlinks (~300KB per dataset)
- **API sources** (Figshare, Zenodo, OSF): REST API manifest fetching
- **WebDAV sources** (data.ru.nl): PROPFIND recursive directory listing

**Output Structure**

.. code-block:: text

   digestion_output/
   ├── ds001785/
   │   ├── ds001785_dataset.json   # Dataset document (1 per dataset)
   │   ├── ds001785_records.json   # Records array (many per dataset)
   │   └── ds001785_summary.json   # Processing summary
   └── BATCH_SUMMARY.json

**CI/CD Workflows**

Automated workflows in ``.github/workflows/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Workflow
     - Description
   * - ``1-fetch-*.yml``
     - Fetch datasets from each source (weekly on Monday)
   * - ``2-digest-*.yml``
     - Clone + digest (triggered after fetch)
   * - ``3-inject-all.yml``
     - Inject to MongoDB (weekly on Tuesday)
   * - ``full-pipeline.yml``
     - Manual full pipeline: Fetch → Digest → Inject

API Gateway Endpoint
--------------------

The public HTTP gateway that fronts the MongoDB metadata service lives at
``|api-base-url|``. Point external tooling, health probes, and API examples at
that hostname instead of the raw server IP so future migrations only require
updating the ``|api-base-url|`` substitution in ``docs/source/links.inc``.

**API Features (v2.1.0+)**

- **Rate Limiting**: Public endpoints are limited to 100 requests/minute per IP
- **Metrics**: Prometheus-compatible metrics at ``/metrics``
- **Health Checks**: Service status at ``/health`` including MongoDB and Redis connectivity
- **Request Tracing**: All responses include ``X-Request-ID`` for debugging
- **Response Timing**: ``X-Response-Time`` header in milliseconds

**Available Endpoints**

.. code-block:: text

   GET  /                                - API information
   GET  /health                          - Health check with service status
   GET  /metrics                         - Prometheus metrics
   GET  /api/{database}/records          - Query records with filters
   GET  /api/{database}/count            - Count matching documents
   GET  /api/{database}/datasets         - List all dataset names
   GET  /api/{database}/metadata/{name}  - Get dataset metadata
   POST /admin/{database}/records        - Insert record (token required)
   POST /admin/{database}/records/bulk   - Bulk insert (token required)


Remote Storage Mounting
-----------------------

Some workflows require mounting Expanse project storage locally:

.. code-block:: bash

   sudo sshfs -o allow_other,IdentityFile=/home/dung/.ssh/id_rsa \
     arno@login.expanse.sdsc.edu:/expanse/projects/nemar /mnt/nemar/

Ensure the identity file path matches your local SSH configuration before
issuing the command.
