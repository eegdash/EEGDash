:html_theme.sidebar_secondary.remove: true

Core API
========

EEGDash provides a comprehensive interface for accessing and processing EEG data through 
a three-tier architecture that combines metadata management, cloud storage, and standardized 
data organization.

Architecture Overview
---------------------

The EEGDash core API is built around a REST API gateway that provides secure, scalable access
to the underlying data infrastructure:

.. code-block:: text

      +-----------------+
      |   REST API      |
      | (FastAPI+Redis) |
      +-----------------+
            |
            v
      +-----------------+      +-----------------+
      |     MongoDB     |      |      Redis      |
      |    (Metadata)   |      |  (Rate Limit)   |
      +-----------------+      +-----------------+
            |
            v
      +-----------v-----------+      +-----------------+
      |       eegdash         |<---->|   S3 Filesystem |
      |     Interface         |      |    (Raw Data)   |
      +-----------------------+      +-----------------+
            |
            v
      +-----------v-----------+
      |      BIDS Parser      |
      +-----------------------+

**REST API Gateway**
    A FastAPI-based REST API (``https://data.eegdash.org``) provides secure access to
    the metadata database. Features include:
    
    - Rate limiting (100 requests/minute for public endpoints)
    - Redis-backed distributed rate limiting for scalability
    - Prometheus metrics for monitoring (``/metrics``)
    - Request tracing with ``X-Request-ID`` headers
    - Health checks (``/health``) for service monitoring

**MongoDB Metadata Layer**
    Centralized NoSQL database storing EEG dataset metadata including subject information,
    session details, task parameters, and experimental conditions. Enables fast querying
    and filtering of large-scale datasets.

**File Cloud Storage**
    Scalable object storage for raw EEG data files. Provides reliable access to large
    datasets with on-demand downloading capabilities, reducing local storage requirements.
    At the moment, AWS S3 is the only supported storage backend.

**BIDS Standardization**
    Brain Imaging Data Structure (BIDS) parser ensuring consistent data organization
    and interpretation across different datasets and experiments.
    Used to perform the digest of BIDS datasets and extract relevant metadata for
    the MongoDB database.

API Endpoints
-------------

The REST API provides the following endpoints:

**Public Endpoints (Rate Limited)**

.. list-table::
   :header-rows: 1
   :widths: 10 40 50

   * - Method
     - Endpoint
     - Description
   * - GET
     - ``/``
     - API information and available endpoints
   * - GET
     - ``/health``
     - Health check with service status
   * - GET
     - ``/metrics``
     - Prometheus-compatible metrics
   * - GET
     - ``/api/{database}/records``
     - Query records with filters
   * - GET
     - ``/api/{database}/count``
     - Count documents matching filter
   * - GET
     - ``/api/{database}/datasets``
     - List all unique dataset names
   * - GET
     - ``/api/{database}/metadata/{dataset}``
     - Get metadata for specific dataset

**Admin Endpoints (Token Required)**

.. list-table::
   :header-rows: 1
   :widths: 10 40 50

   * - Method
     - Endpoint
     - Description
   * - POST
     - ``/admin/{database}/records``
     - Insert single record
   * - POST
     - ``/admin/{database}/records/bulk``
     - Insert multiple records (max 1000)
   * - POST
     - ``/admin/{database}/datasets``
     - Insert dataset metadata
   * - POST
     - ``/admin/{database}/datasets/bulk``
     - Insert multiple datasets (max 1000)

Database Schema
---------------

EEGDash uses a two-level MongoDB schema optimized for different query patterns:

**Datasets Collection** (for discovery/filtering)

One document per dataset containing metadata for browsing:

.. code-block:: python

   from eegdash.records import Dataset, create_dataset

   dataset = create_dataset(
       dataset_id="ds002718",
       name="A multi-subject EEG dataset",
       source="openneuro",
       recording_modality="eeg",
       tasks=["RestingState"],
       subjects_count=32,
   )

**Records Collection** (for fast file loading)

One document per EEG file with storage location:

.. code-block:: python

   from eegdash.records import Record, create_record

   record = create_record(
       dataset="ds002718",
       storage_base="s3://openneuro.org/ds002718",
       bids_relpath="sub-012/eeg/sub-012_task-RestingState_eeg.set",
       subject="012",
       task="RestingState",
   )

Core Modules
------------

The API is organized into focused modules that handle specific aspects of EEG data processing:

* :mod:`~eegdash.api` - Main ``EEGDash`` client for data access and querying
* :mod:`~eegdash.records` - Schema definitions (``Dataset``, ``Record`` TypedDicts)
* :mod:`~eegdash.http_api_client` - HTTP REST API client for database operations
* :mod:`~eegdash.downloader` - S3 and HTTPS download utilities
* :mod:`~eegdash.bids_metadata` - BIDS-compliant metadata handling
* :mod:`~eegdash.const` - Constants and configuration defaults
* :mod:`~eegdash.paths` - File system and storage path management
* :doc:`dataset/api_dataset` - Dataset classes and registry

Configuration
-------------

The API URL can be configured via environment variables:

.. code-block:: bash

   # Override the default API URL
   export EEGDASH_API_URL="https://data.eegdash.org"
   
   # For admin write operations
   export EEGDASH_API_TOKEN="your-admin-token"

API Reference
-------------

.. currentmodule:: eegdash

.. autosummary::
   :toctree: generated/api-core
   :recursive:

   api
   records
   http_api_client
   downloader
   bids_metadata
   const
   logging
   paths
   hbn
