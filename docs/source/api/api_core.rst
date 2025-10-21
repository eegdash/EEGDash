:html_theme.sidebar_secondary.remove: true

Core API
========

EEGDash provides a comprehensive interface for accessing and processing EEG data through 
a three-tier architecture that combines metadata management, cloud storage, and standardized 
data organization.

Architecture Overview
---------------------

The EEGDash core API is built around three interconnected components:

.. code-block:: text

      +-----------------+
      |     MongoDB     |
      |    (Metadata)   |
      +-----------------+
            |
            |
      +-----------v-----------+      +-----------------+
      |       eegdash         |<---->|   S3 Filesystem |
      |     Interface         |      |    (Raw Data)   |
      +-----------------------+      +-----------------+
            |
            |
      +-----------v-----------+
      |      BIDS Parser      |
      +-----------------------+

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
    Use to perform the digest of BIDS datasets and extract relevant metadata for
    the mongodb.

Core Modules
------------

The API is organized into focused modules that handle specific aspects of EEG data processing:

* :mod:`~eegdash.api` - Main interface for data access and manipulation
* :mod:`~eegdash.const` - Constants and enumerations used throughout the package
* :mod:`~eegdash.bids_eeg_metadata` - BIDS-compliant metadata handling  
* :mod:`~eegdash.mongodb` - Database connection and query operations
* :mod:`~eegdash.paths` - File system and storage path management
* :mod:`~eegdash.utils` - General utility functions and helpers

API Reference
-------------

.. currentmodule:: eegdash

.. autosummary::
   :toctree: generated/api-core
   :recursive:

   api
   bids_eeg_metadata
   const
   logging
   hbn
   mongodb
   paths
   utils
