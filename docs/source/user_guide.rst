:html_theme.sidebar_secondary.remove: true
:og:description: Learn how to use EEGDash to query, download, and analyze 700+ BIDS-first EEG/MEG datasets. Covers EEGDashDataset, metadata queries, and reproducible pipelines.

.. meta::
   :description: Learn how to use EEGDash to query, download, and analyze 700+ BIDS-first EEG/MEG datasets. Covers EEGDashDataset, metadata queries, and reproducible pipelines.

.. currentmodule:: eegdash.api


User Guide
==========

This guide walks through the :mod:`eegdash` library and its main data access object, :class:`~eegdash.api.EEGDashDataset`. You will see how to find, access, and manage EEG data for research and analysis.

The EEGDash Object
------------------

:class:`~eegdash.api.EEGDashDataset` is the main tool for loading data for machine learning. For direct access to the metadata database, use the lower-level :class:`~eegdash.api.EEGDash` object. It is the right choice for exploring what is available, running ad-hoc queries, or managing records.

Initializing EEGDash
~~~~~~~~~~~~~~~~~~~~~~~~

Create a client that connects to the public database:

.. code-block:: python

    from eegdash import EEGDash

    # Connect to the public database
    eegdash = EEGDash()

Finding Records
~~~~~~~~~~~~~~~

Use ``find()`` to query the database for records matching specific criteria. Pass keyword arguments for simple filters, or a full MongoDB query dictionary for more advanced searches.

.. code-block:: python

    # Find records for a specific dataset and subject
    records = eegdash.find(dataset="ds002718", subject="012")
    print(f"Found {len(records)} records.")

    # You can also use more complex queries
    query = {"dataset": "ds002718", "subject": {"$in": ["012", "013"]}}
    records_advanced = eegdash.find(query)
    print(f"Found {len(records_advanced)} records with advanced query.")

EEGDash vs. EEGDashDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These two objects do different jobs:

-   :class:`~eegdash.api.EEGDash`: query and manage metadata. Returns a list of dictionaries, one per record.
-   :class:`~eegdash.api.EEGDashDataset`: load EEG data for analysis or machine learning. Returns a PyTorch-compatible dataset where each item can load the underlying EEG signal.

For most data loading work, use :class:`~eegdash.api.EEGDashDataset`.

The EEGDashDataset Object
-------------------------

:class:`~eegdash.api.EEGDashDataset` is the main entry point for working with EEG recordings in :mod:`eegdash`. It is a high-level interface that queries the metadata database and loads matching EEG data, either from a remote source or from a local cache.

Initializing EEGDashDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an instance of :class:`~eegdash.api.EEGDashDataset`. The two main parameters are ``cache_dir`` and ``dataset``.

- ``cache_dir``: local directory where ``eegdash`` stores downloaded data.
- ``dataset``: identifier of the dataset (e.g., ``"ds002718"``).

A basic example:

.. code-block:: python

    from eegdash import EEGDashDataset

    # Initialize the dataset for ds002718
    dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
    )

    print(f"Found {len(dataset)} recordings in the dataset.")

The resulting object holds every recording in ``ds002718``. Files are downloaded to ``./eeg_data/ds002718/`` on first access.

Querying for Specific Data
--------------------------

:class:`~eegdash.api.EEGDashDataset` lets you select a subset of recordings by task, subject, session, or run.

Filtering by Task
~~~~~~~~~~~~~~~~~

You can select recordings tied to a specific experimental task. For example, to get all resting-state recordings:

.. code-block:: python

    # Filter by a single task
    resting_state_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        task="RestingState"
    )

    print(f"Found {len(resting_state_dataset)} resting-state recordings.")

Filtering by Subject
~~~~~~~~~~~~~~~~~~~~

Filter by one or more subjects:

.. code-block:: python

    # Filter by a single subject
    subject_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        subject="012"
    )

    print(f"Found {len(subject_dataset)} recordings for subject 012.")

    # Filter by a list of subjects
    multi_subject_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        subject=["012", "013", "014"]
    )

    print(f"Found {len(multi_subject_dataset)} recordings for subjects 012, 013, and 014.")


Combining Filters
~~~~~~~~~~~~~~~~~

Combine filters for narrower queries. For example, to get resting-state recordings from a specific set of subjects:

.. code-block:: python

    # Combine subject and task filters
    combined_filter_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        subject=["012", "013"],
        task="RestingState"
    )

    print(f"Found {len(combined_filter_dataset)} recordings matching the criteria.")

Advanced Querying with MongoDB Syntax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more complex queries, pass a MongoDB-style query dictionary directly to the ``query`` parameter. Operators such as ``$in`` work here.

.. code-block:: python

    # Use a MongoDB-style query
    query = {
        "dataset": "ds002718",
        "subject": {"$in": ["012", "013"]},
        "task": "RestingState"
    }
    advanced_dataset = EEGDashDataset(cache_dir="./eeg_data", query=query)

    print(f"Found {len(advanced_dataset)} recordings using an advanced query.")


Working with Local Data (Offline Mode)
--------------------------------------

:mod:`eegdash` also works with local data you have already downloaded or manage yourself. Pass ``download=False`` and :class:`~eegdash.api.EEGDashDataset` reads BIDS-compliant files from disk instead of hitting the database or remote storage.

The data must follow a BIDS-like layout inside your ``cache_dir``. If ``cache_dir`` is ``./eeg_data`` and the dataset is ``ds002718``, the files belong under ``./eeg_data/ds002718/``.

Offline mode in practice:

.. code-block:: python

    # Initialize in offline mode
    local_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        download=False
    )

    print(f"Found {len(local_dataset)} local recordings.")

With ``download=False``, :mod:`eegdash` scans ``cache_dir`` for EEG files and builds the dataset from the local file system. Use this for offline environments, air-gapped machines, or your own curated datasets.

Accessing Data from the Dataset
-------------------------------

The :class:`~eegdash.api.EEGDashDataset` object behaves like a list: index into it to access individual recordings. Each item is an :class:`~eegdash.data_utils.EEGDashBaseDataset` that carries the metadata and loads the EEG data on demand.

.. code-block:: python

    if len(dataset) > 0:
        # Get the first recording
        recording = dataset[0]
        
        print(f"Loaded recording for subject: {recording.description['subject']}")

This is how ``eegdash`` plugs into a data analysis pipeline, whether the data is remote or local. For contributor resources, see :doc:`Developer Notes </developer_notes>`.


API Configuration
-----------------

By default, :mod:`eegdash` connects to the public REST API at ``https://data.eegdash.org``.
Override it through environment variables:

.. code-block:: bash

   # Override the default API URL (e.g., for testing)
   export EEGDASH_API_URL="https://data.eegdash.org"

   # Admin write operations (required for dataset ingestion)
   export EEGDASH_API_TOKEN="your-admin-token"

Public endpoints are rate-limited to 100 requests per minute per IP. Service
status is available at ``/health``, and every response carries an
``X-Request-ID`` header you can use for debugging.

For more on the API architecture, see :doc:`API Core </api/api_core>`.


.. seealso::

   :doc:`developer_notes` captures contributor workflows for the core package.
