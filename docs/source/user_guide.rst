.. _user_guide:

:html_theme.sidebar_secondary.remove: true

.. currentmodule:: eegdash.api


User Guide
==========

This guide provides a comprehensive overview of the :mod:`eegdash` library, focusing on its core data access object, :class:`~eegdash.api.EEGDashDataset`. You will learn how to use this object to find, access, and manage EEG data for your research and analysis tasks.

The EEGDash Object
------------------

While :class:`~eegdash.api.EEGDashDataset` is the main tool for loading data for machine learning, the :class:`~eegdash.api.EEGDash` object provides a lower-level interface for directly interacting with the metadata database. It is useful for exploring the available data, performing complex queries, or managing metadata records.

Initializing EEGDash
~~~~~~~~~~~~~~~~~~~~~~~~

You can create a client to connect to the public database like this:

.. code-block:: python

    from eegdash import EEGDash

    # Connect to the public database
    eegdash = EEGDash()

Finding Records
~~~~~~~~~~~~~~~

The ``find()`` method allows you to query the database for records matching specific criteria. You can pass keyword arguments for simple filters or a full MongoDB query dictionary for more advanced searches.

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

It's important to understand the distinction between these two objects:

-   :class:`~eegdash.api.EEGDash`: Use this for querying and managing metadata. It returns a list of dictionaries, where each dictionary is a record from the database.
-   :class:`~eegdash.api.EEGDashDataset`: Use this when you need to load EEG data for analysis or machine learning. It returns a PyTorch-compatible dataset object where each item can load the actual EEG signal.

In general, you will use :class:`~eegdash.api.EEGDashDataset` for most of your data loading needs.

The EEGDashDataset Object
-------------------------

The :class:`~eegdash.api.EEGDashDataset` is the primary entry point for working with EEG recordings in :mod:`eegdash`. It acts as a high-level interface that allows you to query a metadata database and load corresponding EEG data, either from a remote source or from a local cache.

Initializing EEGDashDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get started, you need to create an instance of :class:`~eegdash.api.EEGDashDataset`. The two most important parameters are ``cache_dir`` and ``dataset``.

- ``cache_dir``: This is the local directory where ``eegdash`` will store downloaded data.
- ``dataset``: The identifier of the dataset you want to work with (e.g., ``"ds002718"``).

Here's a basic example of how to initialize the dataset:

.. code-block:: python

    from eegdash import EEGDashDataset

    # Initialize the dataset for ds002718
    dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
    )

    print(f"Found {len(dataset)} recordings in the dataset.")

This will create a dataset object containing all recordings from ``ds002718``. The data files will be downloaded to the ``./eeg_data/ds002718/`` directory when accessed.

Querying for Specific Data
--------------------------

:class:`~eegdash.api.EEGDashDataset` offers powerful filtering capabilities, allowing you to select a subset of recordings based on various criteria. You can filter by task, subject, session, or run.

Filtering by Task
~~~~~~~~~~~~~~~~~

You can easily select recordings associated with a specific experimental task. For example, to get all resting-state recordings:

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

You can also filter the data to get recordings from one or more subjects.

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

You can combine multiple filters to create more specific queries. For instance, to get the resting-state recordings for a specific set of subjects:

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

For more complex queries, you can pass a MongoDB-style query dictionary directly using the ``query`` parameter. This allows for advanced filtering, such as using operators like ``$in``.

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

:mod:`eegdash` also supports working with local data that you have already downloaded or manage separately. By setting ``download=False``, you can instruct :class:`~eegdash.api.EEGDashDataset` to use local BIDS-compliant data instead of accessing the database or remote storage.

To use this feature, your data must be organized in a BIDS-like structure within your ``cache_dir``. For example, if your ``cache_dir`` is ``./eeg_data`` and your dataset is ``ds002718``, the files should be located at ``./eeg_data/ds002718/``.

Here is how to use :class:`~eegdash.api.EEGDashDataset` in offline mode:

.. code-block:: python

    # Initialize in offline mode
    local_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        download=False
    )

    print(f"Found {len(local_dataset)} local recordings.")

When ``download=False``, :mod:`eegdash` will scan the specified directory for EEG files and construct the dataset from the local file system. This is useful for environments without internet access or when you want to work with your own curated datasets.

Accessing Data from the Dataset
-------------------------------

Once you have your :class:`~eegdash.api.EEGDashDataset` object, you can access individual recordings as if it were a list. Each item in the dataset is an :class:`~eegdash.data_utils.EEGDashBaseDataset` object, which contains the metadata and methods to load the actual EEG data.

.. code-block:: python

    if len(dataset) > 0:
        # Get the first recording
        recording = dataset[0]

        # Load the EEG data as a raw MNE object
        raw = recording.load()

        print(f"Loaded recording for subject: {recording.description['subject']}")
        print(f"Sampling frequency: {raw.info['sfreq']} Hz")
        print(f"Number of channels: {len(raw.ch_names)}")

This provides a powerful and flexible way to integrate ``eegdash`` into your data analysis pipelines, whether you are working with remote or local data. For contributor resources, see :doc:`Developer Notes </developer_notes>`.
