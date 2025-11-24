.. _user_guide:

:html_theme.sidebar_secondary.remove: true

.. currentmodule:: eegdash.api


User Guide
==========

Welcome to the EEGDash User Guide! This comprehensive tutorial will help you get started with the :mod:`eegdash` library and learn how to effectively use it for your EEG research and analysis.

**What is EEGDash?**

EEGDash is a data-sharing resource that provides easy access to large-scale EEG datasets for machine learning and deep learning applications. It offers a simple Python API to query, download, and load EEG data from multiple publicly available datasets in a standardized format.

**Who should use this guide?**

This guide is designed for researchers, data scientists, and students who want to:

- Access large-scale EEG datasets for machine learning
- Perform meta-analyses across multiple EEG studies
- Develop and benchmark deep learning models on standardized data
- Explore available EEG data before committing to downloads

**Guide Overview**

.. contents:: **Contents**
   :local:
   :depth: 2

Getting Started
---------------

If you're new to EEGDash, start here! This section will get you up and running quickly.

Quick Start
~~~~~~~~~~~

The fastest way to get started is to load a dataset and access EEG recordings:

.. code-block:: python

    from eegdash import EEGDashDataset

    # Load a dataset
    dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        task="RestingState"
    )

    # Access the first recording
    recording = dataset[0]
    raw = recording.load()  # Returns an MNE Raw object
    
    print(f"Loaded {len(raw.ch_names)} channels at {raw.info['sfreq']} Hz")

That's it! You've just loaded your first EEG recording. The data is automatically downloaded and cached for future use.

**Next Steps:**

- Learn about :ref:`core-concepts` to understand the library architecture
- Explore :ref:`tutorials-examples` for complete workflows
- Check out the :doc:`Dataset Catalog </dataset_summary>` to browse available datasets

.. _core-concepts:

Core Concepts
-------------

EEGDash provides two main interfaces for working with EEG data. Understanding when to use each one is key to using the library effectively.

EEGDashDataset vs EEGDash
~~~~~~~~~~~~~~~~~~~~~~~~~~

EEGDash provides two complementary interfaces:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Interface
     - When to Use
   * - :class:`~eegdash.api.EEGDashDataset`
     - **Most common use case.** Use this when you need to load EEG data for analysis or machine learning. It returns a PyTorch-compatible dataset where each item can load the actual EEG signal as an MNE Raw object.
   * - :class:`~eegdash.api.EEGDash`
     - Use this for querying and exploring metadata only. It returns a list of dictionaries describing available recordings, without loading any signal data. Useful for browsing what's available before committing to downloads.

**Recommendation:** Start with :class:`~eegdash.api.EEGDashDataset` for most workflows. Only use :class:`~eegdash.api.EEGDash` if you need to explore metadata without loading data.

Example: Querying Metadata Only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you just want to explore what data is available without downloading anything:

.. code-block:: python

    from eegdash import EEGDash

    # Connect to the metadata database
    eegdash = EEGDash()

    # Find records for a specific dataset
    records = eegdash.find(dataset="ds002718", subject="012")
    print(f"Found {len(records)} recordings")
    
    # Each record is a dictionary with metadata
    print(records[0].keys())  # Shows available metadata fields

This is useful for exploration but doesn't provide data loading capabilities.

Working with EEGDashDataset
----------------------------

The :class:`~eegdash.api.EEGDashDataset` is your primary tool for loading EEG data. It handles querying, downloading, caching, and loading recordings seamlessly.

Basic Usage
~~~~~~~~~~~

The most basic usage requires only a cache directory and dataset identifier:

.. code-block:: python

    from eegdash import EEGDashDataset

    # Initialize the dataset for ds002718
    dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
    )

    print(f"Dataset contains {len(dataset)} recordings")

This creates a dataset object containing all recordings from ``ds002718``. The data files will be downloaded to ``./eeg_data/ds002718/`` when you access them.

**Key Parameters:**

- ``cache_dir``: Local directory where data will be stored
- ``dataset``: Dataset identifier (e.g., ``"ds002718"``)

Filtering and Querying
~~~~~~~~~~~~~~~~~~~~~~

EEGDashDataset offers powerful filtering to select specific recordings based on experimental parameters.

Filter by Task
^^^^^^^^^^^^^^

Select recordings from specific experimental paradigms:

.. code-block:: python

    # Get all resting-state recordings
    resting_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        task="RestingState"
    )

    print(f"Found {len(resting_dataset)} resting-state recordings")

Filter by Subject
^^^^^^^^^^^^^^^^^

Select data from one or more subjects:

.. code-block:: python

    # Single subject
    subject_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        subject="012"
    )

    # Multiple subjects
    multi_subject_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        subject=["012", "013", "014"]
    )

    print(f"Single subject: {len(subject_dataset)} recordings")
    print(f"Multiple subjects: {len(multi_subject_dataset)} recordings")

Combine Multiple Filters
^^^^^^^^^^^^^^^^^^^^^^^^^

Create specific queries by combining multiple filter criteria:

.. code-block:: python

    # Resting-state recordings from specific subjects
    filtered_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        subject=["012", "013"],
        task="RestingState"
    )

    print(f"Found {len(filtered_dataset)} recordings matching criteria")

Advanced MongoDB Queries
^^^^^^^^^^^^^^^^^^^^^^^^

For complex queries, use MongoDB-style syntax with the ``query`` parameter:

.. code-block:: python

    # Complex query with MongoDB operators
    query = {
        "dataset": "ds002718",
        "subject": {"$in": ["012", "013"]},
        "task": "RestingState"
    }
    advanced_dataset = EEGDashDataset(cache_dir="./eeg_data", query=query)

    print(f"Found {len(advanced_dataset)} recordings with advanced query")

Loading and Accessing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have your dataset, access individual recordings like a list:

.. code-block:: python

    # Get the first recording
    recording = dataset[0]

    # Load the EEG data as an MNE Raw object
    raw = recording.load()

    # Access metadata
    print(f"Subject: {recording.description['subject']}")
    print(f"Task: {recording.description['task']}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Channels: {len(raw.ch_names)}")

    # Iterate over multiple recordings
    for recording in dataset[:3]:
        raw = recording.load()
        print(f"Loaded recording: {recording.description['run']}")

Advanced Topics
---------------

Offline Mode (Local Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Work with local BIDS-formatted data without internet access:

.. code-block:: python

    # Use local BIDS data without querying the database
    local_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        download=False
    )

    print(f"Found {len(local_dataset)} local recordings")

**Requirements:**

- Data must be organized in BIDS structure: ``cache_dir/dataset/``
- No internet connection required
- Useful for offline analysis or custom datasets

.. _tutorials-examples:

Tutorials and Examples
----------------------

Ready to dive deeper? Check out our comprehensive tutorials and examples:

**Beginner Tutorials**

.. toctree::
   :maxdepth: 1

   generated/auto_examples/core/tutorial_eoec
   generated/auto_examples/eeg2025/tutorial_eegdash_offline

**Advanced Examples**

.. toctree::
   :maxdepth: 1

   generated/auto_examples/core/tutorial_feature_extractor_open_close_eye
   generated/auto_examples/core/p300_transfer_learning

**Browse All Examples**

Visit the :doc:`Examples Gallery </generated/auto_examples/index>` to see all available tutorials with code, visualizations, and downloadable notebooks.

Common Workflows
----------------

Here are some common use cases to help you get started quickly:

**Workflow 1: Explore Available Data**

.. code-block:: python

    from eegdash import EEGDash

    # Query metadata to see what's available
    eegdash = EEGDash()
    records = eegdash.find(dataset="ds002718")
    
    # Check unique tasks and subjects
    tasks = set(r['task'] for r in records)
    subjects = set(r['subject'] for r in records)
    print(f"Available tasks: {tasks}")
    print(f"Available subjects: {len(subjects)} subjects")

**Workflow 2: Load Data for Machine Learning**

.. code-block:: python

    from eegdash import EEGDashDataset
    from torch.utils.data import DataLoader

    # Load a specific subset
    dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        task="RestingState",
        subject=["012", "013"]
    )

    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Use in your training loop
    for batch in dataloader:
        # Your training code here
        pass

**Workflow 3: Multi-Dataset Analysis**

.. code-block:: python

    from eegdash import EEGDashDataset

    # Load multiple datasets
    datasets = [
        EEGDashDataset(cache_dir="./eeg_data", dataset="ds002718"),
        EEGDashDataset(cache_dir="./eeg_data", dataset="ds003775"),
    ]

    # Combine for meta-analysis
    all_recordings = []
    for dataset in datasets:
        all_recordings.extend([rec for rec in dataset])
    
    print(f"Total recordings: {len(all_recordings)}")

Additional Resources
--------------------

- **Dataset Catalog**: Browse all available datasets in the :doc:`Dataset Catalog </dataset_summary>`
- **API Reference**: Detailed API documentation in :doc:`API Reference </api/api>`
- **Developer Guide**: Contributing to EEGDash? See :doc:`Developer Notes </developer_notes>`

**Need Help?**

- Join our `Discord community <https://discord.gg/8jd7nVKwsc>`_
- Report issues on `GitHub <https://github.com/eegdash/EEGDash/issues>`_
- Check out the `EEG2025 Competition <https://eeg2025.github.io/>`_ for real-world applications
