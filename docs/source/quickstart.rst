:html_theme.sidebar_secondary.remove: true
:og:description: Quick Start hub for EEGDash. Three Cat A tutorials, four copy-paste recipes, API configuration, and pointers into the gallery, concepts, and datasets.

.. meta::
   :description: Quick Start hub for EEGDash. Three Cat A tutorials, four copy-paste recipes, API configuration, and pointers into the gallery, concepts, and datasets.

.. currentmodule:: eegdash.api


=================
Quick Start Guide
=================

EEGDash is two things at once: a **metadata index** over 700+ BIDS-curated
EEG/MEG datasets that you can query without ever downloading a byte, and a
**dataset loader** that materialises matching recordings into a
PyTorch-compatible :class:`~eegdash.api.EEGDashDataset`. The same library
takes you from a one-line ``find()`` against the public REST API to a
leakage-safe :class:`torch.utils.data.DataLoader`.

This page is the on-ramp. It points you at the curated learning path
(three Start-Here tutorials), gives you four copy-paste recipes for the
questions a hurried reader actually asks first -- *how do I open a
client, find records, filter by task, filter by subject?* -- documents
the environment variables that govern API access, and then hands off to
the rest of the documentation: the full gallery, the Concepts chapter
that explains *why* each design decision matters, and the dataset
catalogue.


The curated learning path
=========================

The Start-Here trio is the difficulty-1 on-ramp described in the
tutorial restructure plan, Category A. Each lesson runs CPU-only in a
few minutes and pairs with a YAML spec, an audit dossier, and the
"Behind this lesson" footer that traces every claim back to a citation.
Read them in order before branching into the rest of the gallery.

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: quickstart-cat-a

   .. grid-item-card:: Find datasets and records
      :link: /generated/auto_examples/tutorials/00_start_here/plot_00_first_search
      :link-type: doc
      :img-top: _static/thumbs/plot_00_first_search.png
      :columns: 12 6 4 4

      Open a metadata-only :class:`~eegdash.api.EEGDash` client and
      surface BIDS entity fields for ``ds002718`` -- subjects, sampling
      rates, total hours -- without downloading any signal.

   .. grid-item-card:: Load one recording and inspect it
      :link: /generated/auto_examples/tutorials/00_start_here/plot_01_first_recording
      :link-type: doc
      :img-top: _static/thumbs/plot_01_first_recording.png
      :columns: 12 6 4 4

      Materialise a single ``EEGDashDataset`` entry, read sampling rate,
      channel count, duration and annotations, and plot the first five
      seconds via ``RecordingPreview``.

   .. grid-item-card:: From dataset to PyTorch DataLoader
      :link: /generated/auto_examples/tutorials/00_start_here/plot_02_dataset_to_dataloader
      :link-type: doc
      :img-top: _static/thumbs/plot_02_dataset_to_dataloader.png
      :columns: 12 6 4 4

      Chain two safe Braindecode preprocessors, cut fixed-length
      windows, and wrap the result in a
      :class:`torch.utils.data.DataLoader` -- one batch shape, no
      training.


Common recipes
==============

The four blocks below cover the questions a new user asks in the first
five minutes: how do I open a client, find a record, filter by task,
filter by subject? Each recipe is self-contained and ends with a link
to the deeper Start-Here tutorial that walks through the same code in
narrative form.


Initializing EEGDash
--------------------

Create a metadata-only client that connects to the public database at
``https://data.eegdash.org``:

.. code-block:: python

    from eegdash import EEGDash

    # Connect to the public database
    eegdash = EEGDash()

For a fully worked example with cohort statistics and BIDS entity
fields, see
:doc:`/generated/auto_examples/tutorials/00_start_here/plot_00_first_search`.


Finding records
---------------

Use :meth:`~eegdash.api.EEGDash.find` to query the database. Pass keyword
arguments for simple filters, or a MongoDB-style dictionary for advanced
queries with operators like ``$in``:

.. code-block:: python

    # Simple keyword filter
    records = eegdash.find(dataset="ds002718", subject="012")
    print(f"Found {len(records)} records.")

    # MongoDB-style query
    query = {"dataset": "ds002718", "subject": {"$in": ["012", "013"]}}
    records_advanced = eegdash.find(query)
    print(f"Found {len(records_advanced)} records with advanced query.")

The same query mechanism feeds the on-ramp tutorial
:doc:`/generated/auto_examples/tutorials/00_start_here/plot_00_first_search`,
which walks through interpreting the BIDS fields you get back.


Filtering by task
-----------------

:class:`~eegdash.api.EEGDashDataset` accepts the same filters as
``find()`` and materialises the matching recordings as a
PyTorch-compatible dataset. Filter by task to focus on, say,
resting-state recordings:

.. code-block:: python

    from eegdash import EEGDashDataset

    resting_state_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        task="RestingState",
    )

    print(f"Found {len(resting_state_dataset)} resting-state recordings.")

For the end-to-end version that loads one recording and inspects its
annotations, see
:doc:`/generated/auto_examples/tutorials/00_start_here/plot_01_first_recording`.


Filtering by subject
--------------------

Filter by a single subject or by a list of subject IDs. Filters compose,
so you can combine ``subject``, ``task``, ``session`` and ``run`` for
narrower queries:

.. code-block:: python

    # One subject
    subject_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        subject="012",
    )
    print(f"Found {len(subject_dataset)} recordings for subject 012.")

    # A list of subjects, narrowed to a task
    multi_subject_dataset = EEGDashDataset(
        cache_dir="./eeg_data",
        dataset="ds002718",
        subject=["012", "013", "014"],
        task="RestingState",
    )
    print(f"Found {len(multi_subject_dataset)} recordings for subjects 012-014.")

To take a single-subject ``EEGDashDataset`` all the way to a PyTorch
``DataLoader`` -- preprocessing, windowing, batch shape -- see
:doc:`/generated/auto_examples/tutorials/00_start_here/plot_02_dataset_to_dataloader`.


API configuration
=================

By default, :mod:`eegdash` connects to the public REST API at
``https://data.eegdash.org``. Override it through environment variables:

.. code-block:: bash

   # Override the default API URL (e.g., for testing)
   export EEGDASH_API_URL="https://data.eegdash.org"

   # Admin write operations (required for dataset ingestion)
   export EEGDASH_API_TOKEN="your-admin-token"

Public endpoints are rate-limited to 100 requests per minute per IP.
Service status is available at ``/health``, and every response carries
an ``X-Request-ID`` header you can use for debugging.

For more on the API architecture, see :doc:`API Core </api/api_core>`.


Where to go next
================

Four hand-offs that cover the rest of the documentation: the full
gallery, the explanation pages, the dataset catalogue, and the audit
trail.

.. grid:: 1 2 2 4
   :gutter: 3
   :class-container: quickstart-next

   .. grid-item-card:: Examples gallery
      :link: /generated/auto_examples/index
      :link-type: doc
      :columns: 12 6 6 3

      The full curated learning path: seven tutorial categories from
      Start-Here to transfer learning, plus how-to recipes, applied
      projects, EEG 2025 Foundation Challenge pipelines, and HPC
      templates.

   .. grid-item-card:: Concepts
      :link: /concepts/index
      :link-type: doc
      :columns: 12 6 6 3

      Diataxis explanation pages: the ``EEGDashDataset`` object model,
      BIDS metadata, leakage and evaluation, preprocessing decisions,
      features versus deep learning. Read these to understand *why*.

   .. grid-item-card:: Datasets catalogue
      :link: dataset_summary
      :link-type: doc
      :columns: 12 6 6 3

      Search the 700+ BIDS-first datasets across EEG, MEG, fNIRS, EMG
      and iEEG modalities, with per-dataset cohort statistics, sampling
      rates, and ready-to-use class IDs.


.. seealso::

   :doc:`developer_notes` captures contributor workflows for the core package.
