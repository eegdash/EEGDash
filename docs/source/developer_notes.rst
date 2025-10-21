:orphan:

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

* ``EEGDash`` (:mod:`eegdash.api`) is the primary client for querying MongoDB
  metadata, coordinating S3 downloads, and performing bulk updates. It handles
  connection management via :class:`~eegdash.mongodb.MongoConnectionManager`
  and reads default settings from ``mne`` config values or ``.env`` files.
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

Configuration defaults live in :mod:`eegdash.const`. The MongoDB bootstrap logic
is centralised in :func:`eegdash.utils._init_mongo_client`, which stores the
resolved connection string in the ``mne`` config directory. When troubleshooting
database access, confirm that the ``EEGDASH_DB_URI`` value is populated.

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

* Sign in to `mongodb.com <https://mongodb.com>`_ using the shared account
  (``sccn3709@gmail.com``; credentials are stored in the team password vault).
* Toggle the target database inside ``scripts/data_ingest.py`` by updating the
  ``eegdash`` or ``eegdashstaging`` reference in ``main.py``.
* Run the ingestion script to populate or refresh records:

  .. code-block:: bash

     python scripts/data_ingest.py

Remote Storage Mounting
-----------------------

Some workflows require mounting Expanse project storage locally:

.. code-block:: bash

   sudo sshfs -o allow_other,IdentityFile=/home/dung/.ssh/id_rsa \
     arno@login.expanse.sdsc.edu:/expanse/projects/nemar /mnt/nemar/

Ensure the identity file path matches your local SSH configuration before
issuing the command.
