:html_theme.sidebar_secondary.remove: true
:og:description: Install EEGDash from source for development or testing unreleased features. Includes contributor setup instructions.

.. meta::
   :description: Install EEGDash from source for development or testing unreleased features. Includes contributor setup instructions.

.. _install_source:

Installing from sources
~~~~~~~~~~~~~~~~~~~~~~~

This page covers installing EEGDash from source, which is what you want for contributing or for trying features that have not yet been released.

For an overview of contributor workflows and project internals, see :doc:`Developer Notes </developer_notes>`.

.. note::

   If you only want to install a released version, see :doc:`Installing from PyPI </install/install_pip>`.


Install a pre-release from PyPI
-------------------------------

.. code-block:: shell

   pip install --pre eegdash

This installs the in-development version of ``eegdash`` from the main branch. It may not be stable.


Install directly from GitHub
----------------------------

Clone the repository and change into it:

.. code-block:: shell

   git clone https://github.com/eegdash/EEGDash && cd EEGDash


Install with pip
----------------

For a one-off install straight from GitHub:

.. code-block:: shell

  pip install git+https://github.com/eegdash/EEGDash.git

From a local clone, install in editable mode so source edits are picked up without reinstalling:

.. code-block:: shell

   pip install -e .

Optional extras let you pull in test and documentation dependencies:

.. code-block:: shell

   pip install -e .[test,docs,dev]

Or install everything, which is what you want for contributing:

.. code-block:: shell

   pip install -e .[all]


Verifying the installation
--------------------------

.. code-block:: shell

   python -c "import eegdash; print(eegdash.__version__)"

.. include:: /links.inc
