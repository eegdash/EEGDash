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

   If you are only trying to install EEGDash, we recommend the :doc:`Installing from PyPI </install/install_pip>` section for details on that.



Install preview version from PyPI 
----------------------------------


.. code-block:: shell

   pip install --pre eegdash

This installs the in-development version of `eegdash` from the main branch, which may not be stable.


Install directly from repository from GitHub
--------------------------------------------

Clone the EEGDash repository and change into it:

.. code-block:: shell

   git clone https://github.com/eegdash/EEGDash && cd EEGDash

You should now be in the root directory of the EEGDash repository.

Installing EEGDash from the source with pip
-------------------------------------------

For a one-off install from source, without doing any development, use ``pip``:

For the latest development version, directly from GitHub:

.. code-block:: shell

  pip install git+https://github.com/eegdash/EEGDash.git

If you have a local clone of the EEGDash git repository:

.. code-block:: shell

   pip install -e .

This will install EEGDash in editable mode, i.e., changes to the source code could be used
directly in python.

You could also install optional dependency, like to import datasets from `test` and `docs`.

.. code-block:: shell

   pip install -e .[test,docs,dev]

There is also optional dependencies for unit testing and building documentation, you could install
them if you want to contribute to EEGDash.

.. code-block:: shell

   pip install -e .[all]


Testing if your installation is working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that EEGDash is installed and running correctly, run the following command:

.. code-block:: shell

   python -m "import eegdash; eegdash.__version__"

.. include:: /links.inc
