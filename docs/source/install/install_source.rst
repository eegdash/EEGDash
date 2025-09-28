:html_theme.sidebar_secondary.remove: true

.. _install_source:

Installing from sources
~~~~~~~~~~~~~~~~~~~~~~~

If you want to test features under development or contribute to the library, or if you want to test the new tools that have been tested in EEGDash and not released yet, this is the right tutorial for you!

.. note::

   If you are only trying to install EEGDash, we recommend the :doc:`Installing from PyPI </install/install_pip>` section for details on that.



Install preview version from PyPI 
----------------------------------


.. code-block:: shell

   pip install --pre eegdash

You should will install the version of `eegdash` that is currently under development at main branch, which may not be stable. 


Install directly from repository from GitHub
--------------------------------------------

Let's suppose that you want to install EEGDash from the source. The first thing you should do is clone the EEGDash repository to your computer and enter inside the repository.

.. code-block:: shell

   git clone https://github.com/sccn/EEGDash && cd EEGDash

You should now be in the root directory of the EEGDash repository.

Installing EEGDash from the source with pip
-------------------------------------------

If you want to only install EEGDash from source once and not do any development
work, then the recommended way to build and install is to use ``pip``

For the latest development version, directly from GitHub:

.. code-block:: shell

  pip install git+https://github.com/sccn/EEGDash.git

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
