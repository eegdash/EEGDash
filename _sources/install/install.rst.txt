:html_theme.sidebar_secondary.remove: true
:og:description: Install EEGDash via pip or from source. Python 3.11+ required. Works on Linux, macOS, and Windows for EEG/MEG dataset access and analysis.

.. meta::
   :description: Install EEGDash via pip or from source. Python 3.11+ required. Works on Linux, macOS, and Windows for EEG/MEG dataset access and analysis.

.. _installation:

================
Installation
================

EEGDash requires Python 3.11 or higher.

The package is on `PyPI <eegdash-pypi_>`_, and the source lives on
`GitHub <eegdash-github_>`_.

Two install paths, depending on what you need:


.. grid:: 2

    .. grid-item-card::
        :text-align: center

        .. rst-class:: font-weight-bold mb-0

            Install via ``pip``

        .. rst-class:: card-subtitle text-muted mt-0

            For Beginners

        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        .. code-block:: shell
        
           pip install eegdash
        
        .. image:: /_static/eegdash_install.gif
           :alt: EEGDash Installer with pip
        
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        .. button-ref:: install_pip
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold

            Installing from PyPI


    .. grid-item-card::
        :text-align: center

        .. rst-class:: font-weight-bold mb-0

           Building from source code

        .. rst-class:: card-subtitle text-muted mt-0

            For Advanced Users

        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        .. image:: https://mne.tools/stable/_images/mne_installer_console.png
           :alt: Terminal Window

        For Python users who want the development version.
        Follow the setup instructions for building from GitHub.
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        .. button-ref:: install_source
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold

            From Source Code

.. toctree::
    :hidden:

    install_pip
    install_source

.. include:: /links.inc
