:html_theme.sidebar_secondary.remove: true

.. title:: EEGDash - Data-sharing interface for M/EEG and related (fNIRS, EMG)

.. raw:: html

    <style type="text/css">
    /* Visually hide H1 but keep for metadata */
    h1 {
      position: absolute !important;   
      width: 1px !important;
      height: 1px !important;
      padding: 0 !important;
      margin: -1px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
      white-space: nowrap !important;
      border: 0 !important;
    }
    </style>

.. container:: eegdash-hero

   .. container:: eegdash-hero-title
      
      .. text is hidden or removed to rely on logo

   .. container:: eegdash-hero-subtitle
   
      Data-sharing interface for M/EEG and related (fNIRS, EMG)

   .. image:: _static/eegdash.svg
      :alt: EEG Dash Logo
      :class: logo mainlogo only-dark
      :align: center
      :width: 600px

   .. image:: _static/eegdash.svg
      :alt: EEG Dash Logo
      :class: logo mainlogo only-light
      :align: center
      :width: 600px

   .. rst-class:: lead text-center font-weight-light my-4
   
      The **EEG-DaSh** is a data-sharing resource for **M/EEG** (EEG, MEG, iEEG, fNIRS, EMG) data, 
      allowing **interoperability** and enabling large-scale computational advancements to preserve and share scientific data 
      from publicly funded research for machine learning and deep learning applications.

   .. container:: eegdash-badges text-center mb-2

      .. image:: https://github.com/eegdash/EEGDash/actions/workflows/tests.yml/badge.svg
         :alt: Test Status
         :target: https://github.com/eegdash/EEGDash/actions/workflows/tests.yml

      .. image:: https://github.com/eegdash/EEGDash/actions/workflows/doc.yaml/badge.svg
         :alt: Doc Status
         :target: https://github.com/eegdash/EEGDash/actions/workflows/doc.yaml


      .. image:: https://img.shields.io/pypi/v/eegdash?color=blue&style=flat-square
         :alt: PyPI
         :target: https://pypi.org/project/eegdash/

      .. image:: https://img.shields.io/pypi/pyversions/eegdash?style=flat-square
         :alt: Python
         :target: https://pypi.org/project/eegdash/


   .. container:: eegdash-badges text-center mb-4

      .. image:: https://pepy.tech/badge/eegdash
         :alt: Downloads
         :target: https://pepy.tech/project/eegdash

      .. image:: https://img.shields.io/github/license/eegdash/eegdash?style=flat-square
         :alt: License
         :target: https://github.com/eegdash/EEGDash/blob/main/LICENSE

      .. image:: https://img.shields.io/github/stars/eegdash/eegdash?style=flat-square
         :alt: GitHub Stats
         :target: https://github.com/eegdash/EEGDash

   .. container:: eegdash-hero-actions
   
      .. button-ref:: install/install
         :color: primary
         :class: sd-btn-lg mr-2
         
         Get Started

      .. button-ref:: dataset_summary
         :color: secondary
         :class: sd-btn-lg
         
         Explore Datasets

.. grid:: 1 2 3 3
    :gutter: 3
    :class-container: pb-5

    .. grid-item-card::  Datasets
        :link: dataset_summary
        :link-type: doc
        :text-align: center
        :class-card: feature-card

        :octicon:`database;1em;sd-text-primary`
        
        Access **500+ standardized M/EEG datasets** from multiple institutions with unified metadata.

    .. grid-item-card:: Preprocessing
        :link: user_guide
        :link-type: doc
        :text-align: center
        :class-card: feature-card

        :octicon:`plug;1em;sd-text-secondary`
        
        Seamless integration with **EEGPrep**, **MNE-Python**, and **BIDS** standards for easy preprocessing.

    .. grid-item-card::  Feature Extraction and Hugging Face Integration
        :link: generated/auto_examples/index
        :link-type: doc
        :text-align: center
        :class-card: feature-card

        :octicon:`gear;1em;sd-text-success`
        
        Reproducible machine learning benchmarks and advanced data analysis pipelines.

.. raw:: html

    <h2 style="text-align: center; margin-top: 3rem; margin-bottom: 2rem;">Institutions</h2>

.. container:: logos-container

    .. container:: logo-item

        .. image:: _static/logos/ucsd_white.svg
            :alt: UCSD
            :class: can-zoom only-dark
            :width: 300px

        .. image:: _static/logos/ucsd_dark.svg
            :alt: UCSD
            :class: can-zoom only-light
            :width: 300px

    .. container:: logo-item

        .. image:: _static/logos/bgu_dark.svg
            :alt: Ben-Gurion University of the Negev (BGU)
            :class: can-zoom only-dark
            :width: 300px

        .. image:: _static/logos/bgu_white.svg
            :alt: Ben-Gurion University of the Negev (BGU)
            :class: can-zoom only-light
            :width: 300px

.. toctree::
   :hidden:

   Installing <install/install>
   User Guide <user_guide>
   API <api/api>
   Dataset Catalog <dataset_summary>
   Examples <generated/auto_examples/index>

