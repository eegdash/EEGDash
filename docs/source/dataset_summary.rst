:hide_sidebar: true
:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true

.. _data_summary:

.. raw:: html

   <script>document.documentElement.classList.add('dataset-summary-page');</script>

.. rst-class:: dataset-summary-article

Datasets Catalog
================

To leverage recent and ongoing advancements in large-scale computational methods and to ensure the preservation of scientific data generated from publicly funded research, the EEG-DaSh data archive will create a data-sharing resource for MEEG (EEG, MEG) data contributed by collaborators for machine learning (ML) and deep learning (DL) applications.

.. raw:: html

   <section class="dataset-counter-grid">
     <article class="dataset-counter-card">
       <div class="dataset-counter-icon" aria-hidden="true">
         <svg class="dataset-counter-svg" viewBox="0 0 24 24" role="presentation" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
           <ellipse cx="12" cy="5" rx="7" ry="3"></ellipse>
           <path d="M5 5v10c0 1.66 3.13 3 7 3s7-1.34 7-3V5"></path>
           <path d="M5 12c0 1.66 3.13 3 7 3s7-1.34 7-3"></path>
         </svg>
       </div>
       <div class="dataset-counter-body">
         <span class="dataset-counter-label">Datasets</span>
         <span class="dataset-counter-value">|datasets_total|</span>
       </div>
     </article>
     <article class="dataset-counter-card">
       <div class="dataset-counter-icon" aria-hidden="true">
         <svg class="dataset-counter-svg" viewBox="0 0 24 24" role="presentation" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="8" cy="9" r="3"></circle>
           <circle cx="16" cy="9" r="3"></circle>
           <path d="M4 19c0-3 2.24-5 4-5s4 2 4 5"></path>
           <path d="M12 19c0-3 2.24-5 4-5s4 2 4 5"></path>
         </svg>
       </div>
       <div class="dataset-counter-body">
         <span class="dataset-counter-label">Subjects</span>
         <span class="dataset-counter-value">|subjects_total|</span>
       </div>
     </article>
     <article class="dataset-counter-card">
       <div class="dataset-counter-icon" aria-hidden="true">
         <svg class="dataset-counter-svg" viewBox="0 0 24 24" role="presentation" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
           <path d="M3 15c2.5 0 2.5-6 5-6s2.5 6 5 6 2.5-6 5-6 2.5 6 5 6"></path>
           <path d="M3 9c2.5 0 2.5-5 5-5s2.5 5 5 5 2.5-5 5-5 2.5 5 5 5"></path>
         </svg>
       </div>
       <div class="dataset-counter-body">
         <span class="dataset-counter-label">Experiment Modalities</span>
         <span class="dataset-counter-value">|modalities_total|</span>
       </div>
     </article>
     <article class="dataset-counter-card">
       <div class="dataset-counter-icon" aria-hidden="true">
         <svg class="dataset-counter-svg" viewBox="0 0 24 24" role="presentation" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
           <path d="M12 4c-3.5 0-7 2-7 6 0 3.5 2.5 6 6 6v4l3-2 3 2v-4c2.5 0 4-2 4-5 0-3.5-2.5-7-9-7z"></path>
         </svg>
       </div>
       <div class="dataset-counter-body">
         <span class="dataset-counter-label">Cognitive Domains</span>
         <span class="dataset-counter-value">|cognitive_total|</span>
       </div>
     </article>
   </section>

.. raw:: html

   <script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>

.. tab-set::

   .. tab-item:: Dataset Table

      .. include:: dataset_summary/table.rst

   .. tab-item:: Participant Distribution

      .. include:: dataset_summary/kde.rst

   .. tab-item:: Dataset Flow

      .. include:: dataset_summary/sankey.rst

   .. tab-item:: Dataset Treemap

      .. include:: dataset_summary/treemap.rst

   .. tab-item:: Scatter of Sample Size vs. Recording Duration

      .. include:: dataset_summary/bubble.rst

The archive is currently still in :bdg-danger:`beta testing` mode, so be kind. 
