.. meta::
   :hide_sidebar: true

:html_theme.sidebar_secondary.remove:
:html_theme.sidebar_primary.remove:

.. _data_summary:


EEG Dash Datasets
==================


Columns definitions:
   - **dataset**: Name of the dataset.
   - **n_records**: Number of EEG records in the dataset.
   - **n_subjects**: Number of subjects in the dataset.
   - **n_tasks**: Number of experimental tasks in the dataset.
   - **nchans_set**: Set of EEG channel counts used in the dataset.
   - **sampling_freqs**: Set of sampling frequencies used in the dataset.
   - **duration_hours_total**: Total duration of all recordings in hours.


Datasets
======================

.. csv-table::
   :file: ../build/dataset_summary.csv
   :header-rows: 1
   :class: sortable



.. raw:: html
   <style>
     /* Make this page full-width and remove side padding */
     :root {
       --pst-page-max-width: 100%;
       --pst-content-max-width: 100%;
     }
     .bd-main .bd-content .bd-article-container {
       max-width: 100%;
       padding-left: 0;
       padding-right: 0;
     }
     /* Ensure the DataTable uses the full width */
     table.sortable { width: 100% !important; }
   </style>

   <link href="https://cdn.datatables.net/v/bm/jq-3.7.0/dt-2.3.2/af-2.7.0/b-3.2.4/b-html5-3.2.4/cr-2.1.1/fh-4.0.3/r-3.0.5/datatables.min.css"
         rel="stylesheet"
         integrity="sha384-aemAM3yl2c0KAZZkR1b1AwMH2u3r1NHOppsl5i6Ny1L5pfqn7SDH52qdaa1TbyN9"
         crossorigin="anonymous">

   <script src="https://cdn.datatables.net/v/bm/jq-3.7.0/dt-2.3.2/af-2.7.0/b-3.2.4/b-html5-3.2.4/cr-2.1.1/fh-4.0.3/r-3.0.5/datatables.min.js"
           integrity="sha384-CKcCNsP1rMRsJFtrN6zMWK+KIK/FjYiV/d8uOp0LZtbEVzbidk105YcuVncAhBR8"
           crossorigin="anonymous"></script>

   <script>
     document.addEventListener('DOMContentLoaded', function () {
       const tables = document.querySelectorAll('table.sortable');
       tables.forEach(function (tbl) {
         // Use the jQuery plugin that ships in the bundle
         $(tbl).DataTable({
           paging: false,
           searching: false,
           info: false,
           ordering: true,
           responsive: true,
           fixedHeader: true,
           // Avoid re-initialization if this script runs more than once
           retrieve: true,
           scrollX: true
         });
       });
     });
   </script>

