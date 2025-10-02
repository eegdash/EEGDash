.. title:: Dataset landscape

.. rubric:: Dataset landscape

The dataset landscape visualisation is now powered by the grouped bubble workflow introduced
in Phase 5. The chart is rendered with Plotly, sizes bubbles by record counts, highlights
modalities with a consistent palette, and supports click-through navigation to individual
dataset pages.

.. raw:: html

   <figure class="eegdash-figure" style="margin: 0 0 1.25rem 0;">

.. raw:: html
   :file: ../_static/dataset_generated/dataset_bubble.html

.. raw:: html

   <figcaption class="eegdash-caption">
     Figure: Interactive grouped bubble landscape. Bubble positions cluster subjects in each dataset,
     marker size encodes record volume on a log scale, colours encode dominant modality, and the legend
     acts as a modality filter. Click a bubble to open the dataset detail page in a new tab.
   </figcaption>
   </figure>
