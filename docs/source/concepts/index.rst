.. _concepts-index:

Concepts
========

These pages explain *why*, not *how*. They describe the ideas, design choices,
and tradeoffs behind EEGDash so that you can reason about your own pipeline,
debug surprising results, and read the rest of the documentation with the
right mental model. For step-by-step recipes, see the
:doc:`Tutorials </generated/auto_examples/index>` and the How-To guides.

The split between "explanation", "tutorials", "how-to guides", and "reference"
follows the `Diataxis framework <https://diataxis.fr>`_. Each part of the
documentation answers a different question:

- **Tutorials** answer *learning* questions. They are guided lessons.
- **How-to guides** answer *task* questions. They are recipes.
- **Reference** answers *information* questions. It is exhaustive lookup.
- **Concepts (this section)** answer *understanding* questions. They explain
  the model, not the keystrokes.

When you find yourself wanting to know "but *why* does it work this way?", a
concept page is the right place to look. When a tutorial says "we use a
subject-aware split here", the underlying argument lives in
:doc:`leakage_and_evaluation`. When a how-to says "set a montage", the
reasoning is in :doc:`preprocessing_decisions`.

Available concept pages
-----------------------

.. toctree::
   :maxdepth: 1

   eegdash_objects
   metadata_and_bids
   leakage_and_evaluation
   preprocessing_decisions
   features_vs_deep_learning

Summary of each page:

- :doc:`eegdash_objects` — The three main objects (``EEGDash``,
  ``EEGDashDataset``, ``EEGChallengeDataset``), what each one returns, and
  when to reach for which.

- :doc:`metadata_and_bids` — How BIDS entities (subject, session, task,
  run) map to EEGDash query keywords, why standardized metadata matters,
  and how participant-level descriptors flow through the dataset.

- :doc:`leakage_and_evaluation` — Why subject-level data leakage destroys
  generalization claims in EEG decoding, why random window splits are
  unsafe, and how within-subject, cross-session, and cross-subject
  evaluation differ.

- :doc:`preprocessing_decisions` — What changes when you pick a high-pass
  cutoff, a montage, or a reference scheme. Defaults are not neutral
  choices: they encode assumptions about the signal and the question.

- :doc:`features_vs_deep_learning` — When handcrafted features
  (band power, CSP, Riemannian) outperform deep nets and vice versa.
  How to pick a baseline that is informative, not just easy.

How to use these pages
----------------------

Read a concept page **before** you start a project so you know which
questions to ask. Re-read it **after** something surprising happens — a
suspiciously high accuracy, a result that disappears across subjects, a
filter that changes class boundaries — so you can localise the assumption
that broke. The pages are deliberately short on syntax. The tutorials
contain runnable code; the how-to guides contain task recipes; this
section contains the reasoning that connects them.

Further reading
---------------

- Diataxis documentation framework. https://diataxis.fr
- Cisotto, G., & Chicco, D. (2024). Ten quick tips for clinical
  electroencephalographic (EEG) data acquisition and signal processing.
  *PeerJ Computer Science*, 10, e2256.
  https://doi.org/10.7717/peerj-cs.2256
- Pernet, C. R., et al. (2019). EEG-BIDS, an extension to the brain
  imaging data structure for electroencephalography. *Scientific Data*,
  6(1), 103. https://doi.org/10.1038/s41597-019-0104-8
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python.
  *Frontiers in Neuroscience*, 7, 267.
  https://doi.org/10.3389/fnins.2013.00267
