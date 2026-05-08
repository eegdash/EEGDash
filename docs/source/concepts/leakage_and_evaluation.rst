.. _concepts-leakage-and-evaluation:

Leakage and evaluation
======================

Subject-level data leakage is the single most common reason that an EEG
decoder claims a high cross-validated accuracy and then collapses on a new
participant. It is also the reason a tutorial that scores 0.95 on a random
window split can score 0.55 on a subject-disjoint split using the *same*
data and the *same* model. This page explains the failure mode, why it is
specific to physiological signals, and how the within-subject /
cross-session / cross-subject distinction maps onto the splitters EEGDash
provides.

The advice below is consistent with Tip 9 of Cisotto & Chicco (2024) [1]_,
which explicitly identifies subject-aware cross-validation as the only
defensible default for clinical EEG. Tutorials that demonstrate the
problem live in
:doc:`/generated/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`,
:doc:`/generated/auto_examples/tutorials/50_evaluation/plot_50_within_subject_evaluation`,
:doc:`/generated/auto_examples/tutorials/50_evaluation/plot_51_cross_subject_evaluation`,
and
:doc:`/generated/auto_examples/tutorials/50_evaluation/plot_52_cross_session_evaluation`.

Why subject leakage destroys generalization claims
--------------------------------------------------

EEG signals carry strong, idiosyncratic, *subject-specific* statistics:
skull thickness, hair impedance, electrode placement, baseline alpha
power, blink habits, posture. A neural network with a few thousand free
parameters easily latches onto those identity features because they
generalise perfectly within a subject — every window from subject A
"smells like" subject A.

Now imagine a binary decoder for "eyes open vs. eyes closed". You shuffle
all windows across all subjects and split 80/20 randomly. The classifier
quickly discovers that a few windows from each subject are in the test
set, and its best strategy is to memorise the spectral fingerprint of
subject A and reuse it for the held-out windows from subject A. It then
reports an apparent accuracy of, say, 0.94. Unfortunately, this number
is a lower-bounded *identification* accuracy plus the actual condition
classification — and on a fresh participant the model may do no better
than chance.

The cleanest demonstration of this is to train the same architecture on
two splits of the same dataset: a leaky random window split and a
subject-disjoint split. The accuracy gap is the leakage tax, and it is
typically 0.20–0.40 in absolute accuracy on real EEG decoding problems.

Why random window splits are unsafe
-----------------------------------

A *window* is a short, overlapping slice of a recording. If you make
2-second windows with 50% overlap, neighbouring windows share a full
second of samples; their feature vectors differ only by smoothing. When
you assign one to "train" and the other to "test", the test score is
almost a noise estimate, not a generalisation estimate.

This problem exists on top of the subject leakage problem: even within a
single subject, randomising windows leaks information across the train/
test boundary because the windows overlap in time. The mitigation is
twofold:

1. Split at the **recording or session level**, not the window level.
2. If a single recording must be split, choose a splitter that respects
   temporal contiguity (e.g., the first 80% by time for train, the last
   20% for test).

EEGDash defers the actual splitting to braindecode and MOABB, but the
conceptual rule is the same regardless of library: a window must inherit
the train/test label of its parent recording, never get assigned
independently.

Within-subject vs. cross-session vs. cross-subject
--------------------------------------------------

These three terms describe what kind of generalisation you are claiming
to measure. They differ in which axis the held-out fold spans:

- **Within-subject** evaluation holds out *time* within a single
  participant. Train on the first portion of recording, test on the
  last portion. Answer: *can the model decode this person's signal
  later in the same session?* This is the easiest setting and the one
  most clinical BCI demos report.

- **Cross-session** evaluation holds out a *different session* of the
  same participant. Train on session 1, test on session 2 (typically
  collected on a different day, with re-applied electrodes). Answer:
  *does the model survive electrode re-application and day-to-day
  drift?* This is the relevant setting for repeated-use BCIs and for
  any real-world deployment where calibration is rare.

- **Cross-subject** evaluation holds out *different participants*.
  Train on subjects A–T, test on subjects U–Z. Answer: *does the model
  generalise to a person it has never seen?* This is the standard for
  any "subject-invariant" or "foundation-model" claim.

Each setting answers a different scientific question, so neither one is
universally correct. The mistake is to *report* one and *implicitly
claim* another. A paper that splits randomly and then advertises a
"general-purpose decoder" is overstating the evaluation; a paper that
holds out a session and accurately calls it cross-session is doing
honest work even if the number is lower.

Practical guidance
------------------

1. Always inspect ``ds.description["subject"]`` and
   ``ds.description["session"]`` before choosing a splitter. If any
   subject appears in more than one fold, the split is leaky by
   construction.

2. Treat the split function as part of the experiment, not a utility.
   Print, log, and version-control the participants in each fold.
   Tutorials such as
   :doc:`/generated/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`
   include an audit step that verifies disjointness.

3. Pick a splitter that matches your scientific question — within-subject,
   cross-session, or cross-subject. If you cannot decide, default to
   cross-subject; it is the strictest of the three and rarely
   misleading.

4. Always include a chance-level baseline and, where possible, a
   simple feature baseline (see :doc:`features_vs_deep_learning`).
   A neural network that beats random by 3 points but loses to a
   logistic regression on band power has not learned the task.

5. Report variance across folds, not just the mean. Subject-level
   variance dominates EEG; a mean accuracy with no error bars is not
   a measurement.

What "metric leakage" looks like in practice
--------------------------------------------

A few diagnostic patterns you should watch for:

- **Suspiciously high accuracy on hard problems.** A decoder for
  emotional state from 30 seconds of resting EEG that scores 0.92 is
  almost certainly leaking subject identity.
- **Accuracy drops on new subjects.** A 25-point drop between
  cross-validated and held-out cohorts is a leakage signal, not an
  overfitting signal.
- **Random labels still score above chance.** If you shuffle the labels
  per recording but keep them constant within a recording, a leaky
  pipeline still scores well above chance because it is fitting
  recording identity.

When you see one of those, re-read this page and re-check the splitter.

Related tutorials
-----------------

- :doc:`/generated/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`
  is the canonical demonstration of leaky vs. safe splits on EEGDash data.
- :doc:`/generated/auto_examples/tutorials/50_evaluation/plot_50_within_subject_evaluation`,
  :doc:`/generated/auto_examples/tutorials/50_evaluation/plot_51_cross_subject_evaluation`,
  and
  :doc:`/generated/auto_examples/tutorials/50_evaluation/plot_52_cross_session_evaluation`
  show the same dataset evaluated under each protocol.
- :doc:`/generated/auto_examples/tutorials/50_evaluation/plot_53_learning_curves`
  illustrates how leakage interacts with sample-size effects.
- :doc:`/generated/auto_examples/tutorials/50_evaluation/plot_54_compare_two_pipelines`
  shows how to compare pipelines once a defensible split is in place.

Further reading
---------------

.. [1] Cisotto, G., & Chicco, D. (2024). Ten quick tips for clinical
   electroencephalographic (EEG) data acquisition and signal processing.
   *PeerJ Computer Science*, 10, e2256.
   https://doi.org/10.7717/peerj-cs.2256

- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P.
  (2017). The need to approximate the use-case in clinical machine
  learning. *GigaScience*, 6(5), 1–9.
  https://doi.org/10.1093/gigascience/gix019
- Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., Falk, T. H., &
  Faubert, J. (2019). Deep learning-based electroencephalography analysis:
  a systematic review. *Journal of Neural Engineering*, 16(5), 051001.
  https://doi.org/10.1088/1741-2552/ab260c
- Pernet, C. R., et al. (2019). EEG-BIDS, an extension to the brain
  imaging data structure for electroencephalography. *Scientific Data*,
  6(1), 103. https://doi.org/10.1038/s41597-019-0104-8
