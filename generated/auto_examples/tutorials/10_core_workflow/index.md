# Core Decoding Workflow

The canonical EEG decoding pipeline in four lessons: preprocess and
window, split without subject leakage, train a baseline against chance,
then persist prepared data so you do not pay the windowing cost on
every rerun. Difficulty 1-2; assumes the *Start Here* trio.

This category encodes the mistakes EEG decoding papers most often make
– random window splits that leak subjects across train and test,
baselines that beat chance only because of a confound, and re-windowing
every session because nothing was cached. The leakage-safe split lesson
is the rubric anchor for E3.27 invariants and tracks the evaluation
guidance in Cisotto and Chicco (2024). Sourced from
`docs/tutorial_restructure_plan.md` Category B (lines 380-410).

What you will learn:

- How to compose preprocessing as a list of Braindecode preprocessors
  (filtering, resampling, channel selection, scaling) and apply it
  consistently across recordings.
- How to cut continuous signal into fixed-length and event-locked
  windows.
- Why subject-aware splitting is non-negotiable for generalisation
  claims, and how to implement one with EEGDash’s split helpers.
- How to train a small baseline model against an explicit chance level
  and report a confidence interval.
- How to persist windows or features to disk and reload them in a later
  session without redoing the pipeline.

Run the lessons in order:

1. `plot_10_preprocess_and_window.py` – preprocessing pipeline and
   window construction.
2. `plot_11_leakage_safe_split.py` – subject-aware train / val /
   test split.
3. `plot_12_train_a_baseline.py` – a small model versus the chance
   level.
4. `plot_13_save_and_reuse_prepared_data.py` – save once, reuse many.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Raw EEG is rarely model-ready: wrong sampling rate, drift and line noise, no fixed reference, sporadic large-amplitude bursts, a continuous timeline instead of fixed epochs. This tutorial walks the canonical EEGDash preprocessing recipe on one recording from OpenNeuro ds002718 (Wakeman &amp; Henson 2015), reachable through NEMAR (Delorme et al. 2022). Every choice is named (Cisotto &amp; Chicco 2024 Tips 4-5), inspected on the array, and the recipe ends with a windowed dataset the next four core-workflow tutorials reuse. The closing diagnostic figure compares the recording before and after a one-call braindecode.preprocessing.EEGPrep pass that wraps ASR (Mullen et al. 2015), bad-channel detection, high-pass, and CAR.">  <div class="sphx-glr-thumbnail-title">How do I preprocess EEG and create model-ready windows?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Random window splits on cross-subject EEG decoders post training-set accuracy near 99% and collapse on a held-out participant. The reason is not exotic: every recording produces hundreds of overlapping windows from the same brain, so a uniform shuffle scatters each subject across both train and test, and the model memorises subject-level fingerprints (heart-rate, alpha amplitude, electrode impedance) instead of the task we actually want to decode.">  <div class="sphx-glr-thumbnail-title">How do I split EEG data without subject leakage?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="A model that scores 0.78 on held-out windows is only useful when you also know what 0.50 (chance) and 0.55 (a transparent linear baseline) look like on the same split. This tutorial trains that linear baseline on three subjects of OpenNeuro ds002718 (Wakeman &amp; Henson 2015), reachable through NEMAR (Delorme et al. 2022). Four bands of log power per channel feed sklearn.linear_model.LogisticRegression (Pedregosa et al. 2011); a 3-fold cross-subject loop with sklearn.model_selection.GroupKFold keeps every subject in exactly one fold. The deliverable is a single three-panel figure that answers three questions on one screen: do the features separate the classes, how does the accuracy vary across held-out subjects, and which trials does the model confuse?">  <div class="sphx-glr-thumbnail-title">How do I train a leakage-safe baseline classifier on EEG?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Preprocessing EEG is expensive. Filtering, resampling, and windowing one subject can take seconds; doing it for every kernel restart wastes hours over the lifetime of a project. The work has already been done once. The question is how to write it to disk so the next session, the next collaborator, and the future you all skip the recompute.">  <div class="sphx-glr-thumbnail-title">How do I save and reload prepared windows + features?</div>
</div>
<!-- thumbnail-parent-div-close --></div>
