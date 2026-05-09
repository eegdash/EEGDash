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
guidance in Cisotto and Chicco (2024).

What you will learn:

- How to compose preprocessing as a list of Braindecode preprocessors
  (filtering, resampling, channel selection, scaling) and apply it
  consistently across recordings.
- How to cut continuous signal into fixed-length and event-locked
  windows.
- Why subject-aware splitting is non-negotiable for generalization
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
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Preprocess EEG and create windows</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 30s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Split EEG without subject leakage</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 30s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Train a leakage-safe baseline</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 5s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Save and reload prepared data</div>
</div>
<!-- thumbnail-parent-div-close --></div>
