Core Decoding Workflow
======================

Four tutorials covering the canonical EEG decoding pipeline: preprocess and
window, split without subject leakage, train a baseline, save and reload
prepared data. Sourced from ``docs/tutorial_restructure_plan.md`` Category B.

Run them in order:

1. ``plot_10_preprocess_and_window.py`` — apply preprocessing and create
   fixed-length / event windows.
2. ``plot_11_leakage_safe_split.py`` — split data so no subject appears in
   both train and test.
3. ``plot_12_train_a_baseline.py`` — train a simple baseline and report
   accuracy alongside chance level.
4. ``plot_13_save_and_reuse_prepared_data.py`` — persist windows or features
   and reload them in a later session.
