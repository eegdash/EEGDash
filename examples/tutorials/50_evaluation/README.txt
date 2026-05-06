Evaluation and Benchmarking
===========================

Five tutorials that teach EEG decoding evaluation as a core skill, taking
inspiration from MOABB (Chevallier, Aristimunha et al. 2024). Sourced from
``docs/tutorial_restructure_plan.md`` Category F (lines 425-442).

Run them in order, building from a single split (plot_11) toward
benchmark-grade pipeline comparison:

1. ``plot_50_within_subject_evaluation.py`` — when single-subject
   diagnostics are appropriate.
2. ``plot_51_cross_subject_evaluation.py`` — the gold standard for
   generalization claims.
3. ``plot_52_cross_session_evaluation.py`` — calibration drift across
   sessions of the same subject.
4. ``plot_53_learning_curves.py`` — performance as a function of training
   subjects, trials, or windows.
5. ``plot_54_compare_two_pipelines.py`` — paired comparison of a feature
   baseline against a neural model on the same split, with paired Wilcoxon.
