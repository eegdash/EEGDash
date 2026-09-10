Evaluation on recorded EEG
==========================

Six independently runnable lessons use explicit small EEGDashDataset subsets.
The SSVEP lessons download approximately 21.1 MB for three participants;
the session and MOABB lessons use about 11 MB of real motor imagery.
Set ``EEGDASH_CACHE_DIR`` to reuse the signals. All scores come from held-out
recorded trials and their observed event labels.

1. ``plot_50_within_subject_evaluation.py``: hold out complete trials within
   each known participant.
2. ``plot_51_cross_subject_evaluation.py``: leave one participant out.
3. ``plot_52_cross_session_evaluation.py``: transfer between genuine sessions
   from one participant.
4. ``plot_53_learning_curves.py``: add nested training-subject subsets while
   keeping one validation participant fixed.
5. ``plot_54_compare_two_pipelines.py``: compare paired subject scores and
   explain the limited resolution of a three-participant statistical test.
6. ``plot_55_moabb_interop.py``: pass EEGDash-loaded signals into MOABB's
   dataset interface and obtain real labelled epochs (requires ``moabb``).

Choose the split unit to match deployment, fit learned transformations only
on training data, and preserve trial identities when creating windows.
These small subsets demonstrate evaluation mechanics, not benchmark claims.
