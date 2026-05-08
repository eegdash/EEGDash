Resting-State and State Decoding
================================

The canonical beginner decoding lesson on resting-state EEG: eyes-open
versus eyes-closed classification, decoded from alpha-rhythm
differences with a band-power baseline. Difficulty 1; assumes the
*Start Here* trio.

Resting-state state decoding is the cleanest neurophysiological
benchmark in the field: the alpha increase on eye closure is large,
reproducible, and present at the single-recording level, so it is the
ideal lesson for verifying that your preprocessing pipeline is doing
something sane before you point it at a noisier event-related task.
Sourced from ``docs/tutorial_restructure_plan.md`` Category D (lines
425-435), with preprocessing guidance from Cisotto and Chicco (2024).

What you will learn:

- How to label EEG segments by resting-state condition (eyes open vs
  eyes closed) from BIDS ``events.tsv`` rows.
- How to compute alpha-band (8-12 Hz) power per channel and visualise
  the eyes-open / eyes-closed difference topographically.
- How to train a logistic-regression baseline on band-power features
  and report subject-level cross-validation accuracy.
- How to read a topomap critically -- where the alpha effect should
  appear and what to do when it doesn't.

Run the lesson:

1. ``plot_30_eyes_open_closed.py`` -- alpha-band classification of
   resting-state EEG.
