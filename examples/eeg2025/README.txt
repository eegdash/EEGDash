EEG2025 Foundation Challenge
============================

Two runnable baselines using observed targets and recorded signals from the
100 Hz EEG2025 R5 mini challenge release. Each script loads an explicit small
participant subset and fits its baseline during execution. The tutorial splits
are instructional and do not reproduce the official challenge leaderboard.

1. ``tutorial_challenge_1.py`` -- predict actual stimulus-to-response reaction
   time from two seconds of prestimulus EEG, holding out one participant.
2. ``tutorial_challenge_2.py`` -- predict observed p-factor from resting EEG
   features with one feature row and one held-out prediction per participant.

For encoder pretraining and adaptation across real resting-state and contrast-
change tasks, see ``plot_71_cross_task_transfer.py`` in the transfer tutorials.
