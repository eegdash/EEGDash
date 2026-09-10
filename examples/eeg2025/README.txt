EEG2025 Foundation Challenge
========================================

Two runnable baselines using observed targets and recorded signals from the
100 Hz EEG2025 R5 mini challenge release. Each script loads an explicit small
participant subset and fits its baseline during execution. The tutorial splits
are instructional and do not reproduce the official challenge leaderboard.

1. ``tutorial_challenge_1.py`` -- predict actual stimulus-to-response reaction
   time from two seconds of prestimulus EEG, holding out one participant.
2. ``tutorial_challenge_2.py`` -- predict observed externalizing scores from resting EEG
   features with one feature row and one held-out prediction per participant.

For encoder pretraining and adaptation across real resting-state and contrast-
change tasks, see ``plot_71_cross_task_transfer.py`` in the transfer tutorials.

These are the final tasks described on the `2025 challenge website
<https://eeg2025.github.io/>`_ and its `starter kit
<https://github.com/eeg2025/startkit>`_. Challenge 2 removed p-factor,
internalizing and attention during the competition. Both examples report MAE
and the starter kit's dimensionless NRMSE (RMSE divided by the evaluated
targets' population standard deviation), with a training-mean reference.
