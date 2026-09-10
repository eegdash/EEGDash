EEG/EMG Foundation Challenge 2026
=============================================

Runnable EEGDash workflows using recorded task-specific signals and observed
labels. Each page explains acquisition, preprocessing, targets, splits, fitting
and evaluation. First execution downloads the explicit subset named there;
cropping a recording does not reduce its download size.

Read both the `competition tracks <https://neural-interfaces26.github.io/tracks.html>`_
and the `NeuralBench starter kit <https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/index.html>`_.
The competition site defines the tasks; NeuralBench documents the executable
public baselines. Checked on September 10, 2026: exclusive competition releases
are scheduled for September 21. Public warm-up scores do not reproduce the
hidden competition leaderboard.

Run the tutorials:

1. ``tutorial_track_1_eeg_to_image.py`` -- retrieve held-out THINGS images from
   recorded EEG, keeping image identities separate during model selection.
2. ``tutorial_track_2_bci.py`` -- classify real motor-imagery trials across
   sessions. This small binary warm-up introduces the split boundary; the
   competition has three commands in Graz / BrainHero data.
3. ``tutorial_track_3_sleep_onset.py`` -- predict capped seconds remaining to
   first N2 from five-second Sleep-EDF windows, with held-out participants and
   binned MAE. The competition uses Muse recordings from unseen sleepers;
   transferring clinical PSG to a wearable also changes hardware and channels.
4. ``tutorial_track_4_emg_to_pose.py`` -- regress 20 observed joint-angle
   trajectories from 16-channel EMG2Pose wrist signals. Preserve published
   splits, exclude invalid inverse-kinematics intervals and report angular MAE.

Track 4 is hand-pose regression. The former typing tutorial used a different
dataset, target and metric and has been replaced. Follow each NeuralBench track
guide for the full model, cohort, evaluation and submission workflow.
