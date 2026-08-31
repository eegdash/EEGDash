EEG/EMG Foundation Challenge 2026
=================================

Runnable EEGDash companions for the four announced 2026 competition tracks.
Each tutorial shows the seed-corpus query, the held-out evaluation axis, and
the track metric on a deterministic synthetic analogue that builds offline.

The synthetic scores are teaching checks, not leaderboard baselines. Use the
official NeuralBench evaluator for competition results:
https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/index.html

What you will learn:

- Track 1: map EEG to image embeddings and measure top-5 retrieval accuracy
  on unseen stimuli.
- Track 2: decode BCI commands across sessions with balanced accuracy.
- Track 3: predict sleep-onset latency across held-out subjects and devices,
  and understand the official weighted, binned error metric.
- Track 4: decode EMG keystrokes across users and measure character error
  rate.

Run the tutorials:

1. ``tutorial_track_1_eeg_to_image.py`` -- cross-stimulus EEG-to-image.
2. ``tutorial_track_2_bci.py`` -- cross-session BCI commands.
3. ``tutorial_track_3_sleep_onset.py`` -- cross-subject onset regression.
4. ``tutorial_track_4_emg_to_text.py`` -- cross-user EMG-to-text.
