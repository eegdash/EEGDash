EEG/EMG Foundation Challenge 2026
=================================

Runnable EEGDash workflows using recorded task-specific signals and observed
labels. First execution downloads the explicit subsets named in each script.
These small instructional evaluations use their own documented splits and
endpoints; use NeuralBench for the official competition protocol:
https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/index.html

Run the tutorials:

1. ``tutorial_track_1_eeg_to_image.py`` -- retrieve original THINGS images from
   EEG using fixed image-pixel features and held-out image identities.
2. ``tutorial_track_2_bci.py`` -- cross-session motor-imagery classification.
3. ``tutorial_track_3_sleep_onset.py`` -- predict time from recording start to
   sustained N2 sleep, evaluated on held-out Sleep-EDF participants. Recording
   start is not lights-out; this is not the wearable competition endpoint.
4. ``tutorial_track_4_emg_to_text.py`` -- decode observed lowercase keystrokes
   across two real users. This aligned baseline receives event times and does
   not replace full sequence transduction or the official CER evaluator.
