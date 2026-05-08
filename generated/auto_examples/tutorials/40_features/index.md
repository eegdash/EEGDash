# Feature Engineering

EEGDash’s feature extraction package as a first-class option, not an
afterthought to deep learning. Three lessons cover feature tables from
windows, preprocessor / dependency trees that avoid recomputation, and
a scikit-learn / LightGBM baseline straight from a feature table.
Difficulty 1-2; assumes the core workflow track.

Hand-crafted features remain the strongest baseline on small EEG
datasets. They are interpretable, sample-efficient, and reproducible,
and a feature pipeline is the right tool the moment you have fewer than
a few thousand windows per class. The dependency-tree lesson teaches
why you should compute the spectrum once and derive band powers from
it, rather than recomputing the FFT five times. Sourced from
`docs/tutorial_restructure_plan.md` Category E (lines 442-458).

What you will learn:

- How to extract a small feature table (band power, statistical
  moments, Hjorth parameters) from windows of an `EEGDashDataset`.
- How to compose preprocessors and feature definitions as a tree where
  shared computations (FFT, covariance) execute once.
- How to feed the feature table into `sklearn.pipeline.Pipeline` or
  LightGBM and run a stratified cross-validation.
- How to compare a feature baseline against a deep model on the same
  split (the *Evaluation* track builds on this).

Run the lessons in order:

1. `plot_40_first_features.py` – a feature table from windows.
2. `plot_41_feature_trees.py` – preprocessor and dependency trees.
3. `plot_42_features_to_sklearn.py` – a scikit-learn / LightGBM
   baseline from feature tables.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Plot_30 closed the loop on the parieto-occipital alpha rhythm Hans Berger first reported in 1929 klimesch2012alpha. This tutorial generalises the recipe: starting from windowed EEG (the same Healthy Brain Network ds005514 resting-state idiom we keep across the gallery, reachable through NEMAR, Delorme et al. 2022; Alexander et al. 2017), it extracts a band-power feature per channel for each of theta, alpha, beta, and gamma. The deliverable is a (n_windows, 4, n_channels) tensor of log10 band power that plot_42 hands to scikit-learn pedregosa2011sklearn without any further reshaping.">  <div class="sphx-glr-thumbnail-title">How do I turn EEG windows into a band-power feature matrix?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Welch&#x27;s method welch1967psd returns one power spectrum per window; band power, spectral entropy, peak frequency, and the 1/f slope (Demanuele et al. 2007; Donoghue et al. 2020) are four scalars derived from that same spectrum. If a feature dictionary asks for four band powers as four independent features, the FFT runs four times per window. The FeatureExtractor dependency tree shares one spectral_preprocessor across every spectral feature so the FFT runs once. The deliverable for this tutorial is one feature table with the four derived columns, plus a three-panel figure that names the hierarchy, shows each feature&#x27;s distribution, and prints the 4x4 Pearson correlation between them.">  <div class="sphx-glr-thumbnail-title">How do classical EEG markers compose on top of one Welch PSD?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="A feature table from plot_40 is one row per window with columns shaped &lt;feature&gt;_&lt;channel&gt;. The data follows the same OpenNeuro ds005514 HBN resting-state contour as plot_40, reachable through NEMAR (Delorme et al. 2022); to keep the run offline-reproducible the feature table is synthesised with the column layout plot_40 saves to parquet. Before reaching for a deep net, this tutorial wires the feature matrix into sklearn.pipeline.Pipeline pedregosa2011sklearn with StandardScaler and LogisticRegression, runs a leave-one- subject-out loop with a leakage-safe split from get_splitter, and reports per-fold accuracy against majority_baseline. The deliverable is a three-panel diagnostic that mirrors the one from plot_12 so the two read together: plot_12 trains the same Pipeline on log-band-power features computed inline, plot_42 trains it on the feature table extracted by plot_40. The question the figure answers is whether a transparent linear baseline clears the chance line on a held-out subject?">  <div class="sphx-glr-thumbnail-title">How do I push EEGDash features through a scikit-learn Pipeline?</div>
</div>
<!-- thumbnail-parent-div-close --></div>
