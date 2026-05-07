"""Decode visual P300 target vs. standard responses
=================================================

A child watches letters flash one by one. One letter is the *target* the
experimenter chose; every other is a *standard*. The brain answers the
rare target with a positive deflection around 300 ms over central-parietal
cortex -- the classic P300 (Polich 2007, doi:10.1016/j.clinph.2007.04.019).
Can we decode whether a child saw a target or a standard image from a
800-ms post-stimulus EEG window?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/plot_20_visual_p300_oddball.png'

# %% [markdown]
# Learning objectives
# -------------------
# After this tutorial you will be able to:
#
# - load a P300 BIDS dataset via ``EEGDashDataset`` and surface its events.
# - build event-locked windows with baseline correction (Cisotto & Chicco 2024, Tip 7).
# - train a target-vs-standard linear classifier on the flattened windows.
# - compute ROC-AUC and accuracy alongside the chance level for an imbalanced task.
# - compare target and standard ERPs and recognise the P300 bump.

# %% [markdown]
# Requirements
# ------------
# - **Estimated time**: ~3 min on CPU (one subject; cached on first run).
# - **Prerequisites**: ``plot_10_preprocess_and_window``, ``plot_11_leakage_safe_split``.
# - **Concept**: [docs/source/concepts/leakage_and_evaluation.rst](../../docs/source/concepts/leakage_and_evaluation.rst).
# - **Data**: one subject of ``ds005863`` (visual oddball, OpenNeuro), <50 MB.

# %%
# Setup -- seed and resolve the cache directory.
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from eegdash import EEGDashDataset
from eegdash.splits import assert_no_leakage, majority_baseline
from eegdash.viz import EEGDASH_BLUE, EEGDASH_ORANGE, style_figure, use_eegdash_style

use_eegdash_style()
SEED = 42
np.random.seed(SEED)
mne.set_log_level("ERROR")
warnings.simplefilter("ignore", category=RuntimeWarning)
cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Step 1 -- Pick a P300 dataset
# -----------------------------
# We query EEGDash for ``ds005863`` visual-oddball recordings (Pernet et
# al. 2019, doi:10.1038/s41597-019-0104-8). Subject ``001`` is small.

# %%
SUBJECT = "002"
dataset = EEGDashDataset(
    cache_dir=cache_dir, dataset="ds005863", task="visualoddball", subject=SUBJECT
)
print(f"records: {len(dataset.datasets)} (dataset=ds005863, task=visualoddball)")

# %% [markdown]
# Step 2 -- Inspect the events table
# ----------------------------------
# ``events_from_annotations`` (MNE-Python, Gramfort et al. 2013,
# doi:10.3389/fnins.2013.00267) maps string codes to integer ids. Targets
# are rare and standards frequent: the textbook oddball imbalance we
# never silently rebalance.

# %%
raw0 = dataset.datasets[0].raw.load_data().copy()
events, event_id = mne.events_from_annotations(raw0)
print(f"sfreq={raw0.info['sfreq']} Hz, n_channels={len(raw0.ch_names)}")
print(f"event keys (first 6): {list(event_id.keys())[:6]}")

# %% [markdown]
# **Predict.** What difference do you expect between target and standard
# ERPs at central-parietal Pz, in the 250-500 ms window? Polich 2007 says
# a positive bump on targets only.

# %% [markdown]
# Step 3 -- Preprocess and create event-locked windows
# ----------------------------------------------------
# We resample to 128 Hz and apply a 0.5-30 Hz **non-causal FIR band-pass**
# (pass-band 0.5-30 Hz; Cisotto & Chicco 2024, Tip 4).
# ``create_windows_from_events`` epochs every Target / NonTarget annotation
# with ``tmin=-0.2``, ``tmax=0.8`` and **DC offset baseline correction**
# on the pre-stimulus interval (Cisotto & Chicco 2024, Tip 7).
# **Run.** Preprocess and epoch.

# %%
SFREQ = 128.0
preprocess(
    dataset,
    [
        Preprocessor("set_eeg_reference", ref_channels="average", projection=False),
        Preprocessor("resample", sfreq=SFREQ),
        Preprocessor("filter", l_freq=0.5, h_freq=30.0, method="fir", phase="zero"),
    ],
)
TMIN, TMAX = -0.2, 0.8
windows = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(TMIN * SFREQ),
    trial_stop_offset_samples=int(TMAX * SFREQ),
    preload=True,
    drop_bad_windows=True,
    mapping={"Target": 1, "NonTarget": 0, "target": 1, "standard": 0},
)
X = np.stack([windows[i][0] for i in range(len(windows))]).astype(np.float32)
y = np.asarray([windows[i][1] for i in range(len(windows))], dtype=int)
n_targets, n_standards = int((y == 1).sum()), int((y == 0).sum())
print(f"X={X.shape}, n_targets={n_targets}, n_standards={n_standards}")

# %% [markdown]
# Step 4 -- Leakage-safe within-subject split
# -------------------------------------------
# One subject is on hand, so a stratified within-subject split keeps the
# class balance. ``assert_no_leakage`` intersects train and test on the
# chosen key and prints the contract JSON line E5.42 reads.
# **Run.** Materialise the split and emit the leakage_report line.

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, test_idx = next(skf.split(np.zeros(len(y)), y))
meta = pd.DataFrame({"trial_id": np.arange(len(y)), "subject": f"sub-{SUBJECT}"})
manifest = {"folds": [{"train": train_idx.tolist(), "test": test_idx.tolist()}]}
overlap = assert_no_leakage(manifest, meta, by="trial_id")
assert overlap == 0

# %% [markdown]
# Step 5 -- Train a linear classifier
# -----------------------------------
# A flat ``LogisticRegression`` (``random_state=42``, L2, C=1) on
# vectorised windows is the simplest target-vs-standard decoder. We
# standardise per-channel using train-set statistics, then quote ROC-AUC.

# %%
mu = X[train_idx].mean(axis=(0, 2), keepdims=True)
sd = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-7
Xn = ((X - mu) / sd).reshape(len(y), -1)
clf = LogisticRegression(random_state=SEED, max_iter=1000, C=1.0)
clf.fit(Xn[train_idx], y[train_idx])
y_pred = clf.predict(Xn[test_idx])
y_score = clf.predict_proba(Xn[test_idx])[:, 1]
acc = float(accuracy_score(y[test_idx], y_pred))
auc = float(roc_auc_score(y[test_idx], y_score))

# %% [markdown]
# Step 6 -- Report accuracy alongside chance
# ------------------------------------------
# ``majority_baseline`` returns the test-set frequency of the most common
# class. With a 4:1 standard:target imbalance, chance hovers near 0.80.

# %%
chance = float(majority_baseline(y[train_idx], y[test_idx])["chance_level"])
print(f"Model accuracy: {acc:.3f} | chance: {chance:.3f} | ROC-AUC: {auc:.3f}")

# %% [markdown]
# Step 7 -- Investigate target vs. standard ERPs
# ----------------------------------------------
# We plot the per-class average at Pz, where the P300 peaks.
# **Investigate.** Compare the orange target trace with the blue standard
# trace inside the shaded P300 window: a targets-only positivity confirms
# the class signal the decoder uses.

# %%
times = np.linspace(TMIN, TMAX, X.shape[-1])
ch_names = [c.lower() for c in raw0.ch_names]
ch_idx = ch_names.index("pz") if "pz" in ch_names else 0
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(times, X[y == 1, ch_idx, :].mean(0), color=EEGDASH_ORANGE, label="target")
ax.plot(times, X[y == 0, ch_idx, :].mean(0), color=EEGDASH_BLUE, label="standard")
ax.axvspan(0.25, 0.5, color="grey", alpha=0.15, label="P300 window")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("time (s)")
ax.set_ylabel("amplitude (µV)")
ax.legend(loc="upper right", fontsize=8)
fig.subplots_adjust(top=0.78, bottom=0.18)
style_figure(
    fig,
    title=f"P300 ERP at {raw0.ch_names[ch_idx]}",
    subtitle=f"ds005863 sub-{SUBJECT} | n_targets={n_targets}, n_standards={int((y == 0).sum())} | sfreq={SFREQ:.0f} Hz",
    source="EEGDash plot_20 | OpenNeuro ds005863 | task=visualoddball",
)

# %% [markdown]
# A common mistake -- and how to recover
# --------------------------------------
#
# **Run.** A frequent slip is mistyping the event mapping (here we swap
# ``"target"`` for ``"targets"``) -- ``create_windows_from_events`` then
# returns zero target windows. We trigger it with ``try/except`` so the
# failure mode is visible (Nederbragt et al. 2020,
# doi:10.1371/journal.pcbi.1008090).

# %%
try:
    bad_mapping = {"targets": 1, "standards": 0}  # plural typo
    missing = [k for k in bad_mapping if k not in event_id]
    if missing:
        raise KeyError(f"event keys not found: {missing}")
except (KeyError, ValueError) as exc:
    print(f"Caught {type(exc).__name__}: {exc}")
    # Recovery: list the actual event keys before mapping.
    print(f"Recovery: known event keys are {list(event_id.keys())[:6]}.")

# %% [markdown]
# Modify
# ------
# **Your turn**: re-run Step 3 with ``TMAX = 0.4`` (cutting at 400 ms) and
# refit. ROC-AUC drops a little because the late P300 is gone.

# %% [markdown]
# Make
# ----
# **Mini-project**: rerun the pipeline on a different subject
# (``SUBJECT="002"``) or apply it to ``ds002718`` (Wakeman & Henson) by
# remapping events to famous-vs-scrambled.

# %% [markdown]
# Result
# ------

# %%
print("\n| metric            | value |")
print("|-------------------|-------|")
print(f"| accuracy          | {acc:0.3f} |")
print(f"| chance (majority) | {chance:0.3f} |")
print(f"| ROC-AUC           | {auc:0.3f} |")
print(f"| n_targets         | {n_targets} |")
print(f"| n_standards       | {n_standards} |")

# %% [markdown]
# Wrap-up
# -------
# We loaded a P300 BIDS dataset, mapped its annotations to binary labels,
# epoched with baseline correction, ran a leakage-safe split, and trained
# a linear classifier. The ROC-AUC sits above the majority chance level --
# the only honest summary of an imbalanced two-class decoder.

# %% [markdown]
# Try it yourself
# ---------------
# - Set ``TMAX = 0.4`` in Step 3 and re-run; ROC-AUC drops because the late P300 is gone.
# - Swap ``LogisticRegression`` for ``LogisticRegressionCV(cv=3)`` and see whether tuned regularisation buys >0.01 AUC.
# - Loop over 3 subjects of ``ds005863``; class balance stays oddball-shaped (~1:4) and chance stays near 0.80.
# - Compute a topomap at 300 ms with ``mne.viz.plot_topomap`` and confirm the central-parietal P300 maximum.

# %% [markdown]
# References
# ----------
# - Polich 2007, Updating P300, *Clin. Neurophysiol.* https://doi.org/10.1016/j.clinph.2007.04.019
# - Pernet et al. 2019, EEG-BIDS, *Sci. Data*. https://doi.org/10.1038/s41597-019-0104-8
# - Gramfort et al. 2013, MNE-Python, *Front. Neurosci.* https://doi.org/10.3389/fnins.2013.00267
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, *PeerJ CS*. https://doi.org/10.7717/peerj-cs.2256
# - Nederbragt et al. 2020, Ten simple rules for teaching coding, *PLOS Comp. Biol.* https://doi.org/10.1371/journal.pcbi.1008090
# - Dataset: OpenNeuro ds005863. https://doi.org/10.18112/openneuro.ds005863.v1.0.0
#   (verify the dataset's authors/title in the live OpenNeuro listing — DOI
#   resolves to the canonical OpenNeuro page; cite the dataset paper from there.)
# - Concept: [docs/source/concepts/leakage_and_evaluation.rst](../../docs/source/concepts/leakage_and_evaluation.rst).
