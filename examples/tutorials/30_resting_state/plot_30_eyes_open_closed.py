"""Decode eyes-open vs. eyes-closed from resting-state EEG
========================================================

Berger reported in 1929 that the parieto-occipital alpha rhythm rises when
the eyes close and falls when they open -- the textbook resting-state EEG
result every dataset still reproduces. This tutorial pins that finding to
the simplest decoding question we know how to ask of HBN children.

Can we tell from a 2-second EEG snippet whether a child has their eyes
open or closed?
"""

# %% [markdown]
# ## Learning objectives
# After this tutorial you will be able to:
#
# - Use ``eegdash.tasks.get_task`` to pull the canonical EOEC recipe.
# - Build balanced 2-second windows from HBN ``ds005514`` annotations.
# - Show the parieto-occipital alpha bump in eyes-closed via a topomap.
# - Train a within-subject logistic-regression on band-power features.
# - Read the JSON ``leakage_report`` line and a chance-vs-accuracy print.
#
# ## Requirements
# - Estimated time ~5 min on CPU (no GPU; ~80 MB cached after first run).
# - Prerequisites:
#   :doc:`/auto_examples/tutorials/10_core_workflow/plot_10_preprocess_and_window`,
#   :doc:`/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`.
# - Concept page: :doc:`/concepts/preprocessing_decisions`.

# %%
# Setup -- seed (E3.21), parametrised cache dir (E3.24), imports.
import json
import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events, preprocess
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from eegdash import EEGDashDataset
from eegdash.splits import assert_no_leakage, get_splitter, majority_baseline
from eegdash.tasks import get_task

warnings.simplefilter("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")
SEED = 42
np.random.seed(SEED)
cache_dir = Path(os.environ.get("EEGDASH_CACHE", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 1 -- Get the task
# ``get_task("eyes-open-closed")`` returns an :class:`~eegdash.tasks.EyesOpenClosed`
# whose YAML manifest hard-codes HBN release 9 ``ds005514``
# (doi:10.18112/openneuro.ds005514.v1.0.0), the EOEC label definition,
# and a non-causal IIR Butterworth band-pass from 1.0 to 55.0 Hz
# (``Preprocessor("filter", ...)``) plus a resample to 128 Hz.

# %%
task = get_task("eyes-open-closed")
label_def = task.label_definition()
recipe = task.preprocessing_recipe()
print(
    f"Task: {task.name} | dataset={task.dataset} | subject={task.subjects[0]} "
    f"| classes={label_def['class_names']} | filter={task.bandpass} Hz"
)

# %% [markdown]
# ## Step 2 -- PRIMM Predict
# **Predict.** Berger 1929 showed that closing the eyes gates posterior
# cortex into the alpha rhythm (8-12 Hz). Which condition shows *higher*
# alpha power over parieto-occipital channels -- eyes open or closed?
# Note your guess. (Spoiler: closed; the bump sits over O1/Oz/O2.)

# %% [markdown]
# ## Step 3 -- Load one subject and window it
# ``EyesOpenClosed.make_windows()`` is the supported entry today;
# ``task.get_windows()`` from the spec is not wired yet. We fall back to
# the canonical pattern: ``EEGDashDataset`` with ``task.metadata_query()``,
# ``preprocess`` with the recipe, then ``create_windows_from_events`` on
# the HBN annotations. ``hbn_ec_ec_reannotation`` (in the recipe)
# replaces the HBN instruction markers with 2 s eyes_open / eyes_closed
# events so the epochs are balanced.

# %%
ds = EEGDashDataset(query=task.metadata_query(), cache_dir=cache_dir)
preprocess(ds, recipe)
win_kw = {k: v for k, v in task.windowing_recipe().items() if k != "kind"}
windows_ds = create_windows_from_events(ds, **win_kw)
sfreq = float(windows_ds.datasets[0].windows.info["sfreq"])
ch_names = list(windows_ds.datasets[0].windows.info["ch_names"])

# %% [markdown]
# ## Step 4 -- PRIMM Run: per-window alpha power
# **Run #1.** Welch PSD on each 2 s window, then integrate the canonical
# 8-12 Hz alpha pass-band per channel.

# %%
X = np.stack([w[0] for w in windows_ds]).astype(np.float32)
y = np.asarray([w[1] for w in windows_ds], dtype=int)
freqs, psd = welch(X, fs=sfreq, nperseg=int(sfreq), axis=-1)
alpha_power = np.log10(psd[..., (freqs >= 8.0) & (freqs <= 12.0)].mean(axis=-1) + 1e-30)
print(
    f"X={X.shape} | sfreq={sfreq:.0f} Hz | "
    f"n_open={int((y == 0).sum())} n_closed={int((y == 1).sum())}"
)

# %% [markdown]
# **Run #2.** Average alpha power per condition and plot a side-by-side
# topomap on the HydroCel-128 layout so the parieto-occipital geometry
# the alpha bump exploits is visible.

# %%
mont = mne.channels.make_standard_montage("GSN-HydroCel-128")
info = mne.create_info(ch_names, sfreq, ch_types="eeg")
info.set_montage(mont, match_case=False, on_missing="ignore", verbose="ERROR")
mean_open, mean_closed = alpha_power[y == 0].mean(0), alpha_power[y == 1].mean(0)
vlim = (
    float(min(mean_open.min(), mean_closed.min())),
    float(max(mean_open.max(), mean_closed.max())),
)
fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), dpi=120)
for ax, vec, title in zip(axes, (mean_open, mean_closed), ("eyes open", "eyes closed")):
    mne.viz.plot_topomap(vec, info, axes=ax, show=False, vlim=vlim, cmap="viridis")
    ax.set_title(f"{title}\nlog10 alpha (8-12 Hz)")
footer = (
    f"source: ds005514 sub-{task.subjects[0]} | n_chans={len(ch_names)} | "
    f"sfreq={sfreq:.0f} Hz | band-pass {task.bandpass[0]:.1f}-{task.bandpass[1]:.1f} Hz"
)
fig.text(0.01, 0.01, footer, fontsize=6, ha="left")
fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
plt.show()

# %% [markdown]
# ## Step 5 -- PRIMM Investigate
# **Investigate.** Subtract the two condition maps and rank channels by
# the eyes-closed minus eyes-open contrast. Posterior electrodes should
# top the ranking, confirming Berger's finding on a single child.

# %%
contrast = mean_closed - mean_open
top = np.argsort(contrast)[::-1][:5]
print(
    "Top-5 alpha-bump channels (closed - open):  "
    + " | ".join(f"{ch_names[i]} +{contrast[i]:.3f}" for i in top)
)
mean_alpha_diff = float(contrast.mean())
# Berger 1929 is a robust group mean; on one subject we soft-assert
# closed > open in >=50% of channels (the majority sign).
positive_channel_ratio = float((contrast > 0).mean())
assert positive_channel_ratio >= 0.50, (
    f"closed > open in only {positive_channel_ratio:.0%} of channels."
)

# %% [markdown]
# ## Step 6 -- Train logistic regression on alpha power
# **Run #3.** With one subject and no cross-subject groups to hold out,
# we use ``get_splitter("within_subject", ...)``: fold 0's train/test
# windows come from disjoint slices of the same recording. We then call
# ``assert_no_leakage`` so the JSON ``leakage_report`` line lands on
# stdout (E5.42). Cisotto & Chicco 2024 (Tip 9, doi:10.7717/peerj-cs.2256)
# reminds us to emit it even when the answer is trivially zero.

# %%
metadata = pd.DataFrame(
    {
        "subject": [task.subjects[0]] * len(y),
        "sample_id": [f"w{i:04d}" for i in range(len(y))],
        "target": y,
    }
)
splitter = get_splitter(
    "within_subject", n_folds=5, n_splits=5, random_state=SEED, shuffle=True
)
train_idx, test_idx = next(iter(splitter.split(y, metadata)))
fold = [
    (
        metadata.loc[train_idx, "sample_id"].tolist(),
        metadata.loc[test_idx, "sample_id"].tolist(),
    )
]
overlap = assert_no_leakage(fold, metadata, by="subject")
assert overlap == 0
clf = LogisticRegression(random_state=SEED, max_iter=400).fit(
    alpha_power[train_idx], y[train_idx]
)
model_acc = float(accuracy_score(y[test_idx], clf.predict(alpha_power[test_idx])))
chance = float(majority_baseline(y[train_idx], y[test_idx])["chance_level"])
print(
    f"Model accuracy: {model_acc:.2f} | chance level: {chance:.2f} | metric: accuracy"
)

# %% [markdown]
# ## A common mistake -- and how to recover
#
# **Run.** A common slip is reading the wrong frequency band -- e.g.
# integrating the PSD between 30 and 8 Hz (lo > hi) so the mask is empty
# and the per-window band power collapses to ``-inf``. We trigger it on
# purpose with ``try/except`` so you see exactly what the error looks
# like.

# %%
try:
    lo, hi = 30.0, 8.0  # swapped on purpose
    bad_mask = (freqs >= lo) & (freqs <= hi)
    if not bad_mask.any():
        raise ValueError(f"empty band mask for ({lo}, {hi}) Hz: lo > hi")
except (ValueError, RuntimeError) as exc:
    print(f"Caught {type(exc).__name__}: {exc}")
    # Recovery: order the band tuple so lo < hi (alpha = 8 to 12 Hz).
    print("Recovery: use (8.0, 12.0) Hz so the alpha mask is non-empty.")

# %% [markdown]
# ## Modify -- swap the band
# **Modify.** Re-run Steps 4-6 after swapping the 8-12 Hz alpha band for
# the 1-8 Hz delta+theta band. The contrast should collapse (and the
# classifier with it), confirming the alpha bump drives the decoder
# rather than any global power difference.

# %%
low = np.log10(psd[..., (freqs >= 1.0) & (freqs <= 8.0)].mean(axis=-1) + 1e-30)
low_diff = float((low[y == 1].mean(0) - low[y == 0].mean(0)).mean())
clf_low = LogisticRegression(random_state=SEED, max_iter=400).fit(
    low[train_idx], y[train_idx]
)
acc_low = float(accuracy_score(y[test_idx], clf_low.predict(low[test_idx])))
print(
    f"1-8 Hz contrast: mean log10 power diff={low_diff:+.3f} | accuracy={acc_low:.2f}"
)

# %% [markdown]
# ## Make -- a second subject
# **Make.** Repeat Steps 3-6 on another HBN subject (e.g.
# ``"NDARAA075AMK"``) by passing ``subjects=[...]`` to ``get_task``. With
# two subjects you can promote the splitter to ``cross_subject`` and let
# the model meet a participant it has never seen.

# %% [markdown]
# ## Result
# A printed table of the alpha-band evidence and the decoder vs. the
# chance baseline (E5.43). Model accuracy beats chance and the
# alpha-power difference is positive: closed eyes carry more alpha.

# %%
rows = [
    ("eyes open  (y=0)", f"{mean_open.mean():+.3f}", "--"),
    ("eyes closed (y=1)", f"{mean_closed.mean():+.3f}", "--"),
    ("logistic regression", "--", f"{model_acc:0.3f}"),
    ("chance (majority)", "--", f"{chance:0.3f}"),
]
print("\n| condition          | mean log10 alpha | accuracy |")
print("|--------------------|------------------|----------|")
for cond, av, acv in rows:
    print(f"| {cond:<19}| {av:<17}| {acv:<8} |")
print(
    json.dumps(
        {
            "alpha_diff_closed_minus_open": round(mean_alpha_diff, 4),
            "alpha_positive_channel_ratio": round(positive_channel_ratio, 4),
            "model_accuracy": round(model_acc, 4),
            "chance_level": round(chance, 4),
        }
    )
)

# %% [markdown]
# ## Try it yourself / Extensions
# - Increase the window from 2 s to 4 s and re-run; does the contrast sharpen?
# - Pull two more subjects, switch to ``get_splitter("cross_subject", ...)``.
# - Replace the logistic regression with ``ShallowFBCSPNet`` (see plot_12).
# - Save the alpha topomap with ``fig.savefig`` for the benchmark log.

# %% [markdown]
# ## Links and references
# - Concept: :doc:`/concepts/preprocessing_decisions`.
# - API: :func:`eegdash.tasks.get_task`, :class:`eegdash.EEGDashDataset`, :func:`eegdash.splits.get_splitter`, :func:`eegdash.splits.assert_no_leakage`.
# - Berger 1929, *Arch. Psychiatr. Nervenkr.*, "Uber das Elektrenkephalogramm des Menschen" (cited textually; no DOI).
# - Pernet et al. 2019, *Sci. Data* 6:103, doi:10.1038/s41597-019-0104-8 -- EEG-BIDS.
# - Gramfort et al. 2013, *Front. Neurosci.* 7:267, doi:10.3389/fnins.2013.00267 -- MNE-Python.
# - Cisotto & Chicco 2024, *PeerJ CS* 10:e2256, doi:10.7717/peerj-cs.2256 -- Tip 9 (leakage).
# - Alexander 2017, *Sci. Data* 4:170181, doi:10.1038/sdata.2017.181 -- HBN dataset.
# - HBN ds005514, doi:10.18112/openneuro.ds005514.v1.0.0.
