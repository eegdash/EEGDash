"""Track 3: sleep-onset latency
================================

**Difficulty 2** | **Runtime: <5s** | **Compute: CPU**

Track 3 predicts seconds from recording start to the first stable N2 epoch.
The competition evaluates held-out subjects recorded with wearable EEG and
uses weighted binned mean absolute error (W-bMAE). See the
`competition site <https://neural-interfaces26.github.io/>`_ and the
`NeuralBench guide <https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html>`_.

This tutorial reports a transparent balanced-bin MAE analogue. Only the
official NeuralBench evaluator produces the competition W-bMAE.

Keywords: EEG2026, sleep, regression
"""

# %% [markdown]
# Seed data in EEGDash
# --------------------
# Sleep-EDF Expanded and the PhysioNet 2018 sleep challenge are available as
# BIDS-first EEGDash records. These snippets open a pre-populated local cache:
#
# .. code-block:: python
#
#    from eegdash import EEGDashDataset
#
#    sleep_edf = EEGDashDataset(
#        cache_dir="./data", dataset="nm000185", download=False
#    )
#    sleep_edf.plot(0)
#    physionet = EEGDashDataset(
#        cache_dir="./data", dataset="nm000225", download=False
#    )
#
# Remove ``download=False`` to fetch missing recordings.
# ``plot(0)`` opens the first cached recording in the Braindecode notebook
# viewer before preprocessing.
# NeuralBench defines the event annotation, wearable evaluation cohort, and
# official scoring weights.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash.viz import use_eegdash_style

use_eegdash_style()
rng = np.random.default_rng(2026)

# %% [markdown]
# Build subject-level onset features
# ----------------------------------
# Each row is one person. Four spectral and behavioural proxies carry a noisy
# onset signal. The last 15 subjects also receive a wearable-device shift.

# %%
n_subjects = 60
subjects = np.asarray([f"sub-{index:03d}" for index in range(n_subjects)])
devices = np.where(np.arange(n_subjects) < 45, "clinical", "wearable")
onset = rng.uniform(120, 1800, size=n_subjects)
X = np.column_stack(
    [
        onset / 600 + rng.normal(scale=0.25, size=n_subjects),
        np.log1p(onset) + rng.normal(scale=0.15, size=n_subjects),
        np.sqrt(onset) / 20 + rng.normal(scale=0.20, size=n_subjects),
        rng.normal(size=n_subjects),
    ]
)
X[45:] += np.array([0.35, -0.15, 0.20, 0.10])

# %% [markdown]
# Hold subjects out and balance onset ranges
# ------------------------------------------
# A plain MAE can be dominated by common onset ranges. This local analogue
# computes MAE inside fixed latency bins and averages the non-empty bins. The
# official evaluator has its own frozen bins and weights.

# %%
train = np.arange(n_subjects) < 45
test = ~train
subject_overlap = set(subjects[train]) & set(subjects[test])
device_overlap = set(devices[train]) & set(devices[test])
split_overlap = subject_overlap | device_overlap
assert not split_overlap

model = make_pipeline(StandardScaler(), Ridge(alpha=4.0))
model.fit(X[train], onset[train])
predicted = model.predict(X[test])

bin_id = np.digitize(onset[test], bins=[300, 600, 900, 1200])
bin_errors = np.asarray(
    [
        np.mean(np.abs(onset[test][bin_id == index] - predicted[bin_id == index]))
        for index in np.unique(bin_id)
    ]
)
score = float(np.mean(bin_errors))
metric_name = "balanced-bin MAE (s)"
held_out_axis = "subject + device"
print(f"{metric_name}: {score:.1f} | split overlap: {len(split_overlap)}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
axes[0].scatter(onset[test], predicted, color="#6B8E8E")
limits = [0, 1900]
axes[0].plot(limits, limits, "--", color="#C45A3C")
axes[0].set(
    xlabel="true onset (s)", ylabel="predicted onset (s)", xlim=limits, ylim=limits
)
axes[1].bar(np.arange(len(bin_errors)), bin_errors, color="#C45A3C")
axes[1].set(xlabel="occupied latency bin", ylabel="MAE (s)")
fig.suptitle("EEG2026 Track 3 — held-out subjects and device shift")
fig.tight_layout()
plt.show()
