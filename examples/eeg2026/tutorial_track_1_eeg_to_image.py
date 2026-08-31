"""Track 1: EEG-to-image retrieval
=================================

**Difficulty 2** | **Runtime: <5s** | **Compute: CPU**

Track 1 maps an EEG response to the embedding of the viewed image. The
competition holds stimuli out, so repeated trials of one image must never
appear on both sides of the split. The official score is top-5 retrieval
accuracy. See the `competition site <https://neural-interfaces26.github.io/>`_
and the `NeuralBench guide <https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html>`_.

This offline example uses a small synthetic embedding problem. Its score is a
pipeline check, not an official baseline.

Keywords: EEG2026, retrieval, vision
"""

# %% [markdown]
# Seed data in EEGDash
# --------------------
# THINGS-EEG2 and the Alljoined corpora are already catalogued by EEGDash.
# Open recordings already present in a local cache without network access:
#
# .. code-block:: python
#
#    from eegdash import EEGDashDataset
#
#    things = EEGDashDataset(
#        cache_dir="./data", dataset="nm000232", download=False
#    )
#    things.plot(0)
#    alljoined = EEGDashDataset(
#        cache_dir="./data", dataset="nm000134", download=False
#    )
#
# Remove ``download=False`` when you are ready to fetch missing recordings.
# ``plot(0)`` opens the first cached recording in the Braindecode notebook
# viewer before preprocessing.
# The official NeuralBench task owns preprocessing, splits, and submission
# files. EEGDash supplies the BIDS-first recording layer.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from eegdash.viz import use_eegdash_style

use_eegdash_style()
rng = np.random.default_rng(2026)

# %% [markdown]
# Build an embedding-shaped analogue
# ----------------------------------
# Each stimulus has a fixed image embedding and six noisy EEG repetitions.
# A linear sensor map turns the embedding into EEG features; ridge regression
# learns the inverse map from training stimuli only.

# %%
n_stimuli, repetitions = 30, 6
embedding_dim, eeg_features = 12, 36
image_embeddings = rng.normal(size=(n_stimuli, embedding_dim))
image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)
sensor_map = rng.normal(size=(embedding_dim, eeg_features))

stimulus = np.repeat(np.arange(n_stimuli), repetitions)
X = image_embeddings[stimulus] @ sensor_map
X += rng.normal(scale=0.45, size=X.shape)
y = image_embeddings[stimulus]

# %% [markdown]
# Hold out stimuli, then retrieve
# -------------------------------
# The first 20 stimuli train the decoder; the last 10 are unseen. Candidate
# retrieval uses all 30 image embeddings, so chance top-5 accuracy is 5/30.

# %%
train_stimuli = set(range(20))
test_stimuli = set(range(20, n_stimuli))
split_overlap = train_stimuli & test_stimuli
assert not split_overlap

train = np.isin(stimulus, list(train_stimuli))
test = ~train
decoder = Ridge(alpha=1.0).fit(X[train], y[train])
predicted = decoder.predict(X[test])
predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)

similarity = predicted @ image_embeddings.T
top_five = np.argpartition(similarity, -5, axis=1)[:, -5:]
score = float(np.mean([target in row for target, row in zip(stimulus[test], top_five)]))
metric_name = "top-5 accuracy"
held_out_axis = "stimulus"
print(f"{metric_name}: {score:.3f} | stimulus overlap: {len(split_overlap)}")

# %% [markdown]
# Read the result
# ---------------
# The left panel shows one test trial's candidate similarities. The right
# panel compares the tutorial score with retrieval chance. Use NeuralBench,
# not this synthetic value, for any leaderboard claim.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
trial = 0
axes[0].bar(np.arange(n_stimuli), similarity[trial], color="#6B8E8E")
axes[0].axvline(stimulus[test][trial], color="#C45A3C", label="true image")
axes[0].set(xlabel="candidate stimulus", ylabel="cosine similarity")
axes[0].legend()
axes[1].bar(["chance", "ridge"], [5 / n_stimuli, score], color=["#B8B8B8", "#C45A3C"])
axes[1].set(ylabel="top-5 accuracy", ylim=(0, 1))
fig.suptitle("EEG2026 Track 1 — cross-stimulus retrieval")
fig.tight_layout()
plt.show()
