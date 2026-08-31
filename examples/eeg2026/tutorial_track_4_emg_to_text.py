"""Track 4: EMG-to-text decoding
================================

**Difficulty 2** | **Runtime: <5s** | **Compute: CPU**

Track 4 decodes typed text from wrist surface EMG. Evaluation users are
unseen during training, so anatomy, typing strategy, and sensor placement all
shift. The official metric is character error rate (CER). See the
`competition site <https://neural-interfaces26.github.io/>`_ and the
`NeuralBench guide <https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_text.html>`_.

Keywords: EEG2026, EMG, text
"""

# %% [markdown]
# Seed data in EEGDash
# --------------------
# The public emg2qwerty corpus is catalogued as ``nm000104``. This snippet
# opens recordings already present in a local cache:
#
# .. code-block:: python
#
#    from eegdash import EEGDashDataset
#
#    emg = EEGDashDataset(
#        cache_dir="./data", dataset="nm000104", download=False
#    )
#    emg.plot(0)
#
# Remove ``download=False`` to fetch missing recordings.
# ``plot(0)`` opens the first recording in the Braindecode notebook viewer;
# when a ``*_desc-pose.json`` sidecar is present, it adds the synchronized
# hand-pose panel beside the EMG traces.
# EEGDash handles the recordings; NeuralBench owns sequence preparation,
# frozen users, the official decoder interface, and submission scoring.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash.viz import use_eegdash_style

use_eegdash_style()
rng = np.random.default_rng(2026)


def edit_distance(reference: str, hypothesis: str) -> int:
    """Return Levenshtein distance using one rolling dynamic-programming row."""
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_char in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


# %% [markdown]
# Build a cross-user typing analogue
# ----------------------------------
# Eight keys each have a latent EMG pattern. Every user adds an anatomical
# offset and sensor-placement transform. Twelve users train the classifier;
# six entirely new users test it.

# %%
alphabet = np.asarray(list("asdfjkl;"))
phrase = "asdfjkl;" * 8
n_users, n_features = 18, 24
key_centres = rng.normal(scale=1.8, size=(len(alphabet), n_features))
rows = []
for user in range(n_users):
    user_offset = rng.normal(scale=0.45, size=n_features)
    user_scale = rng.normal(loc=1.0, scale=0.08, size=n_features)
    for char in phrase:
        label = int(np.flatnonzero(alphabet == char)[0])
        features = key_centres[label] * user_scale + user_offset
        features = features + rng.normal(scale=1.0, size=n_features)
        rows.append((features, label, user))

X = np.stack([row[0] for row in rows])
y = np.asarray([row[1] for row in rows])
users = np.asarray([row[2] for row in rows])
train = users < 12
test = ~train
split_overlap = set(users[train]) & set(users[test])
assert not split_overlap

# %% [markdown]
# Decode aligned characters, then compute CER
# -------------------------------------------
# The compact baseline predicts one character per fixed window. It therefore
# demonstrates cross-user token classification, not the official variable-
# length sequence-transduction model. The edit-distance CER calculation is
# exact and remains valid when you replace the classifier with a sequence
# decoder.
#
# CER is Levenshtein edits divided by reference length. Unlike accuracy, CER
# can exceed 1 when a hypothesis contains many insertions; do not clamp it.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
model.fit(X[train], y[train])
predicted = model.predict(X[test])

references = []
hypotheses = []
for user in np.unique(users[test]):
    user_mask = users[test] == user
    references.append("".join(alphabet[y[test][user_mask]]))
    hypotheses.append("".join(alphabet[predicted[user_mask]]))

edits = sum(
    edit_distance(reference, hypothesis)
    for reference, hypothesis in zip(references, hypotheses)
)
n_reference_chars = sum(map(len, references))
score = float(edits / n_reference_chars)
metric_name = "character error rate"
held_out_axis = "user"
print(f"{metric_name}: {score:.3f} | user overlap: {len(split_overlap)}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
preview = 32
axes[0].plot(np.arange(preview), y[test][:preview], "o-", label="reference")
axes[0].plot(np.arange(preview), predicted[:preview], "x--", label="decoded")
axes[0].set(xlabel="character position", ylabel="key index")
axes[0].legend()
axes[1].bar(["perfect", "logistic"], [0, score], color=["#B8B8B8", "#C45A3C"])
axes[1].set(ylabel="character error rate", ylim=(0, max(0.2, score * 1.2)))
fig.suptitle("EEG2026 Track 4 — cross-user EMG decoding")
fig.tight_layout()
plt.show()
