"""Track 2: cross-session BCI decoding
=====================================

**Difficulty 2** | **Runtime: <5s** | **Compute: CPU**

Track 2 decodes cued mental commands from short EEG windows. Training uses
early sessions and evaluation uses later sessions from the same people, with
no session recalibration. The official metric is balanced accuracy. See the
`competition site <https://neural-interfaces26.github.io/>`_ and the
`NeuralBench guide <https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html>`_.

Keywords: EEG2026, BCI, cross-session
"""

# %% [markdown]
# Seed data in EEGDash
# --------------------
# Stieger2021 is a 62-participant, multi-session motor-imagery corpus. This
# snippet opens recordings already present in ``./data``:
#
# .. code-block:: python
#
#    from eegdash import EEGDashDataset
#
#    stieger = EEGDashDataset(
#        cache_dir="./data", dataset="nm000339", download=False
#    )
#    stieger.plot(0)
#
# Remove ``download=False`` to fetch missing recordings.
# ``plot(0)`` opens the first cached recording in the Braindecode notebook
# viewer before preprocessing.
# Use NeuralBench for the official preprocessing and frozen split. The small
# analogue below makes the cross-session rule visible without downloading
# hundreds of gigabytes.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash.viz import use_eegdash_style

use_eegdash_style()
rng = np.random.default_rng(2026)

# %% [markdown]
# Simulate commands, people, and session drift
# --------------------------------------------
# Three class centres represent mental commands. Subject offsets preserve
# individual anatomy; a session-specific drift shifts every later recording.

# %%
n_subjects, n_sessions = 8, 4
commands = ("motor imagery", "mental calculation", "word association")
n_commands = len(commands)
trials_per_command, n_features = 12, 20
centres = rng.normal(scale=1.4, size=(n_commands, n_features))
subject_offset = rng.normal(scale=0.35, size=(n_subjects, n_features))
session_drift = rng.normal(scale=0.25, size=(n_sessions, n_features))

rows = []
for subject in range(n_subjects):
    for session in range(n_sessions):
        for command in range(n_commands):
            for _ in range(trials_per_command):
                features = centres[command] + subject_offset[subject]
                features = features + session_drift[session]
                features = features + rng.normal(scale=1.0, size=n_features)
                rows.append((features, command, subject, session))

X = np.stack([row[0] for row in rows])
y = np.asarray([row[1] for row in rows])
subjects = np.asarray([row[2] for row in rows])
sessions = np.asarray([row[3] for row in rows])

# %% [markdown]
# Train early, test late
# ----------------------
# Sessions 1-3 train the model and session 4 tests it. Subjects intentionally
# occur on both sides; sessions must not.

# %%
train = sessions < 3
test = sessions == 3
split_overlap = set(sessions[train]) & set(sessions[test])
subject_overlap = set(subjects[train]) & set(subjects[test])
assert not split_overlap
assert subject_overlap == set(range(n_subjects))

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
model.fit(X[train], y[train])
predicted = model.predict(X[test])
cell_scores = [
    balanced_accuracy_score(
        y[test][subjects[test] == subject], predicted[subjects[test] == subject]
    )
    for subject in sorted(subject_overlap)
]
score = float(np.mean(cell_scores))
metric_name = "balanced accuracy"
held_out_axis = "session"
print(f"{metric_name}: {score:.3f} | session overlap: {len(split_overlap)}")

# %%
matrix = confusion_matrix(y[test], predicted, normalize="true")
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
image = axes[0].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
axes[0].set(xlabel="predicted command", ylabel="true command")
axes[0].set_xticks(range(n_commands), commands, rotation=25, ha="right")
axes[0].set_yticks(range(n_commands), commands)
fig.colorbar(image, ax=axes[0], label="row fraction")
axes[1].bar(
    ["chance", "logistic"], [1 / n_commands, score], color=["#B8B8B8", "#C45A3C"]
)
axes[1].set(ylabel="balanced accuracy", ylim=(0, 1))
fig.suptitle("EEG2026 Track 2 — early sessions to later session")
fig.tight_layout()
plt.show()
