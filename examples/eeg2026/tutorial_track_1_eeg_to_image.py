"""Track 1: retrieve actual viewed images from recorded EEG
========================================================

Load THINGS-EEG2 subject 08, session 02, task-test from nm000232. This is one
whole recording (hundreds of MB) plus 30 small original stimulus images.
Observed tot_img_number values in the BIDS event sidecar identify the images;
the release's stimuli.tsv maps those IDs to the actual JPG files.

An 8-by-8 RGB thumbnail supplies a fixed 192-dimensional image representation.
This deliberately simple pixel baseline needs no vision-model checkpoint. Ridge
maps EEG to these real image features. Split image identities 60/20/20 for
training, validation and testing, and retrieve against all 30 candidates.
The source calls this recording "test", but our within-recording image split
is an instructional experiment, not the official challenge train/test split.

Source: https://github.com/nemarDatasets/nm000232
"""

# %%
# Before you start
# ----------------
#
# Use EEGDash, MNE, NumPy, pandas, scikit-learn, Pillow and Matplotlib.
# No GPU or vision checkpoint is required. The chosen BDF is about 272 MB;
# first use additionally fetches the stimulus manifest and 30 original JPGs.
# ``EEGDASH_CACHE_DIR`` retains both the signal and ``things_test_images``.
# Network failures are acquisition errors and are not replaced with substitute
# targets.
#
# This page learns EEG-to-image retrieval with fixed low-resolution pixel
# features. It demonstrates image-identity separation and candidate ranking,
# not semantic CLIP embeddings or an official competition score. The source
# recording's ``task-test`` name describes its original release role; the
# train/validation/test masks below are new, explicitly scoped image subsets.

import os
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash import EEGDash, EEGDashDataset

# %%
# Choose one converted recording and its observed image IDs
# ---------------------------------------------------------
#
# The catalogue also contains original BrainVision source files. Selecting
# a path beginning with ``sub-`` chooses the converted BIDS BDF and its event
# sidecar; the assertion prevents loading both representations of the same run.
# The exact path is printed so the acquisition remains reviewable.
#
# ``tot_img_number`` in that sidecar records which stimulus was presented.
# Normal image trials with IDs 1–30 are kept, excluding target/catch trials.
# Subtracting one only changes these observed IDs into zero-based array indices.
# Numeric trigger order is never guessed to be an image identity.

records = EEGDash().find(
    {"dataset": "nm000232", "subject": "08", "session": "02", "task": "test"}
)
records = [record for record in records if record["bids_relpath"].startswith("sub-")]
assert len(records) == 1
cache = Path(os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")).expanduser()
dataset = EEGDashDataset(records=records, cache_dir=cache)
print(dataset.description.to_string(index=False))
print("Exact signal:", records[0]["bids_relpath"])
raw = dataset.datasets[0].raw.pick("eeg")
print(raw.ch_names, raw.info["sfreq"], np.unique(raw.annotations.description))
# Read the sidecar acquired together with this exact EEG recording.
event_path = Path(raw.filenames[0]).with_name(
    Path(raw.filenames[0]).name.replace("_eeg.bdf", "_events.tsv")
)
trials = pd.read_csv(event_path, sep="\t")
trials = trials[
    trials.trial_type.eq("image") & trials.tot_img_number.between(1, 30)
].copy()
labels = trials.tot_img_number.to_numpy(dtype=int) - 1
assert len(trials) and set(labels) == set(range(30))
# Pin the metadata and JPG revision together; no inferred trigger-to-image mapping.
# %%
# Build targets from the actual stimulus files
# --------------------------------------------
#
# The pinned stimulus manifest maps each observed ID to its original JPG.
# Downloaded bytes must decode as an image before entering the cache. Each RGB
# image is resized to 8 × 8 and flattened, producing 192 coordinates scaled to
# 0–1. Unit-length normalization makes the later dot products comparable across
# candidate images with different overall brightness.
#
# These targets use no EEG and require no learned preprocessing, so candidate
# embeddings can be computed for all 30 images. Access to an unseen candidate
# image at retrieval time is part of the task; access to that image's held-out
# EEG response during fitting would be leakage. Thumbnail features emphasize
# colour and coarse layout and are a weak substitute for semantic visual
# representations, so low retrieval accuracy would not be surprising.

revision = "5732fd1a9b9c873c94b25c7e7e70e550dc64faa7"
root = f"https://raw.githubusercontent.com/nemarDatasets/nm000232/{revision}/stimuli/"
manifest = pd.read_csv(root + "stimuli.tsv", sep="\t").set_index("stimulus_id")
image_cache = cache / "things_test_images"
image_cache.mkdir(parents=True, exist_ok=True)
embeddings = []
for identity in range(1, 31):
    row = manifest.loc[f"stim-test{identity:03d}"]
    destination = image_cache / Path(row.filename).name
    if not destination.exists():
        with urlopen(root + row.filename, timeout=60) as response:
            payload = response.read()
        # Decode before committing to cache: failed acquisitions cannot masquerade as images.
        from io import BytesIO

        with Image.open(BytesIO(payload)) as image:
            image.verify()
        destination.write_bytes(payload)
    with Image.open(destination) as image:
        embeddings.append(
            np.asarray(image.convert("RGB").resize((8, 8)), dtype=float).ravel() / 255
        )
embeddings = np.asarray(embeddings)
assert np.isfinite(embeddings).all() and (np.linalg.norm(embeddings, axis=1) > 0).all()
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
identities = np.arange(30)
# Limit windows to 180 ms: stimuli occur approximately every 200 ms. Earlier
# responses can still overlap because this is a rapid serial presentation task.
# %%
# Extract short visual-response windows
# -------------------------------------
#
# Event sample indices come from the same BDF sidecar. Posterior channels
# O1, Oz and O2 keep the predictor small. Windows run from onset to 180 ms because
# images arrive at roughly 200 ms intervals. This limits overlap with the next
# presentation but cannot remove responses to preceding images in the rapid
# stream; it is not isolated-trial visual decoding.
#
# MNE handles epoch rejection and resamples the retained waveforms to 100 Hz.
# Indexing labels by ``epochs.selection`` preserves their correspondence if an
# epoch is dropped. Flattening yields ``(retained trials, channels × samples)``
# features in volts. Feature scaling is deferred until training rather than
# estimated from the full collection.

events = np.column_stack(
    [
        trials["sample"].to_numpy(dtype=int),
        np.zeros(len(trials), dtype=int),
        np.ones(len(trials), dtype=int),
    ]
)
epochs = mne.Epochs(
    raw,
    events,
    event_id={"image": 1},
    tmin=0,
    tmax=0.18,
    baseline=None,
    picks=["O1", "Oz", "O2"],
    preload=True,
    reject_by_annotation=True,
).resample(100)
labels = labels[epochs.selection]
X = epochs.get_data().reshape(len(epochs), -1)
assert np.isfinite(X).all()
print("Actual EEG features:", X.shape, "actual image features:", embeddings.shape)

# %%
# Split identities, not repetitions. No held-out image appears in training.
# %%
# Split repeated presentations by image identity
# ----------------------------------------------
#
# An image can occur many times in the recording. A random trial split
# would let the decoder see the same image during training and evaluation,
# answering an easier repeated-stimulus question. The seeded permutation instead
# assigns complete image identities: 18 train, 6 validate, and 6 test when all
# 30 are retained. It randomizes only assignments, not signals or labels.
#
# These are within-participant, within-session results. They do not measure
# cross-person or cross-device generalization. Assertions check the image-ID
# boundaries; they cannot erase physiological overlap between successive
# presentations in the original acquisition.

unique = np.unique(labels)
assert len(unique) >= 15
np.random.default_rng(2026).shuffle(unique)
first, second = int(0.6 * len(unique)), int(0.8 * len(unique))
train, valid, test = [
    np.isin(labels, group) for group in np.split(unique, [first, second])
]
assert set(labels[train]).isdisjoint(labels[valid | test])
assert set(labels[valid]).isdisjoint(labels[test])


# %%
# Tune the decoder and rank all candidate images
# ----------------------------------------------
#
# Ridge predicts 192 image coordinates from EEG. Its pipeline estimates
# feature means and scales on training images' responses only. Validation
# retrieval selects among the three prespecified penalties; the selected fitted
# model is then evaluated once on test responses without refitting on validation.
#
# ``cosine_similarity`` compares each predicted representation with the 30
# candidate embeddings. ``top_k_accuracy_score`` ranks those similarities;
# ``labels=identities`` supplies the full candidate set even though a fold
# contains responses to only a subset of images. A trial succeeds if its true
# image appears anywhere in the five highest scores. The uniform-candidate reference is ``5 / 30``, independent of
# the number of test identities. The reported proportion weights responses,
# not distinct image IDs, equally.

best_score, best_model = -np.inf, None
for alpha in [1, 10, 100]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X[train], embeddings[labels[train]])
    score = top_k_accuracy_score(
        labels[valid],
        cosine_similarity(model.predict(X[valid]), embeddings),
        k=5,
        labels=identities,
    )
    if score > best_score:
        best_score, best_model = score, model
score = top_k_accuracy_score(
    labels[test],
    cosine_similarity(best_model.predict(X[test]), embeddings),
    k=5,
    labels=identities,
)
print("Held-out image top-5 accuracy:", score, "candidate images:", len(identities))
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["uniform retrieval", "ridge"], [5 / len(identities), score])
ax.set(ylabel="Test image top-5 accuracy", ylim=(0, 1))
plt.show()

# %%
# Make the next comparison informative
# ------------------------------------
#
# Inspect errors per held-out image before concluding that a representation
# captures visual content. Some images have similar thumbnails, and different
# images may have different numbers of usable responses. Averaging predictions
# within image or reporting image-wise scores answers a different evaluation
# question and should be prespecified.
#
# For an actionable extension, replace thumbnail coordinates with embeddings
# from a named, revision-pinned vision model while retaining the observed ID
# mapping, candidate pool and untouched test images. Select ridge penalties using
# validation images again, and compare both representations on the same test
# responses. Use separate sessions or participants for broader generalization.
#
# Related split-selection example: `Braindecode train, test and tune
# <https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html>`_.
