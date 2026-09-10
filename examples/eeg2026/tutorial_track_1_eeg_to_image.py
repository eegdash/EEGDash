"""Track 1: retrieve actual viewed images from recorded EEG
====================================================================

Load THINGS-EEG2 subject 08, session 02, task-test from nm000232. This is one
whole recording (hundreds of MB) plus 60 original stimulus images.
Observed tot_img_number values in the BIDS event sidecar identify the images;
the release's stimuli.tsv maps those IDs to the actual JPG files.

A frozen DINOv2-giant encoder supplies 1536-dimensional image targets. Ridge
maps EEG to these real image features. Split image identities 60/20/20 for
training, validation and testing. Each retrieval gallery contains only that
split's unseen images. The default uses the official DINOv2-giant target
configuration; EEGDASH_DINO_MODEL=small selects a lower-cost warm-up encoder.
The source calls this recording "test", but our within-recording image split
is an instructional experiment, not the official challenge train/test split.

Source: https://github.com/nemarDatasets/nm000232
"""

# %%
# Before you start
# ----------------------------
#
# Use EEGDash, NeuralSet, Transformers, TorchVision, MNE, NumPy, pandas,
# scikit-learn, Pillow and Matplotlib. The BDF is about 272 MB; first use also
# fetches 60 JPGs and a pretrained checkpoint (about 88 MB for DINOv2-small,
# about 4.5 GB for giant). CPU execution uses one image at a time. NeuralSet
# caches extracted targets, and Hugging Face retains the model checkpoint.
# ``EEGDASH_CACHE_DIR`` retains both the signal and ``things_test_images``.
# Network failures are acquisition errors and are not replaced with substitute
# targets.
#
# The `official track <https://neural-interfaces26.github.io/tracks.html>`_
# retrieves unseen images using frozen DINOv2 features. The `NeuralBench guide
# <https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html>`_
# specifies giant features at relative depth 0.6667, mean token pooling and
# ``imsize=518``, scored across the complete held-out gallery and then averaged
# across participants. Its within-batch validation metric is a different task.
#
# The default giant encoder uses the guide's NeuralSet extraction settings.
# Selecting small reduces download and CPU cost with different targets. Neither mode
# reproduces its contrastive EEG training, source-defined splits or hidden
# Alljoined cohort. This one-session ridge warm-up reports its own held-out
# gallery size alongside its measured score.

import os
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from neuralset.events.etypes import Image as ImageEvent
from neuralset.extractors import HuggingFaceImage
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash import EEGDash, EEGDashDataset

# %%
# Choose one converted recording and its observed image IDs
# ---------------------------------------------------------------------
#
# The catalogue also contains original BrainVision source files. Selecting
# a path beginning with ``sub-`` chooses the converted BIDS BDF and its event
# sidecar; the assertion prevents loading both representations of the same run.
# The exact path is printed so the acquisition remains reviewable.
#
# ``tot_img_number`` in that sidecar records which stimulus was presented.
# Normal image trials with IDs 1–60 are kept, excluding target/catch trials.
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
    trials.trial_type.eq("image") & trials.tot_img_number.between(1, 60)
].copy()
labels = trials.tot_img_number.to_numpy(dtype=int) - 1
assert len(trials) and set(labels) == set(range(60))
# Pin the metadata and JPG revision together; no inferred trigger-to-image mapping.
# %%
# Build targets from the actual stimulus files
# --------------------------------------------------------
#
# The pinned stimulus manifest maps each observed ID to its original JPG.
# Downloaded bytes must decode as images before entering the cache. Image IDs
# come from the observed event sidecar, rather than a guessed trigger ordering.
# The frozen encoder sees only original images; all EEG-to-target fitting and
# hyperparameter selection happen after the image-identity split below.
#
# NeuralSet performs the same relative-layer selection and token averaging in
# either model mode. The small model yields 384 coordinates; giant yields 1536.
# Its image processor can resize/crop after ``imsize``: using the public extractor
# preserves the guide's actual processing path rather than hand-implementing a
# superficially similar resize. Unit normalization gives cosine ranking a common
# scale. These are measured pretrained activations, never random image targets.

revision = "5732fd1a9b9c873c94b25c7e7e70e550dc64faa7"
root = f"https://raw.githubusercontent.com/nemarDatasets/nm000232/{revision}/stimuli/"
manifest = pd.read_csv(root + "stimuli.tsv", sep="\t").set_index("stimulus_id")
image_cache = cache / "things_test_images"
image_cache.mkdir(parents=True, exist_ok=True)
image_events = []
for identity in range(1, 61):
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
    image_events.append(
        ImageEvent(filepath=destination, start=0, duration=1, timeline="stimulus")
    )

model_size = os.environ.get("EEGDASH_DINO_MODEL", "giant")
assert model_size in {"small", "giant"}
model_revision = {
    "small": "ed25f3a31f01632728cabb09d1542f84ab7b00566",
    "giant": "611a9d42f2335e0f921f1e313ad3c1b7178d206d",
}[model_size]
torch.set_num_threads(2)
image_encoder = HuggingFaceImage(
    model_name=f"facebook/dinov2-{model_size}",
    pretrained=True,
    layers=0.6667,
    token_aggregation="mean",
    imsize=518,
    batch_size=1,
    device="cpu",
    hf_config={
        "model_kwargs": {"revision": model_revision},
        "processor_kwargs": {"revision": model_revision},
    },
    infra={"folder": cache / "track1_dinov2", "cluster": None},
)
image_encoder.prepare(image_events)
embeddings = np.stack(
    [image_encoder.get_static(event).numpy() for event in image_events]
)
assert embeddings.shape == (60, 384 if model_size == "small" else 1536)
assert np.isfinite(embeddings).all() and (np.linalg.norm(embeddings, axis=1) > 0).all()
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
# Limit windows to 180 ms: stimuli occur approximately every 200 ms. Earlier
# responses can still overlap because this is a rapid serial presentation task.
# %%
# Extract short visual-response windows
# -------------------------------------------------
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
# Split repeated presentations by image identity
# ----------------------------------------------------------
#
# An image can occur many times in the recording. A random trial split
# would let the decoder see the same image during training and evaluation,
# answering an easier repeated-stimulus question. The seeded permutation instead
# assigns complete image identities: 36 train, 12 validate, and 12 test when all
# 60 are retained. It randomizes only assignments, not signals or labels.
#
# These are within-participant, within-session results. They do not measure
# cross-person or cross-device generalization. Assertions check the image-ID
# boundaries; they cannot erase physiological overlap between successive
# presentations in the original acquisition.

unique = np.unique(labels)
assert len(unique) == 60
np.random.default_rng(2026).shuffle(unique)
first, second = int(0.6 * len(unique)), int(0.8 * len(unique))
train_ids, valid_ids, test_ids = [
    np.sort(group) for group in np.split(unique, [first, second])
]
train, valid, test = [
    np.isin(labels, group) for group in [train_ids, valid_ids, test_ids]
]
assert set(labels[train]).isdisjoint(labels[valid | test])
assert set(labels[valid]).isdisjoint(labels[test])


# %%
# Tune the decoder and rank held-out candidate images
# ---------------------------------------------------------------
#
# Ridge predicts the frozen image coordinates from EEG. Its pipeline estimates
# feature means and scales on training images' responses only. Validation
# retrieval selects among the three prespecified penalties; the selected fitted
# model is then evaluated once on test responses without refitting on validation.
#
# Validation responses rank against the 12 validation images; final test
# responses rank against all 12 test images. Training images are absent from
# both candidate pools. ``searchsorted`` maps the actual image IDs to positions
# in each sorted gallery; those positions match similarity-matrix columns.
# Top-5 counts whether the viewed image occurs among the five highest cosine
# similarities. Its uniform reference here is 5/12, not the official gallery's
# reference. The complete test pool is used for every test trial, irrespective
# of batching. One participant means this is one participant-level proportion;
# a multi-participant evaluation would average participant scores equally.

best_score, best_model = -np.inf, None
for alpha in [1, 10, 100]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X[train], embeddings[labels[train]])
    score = top_k_accuracy_score(
        np.searchsorted(valid_ids, labels[valid]),
        cosine_similarity(model.predict(X[valid]), embeddings[valid_ids]),
        k=5,
        labels=np.arange(len(valid_ids)),
    )
    if score > best_score:
        best_score, best_model = score, model
score = top_k_accuracy_score(
    np.searchsorted(test_ids, labels[test]),
    cosine_similarity(best_model.predict(X[test]), embeddings[test_ids]),
    k=5,
    labels=np.arange(len(test_ids)),
)
print("Encoder:", image_encoder.model_name)
print("Held-out image top-5 accuracy:", score, "test candidates:", len(test_ids))
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["uniform retrieval", "ridge"], [5 / len(test_ids), score])
ax.set(ylabel="Test image top-5 accuracy", ylim=(0, 1))
plt.show()

# %%
# Make the next comparison informative
# ------------------------------------------------
#
# Inspect errors per held-out image before concluding that a representation
# captures visual content. Some images have similar embeddings, and different
# images may have different numbers of usable responses. Averaging predictions
# within image or reporting image-wise scores answers a different evaluation
# question and should be prespecified.
#
# Select ``EEGDASH_DINO_MODEL=small`` before running to compare a compact
# encoder with the official giant target extractor on the same image split.
# Larger target vectors alone do not establish competition performance. The
# next protocol change is to use the release's original train/test image sets,
# train the EEG mapping on separate source recordings, and retain the complete
# held-out gallery. The official NeuralBench preparation step builds its target
# cache; its full-retrieval subject-aggregated metric is the relevant comparison.
#
# Related split-selection example: `Braindecode train, test and tune
# <https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html>`_.
