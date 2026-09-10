"""How do I fine-tune a published pretrained EEG encoder?
======================================================

Adapt the published CBraMod checkpoint to real HBN eyes-open/closed cues.
The checkpoint is https://huggingface.co/braindecode/cbramod-pretrained
(documented by Braindecode's CBraMod model). It was pretrained on TUH EEG;
this example uses the distinct HBN R5 mini cohort. It downloads about 20 MB
of weights and six challenge recordings (roughly 100 MB total).

Three participants train, one selects the epoch, and two are evaluated once.
Compare scratch, a frozen encoder with a learned linear head, and fine-tuning.
The small fixed budget demonstrates adaptation, not a foundation-model benchmark.
"""

# %%
# Before you start
# ----------------
#
# Use EEGDash with Braindecode's Hub and EEGPrep support, PyTorch, NumPy, scikit-learn
# and Matplotlib. The public checkpoint requires network access on first use;
# ``from_pretrained`` caches its weights separately from ``EEGDASH_CACHE_DIR``.
# The latter caches the six EEG recordings. The revision below pins the actual
# checkpoint rather than relying on a changing default branch.
#
# The executable comparison answers a narrow question: what happens when the
# same downstream task is learned from random weights, a frozen published
# representation, or an adaptable published representation? It does not repeat
# the checkpoint's large-scale pretraining. For the architecture and checkpoint
# contract, consult `Braindecode's pretrained-model example
# <https://braindecode.org/dev/auto_examples/model_building/plot_load_pretrained_models.html>`_.

import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from braindecode.models import CBraMod
from braindecode.preprocessing import create_windows_from_events, Resampling
from sklearn.metrics import balanced_accuracy_score

from eegdash import EEGChallengeDataset
from eegdash.const import SUBJECT_MINI_RELEASE_MAP

# %%
# Match the real signal to the encoder input
# ------------------------------------------
#
# E70, E75 and E83 are an explicit small posterior-channel subset, not a
# claim that the complete HBN montage is interchangeable with the pretraining
# montage. The source challenge derivative has already been filtered and sampled
# at 100 Hz. EEGPrep's ``Resampling`` adapter converts to 200 Hz to satisfy
# CBraMod's one-second, 200-sample patch
# contract; it cannot recover information above the source Nyquist frequency.
#
# Only the two seconds from one to three seconds after an eye-state instruction
# are used. That offset avoids the immediate instruction onset. Labels represent
# the observed open/close instructions, not eye tracking. MNE supplies volts;
# ``float32`` microvolt conversion gives the encoder the intended amplitude unit
# without estimating a test-set normalization.

# The EEGPrep adapter converts through EEGLAB and can round event times or
# discard the measurement date. Because this step only resamples, we retain
# the original date and annotations, verify the origin, duration, channel order
# and new rate, then restore the observed event times in seconds. The final
# onset assertion prevents a silent cue shift from changing window labels.

subjects = sorted(SUBJECT_MINI_RELEASE_MAP["R5"])[:6]
dataset = EEGChallengeDataset(
    release="R5",
    mini=True,
    task="RestingState",
    subject=subjects,
    cache_dir=Path(
        os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")
    ).expanduser(),
)
print(dataset.description.to_string(index=False))
for recording in dataset.datasets:
    raw = recording.raw.pick(["E70", "E75", "E83"])
    print(
        recording.description.subject,
        raw.ch_names,
        raw.info["sfreq"],
        np.unique(raw.annotations.description),
    )
    # CBraMod uses 200-sample, one-second patches and microvolt inputs.
    # Upsampling the 100 Hz derivative does not restore frequencies above 50 Hz.
    annotations = raw.annotations.copy()
    measurement_date = raw.info["meas_date"]
    recording_duration = raw.n_times / raw.info["sfreq"]
    assert raw.first_samp == 0
    Resampling(sfreq=200).apply(raw.load_data())
    assert raw.ch_names == ["E70", "E75", "E83"] and raw.info["sfreq"] == 200
    assert raw.first_samp == 0
    assert abs(raw.n_times / raw.info["sfreq"] - recording_duration) <= 1 / 200
    # EEGPrep's file conversion can round annotation times; retain source seconds.
    raw.set_meas_date(measurement_date)
    raw.set_annotations(annotations)
    np.testing.assert_allclose(
        raw.annotations.onset, annotations.onset, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        raw.annotations.duration, annotations.duration, atol=1 / 200, rtol=0
    )
    np.testing.assert_array_equal(raw.annotations.description, annotations.description)
windows = create_windows_from_events(
    dataset,
    mapping={"instructed_toOpenEyes": 0, "instructed_toCloseEyes": 1},
    trial_start_offset_samples=200,
    trial_stop_offset_samples=600,
    window_size_samples=400,
    window_stride_samples=400,
    preload=True,
)
# %%
# Reserve participants before selecting an epoch
# ----------------------------------------------
#
# ``X`` has shape ``(windows, 3, 400)`` and ``y`` contains the two integer
# instruction classes. Participants, rather than windows, determine the masks:
# the first three train, the fourth validates, and the remaining two test. The
# assertions require disjoint identities and both observed classes in each split.
# A missing class is a problem with this small design, not something to repair by
# inventing examples.
#
# All checkpoint selection uses validation balanced accuracy. Test labels are
# read only for the final score of each prespecified regime. Reporting several
# regimes is not permission to choose a winning configuration on test performance
# and call that a new independent result.

metadata = windows.get_metadata()
X = np.stack([windows[i][0] for i in range(len(windows))]).astype("float32") * 1e6
# The source preprocessing leaves some electrodes flat; no test-derived scaling.
y = metadata.target.to_numpy(dtype="int64")
train = metadata.subject.isin(subjects[:3]).to_numpy()
valid = metadata.subject.eq(subjects[3]).to_numpy()
test = metadata.subject.isin(subjects[4:]).to_numpy()
assert set(metadata.subject[train]).isdisjoint(metadata.subject[valid])
assert set(metadata.subject[test]).isdisjoint(metadata.subject[train | valid])
assert all(mask.any() and set(y[mask]) == {0, 1} for mask in [train, valid, test])
assert np.isfinite(X).all()
print("Windows:", X.shape, "observed class counts:", np.unique(y, return_counts=True))

# %%
torch.manual_seed(73)
torch.set_num_threads(2)
# %%
# Load the checkpoint and understand the head dimensions
# ------------------------------------------------------
#
# ``return_encoder_output=True`` exposes CBraMod's patch representations.
# For three channels and two one-second patches, each with 200 representation
# coordinates, flattening gives ``3 × 2 × 200 = 1200`` inputs to the two-class
# linear head. The head returns logits; cross-entropy consumes logits directly,
# so no softmax is inserted in the training loop.
#
# If you change channels or duration, derive the new head width from a real
# encoder forward pass. Merely changing ``n_outputs`` cannot fix a mismatched
# feature shape. The same principle is illustrated in Braindecode's
# `foundation-model fine-tuning walkthrough
# <https://braindecode.org/dev/auto_examples/advanced_training/plot_finetune_foundation_model.html>`_.

pretrained = CBraMod.from_pretrained(
    "braindecode/cbramod-pretrained",
    revision="584cdc415913739a05d84bf0c1cb3db397764507",
    return_encoder_output=True,
    n_chans=3,
    n_times=400,
    sfreq=200,
)
# %%
# Compare optimization regimes with validation-only selection
# -----------------------------------------------------------
#
# The scratch model has the same encoder architecture but no downloaded
# weights. The probe freezes encoder parameters and explicitly sets that encoder
# to evaluation mode after ``model.train()``. Freezing gradients alone would not
# stop dropout or running-statistic updates. Fine-tuning allows both encoder and
# head to change.
#
# AdamW uses a fixed ``1e-4`` learning rate and its default weight decay.
# Approximately eight examples per batch and two candidate epochs keep CPU
# execution small; they are not a recommended training budget. A copied state
# dictionary preserves the best validation epoch, with the first epoch retained
# on a tie. Calling ``eval()`` and ``no_grad()`` makes evaluation deterministic
# with respect to dropout and avoids storing an unnecessary gradient graph.
#
# The three heads are initialized sequentially under one seed, so this is not a
# paired multi-seed estimate of transfer effect. All three nevertheless use the
# same observed windows and participant masks.

results = {}
for regime in ["scratch", "linear probe", "fine-tune"]:
    encoder = (
        CBraMod(n_chans=3, n_times=400, sfreq=200, return_encoder_output=True)
        if regime == "scratch"
        else copy.deepcopy(pretrained)
    )
    model = torch.nn.Sequential(encoder, torch.nn.Flatten(), torch.nn.Linear(1200, 2))
    if regime == "linear probe":
        encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    best_score, best_state = -np.inf, None
    for epoch in range(2):
        model.train()
        if regime == "linear probe":
            encoder.eval()  # Freeze dropout and running statistics as well as gradients.
        for indices in np.array_split(np.flatnonzero(train), max(1, train.sum() // 8)):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                model(torch.from_numpy(X[indices])), torch.from_numpy(y[indices])
            )
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = model(torch.from_numpy(X[valid])).argmax(1).numpy()
        score = balanced_accuracy_score(y[valid], predicted)
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(X[test])).argmax(1).numpy()
    results[regime] = balanced_accuracy_score(y[test], predicted)
    print(
        regime,
        "validation balanced accuracy:",
        best_score,
        "final held-out participants:",
        results[regime],
    )
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(list(results), list(results.values()))
ax.set(ylabel="Held-out participant pooled balanced accuracy", ylim=(0, 1))
plt.show()

# %%
# Read the final bars and design the next experiment
# --------------------------------------------------
#
# Balanced accuracy averages recall for the two classes; 0.5 is the
# uniform two-class reference. Here windows from both test participants are
# pooled before computing recall, so a participant with more windows contributes
# more observations. The result is not a mean of separate participant scores.
# The validation participant is also too small to support a stable ranking of
# epochs or architectures.
#
# For a stronger comparison, retain the test participants and add validation
# participants, longer prespecified budgets and repeated seeds. Compare equal
# head initializations when estimating the effect of pretrained weights. Inspect
# per-participant confusion matrices and channel quality before attributing a
# change to the encoder. Save the selected weights together with the checkpoint
# revision, channel order, rate, window offsets and subject lists if you extend
# this page into a reusable trained model; weights alone omit the input contract.
