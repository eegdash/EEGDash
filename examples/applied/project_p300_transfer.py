"""P300 transfer between participants with MMD adaptation
======================================================

Train on real visual-oddball EEG from subject 054, adapt using unlabelled
EEG from subject 119, and test on subject 123. EEGDashDataset downloads
these three ``ds005863`` recordings (about 69 MB total). Set
``EEGDASH_CACHE_DIR`` to reuse them. CPU is sufficient for this small model. Install
``eegprep[eeglabio]>=0.2.23,<0.3`` for the EEGPrep stages.

Compare source-only training, maximum mean discrepancy (MMD) adaptation,
and a supervised target-domain reference. These are participants from the
same dataset, not separate laboratories. This is an MMD penalty, not an
adversarial method; no discriminator is trained. One held-out participant
cannot establish that adaptation helps the wider population.

This project builds on the visual P300 tutorial and assumes familiarity with
MNE epochs, participant splits and a PyTorch training loop. Install EEGDash
with its PyTorch dependencies and run the blocks in order. You will obtain
three training-objective curves and a held-out balanced-accuracy comparison.
The `source dataset <https://openneuro.org/datasets/ds005863>`_ supplies both
the voltages and the stimulus codes; only the training regime changes.

The key distinction is access to labels. Source-only training sees labels
from 054. MMD also sees signals from 119, but its labels are withheld from the
loss. The supervised reference uses labels from 119. All three models are
evaluated on 123, whose signals and labels are excluded from fitting.
"""

# %%
# 1. Load real stimulus-labelled epochs with matching channels
# ------------------------------------------------------------
# Use the same five named channels in the same order for every participant.
# EEGPrep removes channel-median offsets and applies its common-average
# reference over the EEG sensors before this subset is selected. The adapters
# leave the sample grid unchanged; the explicit check lets us restore the
# original annotations without accumulating conversion-rounding errors. The 0.5–30 Hz filter and -100..0 ms baseline are fixed
# across participants, so a preprocessing choice is not selected from test
# accuracy. These operations respect the original participant boundaries.
#
# In recorded ``SXY`` markers, matching digits identify a target and unequal
# digits identify a standard; only stimulus digits 1..5 are accepted. The
# target becomes class 1 and the standard class 0 after subtracting one.
# Epochs contain -100..800 ms around each actual stimulus onset.
#
# Resampling to 64 Hz keeps this dense model small. An epoch array starts
# with axes (trials, channels, time); flattening joins channel and time into
# the feature axis while retaining the trial axis. Multiplication by 1e6
# converts volts to microvolts before training-only standardization. This
# simple representation uses voltages directly rather than engineered P300
# amplitudes, but it does not exploit convolutional time structure.
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from braindecode.preprocessing import RemoveCommonAverageReference, RemoveDCOffset
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from eegdash import EEGDashDataset

cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache"))
subjects = ["054", "119", "123"]
channels = ["Fz", "CP1", "Pz", "P3", "P4"]
dataset = EEGDashDataset(
    cache_dir=cache_dir,
    dataset="ds005863",
    subject=subjects,
    task="visualoddball",
    n_jobs=1,
)
assert len(dataset.datasets) == 3
cohort = {}
for recording in dataset.datasets:
    raw = recording.raw.copy().load_data().pick("eeg")
    assert set(channels) <= set(raw.ch_names)
    source_annotations = raw.annotations.copy()
    source_date = raw.info["meas_date"]
    source_grid = (raw.info["sfreq"], raw.n_times, raw.first_samp)
    RemoveDCOffset().apply(raw)
    RemoveCommonAverageReference().apply(raw)
    assert (raw.info["sfreq"], raw.n_times, raw.first_samp) == source_grid
    raw.set_meas_date(source_date)
    raw.set_annotations(source_annotations)
    raw.filter(0.5, 30)
    raw.pick(channels)
    mapping = {}
    for name in set(raw.annotations.description):
        code = name.split("/")[-1].replace(" ", "")
        if len(code) == 3 and code[0] == "S" and set(code[1:]) <= set("12345"):
            mapping[name] = 2 if code[1] == code[2] else 1
    assert set(mapping.values()) == {1, 2}
    events, _ = mne.events_from_annotations(raw, event_id=mapping)
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"standard": 1, "target": 2},
        tmin=-0.1,
        tmax=0.8,
        baseline=(-0.1, 0),
        preload=True,
    )
    epochs.resample(64)
    X = epochs.get_data().reshape(len(epochs), -1) * 1e6
    y = epochs.events[:, 2] - 1
    assert np.isfinite(X).all() and set(y) == {0, 1}
    cohort[str(recording.description["subject"])] = (X, y)
assert set(cohort) == set(subjects)
source_subject, adaptation_subject, test_subject = subjects
assert len({source_subject, adaptation_subject, test_subject}) == 3

# %%
# 2. Define the model and a distribution discrepancy
# --------------------------------------------------
# A small neural encoder operates on flattened voltage windows. The biased
# RBF MMD estimator compares hidden activations, including kernel diagonals.
# Its bandwidth and weight are fixed; they are not selected using test scores.
# Model initialization and minibatch ordering are random, but the EEG and
# targets are entirely recorded observations.
#
# MMD is the mean source–source similarity plus mean adaptation–adaptation
# similarity, minus twice the mean cross-participant similarity. The Gaussian
# kernel gives similar hidden vectors larger similarity. Minimizing this term
# encourages the two marginal hidden distributions to overlap; it does not
# guarantee that target and standard trials align with their correct classes.
#
# The 16-unit encoder and two-output head below map a voltage vector into
# class logits. The 0.1 loss weight and unit kernel bandwidth are fixed
# demonstration choices, not tuned values. A collapsed representation can
# have low discrepancy without being useful, which is why the labelled-source
# classification loss remains part of the objective.
torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))


# %%
# 3. Fit each regime without using the test participant
# -----------------------------------------------------
# Source-only and MMD use source labels. Only the supervised target reference
# uses adaptation-subject labels. Scaling is fitted on the labelled training
# participant for each regime. Test data are transformed/predicted only after
# training; their signals are not part of the MMD penalty.
#
# Weighted cross-entropy compensates for the rarer target class using counts
# from the labelled training participant. Adam updates the encoder and head
# together, using up to 32 labelled trials per minibatch and ten passes through
# training data. MMD pairs each labelled minibatch with a randomly selected
# minibatch of real adaptation trials; the observations themselves are unchanged.
# There is no early stopping or validation-based hyperparameter search here.
#
# Resetting the seed gives source-only and MMD the same initial model, although
# their later minibatch orders can differ as MMD makes additional random draws.
# The comparison is a single demonstration run, not an estimate across seeds.
rows = []
histories = {}
for regime in ["source only", "MMD", "target supervised"]:
    training_subject = (
        adaptation_subject if regime == "target supervised" else source_subject
    )
    X_train, y_train = cohort[training_subject]
    scaler = StandardScaler().fit(X_train)
    train_X = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)
    adaptation_X = torch.tensor(
        scaler.transform(cohort[adaptation_subject][0]), dtype=torch.float32
    )
    torch.manual_seed(42)
    encoder = nn.Sequential(nn.Linear(X_train.shape[1], 16), nn.ELU())
    head = nn.Linear(16, 2)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=1e-3
    )
    weights = torch.tensor(
        len(y_train) / (2 * np.bincount(y_train)), dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    history = []
    for epoch in range(10):
        losses = []
        for indices in torch.randperm(len(train_X)).split(32):
            optimizer.zero_grad()
            hidden = encoder(train_X[indices])
            loss = criterion(head(hidden), train_y[indices])
            if regime == "MMD":
                target_indices = torch.randperm(len(adaptation_X))[: len(indices)]
                adaptation_hidden = encoder(adaptation_X[target_indices])
                discrepancy = (
                    torch.exp(-torch.cdist(hidden, hidden).square() / 2).mean()
                    + torch.exp(
                        -torch.cdist(adaptation_hidden, adaptation_hidden).square() / 2
                    ).mean()
                    - 2
                    * torch.exp(
                        -torch.cdist(hidden, adaptation_hidden).square() / 2
                    ).mean()
                )
                loss = loss + 0.1 * discrepancy
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    encoder.eval()
    head.eval()
    with torch.no_grad():
        test_X = torch.tensor(
            scaler.transform(cohort[test_subject][0]), dtype=torch.float32
        )
        prediction = head(encoder(test_X)).argmax(dim=1).numpy()
    rows.append(
        {
            "regime": regime,
            "balanced_accuracy": balanced_accuracy_score(
                cohort[test_subject][1], prediction
            ),
        }
    )
    histories[regime] = history

# %%
# 4. Report the observed comparison
# ---------------------------------
# The target-supervised model is a reference, not a guaranteed upper bound.
# Negative transfer and below-chance results remain valid outcomes.
# Binary balanced accuracy is the mean of target and standard recall, so an
# always-standard classifier scores 0.5 even though standards are more common.
# Compare the MMD and source-only bars on this same held-out participant. A
# lower MMD score is negative transfer for this run, not an execution failure.
#
# Training objectives have different definitions: the MMD curve includes the
# discrepancy penalty, whereas the other curves show classification loss
# alone. Their absolute heights are therefore not comparable model-quality
# scores. Decreasing training loss also does not establish generalization.
results = pd.DataFrame(rows)
print(results.to_string(index=False))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for name, history in histories.items():
    axes[0].plot(range(1, len(history) + 1), history, label=name)
axes[0].set(xlabel="Epoch", ylabel="Training objective")
axes[0].legend()
axes[1].bar(results["regime"], results["balanced_accuracy"])
axes[1].axhline(0.5, color="black", linestyle="--")
axes[1].set(ylabel="Balanced accuracy on subject 123", ylim=(0, 1))
plt.show()

# %%
# 5. Separate model selection from a broader transfer study
# ---------------------------------------------------------
# To test the approach beyond this example, add participants and define
# several source/adaptation/test assignments before examining scores. Reserve
# additional validation participants if choosing the kernel, adaptation weight
# or training duration, then evaluate the selected configuration on untouched
# participants. Record several training seeds to distinguish optimization
# variation from participant variation. The supervised reference answers what
# labelled adaptation data can provide; it is not an upper bound on accuracy.
