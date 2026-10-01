"""Exploratory eye-state decoding in an outcome-selected participant
==================================================================

Train ShallowFBCSPNet on real two-second EEG windows from HBN participant
NDARAC589YMB (ds005514, about 90 MB). Install EEGDash, Braindecode,
PyTorch, MNE, scikit-learn and Matplotlib. CPU training uses 30 epochs;
wall time depends on hardware. ``EEGDASH_CACHE_DIR`` retains the recording.

This participant was selected after inspecting outcomes across twelve people
and several pipeline settings. Neither a fresh block split nor scoring once
removes that selection bias. The result is an exploratory workflow illustration,
not unbiased evidence of within-person or population performance. Prespecify
participants, preprocessing and development/validation blocks for a new study.

We retain original instruction-block identities, reject BAD-annotated epochs,
and score reserved blocks only after a fixed training budget. Filtering is
offline, not a causal real-time deployment pipeline. The labels describe the
instructions, not verified compliance with eye opening/closing.
"""

# %%
# Load one recording and retain its actual instruction boundaries
# ---------------------------------------------------------------
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from braindecode.models import ShallowFBCSPNet
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir

dataset = EEGDashDataset(
    cache_dir=get_default_cache_dir(),
    dataset="ds005514",
    task="RestingState",
    subject="NDARAC589YMB",
)
if len(dataset.datasets) != 1:
    raise ValueError(
        "Expected one RestingState recording; inspect session/run metadata."
    )
raw = dataset.datasets[0].raw.copy().load_data()
channels = [
    "E22",
    "E9",
    "E33",
    "E24",
    "E11",
    "E124",
    "E122",
    "E29",
    "E6",
    "E111",
    "E45",
    "E36",
    "E104",
    "E108",
    "E42",
    "E55",
    "E93",
    "E58",
    "E52",
    "E62",
    "E92",
    "E96",
    "E70",
    "Cz",
]
raw.pick(channels).reorder_channels(channels).resample(128).filter(1, 55)
# Keep the original annotations, including BAD spans. Each new epoch carries
# the index of its original instruction, rather than inferring blocks by // 7.
cue_specs = {"instructed_toCloseEyes": (0, 15), "instructed_toOpenEyes": (1, 5)}
events, block_ids = [], []
for block_id, annotation in enumerate(raw.annotations):
    if annotation["description"] not in cue_specs:
        continue
    label, delay = cue_specs[annotation["description"]]
    onset = annotation["onset"] - raw.first_time
    for offset in np.arange(delay, delay + 14, 2):
        sample = int(round((onset + offset) * raw.info["sfreq"])) + raw.first_samp
        events.append([sample, 0, label + 1])
        block_ids.append(block_id)
if not events:
    raise ValueError(
        "No eye-state instructions found; inspect the source event vocabulary."
    )
order = np.argsort(np.asarray(events)[:, 0])
events = np.asarray(events)[order]
block_ids = np.asarray(block_ids)[order]
epochs = mne.Epochs(
    raw,
    events,
    event_id={"eyes_closed": 1, "eyes_open": 2},
    tmin=0,
    tmax=2 - 1 / 128,
    baseline=None,
    preload=True,
    reject_by_annotation=True,
)
X = epochs.get_data().astype(np.float32) * 1e6
y = epochs.events[:, 2] - 1
blocks = block_ids[epochs.selection]
if not len(X) or not np.isfinite(X).all() or set(y) != {0, 1}:
    raise ValueError(
        "Need finite retained epochs in both eye conditions; inspect the drop log."
    )
print("Retained / requested epochs:", len(epochs), len(events))
print("Drop reasons:", epochs.drop_log)

# %%
# Inspect the waveform, spectral representation and original block coverage
# -------------------------------------------------------------------------
posterior = channels.index("E70")
fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
for code, name in enumerate(["eyes_closed", "eyes_open"]):
    first = np.flatnonzero(y == code)[0]
    axes[0].plot(epochs.times, X[first, posterior], label=name)
    spectrum = epochs[name].compute_psd(fmin=1, fmax=40, picks=["E70"], verbose=False)
    axes[1].semilogy(
        spectrum.freqs, spectrum.get_data().mean(axis=(0, 1)) * 1e12, label=name
    )
axes[0].set(xlabel="Time within epoch (s)", ylabel="E70 (µV)")
axes[1].axvspan(8, 13, alpha=0.15, color="gray")
axes[1].set(xlabel="Frequency (Hz)", ylabel="Mean E70 PSD (µV²/Hz)")
axes[0].legend()
axes[1].legend()
plt.show()

# %%
# Split complete instructions, including after any epoch rejection
# ----------------------------------------------------------------
# Amplitudes remain in microvolts: this example does not standardize every
# window separately. The spectra above allow inspection without assuming an
# alpha difference must occur. All settings are fixed before this run's score.
unique_blocks = np.unique(blocks)
block_labels = np.asarray([y[blocks == block][0] for block in unique_blocks])
if np.bincount(block_labels, minlength=2).min() < 2:
    raise ValueError(
        "Need at least two retained instruction blocks per class for this split."
    )
train_blocks, test_blocks = train_test_split(
    unique_blocks,
    test_size=max(2, int(np.ceil(0.2 * len(unique_blocks)))),
    stratify=block_labels,
    random_state=42,
)
train, test = np.isin(blocks, train_blocks), np.isin(blocks, test_blocks)
print("Training blocks:", train_blocks, "reserved blocks:", test_blocks)
print("Training / reserved windows:", train.sum(), test.sum())
fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
for mask, name, marker in [(train, "Training", "o"), (test, "Reserved", "x")]:
    ax.scatter(
        (epochs.events[mask, 0] - raw.first_samp) / raw.info["sfreq"],
        blocks[mask],
        marker=marker,
        label=name,
    )
ax.set(xlabel="Time from recording start (s)", ylabel="Original instruction ID")
ax.legend()
plt.show()

# %%
# Train for a fixed budget without monitoring reserved blocks
# -----------------------------------------------------------
torch.manual_seed(42)
torch.set_num_threads(2)
loader = DataLoader(
    TensorDataset(torch.from_numpy(X[train]), torch.from_numpy(y[train]).long()),
    batch_size=10,
    shuffle=True,
)
model = ShallowFBCSPNet(len(channels), 2, n_times=X.shape[-1], final_conv_length="auto")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
history = []
for epoch in range(30):
    model.train()
    losses = []
    for signals, targets in loader:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(signals), targets)
        if not torch.isfinite(loss):
            raise ValueError(
                "Nonfinite loss: inspect signal amplitudes and learning rate."
            )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    history.append(float(np.mean(losses)))
model.eval()
with torch.inference_mode():
    predicted = model(torch.from_numpy(X[test])).argmax(dim=1).numpy()
print(
    "Exploratory reserved-block balanced accuracy:",
    balanced_accuracy_score(y[test], predicted),
)
print("Constant-class balanced-accuracy reference: 0.5")
fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
ax.plot(np.arange(1, 31), history)
ax.set(xlabel="Epoch", ylabel="Training cross-entropy")
plt.show()
ConfusionMatrixDisplay.from_predictions(
    y[test], predicted, display_labels=["Closed", "Open"]
)
plt.title("Outcome-selected participant: exploratory reserved blocks")
plt.show()
