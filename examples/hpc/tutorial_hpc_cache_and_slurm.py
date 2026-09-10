"""Eyes-open versus eyes-closed decoding on a cluster
==================================================

Load actual HBN ``ds005514`` RestingState recordings with EEGDashDataset,
reannotate observed open/close instructions, and train a neural classifier.
The default selects three explicit, previously inspected subject IDs;
all selected subjects must load successfully. The three signal files total
281.2 MB plus sidecars. See `HBN ds005514
<https://openneuro.org/datasets/ds005514>`_. Inspect sizes before staging.
This is a full EO/EC workload, not the small real-data CI smoke test.

From the repository root, submit ``sbatch examples/hpc/run_eoec_cpu.slurm``
after configuring the cluster account/partition and Python environment.
Use ``EEGDASH_CACHE_DIR`` for persistent or pre-staged node-local cache,
``NUM_SUBJECTS=3``, ``NUM_TEST_SUBJECTS=1`` and ``SEED=42`` by default.
The test subjects remain untouched until the fixed training schedule ends.
The single split is a workflow demonstration, not a population benchmark.

Prerequisites: tutorial 02's DataLoader, tutorial 11's subject split and
basic PyTorch training. Install the project dependencies, including a
PyTorch build appropriate for the compute node, plus EEGPrep and its
EEGLAB reader (``pip install 'eegprep[eeglabio]>=0.2.23,<0.3'``). A CPU can run the default
cohort; the GPU Slurm template requests a GPU but training selects CUDA only
when PyTorch can access one. Cluster accounts and partitions are site-specific.

The deliverables are an actual voltage-window plot and a final held-out
accuracy in the job log. The script does not save a reusable trained model;
write model state and the preprocessing/subject configuration explicitly
if a later inference job needs them.

"""

from pathlib import Path
import os

import numpy as np
import torch


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_USE_NUMBA", "false")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path.cwd()))
(Path(os.environ["_MNE_FAKE_HOME_DIR"]) / ".mne").mkdir(exist_ok=True)

from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    RemoveDCOffset,
    RemoveDrifts,
    Resampling,
    create_windows_from_events,
)
from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation


# %%
# 1. Configure the bounded real-data workload
# -------------------------------------------
cache_folder = get_default_cache_dir()
cache_folder.mkdir(parents=True, exist_ok=True)
dataset_id = "ds005514"
task = "RestingState"

# Explicit upper bound; failed acquisitions stop the job.
num_subjects = int(os.environ.get("NUM_SUBJECTS", "3"))
num_test_subjects = int(os.environ.get("NUM_TEST_SUBJECTS", "1"))
random_state = int(os.environ.get("SEED", "42"))

# 2 seconds at 128 Hz
window_size_samples = 256

# training params
epochs = int(os.environ.get("EPOCHS", "6"))
batch_size = int(os.environ.get("BATCH_SIZE", "32"))


# %%
# 2. Define preprocessing and observed-instruction reannotation
# -------------------------------------------------------------
# The HBN-specific preprocessor derives windows from recorded instructions.
# For eyes-closed it starts two-second segments 15–27 seconds after the close
# cue; for eyes-open it starts them 5–17 seconds after the open cue. This avoids
# immediate cue transitions. The labels indicate instructed state rather than
# an independent measurement that a participant complied.
#
# We keep a fixed 24-channel montage subset so the neural model sees identical
# sensor ordering. EEGPrep RemoveDCOffset subtracts the per-channel median;
# RemoveDrifts applies a fixed forward/backward high-pass transition from
# 0.5 to 1 Hz. EEGPrep Resampling reduces the rate to 128 Hz with anti-aliasing.
# A final 55 Hz low-pass avoids frequencies near the new 64 Hz Nyquist limit.
# These deterministic per-recording operations use no population-fitted state.
# Original instruction times in seconds are restored after verifying that
# resampling preserved the time origin and duration within one output sample.
# Reannotation then creates state markers on the resampled grid. This is
# component-level EEGPrep preprocessing, not automatic ASR/artifact rejection.
preprocessors = [
    Preprocessor(
        "pick_channels",
        ch_names=[
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
        ],
    ),
    RemoveDCOffset(),
    RemoveDrifts(transition=(0.5, 1.0)),
    Resampling(sfreq=128),
    Preprocessor("filter", l_freq=None, h_freq=55),
]


# %%
# 3. Load the selected participants and create real EO/EC windows
# ---------------------------------------------------------------
# SUBJECTS is an explicit comma-separated cohort. NUM_SUBJECTS selects a prefix
# of that list and must not exceed it. Increase both settings when adding
# participants; the script never searches for replacements after an error.
# The default subjects were inspected for both instruction labels and the
# requested channel names. Inspect those contracts again for a new cohort.
subjects_all = os.environ.get(
    "SUBJECTS", "NDARAE710YWG,NDARAH239PGG,NDARAL897CYV"
).split(",")
assert len(set(subjects_all)) == len(subjects_all)

all_windows = []
all_subject_ids = []
valid_subjects = []

assert 2 <= num_subjects <= len(subjects_all)
assert 1 <= num_test_subjects < num_subjects
selected_subjects = subjects_all[:num_subjects]
print("Selected subjects:", selected_subjects)
for subj in selected_subjects:
    ds_eoec = EEGDashDataset(
        dataset=dataset_id,
        task=task,
        subject=subj,
        cache_dir=cache_folder,
    )
    print(ds_eoec.description)
    original_annotations = []
    original_timing = []
    for recording in ds_eoec.datasets:
        raw = recording.raw
        original_annotations.append(raw.annotations.copy())
        original_timing.append(
            (raw.first_time, raw.n_times / raw.info["sfreq"], raw.info["meas_date"])
        )
        print(raw.ch_names, raw.info["sfreq"], np.unique(raw.annotations.description))
        assert {"instructed_toCloseEyes", "instructed_toOpenEyes"}.issubset(
            raw.annotations.description
        )
    preprocess(ds_eoec, preprocessors)
    # These EEGPrep components never remove time. Preserve source cue times
    # across EEGLAB/MNE conversion rather than accepting sample-origin shifts.
    for recording, annotations, (origin, duration, measurement_date) in zip(
        ds_eoec.datasets, original_annotations, original_timing, strict=True
    ):
        raw = recording.raw
        assert abs(raw.first_time - origin) <= 1 / raw.info["sfreq"]
        assert abs(raw.n_times / raw.info["sfreq"] - duration) <= 1 / raw.info["sfreq"]
        raw.set_meas_date(measurement_date)
        raw.set_annotations(annotations)
        np.testing.assert_allclose(
            raw.annotations.onset, annotations.onset, atol=1 / raw.info["sfreq"], rtol=0
        )
        np.testing.assert_array_equal(
            raw.annotations.description, annotations.description
        )
    preprocess(ds_eoec, [hbn_ec_ec_reannotation()])
    # Each derived annotation is a point marker. The stop offset extends it by
    # 256 samples (two seconds after resampling), while the matching size/stride
    # creates one full window per marker. This differs from a source annotation
    # that already carries a nonzero trial duration: do not add the same offset
    # blindly to another dataset.
    windows_ds = create_windows_from_events(
        ds_eoec,
        mapping={"eyes_closed": 0, "eyes_open": 1},
        trial_start_offset_samples=0,
        trial_stop_offset_samples=window_size_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        on_last_window="drop",
        preload=True,
    )
    n_win = len(windows_ds)
    assert n_win > 0, f"No valid windows for {subj}"
    assert set(windows_ds.get_metadata().target) == {0, 1}
    print("Windows for subject:", n_win)
    all_windows.append(windows_ds)
    all_subject_ids.extend([subj] * n_win)
    valid_subjects.append(subj)

if len(valid_subjects) < 2:
    raise RuntimeError(
        f"Only {len(valid_subjects)} valid subject(s) collected; need >=2."
    )

if num_test_subjects >= len(valid_subjects):
    raise ValueError("NUM_TEST_SUBJECTS must be < number of valid subjects found.")

print("\nUsing valid subjects:", valid_subjects)
print("Total subjects requested:", num_subjects, " | collected:", len(valid_subjects))

# Concatenate
from braindecode.datasets import BaseConcatDataset

concat_ds = BaseConcatDataset(all_windows)
print("Total windows across valid subjects:", len(concat_ds))


# %%
# 4. Inspect one recorded window without a display server
# -------------------------------------------------------
# The saved trace is channel zero of a real window, in volts against sample
# index. It checks waveform extraction, not class separability. Divide the
# horizontal indices by 128 when interpreting elapsed seconds.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(concat_ds) > 2:
    plt.figure()
    plt.plot(concat_ds[2][0][0, :].transpose())
    plt.savefig("sample_epoch.png", dpi=150, bbox_inches="tight")
    print("Saved plot to sample_epoch.png")


# %%
# 5. Keep the test participant outside training
# ---------------------------------------------
# Shuffle participant IDs once, not windows across participants. The default
# seed assigns one person to test and two to train; all windows from each
# person stay together. Multiple windows from the same instructed block are
# correlated, so window count is not the number of independent participants.
rng = np.random.RandomState(random_state)
subjects_shuffled = valid_subjects.copy()
rng.shuffle(subjects_shuffled)

test_subjects = set(subjects_shuffled[:num_test_subjects])
train_subjects = set(subjects_shuffled[num_test_subjects:])
assert train_subjects.isdisjoint(test_subjects)

print("\nTrain subjects:", sorted(train_subjects))
print("Test subjects :", sorted(test_subjects))

indices = np.arange(len(concat_ds))
subj_arr = np.array(all_subject_ids)

train_indices = indices[np.isin(subj_arr, list(train_subjects))]
test_indices = indices[np.isin(subj_arr, list(test_subjects))]

print("Train windows:", len(train_indices), "Test windows:", len(test_indices))


# %%
# 6. Build training and test batches
# ----------------------------------
# Each input has axes (windows, 24 channels, 256 samples) and starts in volts.
# The default cohort has 140 training and 70 test windows. Labels are integer
# class indices: zero for eyes-closed, one for eyes-open. A label mean of 0.5
# means equal counts, not an accuracy result. Only training batches shuffle.
# Materializing arrays is practical here; a much larger cohort should use
# indexed datasets/lazy loaders rather than duplicating every window in RAM.
torch.manual_seed(random_state)
np.random.seed(random_state)

X_train = torch.FloatTensor(np.array([concat_ds[i][0] for i in train_indices]))
X_test = torch.FloatTensor(np.array([concat_ds[i][0] for i in test_indices]))
y_train = torch.LongTensor(np.array([concat_ds[i][1] for i in train_indices]))
y_test = torch.LongTensor(np.array([concat_ds[i][1] for i in test_indices]))

from torch.utils.data import DataLoader, TensorDataset

assert torch.isfinite(X_train).all() and torch.isfinite(X_test).all()
assert set(y_train.tolist()) == set(y_test.tolist()) == {0, 1}
dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

print(
    f"X_train {X_train.shape} | Train batches: {len(train_loader)} | Test batches: {len(test_loader)}"
)
print(
    f"Label balance train: {float(y_train.float().mean()):.2f} | test: {float(y_test.float().mean()):.2f}"
)


# %%
# 7. Match the neural model to the window contract
# ------------------------------------------------
# ShallowFBCSPNet combines temporal/spatial filtering with pooled power-like
# features. The constructor specifies 24 channels, two output scores and
# 256 time samples. final_conv_length="auto" derives the final temporal
# kernel from that input length; changing windows requires changing n_times.
from torch.nn import functional as F
from braindecode.models import ShallowFBCSPNet

model = torch.nn.Sequential(
    torch.nn.LayerNorm(256, eps=1e-14, elementwise_affine=False),
    ShallowFBCSPNet(24, 2, n_times=256, final_conv_length="auto"),
)


# %%
# 8. Train for a fixed schedule and reserve the final evaluation
# --------------------------------------------------------------
# Adamax uses a fixed learning rate of 0.002 and weight decay of 0.001. Six
# epochs are an inexpensive workflow check, not a tuned convergence claim.
# Changing these choices after reading test accuracy uses the test person
# for selection; instead reserve validation people inside the training cohort.
optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, weight_decay=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)

print("Using epochs =", epochs, "| device =", device, "| batch_size =", batch_size)


# %%
# Normalization is per window and channel, along the time axis. The model
# therefore consumes dimensionless standardized signals. Native LayerNorm
# divides by sqrt(population variance + 1e-14 V²), with no learned affine
# parameters. This replaces a custom normalization helper and keeps division
# finite for nearly constant channels. This transform does not
# estimate a shared mean or variance across participants, so applying it to a
# test window does not expose other test windows to the trained model.
#
# Each forward pass produces two class scores per window. Cross-entropy
# compares them with the observed integer target; backpropagation updates only
# the training model. model.train() enables training behavior such as dropout
# and BatchNorm updates. The final model.eval() and no_grad() disable those
# updates and gradient tracking for the reserved participant.
for e in range(epochs):
    model.train()
    correct_train = 0.0
    for x, y in train_loader:
        scores = model(x.to(device=device, dtype=torch.float32))
        y = y.to(device=device, dtype=torch.long)

        loss = F.cross_entropy(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = scores.argmax(dim=1)
        correct_train += (preds == y).sum().item()

    print(f"Epoch {e}, train accuracy: {correct_train / len(dataset_train):.3f}")

# Evaluate the held-out subjects once after the fixed training schedule.
model.eval()
correct_test = 0
with torch.no_grad():
    for x, y in test_loader:
        scores = model(x.to(device=device, dtype=torch.float32))
        correct_test += (scores.argmax(dim=1).cpu() == y).sum().item()
print(f"Final held-out subject accuracy: {correct_test / len(dataset_test):.3f}")

# %%
# 9. Interpret the job outputs and plan validation
# ------------------------------------------------
# Training accuracy is counted during parameter updates, so it is not a
# separate fixed-model evaluation. Final accuracy is the fraction of the
# reserved participant's windows classified correctly. The default test labels
# are balanced; check the printed balance when changing the cohort before
# using 0.5 as a chance reference. Seventy windows from one person do not
# provide seventy independent estimates of population generalization.
#
# If the training score rises while the held-out score stays low, investigate
# a validation split of additional training people before increasing epochs.
# For a research result, repeat subject-disjoint outer folds and report each
# participant's score. Save configuration, logs and sample_epoch.png to persistent
# storage before job scratch is removed; a successful Slurm exit alone is not
# evidence that the classifier generalizes.
