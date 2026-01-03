""".. _tutorial-eoec:

Eyes Open vs. Closed Classification
===================================

EEGDash example for eyes open vs. closed classification.

CHANGES:
- Uses the EEGDash API (no local OpenNeuro mirror required)
- Multi-subject run: auto-discover subjects from API and use ~10 valid subjects
- Skip subjects that produce empty EEGDashDataset (no recordings)
- Skip subjects that produce 0 windows after preprocessing/windowing
- Subject-wise train/test split (no leakage)
- Robust windowing: one 2s window per event (avoids braindecode trial overlap errors)
- Save plot to file (no GUI needed on compute nodes)
"""

from pathlib import Path
import os
import warnings

import numpy as np
import torch

warnings.simplefilter("ignore", category=RuntimeWarning)

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_USE_NUMBA", "false")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path.cwd()))
(Path(os.environ["_MNE_FAKE_HOME_DIR"]) / ".mne").mkdir(exist_ok=True)

from eegdash import EEGDash, EEGDashDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation


# -----------------------------
# Config
# -----------------------------
cache_folder = Path(
    os.getenv("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache")
).resolve()
cache_folder.mkdir(parents=True, exist_ok=True)
dataset_id = "ds005514"
task = "RestingState"

# number of *valid* subjects to use
num_subjects = int(os.environ.get("NUM_SUBJECTS", "10"))
num_test_subjects = int(os.environ.get("NUM_TEST_SUBJECTS", "2"))
random_state = int(os.environ.get("SEED", "42"))

# 2 seconds at 128 Hz
window_size_samples = 256

# training params
epochs = int(os.environ.get("EPOCHS", "6"))
batch_size = int(os.environ.get("BATCH_SIZE", "32"))


# -----------------------------
# Preprocessors
# -----------------------------
preprocessors = [
    hbn_ec_ec_reannotation(),
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
    Preprocessor("resample", sfreq=128),
    Preprocessor("filter", l_freq=1, h_freq=55),
]


# -----------------------------
# Build multi-subject windows (skip empties)
# -----------------------------
eegdash = EEGDash()
records = eegdash.find(dataset=dataset_id, task=task, limit=500)
subjects_all = sorted({rec.get("subject") for rec in records if rec.get("subject")})
if len(subjects_all) == 0:
    raise RuntimeError(
        f"No subjects returned by API for dataset={dataset_id}, task={task}"
    )

print("Discovered subjects (first 20):", subjects_all[:20])

all_windows = []
all_subject_ids = []
valid_subjects = []

for subj in subjects_all:
    if len(valid_subjects) >= num_subjects:
        break

    print(f"\n=== Subject {subj} ===")
    try:
        ds_eoec = EEGDashDataset(
            dataset=dataset_id,
            task=task,
            subject=subj,
            cache_dir=cache_folder,
        )
    except AssertionError as e:
        # This happens when EEGDashDataset finds 0 recordings (empty iterable)
        print(f"[SKIP] EEGDashDataset empty for subject {subj}: {e}")
        continue
    except Exception as e:
        print(
            f"[SKIP] Failed to construct EEGDashDataset for subject {subj}: {type(e).__name__}: {e}"
        )
        continue

    try:
        preprocess(ds_eoec, preprocessors)
        windows_ds = create_windows_from_events(
            ds_eoec,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=window_size_samples,  # one 2s window per event
            preload=True,
        )
    except Exception as e:
        print(
            f"[SKIP] Preprocess/windowing failed for subject {subj}: {type(e).__name__}: {e}"
        )
        continue

    n_win = len(windows_ds)
    if n_win == 0:
        print(f"[SKIP] 0 windows for subject {subj}")
        continue

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


# -----------------------------
# Save a sanity plot (no GUI)
# -----------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(concat_ds) > 2:
    plt.figure()
    plt.plot(concat_ds[2][0][0, :].transpose())
    plt.savefig("sample_epoch.png", dpi=150, bbox_inches="tight")
    print("Saved plot to sample_epoch.png")


# -----------------------------
# Subject-wise train/test split
# -----------------------------
rng = np.random.RandomState(random_state)
subjects_shuffled = valid_subjects.copy()
rng.shuffle(subjects_shuffled)

test_subjects = set(subjects_shuffled[:num_test_subjects])
train_subjects = set(subjects_shuffled[num_test_subjects:])

print("\nTrain subjects:", sorted(train_subjects))
print("Test subjects :", sorted(test_subjects))

indices = np.arange(len(concat_ds))
subj_arr = np.array(all_subject_ids)

train_indices = indices[np.isin(subj_arr, list(train_subjects))]
test_indices = indices[np.isin(subj_arr, list(test_subjects))]

print("Train windows:", len(train_indices), "Test windows:", len(test_indices))


# -----------------------------
# Tensors + loaders
# -----------------------------
# -----------------------------
torch.manual_seed(random_state)
np.random.seed(random_state)

X_train = torch.FloatTensor(np.array([concat_ds[i][0] for i in train_indices]))
X_test = torch.FloatTensor(np.array([concat_ds[i][0] for i in test_indices]))
y_train = torch.LongTensor(np.array([concat_ds[i][1] for i in train_indices]))
y_test = torch.LongTensor(np.array([concat_ds[i][1] for i in test_indices]))

from torch.utils.data import DataLoader, TensorDataset

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


# -----------------------------
# Model
# -----------------------------
from torch.nn import functional as F
from braindecode.models import ShallowFBCSPNet
from torchinfo import summary

model = ShallowFBCSPNet(24, 2, n_times=256, final_conv_length="auto")
summary(model, input_size=(1, 24, 256))


# -----------------------------
# Train
# -----------------------------
optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)

print("Using epochs =", epochs, "| device =", device, "| batch_size =", batch_size)


def normalize_data(x):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-7
    x = (x - mean) / std
    return x.to(device=device, dtype=torch.float32)


for e in range(epochs):
    model.train()
    correct_train = 0.0
    for x, y in train_loader:
        scores = model(normalize_data(x))
        y = y.to(device=device, dtype=torch.long)

        loss = F.cross_entropy(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = scores.argmax(dim=1)
        correct_train += (preds == y).sum().item()

    model.eval()
    correct_test = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            scores = model(normalize_data(x))
            y = y.to(device=device, dtype=torch.long)
            preds = scores.argmax(dim=1)
            correct_test += (preds == y).sum().item()

    train_acc = correct_train / len(dataset_train)
    test_acc = correct_test / len(dataset_test)
    print(f"Epoch {e}, Train accuracy: {train_acc:.2f}, Test accuracy: {test_acc:.2f}")

# -----------------------------
torch.manual_seed(random_state)
np.random.seed(random_state)

X_train = torch.FloatTensor(np.array([concat_ds[i][0] for i in train_indices]))
X_test = torch.FloatTensor(np.array([concat_ds[i][0] for i in test_indices]))
y_train = torch.LongTensor(np.array([concat_ds[i][1] for i in train_indices]))
y_test = torch.LongTensor(np.array([concat_ds[i][1] for i in test_indices]))

from torch.utils.data import DataLoader, TensorDataset

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


# -----------------------------
# Model
# -----------------------------
from torch.nn import functional as F
from braindecode.models import ShallowFBCSPNet
from torchinfo import summary

model = ShallowFBCSPNet(24, 2, n_times=256, final_conv_length="auto")
summary(model, input_size=(1, 24, 256))


# -----------------------------
# Train
# -----------------------------
optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)

print("Using epochs =", epochs, "| device =", device, "| batch_size =", batch_size)


def normalize_data(x):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-7
    x = (x - mean) / std
    return x.to(device=device, dtype=torch.float32)


for e in range(epochs):
    model.train()
    correct_train = 0.0
    for x, y in train_loader:
        scores = model(normalize_data(x))
        y = y.to(device=device, dtype=torch.long)

        loss = F.cross_entropy(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = scores.argmax(dim=1)
        correct_train += (preds == y).sum().item()

    model.eval()
    correct_test = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            scores = model(normalize_data(x))
            y = y.to(device=device, dtype=torch.long)
            preds = scores.argmax(dim=1)
            correct_test += (preds == y).sum().item()

    train_acc = correct_train / len(dataset_train)
    test_acc = correct_test / len(dataset_test)
    print(f"Epoch {e}, Train accuracy: {train_acc:.2f}, Test accuracy: {test_acc:.2f}")
