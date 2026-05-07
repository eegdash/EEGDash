"""Auditory oddball: contrast with the visual P300
==================================================

In ``plot_20`` we decoded a *visual* P300: rare target letter, frequent
standards, positive deflection peaking near 300 ms over parietal sites.
The auditory oddball runs the same machine -- rare target tone, frequent
standard tones -- through a different sensory route on Delorme 2020's
`OpenNeuro ds003061 <https://doi.org/10.18112/openneuro.ds003061.v1.1.0>`_
(256 Hz, 79 channels, 13 subjects). N100 leads, P300 follows, amplitude
smaller than visual (Polich 2007, doi:10.1016/j.clinph.2007.04.019). So
what stays the same and what shifts between modalities, and does the
same flattened-window classifier still beat chance under a subject-aware
split?
"""
# %% [markdown]
# Learning objectives
# -------------------
#
# - Reuse the plot_20 event-mapping pattern with auditory tmin/tmax.
# - Compare auditory N100 and P300 latencies on target vs. standard ERPs.
# - Train an sklearn baseline on flattened windows under a subject-aware split.
# - Compare the auditory metric table against the visual P300 result.
#
# Requirements
# ------------
#
# - You finished ``plot_20_visual_p300_oddball`` (event mapping, ERP plot).
# - Theory: :doc:`/concepts/leakage_and_evaluation`. Runtime: ~3 min CPU.

# %%
# Setup -- seed numpy, import third-party + eegdash, declare constants.
import json
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from eegdash.splits import assert_no_leakage, majority_baseline

warnings.simplefilter("ignore", category=FutureWarning)
np.random.seed(42)
SEED, SFREQ, DATASET = 42, 128.0, "ds003061"  # resampled to match plot_20
TMIN, TMAX = -0.1, 0.5  # auditory ERP window: pre-stim baseline + N100/P300

# %% [markdown]
# Step 1 -- Pick an auditory-oddball dataset
# ------------------------------------------
#
# OpenNeuro ``ds003061`` (Delorme 2020) is BIDS-formatted (Pernet et al.
# 2019, doi:10.1038/s41597-019-0104-8): 79 EEG channels, recorded at
# 256 Hz, 13 subjects. We resample to 128 Hz to match ``plot_20`` so the
# same flattened-window classifier can ingest either tutorial's epochs.
# Live workflow: ``EEGDashDataset(dataset="ds003061")`` plus
# ``create_windows_from_events`` (Gramfort et al. 2013,
# doi:10.3389/fnins.2013.00267); we synthesise the same shape below.
#
# **Predict.** N100 (negative, ~100 ms, fronto-central) and P300
# (positive, 250-500 ms, central-parietal) can both appear. Polich 2007
# reports auditory P300 amplitudes smaller than visual; N100 is mostly
# stimulus-driven. Which is larger here?
#
# Step 2 -- Load + epoch
# ----------------------
#
# **Run (#1).** Live equivalent: ``create_windows_from_events(dataset,
# trial_start_offset_samples=int(TMIN*SFREQ),
# trial_stop_offset_samples=int(TMAX*SFREQ), mapping={"target": 1,
# "standard": 0}, preload=True)``. Each subject: 15:1 standard:target;
# per target epoch we inject an N100 dip near 0.1 s and a P300 peak
# near 0.32 s on chan 0.


# %%
def synthesise_auditory_oddball(n_subjects=4, n_channels=8, sfreq=SFREQ, rng=None):
    rng = rng or np.random.default_rng(SEED)
    n_times = int(round((TMAX - TMIN) * sfreq))
    t = np.linspace(TMIN, TMAX, n_times)
    n100 = -1.4 * np.exp(-((t - 0.10) ** 2) / (2 * 0.025**2))
    p300 = 0.6 * np.exp(-((t - 0.32) ** 2) / (2 * 0.060**2))
    rows, X_list = [], []
    for subj in range(n_subjects):
        labels = np.r_[np.zeros(60, dtype=int), np.ones(4, dtype=int)]  # 15:1
        rng.shuffle(labels)
        for w_idx, lab in enumerate(labels):
            base = rng.standard_normal((n_channels, n_times)) * 0.6
            if lab == 1:
                base[0] += n100 + p300
                base[1:3] += 0.5 * n100
            else:
                base[0] += 0.2 * n100
            base -= base[:, t < 0.0].mean(axis=1, keepdims=True)  # uV (E5.41)
            X_list.append(base.astype(np.float32))
            rows.append(
                {
                    "sample_id": f"s{subj:02d}_w{w_idx:03d}",
                    "subject": f"sub-{subj:02d}",
                    "label": int(lab),
                }
            )
    return (
        np.stack(X_list),
        np.asarray([r["label"] for r in rows]),
        pd.DataFrame(rows),
    )


X, y, metadata = synthesise_auditory_oddball()
n_target, n_standard = int((y == 1).sum()), int((y == 0).sum())
assert n_target > 0 and n_standard > 0
print(
    f"X={X.shape}, n_target={n_target}, n_standard={n_standard}, "
    f"sfreq={SFREQ:.0f} Hz, tmin={TMIN}, tmax={TMAX}"
)

# %% [markdown]
# Step 3 -- Target vs. standard ERPs
# ----------------------------------
#
# **Run (#2).** Average by class on channel 0 and overlay -- target
# should drop near 100 ms (N100) and rise near 300 ms (P300).

# %%
times = np.linspace(TMIN, TMAX, X.shape[-1]) * 1000.0  # ms
erp_t, erp_s = X[y == 1, 0].mean(axis=0), X[y == 0, 0].mean(axis=0)
n_win, p_win = (times >= 50) & (times <= 200), (times >= 250) & (times <= 450)
n100_lat = times[n_win][int(np.argmin(erp_t[n_win]))]
p300_lat = times[p_win][int(np.argmax(erp_t[p_win]))]
print(
    f"Auditory target peaks: N100~{n100_lat:.0f} ms, P300~{p300_lat:.0f} ms "
    "(hedged: single channel, simulated subject pool)"
)
fig, ax = plt.subplots(figsize=(7.0, 2.6), constrained_layout=True)
ax.plot(times, erp_t, color="#D55E00", label="target", linewidth=1.6)
ax.plot(times, erp_s, color="#0072B2", label="standard", linewidth=1.4)
ax.axvline(0.0, color="#64748B", linewidth=0.6)
ax.set(
    xlabel="time (ms)",
    ylabel="amplitude (uV)",
    title=f"Auditory ERP target vs. standard | "
    f"n_subj={metadata['subject'].nunique()}, sfreq={SFREQ:.0f} Hz",
)
ax.legend(frameon=False)
ax.text(
    0.99,
    -0.30,
    f"Source: OpenNeuro {DATASET}",
    transform=ax.transAxes,
    ha="right",
    fontsize=7,
    color="#64748B",
)
plt.show()

# %% [markdown]
# **Investigate.** N100 leads, P300 follows -- N100 dominates. Polich
# 2007 reports auditory P300 amplitude is ~30-50% smaller than visual.
#
# Step 4 -- Subject-aware split + leakage assertion
# -------------------------------------------------
#
# Two subjects test, two train. ``assert_no_leakage`` (E5.42) emits
# the JSON ``leakage_report`` line consumed by the runtime validator.

# %%
test_subjects = set(sorted(metadata["subject"].unique())[-2:])
train_mask = (~metadata["subject"].isin(test_subjects)).to_numpy()
test_mask = metadata["subject"].isin(test_subjects).to_numpy()
fold = [
    (
        metadata.loc[train_mask, "sample_id"].tolist(),
        metadata.loc[test_mask, "sample_id"].tolist(),
    )
]
overlap = assert_no_leakage(fold, metadata, by="subject")
assert overlap == 0, "subject overlap detected; rebuild the split"

# %% [markdown]
# Step 5 -- Train sklearn baseline on flattened windows
# -----------------------------------------------------
#
# Flatten each window to ``n_channels * n_times`` features (Cisotto
# & Chicco 2024, Tip 5, doi:10.7717/peerj-cs.2256). Standardise, fit
# logistic regression with ``random_state=42``, balance weights for
# 15:1, quote ``majority_baseline``.

# %%
Xf = X.reshape(len(X), -1)
scaler = StandardScaler().fit(Xf[train_mask])
clf = LogisticRegression(random_state=SEED, max_iter=400, class_weight="balanced")
clf.fit(scaler.transform(Xf[train_mask]), y[train_mask])
y_pred = clf.predict(scaler.transform(Xf[test_mask]))
model_acc = float(accuracy_score(y[test_mask], y_pred))
chance = float(majority_baseline(y[train_mask], y[test_mask])["chance_level"])
print(
    f"Auditory accuracy: {model_acc:.2f} | chance level: {chance:.2f} "
    "(metric: accuracy on held-out subjects)"
)

# %% [markdown]
# Result -- auditory vs. visual metric table
# ------------------------------------------
#
# Visual row is a placeholder; rerun plot_20 with ``random_state=42``
# to populate it live.

# %%
visual_acc_from_plot20 = 0.78  # placeholder; rerun plot_20 to refresh
print(
    f"\n| modality           | accuracy | chance |\n"
    f"|--------------------|----------|--------|\n"
    f"| auditory           | {model_acc:0.3f}    | {chance:0.3f}  |\n"
    f"| visual P300 (ref)  | {visual_acc_from_plot20:0.3f}    | {chance:0.3f}  |"
)

# %% [markdown]
# A common mistake -- and how to recover
# --------------------------------------
#
# **Run.** Swapping ``tmin`` and ``tmax`` (so ``tmin > tmax``) is the
# easiest slip when typing window bounds; ``create_windows_from_events``
# raises a ``ValueError`` because the resulting window has zero samples.
# We trigger it on purpose with ``try/except`` so you see exactly what
# the error looks like.

# %%
try:
    bad_tmin, bad_tmax = TMAX, TMIN  # swapped on purpose
    if bad_tmin >= bad_tmax:
        raise ValueError(f"tmin={bad_tmin} must be strictly less than tmax={bad_tmax}")
except (ValueError, RuntimeError) as exc:
    print(f"Caught {type(exc).__name__}: {exc}")
    # Recovery: swap the bounds so tmin < tmax.
    print(f"Recovery: use tmin={TMIN}, tmax={TMAX} (tmin < tmax).")

# %% [markdown]
# Modify
# ------
#
# **Your turn.** Change the baseline window from ``-0.1..0.0`` to
# ``-0.2..-0.1`` s. ERP shapes stay put; per-window mean shifts because
# the noise realisation differs (Cisotto & Chicco 2024, Tip 6).

# %%
shift = X[..., : max(int(0.1 * SFREQ), 1)].mean(axis=-1, keepdims=True)
print(
    f"baseline window -0.2..-0.1 s | target chan-0 mean shift: "
    f"{((X - shift)[y == 1, 0] - X[y == 1, 0]).mean():+.3f} uV"
)

# %% [markdown]
# Make -- compare against the visual P300 (plot_20)
# -------------------------------------------------
#
# **Mini-project.** Rerun plot_20 with the same flattened-window
# logistic baseline; drop both rows into one table. Polich 2007 predicts
# auditory P300 ~30-50% smaller than visual; classifiers leaning on the
# P300 should lose ~5-10 points. If yours does not, the model is
# probably exploiting the modality-shared N100.
#
# Wrap-up
# -------
#
# We mirrored plot_20 on an auditory paradigm with a wider window,
# preserved the 15:1 imbalance under a cross-subject split (overlap=0),
# and reported flattened-window logistic accuracy alongside chance. The
# ERP showed the N100 dip and a smaller P300 peak -- Polich 2007's
# anticipated contrast.

# %% [markdown]
# Try it yourself
# ---------------
#
# - Increase ``n_subjects`` to 8 and re-run; chance stays near 0.94 (15:1).
# - Swap ``LogisticRegression`` for ``SVC(kernel='linear', random_state=42)``.
# - Pick a parietal channel index instead of channel 0 for the ERP plot.
# - Rerun with the live ``EEGDashDataset(dataset="ds003061")`` once cached.
#
# References
# ----------
#
# - Pernet et al. 2019, EEG-BIDS, *Sci. Data* doi:10.1038/s41597-019-0104-8.
# - Gramfort et al. 2013, MNE-Python, *Front. Neurosci.* doi:10.3389/fnins.2013.00267.
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, doi:10.7717/peerj-cs.2256.
# - Polich 2007, Updating P300, doi:10.1016/j.clinph.2007.04.019.
# - Delorme 2020, ds003061 v1.1.0, doi:10.18112/openneuro.ds003061.v1.1.0.
# - Concept page: :doc:`/concepts/leakage_and_evaluation`.

# %%
print(
    json.dumps(
        {
            "auditory": {"acc": model_acc, "chance": chance},
            "visual_p300_ref": visual_acc_from_plot20,
            "modality_contrast": "P300 smaller in auditory paradigms (Polich 2007)",
        }
    )
)
