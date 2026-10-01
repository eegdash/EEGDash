"""How do I evaluate participant-level p-factor regression?
========================================================

Estimate observed HBN p-factor from resting EEG using one row per participant.
Six R5 mini participants keep acquisition bounded. These recordings are the
100 Hz challenge derivatives, already filtered at 0.5–50 Hz. First use downloads
six recordings; the exact bytes depend on their duration. This small leave-one-
participant-out exercise is not a challenge leaderboard estimate.
"""

# %%
# Before you start
# ----------------
#
# Use an installed EEGDash environment with MNE, NumPy, scikit-learn and
# Matplotlib. No GPU is required. Set ``EEGDASH_CACHE_DIR`` to reuse the six
# resting-state recordings; the subset needs roughly 100 MB on first download.
# Cropping reduces processing time and memory, not the bytes needed to acquire
# each recording.
#
# The p-factor is an observed participant-level phenotype provided by the
# challenge. It is not a trial label, a diagnosis made from EEG, or a value to
# reconstruct from a participant's identifier. A subject contributes exactly one
# feature row and one target, so long recordings cannot increase that subject's
# weight simply by yielding more windows.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash.paths import get_default_cache_dir

from eegdash import EEGChallengeDataset
from eegdash.features import spectral_preprocessor, spectral_bands_power
from eegdash.const import SUBJECT_MINI_RELEASE_MAP

# %%
# Load observed participant targets and recorded voltages.
# %%
# Load actual targets before computing features
# ---------------------------------------------
#
# The sorted mini-release list provides a reproducible small subset rather
# than selecting participants according to their outcomes. ``target_name`` names
# the observed phenotype; ``description_fields`` makes the identity and target
# available alongside each recording. The printed metadata lets you verify the
# join between signal and participant.
#
# Missing recordings or nonnumeric targets
# must be investigated at the source instead of filled with a group mean or a
# random number. For a larger cohort, specify missing-target exclusions before
# fitting a model and report the resulting number of people.

subjects = sorted(SUBJECT_MINI_RELEASE_MAP["R5"])[:6]
dataset = EEGChallengeDataset(
    release="R5",
    mini=True,
    task="RestingState",
    subject=subjects,
    cache_dir=get_default_cache_dir(),
    description_fields=["subject", "task", "p_factor"],
    target_name="p_factor",
)
print(dataset.description.to_string(index=False))
# %%
# Summarize a fixed resting interval
# ----------------------------------
#
# ``crop(tmax=59)`` retains the recording from time zero through 59 seconds.
# That fixed horizon prevents duration differences from deciding how much EEG
# contributes to each row. It does not isolate a uniform eye condition: the
# recorded resting instructions can occur within the interval.
#
# EEGDash Welch spectra use non-overlapping 200-sample Hamming segments at
# the source's 100 Hz rate, giving 0.5 Hz frequency spacing. Features use four
# ranges: 1–4, 4–8, 8–13 and 13–30 Hz. EEGDash's ``spectral_preprocessor``
# computes the PSD and ``spectral_bands_power`` sums the selected bins.
# Multiplying by their 0.5 Hz spacing approximates band power in V² before
# ``log10``. Each band includes its lower bound and excludes its upper bound,
# so adjacent bands do not share bins. All participants use this convention.
# The small floor
# prevents undefined logarithms for flat channels such as the source reference;
# it does not turn a flat electrode into an informative feature.
#
# Concatenation is band-major, retaining channel order inside each band. The
# recordings use the same challenge montage and channel order.
# This direct public function requires _metadata to resolve sampling/filter
# defaults. FeatureExtractor normally supplies it; here each Raw contributes
# one participant row, so its MNE info is passed explicitly.

features, targets, identities = [], [], []
for recording in dataset.datasets:
    raw = recording.raw.copy().pick("eeg").crop(tmax=59).load_data()
    frequencies, psd = spectral_preprocessor(
        raw.get_data(),
        _metadata={"info": raw.info},
        f_min=1,
        f_max=30,
        nperseg=200,
        noverlap=0,
        window="hamming",
    )
    powers = spectral_bands_power(
        frequencies,
        psd,
        bands={"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)},
    )
    band_power = np.concatenate(list(powers.values())) * (
        frequencies[1] - frequencies[0]
    )
    features.append(np.log10(np.maximum(band_power, 1e-30)))
    targets.append(float(recording.description["p_factor"]))
    identities.append(str(recording.description["subject"]))
    print(
        identities[-1],
        len(raw.ch_names),
        raw.info["sfreq"],
        raw.annotations.description[:8],
    )
# %%
# Check the participant-level design matrix
# -----------------------------------------
#
# ``X`` has shape ``(participants, four bands × channels)`` and ``y`` has
# one observed p-factor per row. With the current 129-channel recordings, this
# means 516 predictors for only six participants. This does not establish
# whether the EEG contains predictive information.
#
# This high-dimensional, tiny-sample setting motivates regularization, but no
# penalty can make six participants sufficient for clinical inference. The page
# demonstrates a subject-independent evaluation boundary; it does not establish
# that the resulting features are invariant to subject identity.

X, y = np.asarray(features), np.asarray(targets)
print("Participant features:", X.shape, "observed targets:", y)
# %%
# All scaling and baseline fitting occur inside the held-out participant fold.
# %%
# Fit inside each held-out participant fold
# -----------------------------------------
#
# Leave-one-out fits six models, each using five people and predicting the
# remaining person once. ``StandardScaler`` is refitted inside each pipeline so
# its means and spreads never include the held-out row. Ridge's ``alpha=10`` is a
# fixed illustrative penalty, not a value selected from these six test errors.
#
# The dummy model separately recomputes the training mean in every fold. Mean
# absolute error is reported in the provided p-factor scale, with equal weight
# per person. A smaller ridge error than the dummy error would be descriptive
# evidence on this subset only; a larger one is equally legitimate. The diagonal
# in the scatter denotes exact prediction, not a fitted regression line.

predicted, baseline = np.empty_like(y), np.empty_like(y)
for train, test in LeaveOneOut().split(X):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    predicted[test] = model.fit(X[train], y[train]).predict(X[test])
    baseline[test] = DummyRegressor().fit(X[train], y[train]).predict(X[test])
print("Participant MAE:", mean_absolute_error(y, predicted))
print("Training-mean MAE:", mean_absolute_error(y, baseline))
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y, predicted, label="Ridge: held-out participant")
ax.scatter(y, baseline, marker="x", label="Training-mean baseline")
ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
ax.set(xlabel="Observed p-factor", ylabel="Predicted p-factor")
ax.legend()
plt.show()

# %%
# Separate model development from clinical interpretation
# -------------------------------------------------------
#
# First increase the number of independently held-out participants. If you
# want to choose the ridge penalty, channels, bands or resting interval, do so
# using an inner training-only validation split and keep the outer participant
# fold untouched. Reusing these outer errors to select features makes them model-
# development results rather than a final evaluation.
#
# A useful extension is to compare observed EEG features with a prespecified
# metadata-only baseline, using the same people and folds. Report missing-target
# exclusions and participant-level uncertainty. Avoid interpreting one fitted
# coefficient as a clinical biomarker when predictors are correlated and the
# sample is this small.
#
# Related evaluation example: `Braindecode train, test and tune
# <https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html>`_.
