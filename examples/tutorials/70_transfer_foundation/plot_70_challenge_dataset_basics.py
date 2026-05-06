"""Get started with EEGChallengeDataset for the EEG2025 challenge
=================================================================

The EEG 2025 Foundation Challenge ships its own loader -- ``EEGChallengeDataset``
-- on top of the same ``eegdash`` infrastructure that powers ``EEGDashDataset``.
Why two classes? The challenge fixes a frozen subject pool per "release" (R1 to
R11, all from the Healthy Brain Network cohort, Alexander et al. 2017),
downsamples each recording to 100 Hz, band-pass filters it 0.5-50 Hz, and
exposes a "mini" subset of 20 subjects per release for fast iteration. Mixing
those preprocessed cubes with raw OpenNeuro pulls would silently break the
leaderboard contract. This tutorial walks through the loader, contrasts it with
``EEGDashDataset``, and verifies that ``mini=True`` is a strict subset of the
full release (Pernet et al. 2019 BIDS, Cisotto and Chicco 2024 clinical EEG
reproducibility tips). So why does the challenge need its own dataset class?
"""

# %% [markdown]
# ## Learning objectives
#
# - Build ``EEGChallengeDataset(release=..., mini=True)`` and read ``.release`` / ``.mini`` back.
# - Use ``mini=True`` to iterate on 20 subjects in seconds rather than the full release.
# - Verify the mini subject pool is a strict subset of the full release pool (frozen list).
# - Compare the challenge loader with ``EEGDashDataset``: a release-to-OpenNeuro map plus a curated mini list, on top of the same lazy-loading machinery.
# - Surface the leaderboard contract: see :doc:`/concepts/eegdash_objects` and the EEG 2025 site for evaluation rules.
#
# ## Requirements
#
# - You finished ``plot_01_first_recording`` and ``plot_02_dataset_to_dataloader``.
# - CPU only; runtime < 1 minute.
# - Network: the constructor queries the eegdash metadata catalog (~1 MB on first run, cached after). The challenge BDF files are *not* eagerly downloaded -- that happens lazily when you call ``.preview()`` or iterate ``.raw``.

# %%
# Setup -- seeds and imports.
import os
import warnings
from pathlib import Path

import numpy as np

import eegdash
from eegdash import EEGChallengeDataset, EEGDashDataset
from eegdash.const import (
    RELEASE_TO_OPENNEURO_DATASET_MAP,
    SUBJECT_MINI_RELEASE_MAP,
)

warnings.simplefilter("ignore", category=FutureWarning)
np.random.seed(42)

cache_dir = os.environ.get("EEGDASH_CACHE_DIR", str(Path.home() / ".eegdash_cache"))
print(f"eegdash version: {eegdash.__version__}")
print(f"cache directory: {cache_dir}")

# %% [markdown]
# ## Step 1 -- Instantiate ``EEGChallengeDataset(release='R5', mini=True)``
#
# **Run.** We pick release ``R5`` because it has 20 mini subjects (every release
# does). ``download=False`` means no S3 traffic now -- we only need the metadata
# catalog to enumerate records and verify the strict-subset claim.

# %%
RELEASE = "R5"
ds_mini = EEGChallengeDataset(release=RELEASE, cache_dir=cache_dir, mini=True)
n_mini_records = len(ds_mini.records)
print(f"release   : {ds_mini.release}")
print(f"mini      : {ds_mini.mini}")
print(f"n_records : {n_mini_records}")
print(f"s3_bucket : {ds_mini.s3_bucket}")
print(f"data_dir  : {ds_mini.data_dir}")

# %% [markdown]
# ## Step 2 -- Predict: how does ``mini=True`` change the records list?
#
# **Predict.** Before running Step 3, write down: how many subjects do you
# expect ``ds_mini`` to cover? Recall that ``SUBJECT_MINI_RELEASE_MAP[release]``
# is a frozen list of 20 names per release.
#
# **Investigate.** ``mini=True`` does not touch the random number generator; it
# substitutes a curated subject list and renames the dataset id to
# ``EEG2025r5mini`` for cache isolation. So your prediction should be exactly
# 20 subjects.

# %%
mini_subjects = sorted(set(ds_mini.description["subject"]))
print(
    f"mini subjects ({len(mini_subjects)}): {mini_subjects[:3]} ... {mini_subjects[-2:]}"
)
assert ds_mini.release == RELEASE, "release attribute must match the request"
assert ds_mini.mini is True, "mini=True must be honoured"
assert len(mini_subjects) == 20, "every challenge release lists 20 mini subjects"

# %% [markdown]
# ## Step 3 -- Show that ``mini`` is a strict subset of ``full``
#
# **Run.** Build the full-release dataset (still ``download=False``) and compare
# subject sets. The full release pulls every subject mapped to OpenNeuro
# dataset ``RELEASE_TO_OPENNEURO_DATASET_MAP[RELEASE]``.

# %%
ds_full = EEGChallengeDataset(release=RELEASE, cache_dir=cache_dir, mini=False)
n_full_records = len(ds_full.records)
full_subjects = sorted(set(ds_full.description["subject"]))
print(f"full release subjects: {len(full_subjects)}")
print(f"full release records : {n_full_records}")
print(f"OpenNeuro dataset id : {RELEASE_TO_OPENNEURO_DATASET_MAP[RELEASE]}")

mini_set = set(mini_subjects)
full_set = set(full_subjects)
assert mini_set.issubset(full_set), "mini subjects must all appear in the full release"
assert len(mini_set) < len(full_set), "mini must be a strict subset, not equal"
ratio = len(mini_set) / len(full_set)
print(f"|mini| / |full|      : {ratio:.2%}")
assert ratio < 0.10, "mini should keep < 10% of the full subject pool"

# %% [markdown]
# ## Step 4 -- Surface the leaderboard contract
#
# **Investigate.** The challenge data is *not* identical to what
# ``EEGDashDataset`` would download for the same OpenNeuro id: every recording
# is downsampled (500 Hz to 100 Hz) and band-pass filtered (0.5-50 Hz pass-band)
# before being shipped via the ``s3://nmdatasets/NeurIPS25`` bucket. That means
# you cannot mix challenge and non-challenge data in a leaderboard submission
# without breaking the contract. The full preprocessing recipe is documented at
# https://github.com/eeg2025/downsample-datasets and the evaluation rules at
# https://eeg2025.github.io.

# %%
print(
    f"records ratio mini/full: {n_mini_records} / {n_full_records} = "
    f"{n_mini_records / n_full_records:.2%}"
)
print(
    f"all dataset ids start with 'EEG2025r{RELEASE[1:]}': "
    f"{all(r['dataset'].startswith(f'EEG2025r{RELEASE[1:]}') for r in ds_mini.records)}"
)
print("evaluation: see https://eeg2025.github.io (leaderboard, splits, deadline)")

# %% [markdown]
# ## Step 5 -- Investigate: how does this differ from ``EEGDashDataset``?
#
# **Investigate.** ``EEGDashDataset`` is a thin query layer over the eegdash
# metadata catalog -- you pass any combination of BIDS entities and it returns
# every match. ``EEGChallengeDataset`` is a pre-baked recipe on top of that:
# it picks the dataset id from a release-to-OpenNeuro map, restricts subjects
# to a frozen curated list when ``mini=True``, redirects S3 to the challenge
# bucket, and pins the cache folder so the preprocessed BDF files do not collide
# with the raw OpenNeuro tree.

# %%
ds_eegdash = EEGDashDataset(
    cache_dir=cache_dir,
    dataset=RELEASE_TO_OPENNEURO_DATASET_MAP[RELEASE],
    subject=mini_subjects[0],
)
print(
    f"EEGDashDataset(dataset='{RELEASE_TO_OPENNEURO_DATASET_MAP[RELEASE]}', "
    f"subject='{mini_subjects[0]}') -> {len(ds_eegdash.records)} records (raw 500 Hz)"
)
print(
    f"EEGChallengeDataset(release='{RELEASE}', mini=True) -> {n_mini_records} "
    "records (preprocessed 100 Hz)"
)
print("Same metadata catalog, two different views of the same subjects.")

# %% [markdown]
# ## A common mistake -- and how to recover (E2.17)
#
# **Run.** Typing the release identifier wrong (``r5`` instead of ``R5``, or
# ``"R12"``) raises ``ValueError`` at construction time -- before any S3 traffic
# happens. We trigger the error on purpose so the failure mode is visible.

# %%
try:
    _bogus = EEGChallengeDataset(
        release="R12",  # there is no R12; the map only goes up to R11
        cache_dir=cache_dir,
        mini=True,
    )
except ValueError as exc:
    print(f"Caught {type(exc).__name__}: {str(exc)[:120]}...")
    available = sorted(
        RELEASE_TO_OPENNEURO_DATASET_MAP.keys(), key=lambda s: int(s[1:])
    )
    print(f"Recovery: use one of {available}")

# %% [markdown]
# ## Modify -- try a different release
#
# **Your turn.** Switch ``RELEASE`` to ``"R2"`` (smaller dataset) and re-check
# the strict-subset invariant. The 20 mini subjects per release are different,
# but the relationship ``mini ⊂ full`` holds for every release in the map.

# %%
ds_alt = EEGChallengeDataset(release="R2", cache_dir=cache_dir, mini=True)
alt_subjects = set(ds_alt.description["subject"])
assert alt_subjects == set(SUBJECT_MINI_RELEASE_MAP["R2"]), (
    "mini subjects must equal SUBJECT_MINI_RELEASE_MAP[release]"
)
print(
    f"R2 mini: {len(alt_subjects)} subjects, dataset id {ds_alt.records[0]['dataset']}"
)

# %% [markdown]
# ## Make -- a tiny preview-only pipeline
#
# Build a one-record preview from the mini release. Inspect the record header
# without pulling the 100 Hz BDF file: ``ds.records`` is enumerated at
# construction time and exposes BIDS entities and metadata catalog fields
# (``subject``, ``task``, ``nchans``, ``sampling_frequency``).

# %%
first_record = ds_mini.records[0]
print(f"first record subject : {first_record['subject']}")
print(f"first record task    : {first_record['task']}")
print(f"first record nchans  : {first_record['nchans']}")
print(
    f"first record sfreq   : {first_record['sampling_frequency']} Hz "
    "(downsampled from 500 Hz)"
)
print("To download and plot: call ds_mini.preview(0) after construction.")

# %% [markdown]
# ## Try it yourself (Extensions)
#
# - **Easier**: print ``SUBJECT_MINI_RELEASE_MAP[RELEASE]`` and confirm it matches ``mini_subjects``.
# - **Same difficulty**: loop over every release in ``RELEASE_TO_OPENNEURO_DATASET_MAP`` and tabulate ``|mini|`` vs ``|full|``.
# - **Harder**: run with ``download=True``, call ``preview(0)``, and confirm ``raw.info["sfreq"] == 100.0`` (the challenge downsample step).
#
# ## Result

# %%
summary = ds_mini.summary(verbose=False)
print(
    f"release={ds_mini.release} mini={ds_mini.mini} "
    f"n_records={summary['n_records']} n_subjects={summary['n_subjects']} "
    f"sampling_rates={dict(summary['sampling_rates'])}"
)

# %% [markdown]
# ## Wrap-up
#
# ``EEGChallengeDataset`` is ``EEGDashDataset`` with three rails attached: a
# release-to-OpenNeuro map, a frozen mini subject list, and the challenge bucket.
# Use ``mini=True`` while iterating; switch to ``mini=False`` for a final
# submission; never mix challenge data with raw ``EEGDashDataset`` pulls -- the
# 0.5-50 Hz pass-band and 100 Hz downsample make the two views incompatible. See
# :doc:`/concepts/eegdash_objects` for the object model behind both classes.
#
# ## References
#
# - Alexander, L. M. et al. (2017). An open resource for transdiagnostic
#   research in pediatric mental health and learning disorders. *Scientific
#   Data* 4:170181, doi:10.1038/sdata.2017.181.
# - Pernet, C. R. et al. (2019). EEG-BIDS: an extension to the brain imaging
#   data structure for electroencephalography. *Scientific Data* 6:103,
#   doi:10.1038/s41597-019-0104-8.
# - Cisotto, G. and Chicco, D. (2024). Ten quick tips for clinical
#   electroencephalographic (EEG) data acquisition and signal processing.
#   *PeerJ Computer Science* 10:e2256, doi:10.7717/peerj-cs.2256.
# - EEG 2025 Foundation Challenge: https://eeg2025.github.io
# - Challenge preprocessing recipe: https://github.com/eeg2025/downsample-datasets
