"""Share spectral preprocessing with a feature tree
================================================

Compare separate and shared Welch computations on the same recorded trials.

These real Nakanishi2015 SSVEP recordings are distributed as the processed
`nm000118 release <https://nemar.org/dataset/nm000118>`_
(`study <https://doi.org/10.1371/journal.pone.0140703>`_).
Filtering, downsampling and latency handling were already applied; do not
shift the event onsets again. CPU is sufficient. Internet is needed for the
first download; ``EEGDASH_CACHE_DIR`` keeps downloads across runs.

The explicit subset uses 1 participant(s), about 7.0 MB of signal files.

Before you start
----------------
Install EEGDash and its dependencies. Tutorial 40 introduces the spectral
extractor API; this page rebuilds one subject's windows and needs no saved
feature table. The deliverable is an equivalence check and measured timing
for two ways of expressing the same band features.
"""

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events

from eegdash.paths import get_default_cache_dir
from eegdash import EEGDashDataset
from functools import partial
from eegdash.features import (
    FeatureExtractor,
    extract_features,
    spectral_bands_power,
    spectral_preprocessor,
)
from time import perf_counter

# %%
# 1. Load and inspect the selected recordings
# -------------------------------------------
# Use a single recording so both extraction paths see precisely the same
# channel order, rate and event labels. Download time is outside the timed
# region. The comparison concerns feature computation, not network performance.
cache_dir = get_default_cache_dir()
subjects = ["1"]
dataset = EEGDashDataset(
    cache_dir=cache_dir,
    dataset="nm000118",
    subject=subjects,
    session="0",
    run="0",
    task="ssvep",
    n_jobs=1,
)
if len(dataset.datasets) != len(subjects):
    raise ValueError(
        "Query did not return one recording per requested subject; inspect dataset.description"
    )
dataset.description[["subject", "session", "run"]]

# %%
raw = dataset.datasets[0].raw
sfreq = raw.info["sfreq"]
channel_names = raw.ch_names
class_names = sorted(set(raw.annotations.description), key=float)
mapping = {name: index for index, name in enumerate(class_names)}
print(f"Channels: {channel_names}; sampling rate: {sfreq} Hz")
print("Observed stimulus frequencies (Hz):", class_names)

# %%
# 2. Window the observed trials
# -----------------------------
# Keep the four-second window contract from tutorial 40. Each extractor reads
# the same trial arrays through Braindecode; changing windows between paths
# would mix a data change with the computation comparison.
# The source event lasts 4.15 seconds; its final 0.15 seconds are unused.
window_size = int(4 * sfreq)
windows = create_windows_from_events(
    dataset,
    mapping=mapping,
    window_size_samples=window_size,
    window_stride_samples=window_size,
    on_last_window="drop",
    preload=True,
)
metadata = windows.get_metadata().reset_index(drop=True)
y = metadata["target"].to_numpy(dtype=int)
if (
    len(windows) != len(metadata)
    or metadata.duplicated(["subject", "session", "run", "i_start_in_trial"]).any()
):
    raise ValueError("Window rows must have aligned, unique recording/start identities")
print(pd.crosstab(metadata["subject"], y))

# %%
# 3. Build equivalent independent and shared spectral features
# ------------------------------------------------------------
# Each ``partial`` fixes a band's boundaries. The flat dictionary wraps each
# band in its own spectral preprocessor, whereas the tree puts the common
# preprocessor above all band leaves. That shared parent is the reason the
# Welch spectrum can be reused; putting similar names in a dictionary alone
# would not eliminate repeated preprocessing.
#
# Both definitions use the same one-second Welch segments and 4–30 Hz range.
# The nested names yield different column prefixes, so the equality check below
# matches each semantic band/channel pair explicitly. Identical table shapes
# would not prove identical feature values.
bands = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
psd = partial(spectral_preprocessor, fs=sfreq, nperseg=int(sfreq), f_min=4, f_max=30)
leaves = {
    name: partial(spectral_bands_power, bands={name: limits})
    for name, limits in bands.items()
}
flat = {
    name: FeatureExtractor({"power": leaf}, preprocessor=psd)
    for name, leaf in leaves.items()
}
tree = FeatureExtractor(leaves, preprocessor=psd)
print(tree)

# %%
# The diagram follows the actual three leaves defined above.
fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
for ax, shared in zip(axes, [False, True]):
    ax.axis("off")
    ax.text(0.05, 0.5, "Windows", ha="center")
    if shared:
        ax.text(0.45, 0.5, "Welch once", ha="center")
        ax.annotate("", (0.33, 0.5), (0.13, 0.5), arrowprops={"arrowstyle": "->"})
    for y_pos, band in zip([0.2, 0.5, 0.8], bands):
        ax.text(0.85, y_pos, band, ha="center")
        if not shared:
            ax.text(0.45, y_pos, "Welch", ha="center")
            ax.annotate("", (0.33, y_pos), (0.13, 0.5), arrowprops={"arrowstyle": "->"})
        ax.annotate(
            "",
            (0.77, y_pos),
            (0.57, 0.5 if shared else y_pos),
            arrowprops={"arrowstyle": "->"},
        )
    ax.set_title("Shared tree" if shared else "Independent paths")
plt.show()

# %%
# 4. Measure both paths and check feature values, not a promised speedup
# ----------------------------------------------------------------------
# The timer surrounds the complete ``extract_features(...).to_dataframe()``
# operation, including batching and table assembly. A ratio above one means
# the shared tree was faster in this run; below one means it was slower. Small
# recordings may be dominated by fixed framework overhead, and the second
# measurement can benefit from warm caches.
#
# The all-close assertion is the correctness result. The two measured bars are
# the performance observation, with no lower bound imposed on the speedup.
# The numerical tolerance accounts for floating-point differences while still
# checking that every named band and channel has equivalent values.
start = perf_counter()
flat_table = extract_features(windows, flat, batch_size=64, n_jobs=1).to_dataframe()
flat_seconds = perf_counter() - start
start = perf_counter()
tree_table = extract_features(windows, tree, batch_size=64, n_jobs=1).to_dataframe()
tree_seconds = perf_counter() - start
for band in bands:
    for channel in channel_names:
        np.testing.assert_allclose(
            flat_table[f"{band}_power_{band}_{channel}"],
            tree_table[f"{band}_{band}_{channel}"],
            rtol=1e-6,
        )
if not (np.isfinite(tree_table.to_numpy()).all()):
    raise ValueError(
        "Unexpected shape or nonfinite values; inspect input signals and extraction settings"
    )
print(
    f"Flat: {flat_seconds:.3f}s; tree: {tree_seconds:.3f}s; ratio: {flat_seconds / tree_seconds:.2f}"
)
fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
ax.bar(["Separate spectra", "Shared spectrum"], [flat_seconds, tree_seconds])
ax.set(
    ylabel="Measured extraction time (s)",
    title="One illustrative run (flat first, tree second)",
)
plt.show()

# %%
# Numeric equivalence is more important than a one-shot timing.
flat_values = np.concatenate(
    [
        flat_table[f"{band}_power_{band}_{channel}"].to_numpy()
        for band in bands
        for channel in channel_names
    ]
)
tree_values = np.concatenate(
    [
        tree_table[f"{band}_{band}_{channel}"].to_numpy()
        for band in bands
        for channel in channel_names
    ]
)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
axes[0].scatter(flat_values, tree_values, s=4, alpha=0.4)
axes[0].plot(
    [flat_values.min(), flat_values.max()],
    [flat_values.min(), flat_values.max()],
    "k--",
)
axes[0].set(xlabel="Independent PSD sums", ylabel="Shared PSD sums")
axes[1].plot(tree_values - flat_values, linewidth=0.5)
axes[1].set(
    xlabel="Aligned band/channel/trial value",
    ylabel="Shared minus independent",
    title="Residual (original feature units)",
)
plt.show()

# %%
# Extend the tree and recheck equivalence
# ---------------------------------------
# Add the same additional band to ``bands`` and rerun both paths. The table
# should gain one column per channel, and the equivalence loop will cover it
# automatically. For a performance study, repeat timings on a representative
# recording and alternate execution order; a single gallery run should not be
# reported as a general speed benchmark.
