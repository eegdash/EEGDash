"""EEG-to-image alignment with one real NM000134 BIDS record
==============================================================

**Difficulty 2** | **Runtime: 3–5 min on first run** | **Compute: CPU**

This start kit inspects the genuine compact test-run recording
"nm000134/sub-09/ses-01/task-images/run-02". It materializes one 32-channel,
256 Hz EDF and parses the 936 "stim_test" BIDS rows that reference 200 real
image IDs. The event stream is deliberately fast, making it useful for
checking timing and stimulus coverage before building an EEG-to-image method.

This is an inspection/start-kit for a test-run EEG-to-image alignment, not an
official train/test split, model, score, or submission. NeuralBench owns those
release-level operations.

Keywords: EEG2026, EEG-to-image, visual decoding, BIDS, NEMAR, EEGDash
"""

# %% [markdown]
# Learning objectives
# -------------------
#
# After this tutorial, you will be able to:
#
# - Materialize exactly one real NM000134 BIDS recording with EEGDashDataset.
# - Parse its "stim_test,<image_id>,..." event rows and audit their timing
#   and image-ID coverage.
# - Inspect a short, recorded EEG trace anchored to one real stimulus event.
# - Open the same EDF in EEGDash and inspect its BIDS stimulus mapping.
# - Hand official splits, models, scores, and submissions to NeuralBench.

# %% [markdown]
# Requirements
# ------------
#
# - About 3–5 minutes on CPU on the first run. The EDF and its BIDS sidecars
#   are cached by EEGDash; the viewer materializes the 200 image IDs referenced
#   by this recording's event rows.
# - Network access is needed once for NEMAR "nm000134".
# - Prerequisite: basic familiarity with continuous EEG and BIDS events.

# %%
# Setup
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir
from eegdash.viz import (
    EEGDASH_BLUE,
    EEGDASH_MINT,
    EEGDASH_ORANGE,
    EEGDASH_PURPLE,
    style_figure,
    use_eegdash_style,
)

use_eegdash_style()
mne.set_log_level("ERROR")

DATASET = "nm000134"
SUBJECT = "09"
SESSION = "01"
TASK = "images"
RUN = "02"
EXPECTED_BIDS_RELPATH = "sub-09/ses-01/eeg/sub-09_ses-01_task-images_run-02_eeg.edf"
EXPECTED_STIM_TEST_EVENTS = 936
EXPECTED_UNIQUE_IMAGE_IDS = 200
EXPECTED_EEG_CHANNELS = 32
SAMPLING_FREQUENCY = 256.0
TRACE_BEFORE_SECONDS = 0.25
TRACE_AFTER_SECONDS = 1.25
CACHE_DIR = get_default_cache_dir()


# %% [markdown]
# Step 1. Materialize one real BIDS test-run recording
# -----------------------------------------------------
#
# This fixed query is intentionally narrow: it selects only participant
# "sub-09", session "01", task "images", and run "02". It is a compact real
# recording for inspecting the BIDS interface, not a declaration of an
# official benchmark partition.

# %%
dataset = EEGDashDataset(
    cache_dir=CACHE_DIR,
    dataset=DATASET,
    subject=SUBJECT,
    session=SESSION,
    task=TASK,
    run=RUN,
    download=True,
    n_jobs=1,
)
dataset.download_all(n_jobs=1)

assert len(dataset.records) == 1, "The fixed query must resolve to one record."
record = dataset.records[0]
assert record["bids_relpath"] == EXPECTED_BIDS_RELPATH
assert float(record["sampling_frequency"]) == SAMPLING_FREQUENCY
assert int(record["nchans"]) == EXPECTED_EEG_CHANNELS

recording_path = dataset.data_dir / record["bids_relpath"]
events_path = recording_path.with_name(
    recording_path.name.replace("_eeg.edf", "_events.tsv")
)
assert recording_path.is_file(), f"Missing downloaded EDF: {recording_path}"
assert events_path.is_file(), f"Missing BIDS events: {events_path}"

raw = dataset.datasets[0].raw
assert raw is not None
sfreq = float(raw.info["sfreq"])
eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
recording_duration_s = raw.n_times / sfreq

assert sfreq == SAMPLING_FREQUENCY
assert len(eeg_picks) == EXPECTED_EEG_CHANNELS
assert raw.n_times >= int(record["ntimes"])

pd.DataFrame(
    [
        {
            "BIDS recording": record["bids_relpath"],
            "duration (s)": recording_duration_s,
            "samples": raw.n_times,
            "sample rate (Hz)": sfreq,
            "EEG channels": len(eeg_picks),
        }
    ]
).round({"duration (s)": 3, "sample rate (Hz)": 0})

# %% [markdown]
# **Investigate.** The record is a full 255-second EDF, not a fabricated
# excerpt. Its BIDS events sidecar supplies the image references used below.

# %% [markdown]
# Step 2. Parse the real image-event stream
# ------------------------------------------
#
# Each stimulus event encodes an image ID in "stim_test,<image_id>,...".
# We retain only those rows, convert their onset/sample values to numeric
# coordinates, and check the known event and image-ID totals directly from the
# downloaded sidecar.

# %%
events = pd.read_csv(events_path, sep="\t")
required_columns = {"onset", "duration", "trial_type", "sample"}
assert required_columns.issubset(events.columns), (
    f"{events_path} lacks required BIDS columns: {required_columns - set(events.columns)}"
)

stim_events = events.loc[
    events["trial_type"].astype(str).str.startswith("stim_test,")
].copy()
stim_events["onset"] = pd.to_numeric(stim_events["onset"], errors="raise")
stim_events["sample"] = pd.to_numeric(stim_events["sample"], errors="raise").astype(
    int
)
stim_events["image_id"] = stim_events["trial_type"].str.extract(
    r"^stim_test,(\d+),", expand=False
)
assert stim_events["image_id"].notna().all()
stim_events["image_id"] = pd.to_numeric(
    stim_events["image_id"], errors="raise"
).astype(int)

assert len(stim_events) == EXPECTED_STIM_TEST_EVENTS
assert stim_events["image_id"].nunique() == EXPECTED_UNIQUE_IMAGE_IDS
assert stim_events["image_id"].min() == 16540
assert stim_events["image_id"].max() == 16739

event_intervals_s = stim_events["onset"].sort_values().diff().dropna()
image_repeats = stim_events.groupby("image_id").size().sort_index()

pd.DataFrame(
    {
        "stimulus events": [len(stim_events)],
        "unique image IDs": [image_repeats.size],
        "median event interval (s)": [event_intervals_s.median()],
        "median repeats per image": [image_repeats.median()],
    }
).round({"median event interval (s)": 3, "median repeats per image": 1})

# %% [markdown]
# Step 3. Diagnose real event cadence and image coverage
# -------------------------------------------------------
#
# **Run.** The left panel exposes the actual inter-event timing, including
# pauses in the rapid stream. The right panel shows how often each real image
# ID appears in this one recording. It is an event-sidecar audit, not a model
# or a competition score.

# %%
fig, (cadence_axis, coverage_axis) = plt.subplots(1, 2, figsize=(11, 4.3))
style_figure(
    fig,
    title="Real NM000134 stimulus cadence and image coverage",
    subtitle="936 BIDS stimulus events across 200 image IDs in one test-run EDF",
    source="NEMAR nm000134; quantities are parsed from this recording's events.tsv.",
    grid_axis="y",
)
fig.subplots_adjust(top=0.72, bottom=0.24, wspace=0.34)

cadence_axis.plot(
    np.arange(1, len(stim_events)),
    event_intervals_s.to_numpy(),
    color=EEGDASH_BLUE,
    lw=0.8,
)
cadence_axis.axhline(
    event_intervals_s.median(),
    color=EEGDASH_ORANGE,
    lw=1.1,
    linestyle="--",
    label=f"median {event_intervals_s.median():.3f} s",
)
cadence_axis.set(
    xlabel="Stimulus-event order",
    ylabel="Seconds since previous stimulus",
    title="Observed event cadence",
)
cadence_axis.legend(frameon=False, loc="upper right")

coverage_axis.bar(
    image_repeats.index,
    image_repeats.to_numpy(),
    width=0.9,
    color=EEGDASH_MINT,
    edgecolor="none",
)
coverage_axis.axhline(
    image_repeats.mean(),
    color=EEGDASH_PURPLE,
    lw=1.1,
    linestyle="--",
    label=f"mean {image_repeats.mean():.2f} repeats",
)
coverage_axis.set(
    xlim=(image_repeats.index.min() - 1, image_repeats.index.max() + 1),
    xlabel="BIDS image ID",
    ylabel="Stimulus events",
    title="Coverage of referenced images",
)
coverage_axis.legend(frameon=False, loc="upper right")
plt.show()

# %% [markdown]
# Step 4. Inspect a short EEG trace anchored to a real image event
# -----------------------------------------------------------------
#
# We choose the first "stim_test" row with enough recorded context on both
# sides, then index the EDF with its BIDS "sample" value. This is a signal
# check around one observed image presentation, not an embedding, target, or
# prediction.

# %%
usable_events = stim_events.loc[
    (stim_events["onset"] >= TRACE_BEFORE_SECONDS)
    & (stim_events["onset"] + TRACE_AFTER_SECONDS <= recording_duration_s)
]
assert not usable_events.empty
anchor_event = usable_events.iloc[0]
anchor_sample = int(anchor_event["sample"])
pre_samples = int(round(TRACE_BEFORE_SECONDS * sfreq))
post_samples = int(round(TRACE_AFTER_SECONDS * sfreq))
start_sample = anchor_sample - pre_samples
stop_sample = anchor_sample + post_samples
trace_picks = eeg_picks[:4]

assert start_sample >= 0
assert stop_sample <= raw.n_times
trace = raw.get_data(picks=trace_picks, start=start_sample, stop=stop_sample)
trace_time_s = np.arange(trace.shape[-1]) / sfreq - TRACE_BEFORE_SECONDS
trace_microvolts = trace * 1e6
trace_names = [raw.ch_names[pick] for pick in trace_picks]
spacing = max(float(np.nanpercentile(np.abs(trace_microvolts), 95)) * 2.6, 10.0)

fig, axis = plt.subplots(figsize=(10.8, 4.6))
style_figure(
    fig,
    title="Recorded EEG around one real image event",
    subtitle=(
        f"image ID {int(anchor_event.image_id)} at "
        f"{float(anchor_event.onset):.3f} seconds in the BIDS event stream"
    ),
    source="Four EEG channels from the NM000134 EDF; amplitudes shown in µV.",
    grid_axis="x",
)
fig.subplots_adjust(top=0.72, bottom=0.22, left=0.15)

for channel_index, (channel_name, trace_channel, color) in enumerate(
    zip(
        trace_names,
        trace_microvolts,
        (EEGDASH_BLUE, EEGDASH_MINT, EEGDASH_PURPLE, EEGDASH_ORANGE),
        strict=True,
    )
):
    offset = (len(trace_names) - channel_index - 1) * spacing
    axis.plot(trace_time_s, trace_channel + offset, color=color, lw=0.75)
    axis.text(
        trace_time_s[0] - 0.02,
        offset,
        channel_name,
        ha="right",
        va="center",
        fontsize=8,
    )

axis.axvline(0, color="#475569", lw=1.0, linestyle="--", label="BIDS event")
axis.set(
    xlabel="Seconds relative to image-event sample",
    ylabel="Recorded EEG (µV; offset traces)",
    yticks=[],
)
axis.legend(frameon=False, loc="upper right")
plt.show()
raw.close()

# %% [markdown]
# Step 5. Inspect the EDF and its BIDS stimulus mapping in EEGDash
# -----------------------------------------------------------------
#
# The inline viewer opens the same real EDF and BIDS sidecars. When you pan or
# move its cursor, it chooses the BIDS image nearest the rendered EEG
# window/cursor. Its images are materialized only from the image IDs referenced
# by this recording's event rows; this tutorial never copies or proxies JPEGs.

# %%
dataset.plot(index=0, height=520)

# %% [markdown]
# Result: move release-level benchmarking to NeuralBench
# -------------------------------------------------------
#
# EEGDash has provided one real EDF, an event cadence/coverage audit, a short
# event-anchored EEG trace, and an interactive BIDS stimulus inspection. This
# is not an official train/test split, model, score, or submission.
# NeuralBench owns the official split, model, score, and submission workflow
# for the release; use its task tooling for those operations rather than
# inferring them from this single-record inspection.
