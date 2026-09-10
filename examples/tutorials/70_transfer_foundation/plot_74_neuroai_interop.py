"""How do I export recorded EEG for NeuroAI interoperability?
==========================================================

Load one real motor-imagery recording and export voltages plus an event table
with timeline, start and duration fields. This is an explicit MNE-to-PyTorch
boundary check through NeuralSet Segmenter and EegExtractor, followed by a
PyTorch DataLoader. Install neuralset to run it. EEGDash handles discovery;
NeuralSet reads the acquired signal file and extracts the event-aligned volts.
The example does not run the NeuralBench evaluator. NeuralSet preserves
float32 voltages and channel geometry;
rich MNE fields such as bad-channel flags and filter history need a separate
sidecar. Event masks are not EEG.

The processed BNCI2014-004 subset nm000135 contains C3/Cz/C4 at 250 Hz and costs
about 5.6 MB for subject 1, session 0train, run 0.
"""

# %%
# Before you start
# ----------------
#
# Use EEGDash, Braindecode, PyTorch and NeuralSet in a compatible installed
# environment; NeuralSet is an additional dependency, not supplied by the base
# EEGDash interface alone. Its ``Segmenter``, ``EegExtractor`` and dataset
# collation APIs are used directly. This workflow was exercised with NeuralSet
# 0.3.1, but that observation does not replace respecting its declared dependency
# requirements when creating a new environment.
#
# Keep about 6 MB of signal cache and allow CPU processing. The goal is equality
# of recorded voltages across library boundaries, not a decoding score. No model
# or train/test split is needed to verify the adapter. The resulting windows
# would still require a leakage-safe split before any supervised training.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neuralset as ns
from torch.utils.data import DataLoader
from braindecode.preprocessing import create_windows_from_events

from neuralset.events.etypes import Eeg
from neuralset.extractors.neuro import MneTimedArray

from eegdash import EEGDashDataset

# %%
# Keep recording identity and event time explicit
# -----------------------------------------------
#
# The query selects one genuine recording. ``timeline`` identifies that
# recording throughout the event table; using a unique timeline per file prevents
# an event from being matched to another participant's signal when scaling up.
# The event-table ``start`` and ``duration`` fields are in seconds, while MNE and
# Braindecode also expose sample indices.
#
# The zero-first-sample assertion makes the convention explicit for this example.
# A cropped recording with a nonzero sample origin needs a deliberate conversion;
# reusing these offsets unchanged would misalign signals and events. The TSV
# export preserves original BIDS descriptions for inspection, but does not
# contain voltage samples.

cache = Path(os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")).expanduser()
dataset = EEGDashDataset(
    dataset="nm000135",
    subject="1",
    session="0train",
    run="0",
    task="imagery",
    cache_dir=cache,
)
assert len(dataset.datasets) == 1
raw = dataset.datasets[0].raw
print(dataset.description.to_string(index=False))
print(raw.ch_names, raw.info["sfreq"], np.unique(raw.annotations.description))
assert raw.first_samp == 0, "This example uses recording-relative event times"
timeline = "nm000135/sub-1/ses-0train/run-0"
events = pd.DataFrame(
    {
        "timeline": timeline,
        "start": raw.annotations.onset - raw.first_time,
        "duration": raw.annotations.duration,
        "type": "Stimulus",
        "bids_description": raw.annotations.description,
    }
)
events.to_csv(cache / "plot_74_events.tsv", sep="\t", index=False)

# %%
# Braindecode extracts actual event-aligned two-second voltage windows.
# %%
# Construct an independent voltage reference
# ------------------------------------------
#
# At 250 Hz, a two-second window has 500 samples. Equal size and stride
# produce non-overlapping windows within each eligible imagery interval; there
# can be more than one window per trial. ``on_last_window="drop"`` avoids a
# partial final window. Left-hand and right-hand labels come from the recorded
# annotations, with explicit zero/one codes.
#
# The expected tensor layout is ``(windows, channels, samples)`` in volts.
# ``MneTimedArray`` checks a second boundary by carrying those recorded values and
# channel geometry through NeuralSet's array representation. This conversion
# retains float32 precision; rich MNE metadata are not all preserved. The
# round-trip assertion checks values rather than assuming that matching shapes
# imply matching EEG.

windows = create_windows_from_events(
    dataset,
    mapping={"left_hand": 0, "right_hand": 1},
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=500,
    window_stride_samples=500,
    on_last_window="drop",
    preload=True,
)
metadata = windows.get_metadata()
volts = np.stack([windows[i][0] for i in range(len(windows))])
labels = np.asarray([windows[i][1] for i in range(len(windows))])
assert np.isfinite(volts).all() and set(labels) == {0, 1}
neural_array = MneTimedArray.from_native(raw.copy().pick("eeg"))
reconstructed = neural_array.to_native()
np.testing.assert_allclose(
    reconstructed.get_data(), raw.get_data(picks="eeg"), rtol=1e-6, atol=1e-12
)
assert reconstructed.ch_names == raw.copy().pick("eeg").ch_names

# %%
# Build a NeuralSet event table with the acquired EEG file and window anchors.
# Every anchor and label comes from the real Braindecode window metadata.
# EegExtractor reads the signal payload; Stimulus rows only define timing.
# %%
# Tell NeuralSet where the voltage lives
# --------------------------------------
#
# An ``Eeg`` event points to the acquired signal file and its rate. The
# ``Stimulus`` rows supply anchors and real labels from Braindecode metadata.
# Only those rows trigger segments. They do not themselves provide EEG: the
# ``EegExtractor`` reads the signal named by the recording event.
#
# The extractor uses the native sampling frequency, original channel order and
# ``scaler=None`` because a boundary check must not introduce a new signal
# transform. ``prepare()`` prepares the extractor's data before iteration. Segment
# count and trigger-code assertions check that preparation has neither lost nor
# reordered the requested labelled windows.

recording_event = Eeg(
    filepath=str(raw.filenames[0]),
    start=0.0,
    duration=raw.n_times / raw.info["sfreq"],
    frequency=raw.info["sfreq"],
    subject="1",
    timeline=timeline,
).to_dict()
anchors = pd.DataFrame(
    {
        "type": "Stimulus",
        "timeline": timeline,
        "start": metadata["i_start_in_trial"].to_numpy() / raw.info["sfreq"],
        "duration": 2.0,
        "code": labels,
    }
)
neural_events = ns.events.standardize_events(
    pd.concat([pd.DataFrame([recording_event]), anchors], ignore_index=True)
)
segmenter = ns.Segmenter(
    trigger_query="type == 'Stimulus'",
    start=0.0,
    duration=2.0,
    extractors={
        "eeg": ns.extractors.EegExtractor(
            frequency="native",
            scaler=None,
            channel_order="original",
            mne_cpus=1,
        )
    },
)
neural_dataset = segmenter.apply(neural_events)
neural_dataset.prepare()
assert len(neural_dataset) == len(windows)
np.testing.assert_array_equal(
    [segment.trigger.code for segment in neural_dataset.segments], labels
)
# %%
# Batch with the dataset collation contract
# -----------------------------------------
#
# NeuralSet items are structured examples rather than plain NumPy arrays,
# so the DataLoader uses the dataset's own ``collate_fn``. Batches of 16 bound
# memory; zero workers keep this small demonstration straightforward and
# ``shuffle=False`` preserves the reference order. Voltages are read from
# ``batch.data["eeg"]``, not from an event-mask field.
#
# The full-array comparison tests all windows against independently extracted
# Braindecode data. A direct MNE sample slice adds an explicit first-window time
# check. Relative tolerance ``1e-6`` and absolute tolerance ``1e-12`` volts allow
# float32 conversion without hiding millivolt/microvolt mistakes, reordered
# channels or sample shifts.

loader = DataLoader(
    neural_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    collate_fn=neural_dataset.collate_fn,
)
restored = np.concatenate([batch.data["eeg"].numpy() for batch in loader])
np.testing.assert_allclose(restored, volts, rtol=1e-6, atol=1e-12)
# Independently compare the first window with MNE's sample slice.
start = int(metadata.iloc[0]["i_start_in_trial"])
expected = raw.get_data(picks="eeg", start=start, stop=start + 500)
np.testing.assert_allclose(restored[0], expected, rtol=1e-6, atol=1e-12)
print("NeuralSet voltage tensor (windows, channels, samples):", restored.shape)
print("Observed class counts:", np.unique(labels, return_counts=True))
np.savez(
    cache / "plot_74_voltages.npz",
    volts=restored,
    labels=labels,
    sfreq=raw.info["sfreq"],
    channels=raw.ch_names,
    timeline=timeline,
)

# %%
fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
times = np.arange(500) / raw.info["sfreq"]
axes[0].plot(times, expected[0] * 1e6, label="MNE voltage")
axes[0].plot(times, restored[0, 0] * 1e6, "--", label="NeuralSet → DataLoader")
axes[0].set(ylabel="C3 (µV)")
axes[0].legend()
axes[1].plot(times, (restored[0, 0] - expected[0]) * 1e6)
axes[1].set(xlabel="Time (s)", ylabel="Difference (µV)")
plt.show()

# %%
# Reuse the export without losing its meaning
# -------------------------------------------
#
# The NPZ stores volts, labels, sampling frequency, channel names and the
# timeline; the TSV preserves event timing and original descriptions. The plot
# compares independently extracted first-channel values and their actual
# difference. It is a numerical interoperability check on this recording, not
# validation of every file format or every preprocessing configuration.
#
# To extend to multiple recordings, construct a distinct ``Eeg`` event and
# timeline for each file and carry subject and original-trial identities alongside
# windows. Split complete subjects or trials before training, because multiple
# windows from one trial share their source. Explicitly decide how to preserve
# bad-channel flags, annotations and preprocessing history before treating the
# export as a replacement for the original MNE object.
#
# Related data-boundary example: `Braindecode training on MNE epochs
# <https://braindecode.org/stable/auto_examples/model_building/plot_basic_training_epochs.html>`_.

# %%
# Continue to self-supervised pretraining
# ---------------------------------------
#
# Follow NeuroAI's `Training a model: masked prediction on EEG
# <https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_pretrain_mae.html>`_
# for a worked pretraining loop, checkpoint export and downstream evaluation.
# Masked prediction reconstructs hidden portions of recorded EEG; the imagery
# labels above are not reconstruction targets.
#
# 1. Follow the guide's installation instructions for the repository's
#    ``ssl_example`` project and its training dependencies. From its
#    ``neuraltrain-repo`` directory, start with the real-data debug run:
#
#    .. code-block:: console
#
#       python -m ssl_example.grids.test_run
#
# 2. Configure the studies, subject splits and data/cache/output paths before
#    using the guide's download and full-training commands. Its default corpus
#    needs roughly 1.1 TB; the debug run uses a small real MNE recording.
#
# 3. Use the printed ``encoder.ckpt`` path in the guide's downstream evaluation
#    command. Match the encoder configuration and preprocessing to pretraining;
#    a reconstruction loss alone does not establish decoding performance.
#
# Adapting this conversion requires more than passing ``loader`` to that script.
# Here batches contain ``eeg`` at 250 Hz, with 500 samples and stimulus anchors.
# The guide constructs ``input`` batches, channel positions and recording-strided
# windows at 120 Hz for its patch encoder. Adapt its study/extractor configuration
# to the acquired recordings, retain subject-level separation, and verify the
# resulting batch contract before training. The NPZ is a voltage export, not a
# pretrained checkpoint or a registered NeuroAI study.
