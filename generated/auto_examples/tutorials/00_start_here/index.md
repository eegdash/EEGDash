# Start Here

Three short, CPU-only lessons that take you from a fresh `pip install
eegdash` to a working PyTorch `DataLoader` over real BIDS-curated EEG
records. Each tutorial runs in under a few minutes, executes on the
docs CI matrix on every build, and has a matching YAML spec under
`docs/tutorials/_spec/`.

This category is the gateway to every other section in this gallery.
The core decoding workflow assumes you have already loaded a recording;
the features and evaluation tracks assume you understand windowing; the
transfer / foundation track assumes you can wire an `EEGDashDataset`
into a dataloader. Sourced from `docs/tutorial_restructure_plan.md`
Category A (lines 360-380), with BIDS metadata handling per Pernet et
al. (2019).

What you will learn:

- How to query the EEGDash index for datasets and records without
  downloading raw signals.
- How to load one BIDS recording, inspect its sampling rate, channels,
  and events, and verify it before scaling up.
- How to apply Braindecode-safe preprocessors and convert continuous
  signal into fixed-length windows.
- How to wrap an `EEGDashDataset` in a `torch.utils.data.DataLoader`
  and confirm one batch’s shape.
- The vocabulary – BIDS entities, record vs dataset documents, windows
  – you’ll see in every later tutorial.

Run the lessons in order:

1. `plot_00_first_search.py` – find datasets and records.
2. `plot_01_first_recording.py` – load one recording and inspect it.
3. `plot_02_dataset_to_dataloader.py` – build windows and a
   PyTorch `DataLoader`.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="EEGDash exposes a metadata index over hundreds of BIDS-curated EEG datasets, served by the public REST API at https://data.eegdash.org. The same catalogue powers NEMAR, the EEGLAB-ecosystem portal that hosts EEG/MEG datasets with browsing, compute, and provenance tools (Delorme et al., 2022). The EEGDash client searches, filters, and summarises that index without downloading a single sample.">  <div class="sphx-glr-thumbnail-title">How do I find datasets in EEGDash?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Once a search returns a candidate dataset (see plot_00), the next question is practical: what does one recording actually contain? This tutorial loads a single BIDS file from OpenNeuro ds004504 (Miltiadous et al. 2023; Alzheimer / frontotemporal dementia / healthy controls) through the catalogue shared with NEMAR delorme2022nemar, unwraps the mne.io.Raw object, and inspects channels, montage, and spectrum. The dataset is small (88 subjects, ~10 min per subject, 19 channels at 500 Hz on a standard 10-20 layout) so the first fetch is under 25 MB and every later step is a hot-cache read.">  <div class="sphx-glr-thumbnail-title">How do I load one EEG recording from EEGDash?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="A model trains on tensor batches, not on continuous voltage traces. This tutorial closes the gap on one BIDS recording from OpenNeuro ds002718 wakeman2015, reachable through NEMAR delorme2022nemar: two safe preprocessors, a fixed-length window step, a torch.utils.data.DataLoader paszke2019pytorch, and an optional Zarr cache that turns batch reads into a few milliseconds of random access. We do not train a model. The deliverable is one batch&#x27;s shape and dtype.">  <div class="sphx-glr-thumbnail-title">How do I turn one EEG recording into a PyTorch DataLoader?</div>
</div>
<!-- thumbnail-parent-div-close --></div>
