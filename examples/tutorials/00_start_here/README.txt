Start Here
==========

Three short, CPU-only lessons that take you from a fresh ``pip install
eegdash`` to a working PyTorch ``DataLoader`` over real BIDS-curated EEG
records. Each tutorial runs in under a few minutes, executes on the
docs CI matrix on every build, and has a matching YAML spec under
``docs/tutorials/_spec/``.

This category is the gateway to every other section in this gallery.
The core decoding workflow assumes you have already loaded a recording;
the features and evaluation tracks assume you understand windowing; the
transfer / foundation track assumes you can wire an ``EEGDashDataset``
into a dataloader. Sourced from ``docs/tutorial_restructure_plan.md``
Category A (lines 360-380), with BIDS metadata handling per Pernet et
al. (2019).

What you will learn:

- How to query the EEGDash index for datasets and records without
  downloading raw signals.
- How to load one BIDS recording, inspect its sampling rate, channels,
  and events, and verify it before scaling up.
- How to apply Braindecode-safe preprocessors and convert continuous
  signal into fixed-length windows.
- How to wrap an ``EEGDashDataset`` in a ``torch.utils.data.DataLoader``
  and confirm one batch's shape.
- The vocabulary -- BIDS entities, record vs dataset documents, windows
  -- you'll see in every later tutorial.

Run the lessons in order:

1. ``plot_00_first_search.py`` -- find datasets and records.
2. ``plot_01_first_recording.py`` -- load one recording and inspect it.
3. ``plot_02_dataset_to_dataloader.py`` -- build windows and a
   PyTorch ``DataLoader``.
