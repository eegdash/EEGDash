Event-related EEG analysis
==========================

Load real visual and auditory oddball recordings through EEGDashDataset.
Inspect actual annotations, define event-locked epochs and compare measured
responses. The visual example evaluates a classifier on held-out subjects;
the auditory example inspects one recorded run. The studies differ in more
than stimulus modality, so their comparison is descriptive, not controlled.

1. ``plot_20_visual_p300_oddball.py``: three visual recordings, about 69 MB.
2. ``plot_21_auditory_oddball.py``: one auditory run, about 63.4 MB.

Keep EEGDASH_CACHE_DIR between runs. Neither example prescribes an ERP
amplitude or substitutes reference values for the measured result.
