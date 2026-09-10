EEGDash examples: recorded data from discovery to evaluation
============================================================

All examples use real data or metadata acquired through EEGDash APIs.
Signals, event/participant targets and reported results are never generated as
substitutes for unavailable data. Random model initialization, training order
and legitimate data splitting remain ordinary parts of machine learning.

The examples use EEGDash's public feature APIs and EEGPrep's Braindecode
adapters where preprocessing is appropriate. They avoid local training,
feature and plotting helpers. The MOABB example retains the dataset adapter
methods required by MOABB's interface. Metadata-only pages, already-processed
derivatives and non-EEG signals do not receive unnecessary EEG cleaning.

Install the project's documentation extra for the complete gallery, including
EEGPrep's EEG/MNE conversion support.
Individual pages explain their preprocessing and optional dependencies.

Start with the numbered tutorials. The general workflow uses explicit small
Nakanishi2015 SSVEP subsets (nm000118; about 7 MB per participant). Cross-session
and cache examples use BNCI2014-004 (nm000135; about 5 MB per session). These
are recorded datasets distributed in processed/converted form; each tutorial
explains its provenance and event mapping.

Set EEGDASH_CACHE_DIR to a persistent directory and reuse it between examples.
Tutorial 40 writes the feature table consumed by 42; execute 40 first. Other
examples specify their own real source and any optional dependencies.

CI executes the basic/core/features/evaluation and how-to scripts on the same
small real subset used by readers (about 32 MB across the two datasets), plus
Track 2 on the same imagery data and Track 4 on about 24 MB of real EMG2Pose.
It checks every example for signal/target-generation code. Other task-specific
examples are rendered in CI but require larger downloads for execution. The full
local gallery runs every public script by default; EEGDASH_GALLERY_FILENAME_PATTERN
can select a subset explicitly. No CI failure is replaced with invented data.

P300, HBN, image-retrieval, sleep and EMG pages retain their actual scientific
tasks. Their larger source recordings are disclosed in the introductions;
cropping reduces computation, not the initial source-file download.
