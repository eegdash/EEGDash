Applied Projects
================

Self-contained projects using recorded EEG and observed participant metadata.
Age, sex and p-factor projects select six ds005505 resting-state participants
(approximately 595.8 MB of signal files) and evaluate at participant level.
The clinical summary uses metadata only. P300 transfer uses recorded oddball
events. The eyes-open/closed CNN project trains ShallowFBCSPNet within one
ds005514 participant (about 90 MB) using block-wise train/test splits. Set EEGDASH_CACHE_DIR to reuse acquisitions. These small selected
cohorts demonstrate analysis workflows, not clinical or population claims.
