Core Decoding Workflow
======================

Preprocess real nm000118 SSVEP data, compare trial and subject splits, fit a spectral baseline, and persist prepared windows. Each script runs independently on one to three participants (about 7–21 MB) and uses EEGDASH_CACHE_DIR for persistent storage. Subject-disjoint evaluation uses scikit-learn group splitting.
