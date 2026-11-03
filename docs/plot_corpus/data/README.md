# EEGDash Corpus data (generated copy, do not hand-edit)

These files are a generated copy. The source of truth is the corpus tables of the
REVE replication repository (`corpus/` there, the atlas table and position bank under `corpus/corpus-atlas/`), checked by its own table checker;
`sync_data.py` copies the derived files from it (`atlas.csv` as its public projection: the source's
internal `note` column and the columns of unreleased stores dropped, a public `reason` derived in place
of the note; `eegdash-corpus-v0.csv` without its internal `note` column), scans every published file for internal names, paths, dates, unreleased stores and retired
labels, and records what it copied in `MANIFEST.json`.
The pages describe one release, EEGDash Corpus v0.

| file | what it is | rows |
|---|---|---|
| `atlas.csv` | one row per dataset of the universe (every dataset named by REVE's paper or its release card): status (one of four: converted and released in v0, blocked by the licence, gated by an access agreement, excluded), licence, counts, hours, provenance, reason (why a dataset is not in the corpus, empty for a converted one), link; every row counts (a `tracked` column left by an earlier sync is ignored) | one per dataset |
| `eegdash-corpus-v0-channels.csv` | one row per bucket of the released v0 store (`eegdash-corpus-v0.zarr`): one exact set of channel names of one dataset, its sessions, hours and channel names | one per bucket |
| `eegdash-corpus-v0-recordings.csv` | one row per converted dataset: recordings and subjects, recording length in minutes (10th, 25th, 50th, 75th, 90th percentile and longest), recordings and hours per subject (median and most), tokens per epoch in the converted store. Optional: the recording, per-subject and token charts say so when absent | one per dataset |
| `eegdash-corpus-demographics.csv` | one row per dataset whose records state ages or sex: year, ages known and their quantiles (min, 25th, 50th, 75th percentile, max, in years), male and female counts, tasks; empty cells where unknown. Optional | one per dataset with data |
| `eegdash-corpus-v0.csv` | the corpus table (source `corpus/eegdash-corpus-v0.csv`), one row per dataset of v0: the list `eegdash.corpus.EEGDashCorpus` takes and the v0 page offers as a download (`prepare_corpus.py` copies it, byte for byte, into the generated static folder, and refuses a table whose ids are not v0's rows of `atlas.csv`). Columns: `dataset_id` (the id the corpus object reads), `eegdash_id` (the id EEGDash serves it under: `on…` for an OpenNeuro dataset re-hosted on NEMAR), `short_name`, `name`, `paradigm`, `archive`, `datatypes`, `status`, `n_subjects`, `n_sessions`, `hours`, `mean_channels`, `native_sfreq`, `raw_GB`, `licence` (SPDX), `distribution` and `publishable` (the terms the licence sets on a copy) | one per dataset of v0 |
| `electrodes.json` | one entry per electrode position used by the v0 store's channel sets: spellings, `xyz` (metres, head-centred, rounded to 1e-4 m), hours, datasets, buckets; and `unplaced`, the channel names the position bank lacks (placed by the converter from their dataset's own coordinates, not drawn). Positions derive from the converter's bank (REVE's public `brain-bzh/reve-positions` and MNE's BioSemi-160/256 templates); spellings that differ only in case share a coordinate and are merged | one per position |
| `MANIFEST.json` | source commit, sync date, sha256 and row count per file | |

Resync (from a checkout of the source repository next to this one):

```
python docs/plot_corpus/sync_data.py --from ../replicating-reve   # refuses a dirty source tree; --allow-dirty to override
python docs/plot_corpus/sync_data.py --check                      # re-hash against MANIFEST.json
```

The optional tables are copied when the source carries them and listed in `MANIFEST.json` only then.
The pages are rebuilt from these files by `docs/prepare_corpus.py` (run by `make html`); no step touches the network.

Provenance labels in `atlas.csv` (`provenance` column, one label per numeric column): `zarr` measured in the
converted Zarr store, `eegdash` EEGDash's catalogue records, `card` REVE's release card, `stated` REVE's paper,
`download` size of a downloaded copy.
