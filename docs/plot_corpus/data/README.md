# EEGDash Corpus data (generated copy, do not hand-edit)

These files are a generated copy. The source of truth is the corpus tables of the
REVE replication repository (`corpus/` there, the atlas table and position bank under `corpus/corpus-atlas/`), checked by its own table checker;
`sync_data.py` copies the derived files from it, scans every cell for internal names
and paths, and records what it copied in `MANIFEST.json`.

| file | what it is | rows |
|---|---|---|
| `atlas.csv` | one row per dataset of the universe (every dataset named by REVE's paper or release card, plus the open datasets considered for the next version): status, licence, counts, hours, provenance, note, link; every row counts (a `tracked` column left by an earlier sync is ignored) | one per dataset |
| `eegdash-corpus-v0-channels.csv` | one row per bucket of the converted v0 store (`reve-public-nc.zarr`; I-CARE under its EEGDash id `nm000289`): one exact set of channel names of one dataset, its sessions, hours and channel names | one per bucket |
| `eegdash-corpus-v1-channels.csv` | one row per channel set of the datasets v1 adds, read from EEGDash's records (channel names the position bank knows, records and hours carrying the set); the v1 electrode map's input until those datasets are converted | one per channel set |
| `eegdash-corpus-v0-recordings.csv` | one row per converted dataset: recordings and subjects, recording length in minutes (10th, 25th, 50th, 75th, 90th percentile and longest), recordings and hours per subject (median and most), tokens per epoch in the converted store. Optional: the recording, per-subject and token charts say so when absent | one per dataset |
| `eegdash-corpus-v1-recordings.csv` | the same columns for the datasets v1 adds only, from EEGDash's records; `tokens` empty (tokens are counted at conversion) | one per added dataset |
| `eegdash-corpus-demographics.csv` | one row per dataset whose records state ages or sex: year, ages known and their quantiles (min, 25th, 50th, 75th percentile, max, in years), male and female counts, tasks; empty cells where unknown. Optional | one per dataset with data |
| `electrodes.json` | one entry per electrode position used by those channel sets: spellings, `xyz` (metres, head-centred, rounded to 1e-4 m), hours, datasets, buckets of the v0 store, and `added_hours`, `added_datasets`, `added_buckets` from the datasets v1 adds. Positions derive from REVE's public position bank (`brain-bzh/reve-positions`); spellings that differ only in case share a coordinate and are merged | one per position |
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
