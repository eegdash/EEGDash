# iEEG channel-position fixtures (real deposit sidecars)

Read on 2026-11-03 from the public NEMAR mirrors (`github.com/nemarDatasets/<id>`),
unchanged except for the file names (`<dataset>_<BIDS file name>`).

| Files | Case |
|---|---|
| `on006910_sub-098_ses-3_*` | channels.tsv lists 128 contacts `A1`..`B64` (20 marked bad); the EDF holds 49 bipolar `POL B10-B18`.. channels |
| `edf_header_labels.json` | the EDF signal labels of on006910 `sub-098_ses-3_task-auditory_ieeg.edf` and on004551 `sub-098_ses-01_task-sleep_run-01_ieeg.edf`, read from the NEMAR annex object (identical to the OpenNeuro original) |
| `on004551_sub-098_ses-01_*` | EDF labels `EEG A1-Ref`, `POL A3`.. for contacts `A1`, `A3`.. |
| `on004789_sub-R1148P_*`, `on004809_sub-R1106M_*` | `acq-bipolar` channels `RAFA1-RAFA2`.. over `space-MNI152NLin6ASym` contacts |
| `nm000358_sub-03_ses-5_*` | channels named exactly like contacts; `LAD_080`..`LAD_095` are `n/a` |
