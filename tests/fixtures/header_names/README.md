# Header channel-name fixtures (real deposit files)

Read on 2026-11-03 from the public NEMAR mirrors, unchanged except for the file names (`<dataset>_<BIDS file name>`).
The vhdr files come from the NEMAR annex object (their bytes match the git-annex key); channels.tsv from `github.com/nemarDatasets/<id>`.

| Files | Case |
|---|---|
| `on003702_sub-17_task-SocialMemoryCuing_*` | vhdr: 64 channels; channels.tsv: 59 (no Fp1, Fpz, M1, M2, EOG) |
| `on004917_sub-17_task-pdm_*` | vhdr: 64 channels; channels.tsv: 66 (adds `Ref`, `Gnd`) |
