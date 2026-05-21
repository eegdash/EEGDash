# fNIRS fixture attribution

## openneuro_real.snirf

- **Source**: OpenNeuro [`ds007554`](https://openneuro.org/datasets/ds007554)
- **Path in dataset**: `sub-001/ses-01/nirs/sub-001_ses-01_task-activemotor_nirs.snirf`
- **License**: CC0 1.0 Universal (OpenNeuro standard)
- **Size**: 748,624 bytes (~731 KB)
- **DOI**: [10.18112/openneuro.ds007554.v1.0.0](https://doi.org/10.18112/openneuro.ds007554.v1.0.0)
- **Dataset**: "Multimodal dataset from the CMx7-MM Experiment" by Ajra et al.
- **Format**: SNIRF v1.0 (HDF5)
- **Signal**: fNIRS ~10 Hz, 32 channels (16 sources × 2 wavelengths: 850 nm / 760 nm), 2151 samples (~3.5 min)

## Why this fixture exists

`tests/test_snirf_real_fixture.py` validates `_snirf_parser.py` against real
BIDS data instead of a synthetic h5py construction. Lesson #1 in
`ROBUSTNESS/NEXT-SPRINT-PLAN.md`: synthetic fixtures only validate the
parser against itself. The C5.1 MEF3 fixture caught a real-data sfreq
offset bug; this SNIRF fixture (landed 2026-05-22) caught a real-data
`n_times` extraction bug (`raw.n_times` / `len(time_data)` were
available but never written to the result dict).

## Recovery

If the fixture goes missing:

```bash
cd scripts/ingestions/tests/fixtures && mkdir -p fnirs && \
  curl -L -o fnirs/openneuro_real.snirf \
    'https://s3.amazonaws.com/openneuro.org/ds007554/sub-001/ses-01/nirs/sub-001_ses-01_task-activemotor_nirs.snirf'
```

`pytest tests/test_snirf_real_fixture.py -v` will skip until the file exists.
