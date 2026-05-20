# EEG / iEEG / MEG fixture attribution

Files under `tests/fixtures/{eeg,ieeg,meg}/` are derived from open-access
datasets and used as fuzz-test corpus seeds + parser-contract regression
inputs. Binary files are truncated to the smallest useful prefix
(typically the first 32–64 KB) so the repository stays compact.

All sources permit redistribution. Per OpenNeuro the default license is
**CC0 1.0 Universal**; the MNE-Python test data is licensed
**BSD-3-Clause**. Each fixture's source dataset is listed below.

We attribute even where the license does not require it.

## Inventory — by format × modality

| Format | Modality | Fixture file | Size | Source | License |
|---|---|---|---:|---|---|
| EDF | EEG | `eeg/sub-01_ses-01_task-offline_run-01_eeg.edf` | 32 KB | [ds002034](https://openneuro.org/datasets/ds002034) | CC0 |
| BDF (24-bit EDF) | EEG | `eeg/sub-001_ses-01_task-meditation_eeg.bdf` | 32 KB | [ds001787](https://openneuro.org/datasets/ds001787) | CC0 |
| BrainVision (.vhdr) | EEG | `eeg/sub-xp101_task-motorloc_eeg.vhdr` | 12 KB | [ds002336](https://openneuro.org/datasets/ds002336) | CC0 |
| BrainVision (.vmrk) | EEG | `eeg/sub-xp101_task-motorloc_eeg.vmrk` | 6 KB | [ds002336](https://openneuro.org/datasets/ds002336) | CC0 |
| BrainVision (.eeg)  | EEG | `eeg/sub-xp101_task-motorloc_eeg.eeg` | 32 KB | [ds002336](https://openneuro.org/datasets/ds002336) | CC0 |
| EEGLAB (.set) | EEG | `eeg/sub-001_task-AuditoryVisualShift_run-01_eeg.set` | 64 KB | [ds002893](https://openneuro.org/datasets/ds002893) | CC0 |
| BrainVision (.vhdr) | iEEG | `ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr` | 2.6 KB | [ds003688](https://openneuro.org/datasets/ds003688) | CC0 |
| BrainVision (.vmrk) | iEEG | `ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vmrk` | 1.0 KB | [ds003688](https://openneuro.org/datasets/ds003688) | CC0 |
| BrainVision (.eeg)  | iEEG | `ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.eeg` | 32 KB | [ds003688](https://openneuro.org/datasets/ds003688) | CC0 |
| FIFF (projection vectors) | MEG | `meg/test-proj.fif` | 4.5 KB | [mne-tools/mne-python](https://github.com/mne-tools/mne-python) | BSD-3 |
| FIFF (annotations) | MEG | `meg/test_raw-annot.fif` | 273 B | [mne-tools/mne-python](https://github.com/mne-tools/mne-python) | BSD-3 |
| FIFF (events) | MEG | `meg/test-eve.fif` | 543 B | [mne-tools/mne-python](https://github.com/mne-tools/mne-python) | BSD-3 |

**Total committed:** ~220 KB across 12 files spanning 4 formats × 3 modalities.

## Dataset citations

- **ds002034** — Schneider C., Pereira M., Tonin L., Millán J.d.R.
  *Real-time EEG feedback on alpha power lateralization (see DOI for title)
  Brain Topogr (2019). doi:10.1007/s10548-019-00725-9
  Accession: doi:10.18112/openneuro.ds002034.v1.0.3

- **ds001787** — *Meditation EEG* (Biosemi 64-channel + facial EMG,
  256 Hz, BDF format). Accession: doi:10.18112/openneuro.ds001787

- **ds002336** — Lioi G., Cury C., Perronnet L., Mano M., Bannier E.,
  Lécuyer A., Barillot C.
  *Simultaneous MRI-EEG during a motor imagery neurofeedback task: an
  open access brain imaging dataset for multi-modal data integration.*
  bioRxiv 862375. doi:10.1101/862375

- **ds002893** — Ceponiene R., Westerfield M., Torki M., Townsend J.
  *Modality-specificity of sensory aging in vision and audition:
  Evidence from event-related potentials.*
  Brain Research 1215 (2008) 53-68. doi:10.1016/j.brainres.2008.02.010
  Accession: doi:10.18112/openneuro.ds002893.v2.0.0

- **ds003688** — Hermes D., Petridou N., Kay K.N., Winawer J.
  *Open multimodal iEEG-fMRI dataset from naturalistic stimulation
  with a short audiovisual film.*
  Accession: doi:10.18112/openneuro.ds003688

- **MNE-Python test data** — Gramfort et al., MNE-Python project,
  BSD-3-Clause. Original files from
  [mne/io/tests/data/](https://github.com/mne-tools/mne-python/tree/main/mne/io/tests/data).
  See [LICENSE](https://github.com/mne-tools/mne-python/blob/main/LICENSE.txt).

## Known parser issues surfaced by adding these fixtures

- **fiff.js cannot read real FIFF files.** The parser at `formats/fiff.js:66`
  validates the file by looking for literal ASCII `"FIFF"` magic bytes
  in the first 4 bytes. Real FIFF files start with a TAG (typically
  `FIFF_FILE_ID`, `kind=100`) — the first 4 bytes are `00 00 00 64`,
  not `46 49 46 46`. None of the three FIFF fixtures parse with the
  current `fiff.js` (`Not a valid FIFF file` thrown). Tracked separately
  as a parser bug; the fixtures still seed the fuzz suite usefully —
  fast-check mutates around the FIFF tag structure regardless of
  whether the magic check is correct.

## How to re-fetch / extend

```bash
mkdir -p tests/fixtures/{eeg,ieeg,meg}

# EEG — EDF
curl -sSL -H "Range: bytes=0-32767" -o tests/fixtures/eeg/sub-01_ses-01_task-offline_run-01_eeg.edf \
  "https://s3.amazonaws.com/openneuro.org/ds002034/sub-01/ses-01/eeg/sub-01_ses-01_task-offline_run-01_eeg.edf"

# EEG — BDF
curl -sSL -H "Range: bytes=0-32767" -o tests/fixtures/eeg/sub-001_ses-01_task-meditation_eeg.bdf \
  "https://s3.amazonaws.com/openneuro.org/ds001787/sub-001/ses-01/eeg/sub-001_ses-01_task-meditation_eeg.bdf"

# EEG — BrainVision triple
curl -sSL -o tests/fixtures/eeg/sub-xp101_task-motorloc_eeg.vhdr \
  "https://s3.amazonaws.com/openneuro.org/ds002336/sub-xp101/eeg/sub-xp101_task-motorloc_eeg.vhdr"
curl -sSL -o tests/fixtures/eeg/sub-xp101_task-motorloc_eeg.vmrk \
  "https://s3.amazonaws.com/openneuro.org/ds002336/sub-xp101/eeg/sub-xp101_task-motorloc_eeg.vmrk"
curl -sSL -H "Range: bytes=0-32767" -o tests/fixtures/eeg/sub-xp101_task-motorloc_eeg.eeg \
  "https://s3.amazonaws.com/openneuro.org/ds002336/sub-xp101/eeg/sub-xp101_task-motorloc_eeg.eeg"

# EEG — EEGLAB
curl -sSL -H "Range: bytes=0-65535" -o tests/fixtures/eeg/sub-001_task-AuditoryVisualShift_run-01_eeg.set \
  "https://s3.amazonaws.com/openneuro.org/ds002893/sub-001/eeg/sub-001_task-AuditoryVisualShift_run-01_eeg.set"

# iEEG — BrainVision triple
curl -sSL -o tests/fixtures/ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr \
  "https://s3.amazonaws.com/openneuro.org/ds003688/sub-01/ses-iemu/ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr"
curl -sSL -o tests/fixtures/ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vmrk \
  "https://s3.amazonaws.com/openneuro.org/ds003688/sub-01/ses-iemu/ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vmrk"
curl -sSL -H "Range: bytes=0-32767" -o tests/fixtures/ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.eeg \
  "https://s3.amazonaws.com/openneuro.org/ds003688/sub-01/ses-iemu/ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.eeg"

# MEG — FIFF (small BSD-licensed samples from MNE-Python)
curl -sSL -o tests/fixtures/meg/test-proj.fif \
  "https://raw.githubusercontent.com/mne-tools/mne-python/main/mne/io/tests/data/test-proj.fif"
curl -sSL -o tests/fixtures/meg/test_raw-annot.fif \
  "https://raw.githubusercontent.com/mne-tools/mne-python/main/mne/io/tests/data/test_raw-annot.fif"
curl -sSL -o tests/fixtures/meg/test-eve.fif \
  "https://raw.githubusercontent.com/mne-tools/mne-python/main/mne/io/tests/data/test-eve.fif"
```

## TODO

- **NIRS** — separate file format (.snirf / .nirs), not currently
  supported by the viewer. Add when the reader exists.
- **CTF MEG** — `.ds` directory format, not currently supported.
- **EDF/iEEG** — most iEEG datasets use BrainVision; if you find a
  small EDF-format iEEG dataset, add it for parser-cross-modality coverage.
