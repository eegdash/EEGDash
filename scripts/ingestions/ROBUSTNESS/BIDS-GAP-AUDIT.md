# BIDS specification — what we extract vs. what BIDS specifies

Audit done 2026-05-22 against BIDS v1.6.0 spec (EEG / iEEG / MEG / fNIRS
extensions). Compares the fields currently captured by `extract_record`
and `extract_dataset_metadata` against the BIDS-required, BIDS-
recommended, and BIDS-optional fields.

The Pydantic models use `extra="allow"` so adding fields is non-
breaking. The question is which fields are **worth** capturing.

## Currently captured

### Record level (`*_eeg.json` / etc. + entities)

| Field | Source | BIDS requirement |
|---|---|---|
| `subject`, `session`, `task`, `run`, `acquisition` | filename entities | required by BIDS naming |
| `bids_relpath` | path | (internal) |
| `datatype`, `suffix`, `extension`, `recording_modality` | filename | required |
| `sampling_frequency` | sidecar / channels.tsv / binary parser cascade | **required by BIDS** |
| `nchans` | sidecar / channels.tsv / binary parser cascade | (recommended) |
| `ntimes`, `ch_names` | binary parser / MNE | (derived) |
| `participant_tsv` | full row from participants.tsv | recommended fields |
| `storage` (base, backend, dep_keys, annex_keys, sidecar_inline) | (internal) | n/a |
| `_metadata_provenance` (P0.1) | (internal cascade) | n/a |
| `_data_integrity_issues` | companion-file validation | n/a |

### Dataset level (`dataset_description.json` + walks)

| Field | Source | BIDS requirement |
|---|---|---|
| `name` | dataset_description.json | required |
| `bids_version` | dataset_description.json | required |
| `license` | dataset_description.json | recommended |
| `authors` | dataset_description.json | recommended |
| `funding` | dataset_description.json | optional |
| `dataset_doi` | dataset_description.json | optional |
| `readme` | README file | recommended |
| `recording_modality`, `datatypes`, `tasks`, `sessions` | filename walks | derived |
| `subjects_count`, `ages`, `sex_distribution`, `handedness_distribution` | participants.tsv | (recommended cols) |

## Gaps — what BIDS specifies but we DON'T extract

### High-leverage gaps (worth capturing in C6.1)

**From `*_eeg.json` / `*_ieeg.json` / `*_meg.json` sidecars** (the
modality-specific JSON next to each recording):

| BIDS field | BIDS req level | Why it matters operationally |
|---|---|---|
| `PowerLineFrequency` | **REQUIRED** | 50 vs 60 Hz — filter design. Critical for any spectral analysis. |
| `EEGReference` | **REQUIRED for EEG** | Cz / mastoid / common-average / linked-mastoid. Determines what re-referencing is possible downstream. |
| `SoftwareFilters` | **REQUIRED** | Filters ALREADY applied (e.g. `{"HighPass": 0.1, "LowPass": 60}`). Without this, downstream analyses double-filter silently. |
| `HardwareFilters` | recommended | Analog filter spec — anti-aliasing cutoff matters for high-sfreq data. |
| `Manufacturer` | recommended | Brain Products / BioSemi / EGI / etc. Vendor quirks: BioSemi 24-bit BDF endianness, EGI HydroCel polarity. |
| `ManufacturersModelName` | recommended | Specific amplifier model — gain ranges, filter chains. |
| `EEGPlacementScheme` | recommended | "10-20", "10-10", "GSN HydroCel 256". Direct match to the existing `_montage.py` template-matching code path. |

**From `channels.tsv`** (per-channel TSV alongside the recording):

| BIDS field | Why it matters |
|---|---|
| Per-channel `type` (EEG/EOG/EMG/ECG/TRIG/MISC) | We use this in `_parse_channels_tsv_for_eeg` to filter EEG channels for montage matching but DON'T expose the type-distribution at the record level. The count of EEG vs reference vs trigger channels is a useful summary. |
| Per-channel `status` (good/bad) | `_bids.py:count_bad_channels` exists but `extract_record` doesn't call it. We're losing the bad-channel count silently — critical metadata for downstream QA. |
| Per-channel `units` | µV vs V — amplitude calibration. |
| Per-channel `low_cutoff` / `high_cutoff` / `notch` | Per-channel already-applied filters. |

**From `dataset_description.json`** (extra fields we don't currently grab):

| BIDS field | Why it matters |
|---|---|
| `Acknowledgements` | Attribution. Required by some funders to acknowledge in publications. |
| `HowToAcknowledge` | Citation guidance for downstream users of the data. |
| `EthicsApprovals` | Regulatory metadata — IRB approval IDs. Important for clinical reuse. |
| `ReferencesAndLinks` | Paper / preprint URLs. Important for context. |

### Lower-leverage gaps (consider in a later round)

- `*_eeg.json`: `EEGGround`, `CapManufacturer`, `CapManufacturersModelName`, `SubjectArtefactDescription`, `InstitutionName`, `InstitutionAddress`, `InstitutionalDepartmentName`, `RecordingType`, `EpochLength`, `DeviceSerialNumber`
- `coordsystem.json`: already partially captured by `_montage.py:_parse_coordsystem_json`
- `electrodes.tsv`: captured by `_montage.py` extractors
- `events.tsv`: completely uncaptured at digest time (only inlined via `sidecar_inline` for NEMAR)
- `events.json`: column descriptions for events.tsv — uncaptured

## What captures these today by accident

- **`sidecar_inline`**: NEMAR-only enrichment. The full bytes of selected
  sidecars are inlined into the Record. This means the data IS in
  MongoDB for NEMAR-ingested records but is NOT structured / queryable.
  You can't query "datasets where PowerLineFrequency = 50" — you'd have
  to scan all sidecars and re-parse.
- **`apex_sidecar_inline`**: similar, for dataset-level sidecars
  (dataset_description, README, etc.) — also raw bytes only.

## Recommendation — C6.1 scope

Capture the 5 highest-leverage missing fields **into the structured
Record / Dataset documents** (not just inlined as bytes):

### Per Record:
1. `power_line_frequency` (50 / 60 / null)
2. `eeg_reference` (string)
3. `software_filters` (dict)
4. `hardware_filters` (dict)
5. `manufacturer` + `manufacturers_model_name` (strings)
6. `eeg_placement_scheme` (string — feeds `_montage.py` template matching)
7. `bad_channels_count` (int) + `bad_channels` (list[str]) — from `count_bad_channels`

### Per Dataset:
1. `acknowledgements` (string)
2. `how_to_acknowledge` (string)
3. `ethics_approvals` (list[str])
4. `references_and_links` (list[str])

All as optional fields (`Field(default=None)`); existing records remain valid.

## Backward compatibility

The Pydantic models already use `extra="allow"`. Adding optional fields
in the digester won't break existing records. The schema can stay
permissive; downstream tooling can opt-in to the new fields.

## What this enables in MongoDB

After C6.1 lands, these queries become possible:

```javascript
// Find all 60Hz powerline datasets (US recordings)
db.records.find({power_line_frequency: 60})

// Find all datasets with high-pass filter ≤ 0.1 Hz (DC-coupled analysis)
db.records.find({"software_filters.HighPass": {$lte: 0.1}})

// Find all EGI HydroCel 256-channel datasets
db.records.find({manufacturer: "EGI", eeg_placement_scheme: /HydroCel 256/})

// Find all records with ≥ 5 bad channels (data quality flag)
db.records.find({bad_channels_count: {$gte: 5}})

// Find all datasets that require specific acknowledgement
db.datasets.find({how_to_acknowledge: {$exists: true}})
```

None of these are possible today against the structured fields. They'd
require scanning + parsing the inlined sidecar bytes.

## Production driver

This is **the real driver** for C6.1: every researcher selecting data
from EEGdash needs to know `EEGReference` and `SoftwareFilters`
**before downloading**. Today they have to download, parse the sidecar,
then decide. Capturing these at digest time turns multi-minute browse
loops into milliseconds.

## See also

- `eegdash/schemas.py` — RecordModel + DatasetModel
- `3_digest.py:extract_record` — where the new sidecar parsing slots in
- `_bids.py:count_bad_channels` — bad-channels helper waiting to be called
- `_montage.py:_parse_channels_tsv_for_eeg` — channel-type filter (overlap with this audit)
