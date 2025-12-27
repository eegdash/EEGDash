# EEGDash Dataset Ingestion Pipeline

This directory contains the **simplified 3-step pipeline** for ingesting BIDS datasets from multiple sources into EEGDash MongoDB.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES (8 total)                       │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  OpenNeuro   │   NEMAR      │  EEGManyLabs │   Figshare         │
│  (GitHub)    │   (GitHub)   │  (GIN)       │   (REST API)       │
├──────────────┼──────────────┼──────────────┼────────────────────┤
│   Zenodo     │    OSF       │   SciDB      │   data.ru.nl       │
│  (REST API)  │  (REST API)  │ (Open API)   │   (WebDAV)         │
└──────────────┴──────────────┴──────────────┴────────────────────┘
         │
         │ Step 1: FETCH (per-source scripts)
         ▼
┌─────────────────┐
│  consolidated/  │  ← Dataset metadata JSON files
│  *_datasets.json│     (8 files, one per source)
└────────┬────────┘
         │
         │ Step 2: CLONE (smart, no data download)
         ▼
┌─────────────────┐
│  data/cloned/   │  ← Manifests + shallow clones
│  {dataset_id}/  │     (symlinks only for git sources)
│    manifest.json│
└────────┬────────┘
         │
         │ Step 3: DIGEST (metadata extraction)
         ▼
┌──────────────────────┐
│  digestion_output/   │  ← MongoDB-ready JSON documents
│  {dataset_id}/       │
│    *_dataset.json    │  ← Dataset (1 per dataset)
│    *_records.json    │  ← Records (many per dataset)
└────────┬─────────────┘
         │
         │ Step 4: INJECT (upload to MongoDB)
         ▼
┌─────────────────────────┐
│  MongoDB                │
│  data.eegdash.org       │
│  ├─ datasets collection │  ← For discovery/filtering
│  └─ records collection  │  ← For fast loading
└─────────────────────────┘
```

## Pipeline Scripts

### Step 1: Fetch Dataset Metadata

Per-source scripts in `1_fetch_sources/`:

| Script | Source | Method |
|--------|--------|--------|
| `openneuro.py` | OpenNeuro | GraphQL API |
| `nemar.py` | NEMAR | GitHub API |
| `eegmanylabs.py` | EEGManyLabs | GIN API |
| `figshare.py` | Figshare | REST API |
| `zenodo.py` | Zenodo | REST API |
| `osf.py` | OSF | REST API |
| `scidb.py` | ScienceDB | Query Service API |
| `datarn.py` | data.ru.nl | REST API |

**Output**: `consolidated/{source}_datasets.json`

```bash
# Fetch all sources
for script in scripts/ingestions/1_fetch_sources/*.py; do
  python "$script"
done
```

---

### Step 2: Smart Clone (No Data Download)

**Script**: `2_clone.py`

Smart cloning that gets file structure WITHOUT downloading raw data:

| Source | Strategy | Result |
|--------|----------|--------|
| OpenNeuro, NEMAR, GIN | Shallow git clone + `GIT_LFS_SKIP_SMUDGE=1` | ~300KB symlinks per dataset |
| Figshare | REST API `/v2/articles/{id}` | File manifest |
| Zenodo | REST API `/api/records/{id}` | File manifest |
| OSF | REST API recursive folder traversal | File manifest |
| SciDB | Stub (no public file API) | Metadata only |
| data.ru.nl | WebDAV PROPFIND recursive | File manifest |

```bash
# Clone all datasets from all sources
python scripts/ingestions/2_clone.py \
  --output data/cloned \
  --workers 4

# Clone specific sources only
python scripts/ingestions/2_clone.py \
  --output data/cloned \
  --sources openneuro nemar

# With BIDS validation
python scripts/ingestions/2_clone.py \
  --output data/cloned \
  --validate-bids
```

**Output**: `data/cloned/{dataset_id}/manifest.json`

---

### Step 3: Digest Metadata

**Script**: `3_digest.py`

Extract BIDS metadata from cloned datasets and produce **two separate document types**:

1. **Dataset**: Per-dataset metadata for discovery/filtering
2. **Records**: Per-file metadata optimized for fast loading

```bash
# Digest all cloned datasets
python scripts/ingestions/3_digest.py \
  --input data/cloned \
  --output digestion_output

# Specific datasets only
python scripts/ingestions/3_digest.py \
  --input data/cloned \
  --output digestion_output \
  --datasets ds002718 ds005506

# Parallel processing
python scripts/ingestions/3_digest.py \
  --input data/cloned \
  --output digestion_output \
  --workers 4
```

**Output**:
- `digestion_output/{dataset_id}/{dataset_id}_dataset.json` - Dataset document
- `digestion_output/{dataset_id}/{dataset_id}_records.json` - Records array
- `digestion_output/{dataset_id}/{dataset_id}_summary.json` - Processing summary

---

### Step 4: Inject to MongoDB

**Script**: `4_inject.py`

Upload digested datasets and records to **separate MongoDB collections** via API Gateway.

```bash
# Inject to staging (both datasets and records)
python scripts/ingestions/4_inject.py \
  --input digestion_output \
  --database eegdash_staging

# Inject to production
python scripts/ingestions/4_inject.py \
  --input digestion_output \
  --database eegdash

# Dry run (validate only)
python scripts/ingestions/4_inject.py \
  --input digestion_output \
  --database eegdash_staging \
  --dry-run

# Inject only datasets (skip records)
python scripts/ingestions/4_inject.py \
  --input digestion_output \
  --database eegdash_staging \
  --only-datasets

# Inject only records (skip datasets)
python scripts/ingestions/4_inject.py \
  --input digestion_output \
  --database eegdash_staging \
  --only-records
```

---

## Schema (Two-Level Hierarchy)

### Dataset Schema (per-dataset, for discovery)

```json
{
  "dataset_id": "ds002718",
  "name": "EEG Dataset Name",
  "source": "openneuro",
  "recording_modality": "eeg",
  "modalities": ["eeg"],
  "bids_version": "1.6.0",
  "license": "CC0",
  "authors": ["Author 1", "Author 2"],
  "tasks": ["RestingState", "ActiveTask"],
  "sessions": ["01", "02"],
  "total_files": 128,
  "demographics": {
    "subjects_count": 32,
    "ages": [22, 24, 28, ...],
    "age_min": 18,
    "age_max": 45,
    "age_mean": 28.5,
    "sex_distribution": {"m": 16, "f": 16}
  },
  "timestamps": {
    "digested_at": "2024-01-15T10:30:00Z"
  },
  "external_links": {
    "source_url": "https://openneuro.org/datasets/ds002718"
  }
}
```

### Record Schema (per-file, for loading)

```json
{
  "dataset": "ds002718",
  "data_name": "ds002718_sub-012_task-RestingState_eeg.set",
  "bids_relpath": "sub-012/eeg/sub-012_task-RestingState_eeg.set",
  "datatype": "eeg",
  "suffix": "eeg",
  "extension": ".set",
  "recording_modality": "eeg",
  "entities": {
    "subject": "012",
    "task": "RestingState",
    "session": null,
    "run": null
  },
  "entities_mne": {
    "subject": "012",
    "task": "RestingState"
  },
  "storage": {
    "backend": "s3",
    "base": "s3://openneuro.org/ds002718",
    "raw_key": "sub-012/eeg/sub-012_task-RestingState_eeg.set",
    "dep_keys": [
      "sub-012/eeg/sub-012_task-RestingState_events.tsv",
      "sub-012/eeg/sub-012_task-RestingState_eeg.fdt"
    ]
  },
  "digested_at": "2024-01-15T10:30:00Z"
}
```

**Note on `dep_keys`**: The digester automatically detects companion files required for loading:
- `.fdt` files for EEGLAB `.set` format
- `.vmrk` and `.eeg` files for BrainVision `.vhdr` format
- BIDS sidecar files (`_events.tsv`, `_channels.tsv`, `_electrodes.tsv`, `_coordsystem.json`)

---

### Step 5: Validate Output (Optional)

**Script**: `validate_output.py`

Validates digested output before injection:

```bash
# Validate all digested datasets
python scripts/ingestions/validate_output.py

# Validate specific directory
python scripts/ingestions/validate_output.py --input digestion_output
```

Checks for:
- Missing mandatory fields (`dataset`, `bids_relpath`, `storage`, `recording_modality`)
- Invalid storage URLs
- Empty datasets (0 records)
- ZIP placeholders that need extraction
```

---

## GitHub Actions Workflows

Automated pipelines in `.github/workflows/`:

| Workflow | Schedule | Action |
|----------|----------|--------|
| `1-fetch-openneuro.yml` | Weekly | Fetch OpenNeuro datasets |
| `1-fetch-nemar.yml` | Weekly | Fetch NEMAR datasets |
| `1-fetch-eegmanylabs.yml` | Weekly | Fetch EEGManyLabs datasets |
| `1-fetch-figshare.yml` | Weekly | Fetch Figshare datasets |
| `1-fetch-zenodo.yml` | Weekly | Fetch Zenodo datasets |
| `1-fetch-osf.yml` | Weekly | Fetch OSF datasets |
| `1-fetch-scidb.yml` | Weekly | Fetch SciDB datasets |
| `1-fetch-datarn.yml` | Weekly | Fetch data.ru.nl datasets |

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `validate_output.py` | Validate digested records before injection |
| `_serialize.py` | Deterministic JSON serialization |
| `compare_ground_truth_to_generated.py` | Validate digested records against ground truth |
| `aggregate_gt_comparison_stats.py` | Aggregate validation stats |
| `test_digest_openneuro.py` | Test digestion on sample datasets |

---

## Directory Structure

```
scripts/ingestions/
├── 1_fetch_sources/          # Per-source fetch scripts
│   ├── openneuro.py
│   ├── nemar.py
│   ├── eegmanylabs.py
│   ├── figshare.py
│   ├── zenodo.py
│   ├── osf.py
│   ├── scidb.py
│   └── datarn.py
├── 2_clone.py                # Smart clone/manifest
├── 3_digest.py               # BIDS metadata extraction
├── 4_inject.py               # MongoDB upload
├── validate_output.py        # Output validation
├── _serialize.py             # JSON serialization utils
└── README.md                 # This file
```

---

## Technical Notes

### Git-Annex Symlink Handling

For OpenNeuro and other git-based sources, the clone step creates **broken symlinks** 
(pointers to `.git/annex/objects/`) rather than downloading actual data. The digester 
handles these correctly:

- Uses `Path.is_symlink()` to detect git-annex files
- Extracts metadata from symlink paths without requiring actual file content
- Companion files (`.fdt`, `.vmrk`, `.eeg`) are detected even as broken symlinks

---

## Quick Start

```bash
# 1. Fetch all sources
for script in scripts/ingestions/1_fetch_sources/*.py; do
  python "$script"
done

# 2. Smart clone (no data download)
python scripts/ingestions/2_clone.py --output data/cloned

# 3. Digest metadata
python scripts/ingestions/3_digest.py --input data/cloned --output digestion_output

# 4. Validate output (optional but recommended)
python scripts/ingestions/validate_output.py

# 5. Upload to staging
python scripts/ingestions/4_inject.py --input digestion_output --database eegdash_staging
```
