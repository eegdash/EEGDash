# OpenNeuro Dataset Digestion Pipeline

This directory contains scripts for the EEGDash dataset ingestion pipeline, designed to extract metadata from OpenNeuro BIDS datasets and prepare it for MongoDB ingestion via the API Gateway.

## Architecture Overview

```
┌─────────────────┐
│  OpenNeuro      │
│  GitHub Repos   │
│ (git-annex)     │
└────────┬────────┘
         │ 1. Clone (symlinks only)
         ▼
┌─────────────────┐
│  Local BIDS     │
│  Datasets       │
│ (test_diggestion)│
│ *.set → annex/* │ ← Symlinks to annexed data
│ *.json, *.tsv   │ ← Actual sidecar metadata
└────────┬────────┘
         │ 2. Digest (metadata only)
         ▼
┌─────────────────┐
│  JSON Files     │
│  - Core         │
│  - Enriched     │
│  - Full         │
└────────┬────────┘
         │ 3. Upload
         ▼
┌─────────────────┐
│  API Gateway    │
│ data.eegdash.org│
└────────┬────────┘
         │ 4. Store
         ▼
┌─────────────────┐
│  MongoDB        │
│  - eegdashstaging│
│  - eegdash      │
└─────────────────┘
```

### Two Operating Modes

The `EEGBIDSDataset` class supports two modes via the `allow_symlinks` parameter:

**Digestion Mode** (`allow_symlinks=True`):
- Purpose: Extract metadata without loading raw EEG data
- Use case: Initial dataset ingestion, metadata collection
- Accepts: Broken symlinks (git-annex), actual files
- Extracts from: JSON sidecars, TSV files, BIDS structure
- Requirements: Only metadata files need to exist

**Loading Mode** (`allow_symlinks=False`, default):
- Purpose: Load actual EEG data for analysis
- Use case: Training models, running analyses, data processing
- Accepts: Only actual readable files
- Requires: Complete data files, not just symlinks
- Used by: `EEGDashDataset`, analysis scripts

This separation allows efficient metadata collection from thousands of datasets without downloading terabytes of EEG data.

## Scripts

### 1. Clone Datasets: `clone_openneuro_datasets.py`

Clones OpenNeuro datasets from GitHub to local storage.

**Important Note on Git-Annex:**
Most OpenNeuro datasets use git-annex for large file management. After cloning, EEG data files are symlinks pointing to annexed objects that aren't downloaded by default. This is intentional - the digestion pipeline extracts metadata from BIDS sidecar files (JSON, TSV) without needing the actual EEG data.

**Usage:**
```bash
python clone_openneuro_datasets.py \
    --output-dir test_diggestion \
    --datasets-file consolidated/openneuro_datasets.json \
    --timeout 300
```

**Features:**
- Timeout protection (default: 5 minutes per dataset)
- Automatic retry on failure
- Progress tracking
- Results logging

**Output:**
- Cloned datasets in `test_diggestion/`
- `clone_results.json` - detailed cloning results
- `retry.json` - list of failed datasets for retry

---

### 2. Digest Dataset: `digest_single_dataset.py`

Extracts metadata from a single BIDS dataset and generates JSON files optimized for MongoDB ingestion.

**Usage:**
```bash
python digest_single_dataset.py ds002718 \
    --dataset-dir test_diggestion/ds002718 \
    --output-dir digestion_output/ds002718
```

**Arguments:**
- `dataset_id` (required): Dataset identifier (e.g., `ds002718`)
- `--dataset-dir` (required): Path to local BIDS dataset
- `--output-dir` (optional): Output directory (default: `digestion_output`)

**How It Handles Git-Annex:**
The script uses `allow_symlinks=True` when initializing the BIDS dataset, which enables metadata extraction from symlinked files without requiring the actual data. This is achieved through:
- Reading BIDS sidecar JSON files (e.g., `sub-001_task-rest_eeg.json`)
- Extracting technical parameters: `SamplingFrequency`, `EEGChannelCount`, `RecordingDuration`
- Reading participant information from `participants.tsv`
- Parsing BIDS file structure for subject, task, session, run information

**Output Files:**

1. **`{dataset_id}_core.json`** - Core metadata for MongoDB
   - Essential fields only (data_name, dataset, bidspath, etc.)
   - Optimized for fast querying
   - Ready for bulk upload to MongoDB

2. **`{dataset_id}_enriched.json`** - Extended metadata
   - Participant information
   - EEG technical details
   - BIDS dependencies
   - Additional JSON metadata from sidecar files

3. **`{dataset_id}_full_manifest.json`** - Complete metadata
   - All extracted information combined
   - For reference and debugging

4. **`{dataset_id}_summary.json`** - Processing summary
   - Record counts
   - Error statistics
   - Upload instructions

---

## Metadata Structure

### Core Metadata Fields (Always Loaded)

These fields are stored in MongoDB for efficient querying:

```json
{
  "data_name": "ds002718_sub-012_task-RestingState_eeg.set",
  "dataset": "ds002718",
  "bidspath": "ds002718/sub-012/eeg/sub-012_task-RestingState_eeg.set",
  "subject": "012",
  "task": "RestingState",
  "session": null,
  "run": null,
  "modality": "eeg",
  "sampling_frequency": 500.0,
  "nchans": 64,
  "ntimes": 150000
}
```

### Enriched Metadata Fields (Loaded On-Demand)

Additional information loaded only when needed:

```json
{
  "data_name": "ds002718_sub-012_task-RestingState_eeg.set",
  "participant_tsv": {
    "age": 25,
    "sex": "M",
    "handedness": "R"
  },
  "eeg_json": {
    "PowerLineFrequency": 60,
    "EEGReference": "Average",
    "EEGGround": "AFz",
    "InstitutionName": "University Example"
  },
  "bidsdependencies": [
    "sub-012/eeg/sub-012_task-RestingState_channels.tsv",
    "sub-012/eeg/sub-012_task-RestingState_events.tsv"
  ]
}
```

---

## MongoDB Ingestion via API Gateway

### Upload Core Metadata

After digestion, upload the core metadata to MongoDB using the API Gateway:

**For Staging Database:**
```bash
curl -X POST https://data.eegdash.org/admin/eegdashstaging/records/bulk \
     -H "Authorization: Bearer AdminWrite2025SecureToken" \
     -H "Content-Type: application/json" \
     -d @digestion_output/ds002718/ds002718_core.json
```

**For Production Database:**
```bash
curl -X POST https://data.eegdash.org/admin/eegdash/records/bulk \
     -H "Authorization: Bearer AdminWrite2025SecureToken" \
     -H "Content-Type: application/json" \
     -d @digestion_output/ds002718/ds002718_core.json
```

**Expected Response:**
```json
{
  "success": true,
  "database": "eegdashstaging",
  "insertedCount": 42,
  "message": "42 records inserted successfully"
}
```

---

## Complete Workflow Example

### Step 1: Clone Datasets

```bash
# Clone all datasets from OpenNeuro
python scripts/ingestions/clone_openneuro_datasets.py \
    --output-dir test_diggestion \
    --timeout 300

# Check results
cat test_diggestion/clone_results.json
```

### Step 2: Digest Individual Datasets

```bash
# Digest a single dataset
python scripts/ingestions/digest_single_dataset.py ds002718 \
    --dataset-dir test_diggestion/ds002718 \
    --output-dir digestion_output/ds002718

# Output will be in digestion_output/ds002718/
ls digestion_output/ds002718/
# ds002718_core.json
# ds002718_enriched.json
# ds002718_full_manifest.json
# ds002718_summary.json
```

### Step 3: Review Output

```bash
# Check processing summary
cat digestion_output/ds002718/ds002718_summary.json

# Inspect core metadata (first 5 records)
cat digestion_output/ds002718/ds002718_core.json | jq '.records[:5]'
```

### Step 4: Upload to MongoDB

```bash
# Upload to staging database
curl -X POST https://data.eegdash.org/admin/eegdashstaging/records/bulk \
     -H "Authorization: Bearer AdminWrite2025SecureToken" \
     -H "Content-Type: application/json" \
     -d @digestion_output/ds002718/ds002718_core.json

# Verify upload
curl -H "Authorization: Bearer Competition2025AccessToken" \
     "https://data.eegdash.org/api/eegdashstaging/count?filter=%7B%22dataset%22%3A%22ds002718%22%7D"
```

---

## Batch Processing Multiple Datasets

To process multiple datasets, create a shell script:

```bash
#!/bin/bash
# batch_digest.sh

DATASETS_DIR="test_diggestion"
OUTPUT_DIR="digestion_output"

for dataset_dir in $DATASETS_DIR/ds*/; do
    dataset_id=$(basename "$dataset_dir")
    echo "Processing $dataset_id..."
    
    python scripts/ingestions/digest_single_dataset.py "$dataset_id" \
        --dataset-dir "$dataset_dir" \
        --output-dir "$OUTPUT_DIR/$dataset_id"
    
    if [ $? -eq 0 ]; then
        echo "✓ $dataset_id digested successfully"
    else
        echo "✗ $dataset_id digestion failed"
    fi
done
```

Run with:
```bash
chmod +x batch_digest.sh
./batch_digest.sh
```

---

## Error Handling

### Common Errors

**1. Dataset directory not found**
```
Error: Dataset directory not found: test_diggestion/ds002718
```
Solution: Ensure the dataset has been cloned first.

**2. Invalid BIDS structure**
```
Error creating BIDS dataset: No EEG recordings found
```
Solution: Verify the dataset contains valid EEG data in BIDS format.

**3. Metadata extraction failure**
```
✗ Error extracting metadata for sub-001_eeg.set: channels.tsv not found
```
Solution: Check that all required BIDS sidecar files are present.

### Error Logs

Each digestion creates a summary with errors:

```json
{
  "status": "success",
  "dataset_id": "ds002718",
  "record_count": 40,
  "error_count": 2,
  "outputs": {...},
  "errors": [
    {
      "file": "sub-999/eeg/sub-999_task-rest_eeg.set",
      "error": "channels.tsv not found"
    }
  ]
}
```

---

## Performance Optimization

### Parallel Processing

Process multiple datasets in parallel:

```bash
#!/bin/bash
# parallel_digest.sh

DATASETS_DIR="test_diggestion"
OUTPUT_DIR="digestion_output"
MAX_JOBS=4

find "$DATASETS_DIR" -maxdepth 1 -type d -name "ds*" | \
    parallel -j $MAX_JOBS '
        dataset_id=$(basename {})
        python scripts/ingestions/digest_single_dataset.py "$dataset_id" \
            --dataset-dir {} \
            --output-dir "'$OUTPUT_DIR'/$dataset_id"
    '
```

Requires: `sudo apt install parallel` (or `brew install parallel` on macOS)

---

## Data Model Reference

### MongoDB Collection Schema

```javascript
{
  // Core metadata (indexed)
  "_id": ObjectId("..."),
  "data_name": String,      // Unique identifier (indexed)
  "dataset": String,         // Dataset ID (indexed)
  "bidspath": String,        // S3 path to file
  "subject": String,         // Subject ID
  "task": String,            // Task name
  "session": String | null,  // Session ID
  "run": String | null,      // Run number
  "modality": String,        // Data modality
  "sampling_frequency": Number,
  "nchans": Number,
  "ntimes": Number,
  
  // Enriched metadata (optional)
  "participant_tsv": Object,  // Participant info
  "eeg_json": Object,         // EEG technical metadata
  "bidsdependencies": Array   // Related BIDS files
}
```

### Indexes

Core fields are indexed for fast querying:
- `data_name` (unique)
- `dataset`
- `subject`
- `task`

---

## API Reference

See `EEGDash-mongoDB-files/API_DOCUMENTATION.md` for complete API documentation.

**Key Endpoints:**

- **Upload Records (Bulk):** `POST /admin/{database}/records/bulk`
- **Query Records:** `GET /api/{database}/records`
- **Count Records:** `GET /api/{database}/count`
- **Get Metadata:** `GET /api/{database}/metadata/{dataset}`

---

## Troubleshooting

### Check Dataset Status

```bash
# List all datasets in staging
curl -H "Authorization: Bearer Competition2025AccessToken" \
     https://data.eegdash.org/api/eegdashstaging/datasets

# Count records for a dataset
curl -H "Authorization: Bearer Competition2025AccessToken" \
     "https://data.eegdash.org/api/eegdashstaging/count?filter=%7B%22dataset%22%3A%22ds002718%22%7D"
```

### Verify File Structure

```bash
# Check BIDS validity
python -c "
from eegdash.dataset.bids_dataset import EEGBIDSDataset
ds = EEGBIDSDataset('test_diggestion/ds002718', 'ds002718')
print(f'Files found: {len(ds.get_files())}')
for f in ds.get_files()[:3]:
    print(f'  - {f}')
"
```

---

## Contributing

When adding new features to the digestion pipeline:

1. Ensure backward compatibility with existing JSON structure
2. Update both core and enriched metadata schemas
3. Test with multiple datasets of varying sizes
4. Document any new fields in this README

---

## Related Documentation

- **Architecture:** `EEGDash-mongoDB-files/ARCH_DOCUMENTATION.md`
- **API Documentation:** `EEGDash-mongoDB-files/API_DOCUMENTATION.md`
- **Main README:** `README.md`


https://data.ru.nl/collections/published?access=OPEN_ACCESS,REGISTERED_ACCESS,RESTRICTED_ACCESS&free_keywords=BIDS