# Ingestion Pipeline Refactoring Plan

## Status: ✅ COMPLETED

All 3 steps have been implemented:
- Step 1: Fixed storage URLs in `3_digest.py`
- Step 2: Preserved all metadata in `_file_utils.py`
- Step 3: Created `validate_output.py` for quality checks

## Diagnosis Summary

### Current Issues Identified

1. **CRITICAL BUG - Wrong Storage URLs**: In `3_digest.py` lines 602-604, the `else` branch defaults to `https://files.osf.io/{dataset_id}` for ALL sources not explicitly handled (figshare, zenodo, osf). This means **OpenNeuro, NEMAR, GIN, SciDB, datarn** datasets all get wrong OSF URLs.

2. **Metadata Loss**: The clone step creates manifests but doesn't preserve all metadata from the consolidated files (e.g., `authors`, `license`, `dataset_doi`, `external_links`, etc.).

3. **Source Detection Duplication**: Each script has its own source detection logic:
   - `1_fetch_sources/*.py`: Each fetcher sets `source` explicitly
   - `2_clone.py`: Has `detect_source()` function that tries to infer
   - `3_digest.py`: Has another `detect_source()` function

4. **No Validation/Quality Tests**: No automated tests to verify:
   - Storage URLs match the source
   - Required metadata fields are present
   - Data consistency across pipeline stages

5. **99 Manifests Missing Source**: Some manifests don't have a `source` field at all.

### Data Flow Analysis

```
FETCH (1_fetch)         CLONE (2_clone)           DIGEST (3_digest)
─────────────────       ─────────────────         ─────────────────
Source APIs/repos  -->  consolidated/*.json  -->  data/cloned/*/manifest.json  -->  digestion_output/
                        (RICH metadata)           (LOSES metadata)                  (WRONG URLs)
```

**What should happen:**
- Fetch: Collect all available metadata from source APIs
- Clone: Create file manifests AND preserve ALL metadata from consolidated
- Digest: Use preserved metadata to build correct storage URLs and records

---

## Refactoring Plan - 3 Steps

### STEP 1: Fix Storage URLs and Source Detection

**Goal**: Fix the immediate bug causing wrong storage URLs.

**Changes**:
1. Fix `3_digest.py` to handle ALL sources correctly:
   ```python
   STORAGE_CONFIGS = {
       "openneuro": {"backend": "s3", "base": "s3://openneuro.org"},
       "nemar": {"backend": "s3", "base": "s3://nemar"},  # Also S3!
       "gin": {"backend": "https", "base": "https://gin.g-node.org"},
       "figshare": {"backend": "https", "base": "https://figshare.com/ndownloader/files"},
       "zenodo": {"backend": "https", "base": "https://zenodo.org/records"},
       "osf": {"backend": "https", "base": "https://files.osf.io"},
       "scidb": {"backend": "https", "base": "https://www.scidb.cn"},
       "datarn": {"backend": "webdav", "base": "https://webdav.data.ru.nl"},
   }
   ```

2. Make storage URL building use manifest metadata when available (for sources with custom URLs like Figshare/Zenodo).

**Deliverable**: Fixed `3_digest.py` with correct storage URL logic.

---

### STEP 2: Preserve Metadata Through Pipeline

**Goal**: Ensure ALL metadata from consolidated files flows through to manifests.

**Changes**:
1. Modify `2_clone.py` to preserve full dataset metadata in manifest:
   - Pass entire dataset dict as `metadata` to `build_manifest()`
   - Store `external_links`, `dataset_doi`, `authors`, `license`, etc.

2. Modify `_file_utils.py` `build_manifest()` to include full metadata:
   ```python
   def build_manifest(dataset_id, source, files, metadata=None):
       manifest = {
           "dataset_id": dataset_id,
           "source": source,
           "files": normalized_files,
           # Preserve ALL metadata from fetch step
           **{k: v for k, v in (metadata or {}).items() 
              if k not in ["dataset_id", "source", "files"]}
       }
   ```

3. Update `3_digest.py` to use preserved metadata:
   - Use `manifest.get("external_links", {}).get("source_url")` for download URLs
   - Use `manifest.get("dataset_doi")` for DOIs
   - Use `manifest.get("storage_base")` if source provides it

**Deliverable**: Updated clone and digest scripts that preserve metadata.

---

### STEP 3: Add Quality Tests and Validation

**Goal**: Ensure pipeline produces consistent, correct output.

**Changes**:
1. Create `scripts/ingestions/validate_output.py`:
   - Check all records have valid storage URLs matching their source
   - Check required fields are present (dataset_id, source, storage.base)
   - Check no duplicate records
   - Check metadata consistency (subjects_count matches actual subjects)

2. Add source-specific validation:
   ```python
   VALID_STORAGE_PATTERNS = {
       "openneuro": r"^s3://openneuro\.org/ds\d+",
       "nemar": r"^s3://nemar/nm\d+",  # S3 storage
       "osf": r"^https://files\.osf\.io/",
       "figshare": r"^https://(figshare\.com|ndownloader)",
       "zenodo": r"^https://zenodo\.org/",
       "scidb": r"^https://(www\.)?scidb\.cn/",
       "datarn": r"^https://webdav\.data\.ru\.nl/",
       "gin": r"^https://gin\.g-node\.org/",
   }
   ```

3. Create summary report:
   - Total records per source
   - Total datasets per source
   - Any validation errors
   - Missing/invalid fields

**Deliverable**: `validate_output.py` script that can be run after digestion.

---

## Execution Order

1. **Delete old outputs**: `rm -rf digestion_output consolidated data/cloned`
2. **Execute Step 1**: Fix storage URLs in `3_digest.py`
3. **Verify Step 1**: Re-run digestion on a small sample, check URLs
4. **Execute Step 2**: Update clone and file_utils to preserve metadata
5. **Verify Step 2**: Check manifests contain full metadata
6. **Execute Step 3**: Create validation script
7. **Verify Step 3**: Run full pipeline and validate output

---

## Expected Final Output

After refactoring, each digested record should have:

```json
{
  "dataset": "ds000117",
  "source": "openneuro",
  "storage": {
    "backend": "s3",
    "base": "s3://openneuro.org/ds000117",
    "raw_key": "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif"
  },
  "recording_modality": "meg",
  ...
}
```

NOT:
```json
{
  "storage": {
    "backend": "s3",
    "base": "https://files.osf.io/ds000117",  // WRONG!
    ...
  }
}
```
