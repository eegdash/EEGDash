# data.ru.nl Integration Guide

## Overview

`datarn.py` provides integration with [data.ru.nl](https://data.ru.nl) (Radboud University Research Data Management system) for discovering and importing BIDS datasets.

## Important Notes

**data.ru.nl uses a custom Dataverse-based RDM system with NO public REST API.** This means:
- ❌ No automated bulk discovery available (unlike OpenNeuro, Zenodo, OSF)
- ✅ Manual curation recommended
- ✅ Optional browser automation support (requires Chrome + Selenium)

## Usage

### 1. Create a Template (Recommended Approach)

```bash
python datarn.py --create-template
```

This generates `datarn_manifest_template.json` with a sample dataset entry that you can edit.

### 2. Edit the Manifest

Open `datarn_manifest_template.json` and add your BIDS datasets:

```json
{
  "metadata": {
    "source": "datarn",
    "source_name": "Radboud University RDM (data.ru.nl)",
    "description": "Manually curated BIDS datasets from data.ru.nl",
    "harvest_method": "manual_curation",
    "harvest_date": "2025-12-26T09:53:30.571469+00:00",
    "note": "Add datasets manually by filling in the datasets array below"
  },
  "datasets": [
    {
      "dataset_id": "radboud_eeg_dataset_001",
      "name": "Radboud EEG Dataset - Example",
      "url": "https://data.ru.nl/dataset.xhtml?persistentId=doi:10.34894/EXAMPLE",
      "doi": "10.34894/EXAMPLE",
      "modalities": ["eeg"],
      "authors": ["John Doe", "Jane Smith"],
      "description": "EEG dataset from Radboud University studying neural processes"
    },
    {
      "dataset_id": "radboud_meg_dataset_001",
      "name": "Radboud MEG Dataset - Example",
      "url": "https://data.ru.nl/dataset.xhtml?persistentId=doi:10.34894/EXAMPLE2",
      "doi": "10.34894/EXAMPLE2",
      "modalities": ["meg"],
      "authors": ["Research Team"],
      "description": "MEG dataset from Radboud University"
    }
  ]
}
```

### 3. Process the Manifest

```bash
python datarn.py --manifest datarn_manifest_template.json --digested-at "2025-12-26T10:00:00Z"
```

This outputs to `consolidated/datarn_datasets.json` with all datasets in the EEGDash schema format.

## Finding Datasets on data.ru.nl

Visit [data.ru.nl](https://data.ru.nl) and search for:
1. **"BIDS"** - General BIDS-compliant datasets
2. **"EEG"** - Electroencephalography datasets  
3. **"MEG"** - Magnetoencephalography datasets
4. **"neural recording"** - Neural data in any modality

Look for datasets with:
- DOI (for persistent citation)
- Clear author/creator information
- Description of modalities and sample counts
- Public or registered access level

## Dataset Fields

When adding datasets to the manifest, include:

| Field | Required | Description |
|-------|----------|-------------|
| `dataset_id` | ✅ | Unique identifier (use DOI suffix or descriptive name) |
| `name` | ✅ | Human-readable dataset name |
| `url` | ✅ | Direct link to data.ru.nl dataset page |
| `doi` | ✅ | Digital Object Identifier |
| `modalities` | ✅ | List: `["eeg"]`, `["meg"]`, `["emg"]`, `["fnirs"]`, `["lfp"]`, `["ieeg"]`, etc. |
| `authors` | ✅ | List of dataset creators/contributors |
| `description` | ⭕ | Brief description (auto-detected from modalities if not provided) |

## Advanced Options

### Custom Output Location

```bash
python datarn.py --manifest my_datasets.json --output custom_output.json
```

### Set Deterministic Timestamp

```bash
python datarn.py --manifest datarn_manifest_template.json --digested-at "2025-12-26T10:00:00Z"
```

For CI/CD pipelines, use a fixed timestamp for reproducible output.

### Selenium-Based Browser Automation (Optional)

If Selenium is installed, the script can attempt to fetch additional metadata from dataset pages:

```bash
pip install selenium
# Also requires Chrome and ChromeDriver
python datarn.py --manifest datarn_manifest_template.json
```

This will automatically fill in missing authors/descriptions from the web pages.

## Output Schema

The script outputs JSON files in the EEGDash Dataset schema:

```json
[
  {
    "dataset_id": "radboud_eeg_001",
    "name": "Radboud EEG Dataset",
    "source": "datarn",
    "recording_modality": "eeg",
    "modalities": ["eeg"],
    "authors": ["John Doe"],
    "dataset_doi": "10.34894/EXAMPLE",
    "external_links": {
      "source_url": "https://data.ru.nl/dataset.xhtml?persistentId=doi:10.34894/EXAMPLE"
    },
    "demographics": {
      "subjects_count": 0,
      "ages": []
    },
    "timestamps": {
      "digested_at": "2025-12-26T10:00:00Z"
    },
    "sessions": [],
    "tasks": [],
    "funding": []
  }
]
```

## Workflow for CI/CD Integration

1. **Maintain manifest in repository:**
   ```
   scripts/ingestions/1_fetch_sources/datarn_manifest.json
   ```

2. **Create GitHub Actions workflow:**
   ```yaml
   - name: Fetch data.ru.nl datasets
     run: |
       python scripts/ingestions/1_fetch_sources/datarn.py \
         --manifest scripts/ingestions/1_fetch_sources/datarn_manifest.json \
         --digested-at $(date -u +'%Y-%m-%dT%H:%M:%SZ')
   ```

3. **Commit manifest updates to trigger fetches**

## Troubleshooting

### "No datasets found" error
- Check that manifest file exists and is valid JSON
- Verify dataset URLs are correct (should be data.ru.nl/dataset.xhtml)
- Ensure DOI format is correct (10.XXXXX/XXXXX)

### Missing modalities
- Provide modalities explicitly in manifest
- Don't rely on auto-detection from description for data.ru.nl datasets

### Timestamp issues
- Use ISO 8601 format: `YYYY-MM-DDTHH:MM:SSZ`
- Always use UTC (Z suffix)
- Example: `2025-12-26T10:00:00Z`

## See Also

- [Main Fetch Sources README](README.md)
- [EEGDash Records Schema](../../../eegdash/records.py)
- [data.ru.nl Homepage](https://data.ru.nl)
- [BIDS Specification](https://bids-standard.github.io/)
