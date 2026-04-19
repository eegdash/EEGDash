<a id="api"></a>

# API Reference

The EEGDash API reference curates everything you need to integrate, extend,
and automate EEGDash—from core dataset helpers to feature extraction and rich
dataset metadata. The focus is interoperability, extensibility, and ease of use.

<h2 class="hf-section-title">What's inside EEGDash</h2>
<p class="hf-section-subtitle">Everything you need to discover, prepare, and benchmark EEG and MEG data.</p>

<svg version="1.1" width="1.5em" height="1.5em" class="sd-octicon sd-octicon-search sd-text-primary" viewBox="0 0 24 24" aria-hidden="true"><path d="M10.25 2a8.25 8.25 0 0 1 6.34 13.53l5.69 5.69a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215l-5.69-5.69A8.25 8.25 0 1 1 10.25 2ZM3.5 10.25a6.75 6.75 0 1 0 13.5 0 6.75 6.75 0 0 0-13.5 0Z"></path></svg>

Search metadata, modalities, tasks, and cohorts with unified filters.

<svg version="1.1" width="1.5em" height="1.5em" class="sd-octicon sd-octicon-plug sd-text-primary" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 11.5H2.938c-.794 0-1.438.644-1.438 1.437v8.313a.75.75 0 0 1-1.5 0v-8.312A2.939 2.939 0 0 1 2.937 10H7V6.151c0-.897.678-1.648 1.57-1.74l6.055-.626 1.006-1.174A1.752 1.752 0 0 1 16.96 2h1.29c.966 0 1.75.784 1.75 1.75V6h3.25a.75.75 0 0 1 0 1.5H20V14h3.25a.75.75 0 0 1 0 1.5H20v2.25a1.75 1.75 0 0 1-1.75 1.75h-1.29a1.75 1.75 0 0 1-1.329-.611l-1.006-1.174-6.055-.627A1.749 1.749 0 0 1 7 15.348Zm9.77-7.913v.001l-1.201 1.4a.75.75 0 0 1-.492.258l-6.353.657a.25.25 0 0 0-.224.249v9.196a.25.25 0 0 0 .224.249l6.353.657c.191.02.368.112.493.258l1.2 1.401a.252.252 0 0 0 .19.087h1.29a.25.25 0 0 0 .25-.25v-14a.25.25 0 0 0-.25-.25h-1.29a.252.252 0 0 0-.19.087Z"></path></svg>

One-command pipelines with EEGPrep, MNE, and BIDS alignment.

<svg version="1.1" width="1.5em" height="1.5em" class="sd-octicon sd-octicon-rocket sd-text-primary" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.322.75h1.176a1.75 1.75 0 0 1 1.75 1.749v1.177a10.75 10.75 0 0 1-2.925 7.374l-1.228 1.304a23.699 23.699 0 0 1-1.596 1.542v5.038c0 .615-.323 1.184-.85 1.5l-4.514 2.709a.75.75 0 0 1-1.12-.488l-.963-4.572a1.305 1.305 0 0 1-.14-.129L8.04 15.96l-1.994-1.873a1.305 1.305 0 0 1-.129-.14l-4.571-.963a.75.75 0 0 1-.49-1.12l2.71-4.514c.316-.527.885-.85 1.5-.85h5.037a23.668 23.668 0 0 1 1.542-1.594l1.304-1.23A10.753 10.753 0 0 1 20.321.75Zm-6.344 4.018v-.001l-1.304 1.23a22.275 22.275 0 0 0-3.255 3.851l-2.193 3.29 1.859 1.744a.545.545 0 0 1 .034.034l1.743 1.858 3.288-2.192a22.263 22.263 0 0 0 3.854-3.257l1.228-1.303a9.251 9.251 0 0 0 2.517-6.346V2.5a.25.25 0 0 0-.25-.25h-1.177a9.252 9.252 0 0 0-6.344 2.518ZM6.5 21c-1.209 1.209-3.901 1.445-4.743 1.49a.236.236 0 0 1-.18-.067.236.236 0 0 1-.067-.18c.045-.842.281-3.534 1.49-4.743.9-.9 2.6-.9 3.5 0 .9.9.9 2.6 0 3.5Zm-.592-8.588L8.17 9.017c.23-.346.47-.685.717-1.017H5.066a.25.25 0 0 0-.214.121l-2.167 3.612ZM16 15.112c-.333.248-.672.487-1.018.718l-3.393 2.262.678 3.223 3.612-2.167a.25.25 0 0 0 .121-.214ZM17.5 8a1.5 1.5 0 1 1-3.001-.001A1.5 1.5 0 0 1 17.5 8Z"></path></svg>

Export model-ready features and compare baselines across datasets.

![BIDS](_static/bids_logo_black.svg)

Keep metadata consistent and portable across teams and tools.

The API is organized into three main components:

<span class="fa-solid fa-microchip api-grid-card_\_icon" aria-hidden="true"></span>

**Core API**

Build, query, and manage EEGDash datasets and utilities.

[→ Explore Core API](api_core.md)

<span class="fa-solid fa-wave-square api-grid-card_\_icon" aria-hidden="true"></span>

**Feature engineering**

Extract statistical, spectral, and machine-learning-ready features.

[→ Explore Feature Engineering](api_features.md)

<span class="fa-solid fa-database api-grid-card_\_icon" aria-hidden="true"></span>

**Dataset catalog**

Browse dynamically generated dataset classes with rich metadata.

[→ Explore the Dataset API](dataset/api_dataset.md)

## REST API Endpoints

The EEGDash metadata server exposes a FastAPI REST interface for discovery and
querying. Base URL: [https://data.eegdash.org](https://data.eegdash.org). Below is a concise map of the main
entrypoints and their purpose.

### Meta Endpoints

- `GET /`
  Returns API name, version, and available databases.
- `GET /health`
  Returns API health and MongoDB connection status.
- `GET /metrics`
  Prometheus metrics (if enabled).

### Public Data Endpoints

- `GET /api/{database}/records`
  Query records (files) with filter and pagination.
- `GET /api/{database}/count`
  Count records matching a filter.
- `GET /api/{database}/datasets/names`
  List unique dataset names from records.
- `GET /api/{database}/metadata/{dataset}`
  Get metadata for a single dataset (from records).
- `GET /api/{database}/datasets/summary`
  Get summary statistics and metadata for all datasets (with pagination, filtering).
  Query params: `limit` (1-1000), `skip`, `modality` (eeg/meg/ieeg), `source` (openneuro/nemar/zenodo/etc.).
  Response includes aggregate totals for datasets, subjects, files, and size.
- `GET /api/{database}/datasets/summary/{dataset_id}`
  Get detailed summary for a specific dataset.
  `dataset_id` may be the dataset ID or dataset name.
- `GET /api/{database}/datasets/{dataset_id}`
  Get a specific dataset document by ID.
- `GET /api/{database}/datasets`
  List dataset documents (with filtering and pagination).
- `GET /api/{database}/datasets/stats/records`
  Get aggregated `nchans` and `sampling_frequency` counts for all datasets.
  Used to generate summary tables efficiently.

### Admin Endpoints (require Bearer token)

- `POST /admin/{database}/records`
  Insert a single record (file document).
- `POST /admin/{database}/records/bulk`
  Insert multiple records (max 1000 per request).
- `POST /admin/{database}/datasets`
  Insert or update a single dataset document (upsert by `dataset_id`).
- `POST /admin/{database}/datasets/bulk`
  Insert or update multiple dataset documents (max 500 per request).
- `PATCH /admin/{database}/records`
  Update records matching a filter (only `$set` allowed).
- `GET /admin/security/blocked`
  List blocked IPs and offense counts.
- `POST /admin/security/unblock`
  Unblock a specific IP.

## Related Guides

- [Tutorial gallery](../generated/auto_examples/index.md)
- [Dataset summary](../dataset_summary.md)
- [Installation guide](../install/install.md)
