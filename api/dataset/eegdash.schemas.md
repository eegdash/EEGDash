# eegdash.schemas module

## EEGDash Data Schemas

This module defines the core data structures used throughout EEGDash to represent
neuroimaging datasets and individual recording files.

It provides two types of schemas for each core object:

1. **Pydantic Models** (`*Model`): Used for strict data validation, serialization,
   and schema generation (e.g., for APIs).
2. **TypedDict Definitions**: Used for high-performance internal usage, static type
   checking, and efficient loading of large metadata collections.

### Core Concepts

The data model is organized into a two-level hierarchy:

* **Dataset**: Represents a collection of data (e.g., “ds001785”). It contains
  study-level metadata such as:
  \*   Identity (ID, name, source)
  \*   Demographics (subject ages, sex distribution)
  \*   Clinical (diagnosis, purpose)
  \*   Experiment Paradigm (tasks, stimuli)
  \*   Provenance (timestamps, authors)
* **Record**: Represents a single data file within a dataset (e.g., a specific
  .vhdr or .edf file). It is optimized for fast access and contains:
  \*   File location (storage backend, path)
  \*   BIDS Entities (subject, session, task, run)
  \*   Basic signal properties (sampling rate, channel names)

### Usage

Creating a Dataset:

```python
from eegdash.schemas import create_dataset

ds = create_dataset(
    dataset_id="ds001",
    name="My Study",
    subjects_count=20,
    ages=[20, 25, 30],
    recording_modality=["eeg"],
)
```

Creating a Record:

```python
from eegdash.schemas import create_record

rec = create_record(
    dataset="ds001",
    storage_base="https://my.storage.com",
    bids_relpath="sub-01/eeg/sub-01_task-rest_eeg.edf",
    subject="01",
    task="rest",
)
```

<!-- !! processed by numpydoc !! -->

### *class* eegdash.schemas.Clinical

Bases: `TypedDict`

Clinical classification metadata (dataset-level).

#### Deprecated
Deprecated since version Use: the `tags` field with `pathology` key instead.

#### is_clinical

True if the dataset contains clinical population data.

* **Type:**
  bool

#### purpose

The clinical condition or purpose (e.g., “epilepsy”, “depression”).

* **Type:**
  str | None

<!-- !! processed by numpydoc !! -->

#### is_clinical *: bool*

#### purpose *: str | None*

### *class* eegdash.schemas.Dataset

Bases: `TypedDict`

TypedDict schema for a full Dataset document.

This Dictionary represents all metadata available for a study/dataset.

#### dataset_id

Unique identifier (e.g., “ds001785”).

* **Type:**
  str

#### name

Descriptive title of the dataset.

* **Type:**
  str

#### canonical_name

Canonical / community-recognised name(s) for the dataset, each a valid
Python identifier (e.g. `["BrainTreeBank"]`, `["SleepEDF",
"SleepEDFPlus"]`). Used to register importable class aliases alongside
the `DS…`-style ID. Empty list or `None` means no alias is
registered.

* **Type:**
  list[str] | None

#### source

Origin source (e.g., “openneuro”, “nemar”).

* **Type:**
  str

#### readme

Content of the dataset’s README file.

* **Type:**
  str | None

#### recording_modality

List of recording modalities (e.g., [“eeg”, “meg”]).

* **Type:**
  list[str]

#### datatypes

BIDS datatypes present (e.g., [“eeg”, “anat”]).

* **Type:**
  list[str]

#### experimental_modalities

Stimulus types used (e.g., [“visual”, “auditory”]).

* **Type:**
  list[str] | None

#### bids_version

Version of the BIDS standard used.

* **Type:**
  str | None

#### license

License string (e.g., “CC0”).

* **Type:**
  str | None

#### authors

List of author names.

* **Type:**
  list[str]

#### funding

List of funding sources.

* **Type:**
  list[str]

#### dataset_doi

Digital Object Identifier for the dataset.

* **Type:**
  str | None

#### associated_paper_doi

DOI of the paper associated with the dataset.

* **Type:**
  str | None

#### tasks

List of task names found in the dataset.

* **Type:**
  list[str]

#### sessions

List of session names.

* **Type:**
  list[str]

#### total_files

Total file count.

* **Type:**
  int | None

#### size_bytes

Total dataset size in bytes.

* **Type:**
  int | None

#### data_processed

Indicates if the data has been pre-processed.

* **Type:**
  bool | None

#### study_domain

General domain of the study.

* **Type:**
  str | None

#### study_design

Description of the study design.

* **Type:**
  str | None

#### contributing_labs

List of labs contributing to the dataset.

* **Type:**
  list[str] | None

#### n_contributing_labs

Count of contributing labs.

* **Type:**
  int | None

#### demographics

Summary of subject demographics.

* **Type:**
  Demographics

#### tags

Classification tags (pathology, modality, type).

* **Type:**
  Tags

#### clinical

Clinical classification details (deprecated, use tags instead).

* **Type:**
  Clinical

#### external_links

Links to external resources.

* **Type:**
  ExternalLinks

#### repository_stats

Stats for the source repository (if applicable).

* **Type:**
  RepositoryStats | None

#### senior_author

Name of the senior author.

* **Type:**
  str | None

#### contact_info

Contact emails or names.

* **Type:**
  list[str] | None

#### timestamps

Timestamps for data processing and creation.

* **Type:**
  Timestamps

#### nemar_citation_count

Number of papers citing this dataset (from NEMAR citations repository).

* **Type:**
  int | None

<!-- !! processed by numpydoc !! -->

#### associated_paper_doi *: str | None*

#### authors *: list[str]*

#### bids_version *: str | None*

#### canonical_name *: list[str] | None*

#### clinical *: Clinical*

#### contact_info *: list[str] | None*

#### contributing_labs *: list[str] | None*

#### data_processed *: bool | None*

#### dataset_doi *: str | None*

#### dataset_id *: str*

#### datatypes *: list[str]*

#### demographics *: Demographics*

#### experimental_modalities *: list[str] | None*

#### external_links *: ExternalLinks*

#### funding *: list[str]*

#### ingestion_fingerprint *: str | None*

#### license *: str | None*

#### n_contributing_labs *: int | None*

#### name *: str*

#### nemar_citation_count *: int | None*

#### readme *: str | None*

#### recording_modality *: list[str]*

#### repository_stats *: RepositoryStats | None*

#### senior_author *: str | None*

#### sessions *: list[str]*

#### size_bytes *: int | None*

#### source *: str*

#### storage *: Storage | None*

#### study_design *: str | None*

#### study_domain *: str | None*

#### tags *: Tags*

#### tasks *: list[str]*

#### timestamps *: Timestamps*

#### total_files *: int | None*

### *class* eegdash.schemas.DatasetModel(, dataset_id: Annotated[str, MinLen(min_length=1)], source: Annotated[str, MinLen(min_length=1)], recording_modality: Annotated[list[str], MinLen(min_length=1)], ingestion_fingerprint: str | None = None, senior_author: str | None = None, contact_info: list[str] | None = None, timestamps: dict[str, Any] | None = None, storage: StorageModel | None = None, \*\*extra_data: Any)

Bases: `BaseModel`

Pydantic model for dataset-level metadata.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### contact_info *: list[str] | None*

#### dataset_id *: str*

#### ingestion_fingerprint *: str | None*

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### recording_modality *: list[str]*

#### senior_author *: str | None*

#### source *: str*

#### storage *: StorageModel | None*

#### timestamps *: dict[str, Any] | None*

### *class* eegdash.schemas.Demographics

Bases: `TypedDict`

Subject demographics summary for a dataset.

#### subjects_count

Total number of subjects.

* **Type:**
  int

#### ages

List of all subject ages (if available).

* **Type:**
  list[int]

#### age_min

Minimum age in the cohort.

* **Type:**
  int | None

#### age_max

Maximum age in the cohort.

* **Type:**
  int | None

#### age_mean

Mean age of subjects.

* **Type:**
  float | None

#### species

Species of subjects (e.g., “Human”, “Mouse”).

* **Type:**
  str | None

#### sex_distribution

Count of subjects by sex (e.g., {“m”: 50, “f”: 45}).

* **Type:**
  dict[str, int] | None

#### handedness_distribution

Count of subjects by handedness (e.g., {“r”: 80, “l”: 15}).

* **Type:**
  dict[str, int] | None

<!-- !! processed by numpydoc !! -->

#### age_max *: int | None*

#### age_mean *: float | None*

#### age_min *: int | None*

#### ages *: list[int]*

#### handedness_distribution *: dict[str, int] | None*

#### sex_distribution *: dict[str, int] | None*

#### species *: str | None*

#### subjects_count *: int*

### *class* eegdash.schemas.Entities

Bases: `TypedDict`

BIDS entities parsed from the file path.

#### subject

Subject label (e.g., “01”).

* **Type:**
  str | None

#### session

Session label (e.g., “pre”).

* **Type:**
  str | None

#### task

Task label (e.g., “rest”).

* **Type:**
  str | None

#### run

Run label (e.g., “1” or “01”).

* **Type:**
  str | None

#### acquisition

Acquisition label (e.g., “bipolar”, “PSG”).

* **Type:**
  str | None

<!-- !! processed by numpydoc !! -->

#### acquisition *: str | None*

#### run *: str | None*

#### session *: str | None*

#### subject *: str | None*

#### task *: str | None*

### *class* eegdash.schemas.EntitiesModel(, subject: str | None = None, session: str | None = None, task: str | None = None, run: str | None = None, acquisition: str | None = None, \*\*extra_data: Any)

Bases: `BaseModel`

Pydantic model for BIDS entities.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### acquisition *: str | None*

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### run *: str | None*

#### session *: str | None*

#### subject *: str | None*

#### task *: str | None*

### *class* eegdash.schemas.ExternalLinks

Bases: `TypedDict`

Relevant external hyperlinks for the dataset.

#### source_url

URL to the primary data source (e.g. OpenNeuro page).

* **Type:**
  str | None

#### osf_url

URL to the Open Science Framework project.

* **Type:**
  str | None

#### github_url

URL to the associated GitHub repository.

* **Type:**
  str | None

#### paper_url

URL to the primary publication.

* **Type:**
  str | None

<!-- !! processed by numpydoc !! -->

#### github_url *: str | None*

#### osf_url *: str | None*

#### paper_url *: str | None*

#### source_url *: str | None*

### *class* eegdash.schemas.ManifestFileModel(, path: str | None = None, name: str | None = None, \*\*extra_data: Any)

Bases: `BaseModel`

Pydantic model for a file entry in a manifest.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### name *: str | None*

#### path *: str | None*

#### path_or_name() → str

Return the path or name of the file.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.schemas.ManifestModel(, source: str | None = None, files: list[str | ManifestFileModel], \*\*extra_data: Any)

Bases: `BaseModel`

Pydantic model for a dataset file manifest.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### files *: list[str | ManifestFileModel]*

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### source *: str | None*

### *class* eegdash.schemas.Record

Bases: `TypedDict`

TypedDict schema for a Record document.

Represents a single data file and its metadata. This structure is kept flat
and minimal to ensure fast loading times when querying millions of records.

#### dataset

Foreign key matching `Dataset.dataset_id`.

* **Type:**
  str

#### data_name

Unique name for the data item (e.g., “ds001_sub-01_task-rest”).

* **Type:**
  str

#### bidspath

Legacy path identifier (e.g., “ds001/sub-01/eeg/…”).

* **Type:**
  str

#### bids_relpath

Standard BIDS relative path (e.g., “sub-01/eeg/…”).

* **Type:**
  str

#### datatype

BIDS datatype (e.g., “eeg”).

* **Type:**
  str

#### suffix

Filename suffix (e.g., “eeg”).

* **Type:**
  str

#### extension

File extension (e.g., “.vhdr”).

* **Type:**
  str

#### recording_modality

Modality of the recording.

* **Type:**
  list[str] | None

#### entities

BIDS entities dict (subject, session, etc.).

* **Type:**
  Entities

#### entities_mne

BIDS entities sanitized for compatibility with MNE-Python (e.g. numeric numeric runs).

* **Type:**
  Entities

#### storage

Storage location details.

* **Type:**
  Storage

#### ch_names

List of channel names.

* **Type:**
  list[str] | None

#### sampling_frequency

Sampling rate in Hz.

* **Type:**
  float | None

#### nchans

Channel count.

* **Type:**
  int | None

#### ntimes

Number of time points.

* **Type:**
  int | None

#### digested_at

Timestamp of when this record was processed.

* **Type:**
  str

#### montage_hash

Foreign key into the `montages` collection, pointing at the BIDS
`*_electrodes.tsv` layout this record was recorded with. `None`
when the dataset publishes no scalp electrode positions (e.g.
iEEG depth-electrode datasets or MEG-only recordings).

* **Type:**
  str | None

<!-- !! processed by numpydoc !! -->

#### bids_relpath *: str*

#### bidspath *: str*

#### ch_names *: list[str] | None*

#### data_name *: str*

#### dataset *: str*

#### datatype *: str*

#### digested_at *: str*

#### entities *: Entities*

#### entities_mne *: Entities*

#### extension *: str*

#### montage_hash *: str | None*

#### nchans *: int | None*

#### ntimes *: int | None*

#### recording_modality *: list[str] | None*

#### sampling_frequency *: float | None*

#### storage *: Storage*

#### suffix *: str*

### *class* eegdash.schemas.RecordModel(, dataset: Annotated[str, MinLen(min_length=1)], bids_relpath: Annotated[str, MinLen(min_length=1)], storage: StorageModel, recording_modality: Annotated[list[str], MinLen(min_length=1)], datatype: str | None = None, suffix: str | None = None, extension: str | None = None, entities: EntitiesModel | dict[str, Any] | None = None, \*\*extra_data: Any)

Bases: `BaseModel`

Pydantic model for a single recording file.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### bids_relpath *: str*

#### dataset *: str*

#### datatype *: str | None*

#### entities *: EntitiesModel | dict[str, Any] | None*

#### extension *: str | None*

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### recording_modality *: list[str]*

#### storage *: StorageModel*

#### suffix *: str | None*

### *class* eegdash.schemas.RepositoryStats

Bases: `TypedDict`

Statistics for git-based repositories (e.g. GIN).

#### stars

Number of stars.

* **Type:**
  int

#### forks

Number of forks.

* **Type:**
  int

#### watchers

Number of watchers.

* **Type:**
  int

<!-- !! processed by numpydoc !! -->

#### forks *: int*

#### stars *: int*

#### watchers *: int*

### *class* eegdash.schemas.Storage

Bases: `TypedDict`

Remote storage location details.

#### backend

Storage backend protocol.

* **Type:**
  {‘s3’, ‘https’, ‘local’}

#### base

Base URI (e.g., “s3://openneuro.org/ds000001”).

* **Type:**
  str

#### raw_key

Path relative to base to reach the file.

* **Type:**
  str

#### dep_keys

Paths relative to base for sidecar files (e.g., .json, .vhdr).

* **Type:**
  list[str]

<!-- !! processed by numpydoc !! -->

#### backend *: Literal['s3', 'https', 'local']*

#### base *: str*

#### dep_keys *: list[str]*

#### raw_key *: str*

### *class* eegdash.schemas.StorageModel(\*, backend: ~typing.Annotated[str, ~annotated_types.MinLen(min_length=1)], base: ~typing.Annotated[str, ~annotated_types.MinLen(min_length=1)], raw_key: ~typing.Annotated[str, ~annotated_types.MinLen(min_length=1)], dep_keys: list[str] = <factory>, \*\*extra_data: ~typing.Any)

Bases: `BaseModel`

Pydantic model for storage location details.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### backend *: str*

#### base *: str*

#### dep_keys *: list[str]*

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### raw_key *: str*

### *class* eegdash.schemas.Timestamps

Bases: `TypedDict`

Processing and lifecycle timestamps.

#### digested_at

ISO 8601 timestamp of when the data was processed by EEGDash.

* **Type:**
  str

#### dataset_created_at

ISO 8601 timestamp of when the dataset was originally created.

* **Type:**
  str | None

#### dataset_modified_at

ISO 8601 timestamp of when the dataset was last updated.

* **Type:**
  str | None

<!-- !! processed by numpydoc !! -->

#### dataset_created_at *: str | None*

#### dataset_modified_at *: str | None*

#### digested_at *: str*

### eegdash.schemas.create_dataset(, dataset_id: str, name: str | None = None, canonical_name: list[str] | None = None, source: str = 'openneuro', readme: str | None = None, recording_modality: list[str] | None = None, datatypes: list[str] | None = None, modalities: list[str] | None = None, experimental_modalities: list[str] | None = None, bids_version: str | None = None, license: str | None = None, authors: list[str] | None = None, funding: list[str] | None = None, dataset_doi: str | None = None, associated_paper_doi: str | None = None, tasks: list[str] | None = None, sessions: list[str] | None = None, total_files: int | None = None, size_bytes: int | None = None, data_processed: bool | None = None, study_domain: str | None = None, study_design: str | None = None, subjects_count: int | None = None, ages: list[int] | None = None, age_mean: float | None = None, species: str | None = None, sex_distribution: dict[str, int] | None = None, handedness_distribution: dict[str, int] | None = None, contributing_labs: list[str] | None = None, tags_pathology: list[str] | None = None, tags_modality: list[str] | None = None, tags_type: list[str] | None = None, is_clinical: bool | None = None, clinical_purpose: str | None = None, source_url: str | None = None, osf_url: str | None = None, github_url: str | None = None, paper_url: str | None = None, stars: int | None = None, forks: int | None = None, watchers: int | None = None, senior_author: str | None = None, contact_info: list[str] | None = None, digested_at: str | None = None, dataset_created_at: str | None = None, dataset_modified_at: str | None = None, storage: Storage | None = None) → Dataset

Create a Dataset document.

This helper function constructs a `Dataset` TypedDict with default values
and logic to handle nested structures like demographics, clinical info, and
external links.

* **Parameters:**
  * **dataset_id** (*str*) – Dataset identifier (e.g., “ds001785”).
  * **name** (*str* *,* *optional*) – Dataset title/name.
  * **canonical_name** (*list* *[**str* *]* *,* *optional*) – Canonical / community-recognised name(s) for the dataset (each a valid
    Python identifier, e.g. `["BrainTreeBank"]` or `["SleepEDF",
    "SleepEDFPlus"]`). Used by the dataset class registry to expose
    importable aliases. Empty list or `None` registers no aliases.
  * **source** (*str* *,* *default "openneuro"*) – Data source (“openneuro”, “nemar”, “gin”).
  * **recording_modality** (*list* *[**str* *]* *,* *optional*) – Recording types (e.g., [“eeg”, “meg”, “ieeg”]).
  * **datatypes** (*list* *[**str* *]* *,* *optional*) – BIDS datatypes present in the dataset (e.g., [“eeg”, “anat”, “beh”]).
  * **experimental_modalities** (*list* *[**str* *]* *,* *optional*) – Stimulus/experimental modalities (e.g., [“visual”, “auditory”, “tactile”]).
  * **bids_version** (*str* *,* *optional*) – BIDS version of the dataset.
  * **license** (*str* *,* *optional*) – Dataset license (e.g., “CC0”, “CC-BY-4.0”).
  * **authors** (*list* *[**str* *]* *,* *optional*) – Dataset authors.
  * **funding** (*list* *[**str* *]* *,* *optional*) – Funding sources.
  * **dataset_doi** (*str* *,* *optional*) – Dataset DOI.
  * **associated_paper_doi** (*str* *,* *optional*) – DOI of associated publication.
  * **tasks** (*list* *[**str* *]* *,* *optional*) – Tasks in the dataset.
  * **sessions** (*list* *[**str* *]* *,* *optional*) – Sessions in the dataset.
  * **total_files** (*int* *,* *optional*) – Total number of files.
  * **size_bytes** (*int* *,* *optional*) – Total size in bytes.
  * **data_processed** (*bool* *,* *optional*) – Whether data is processed.
  * **study_domain** (*str* *,* *optional*) – Study domain/topic.
  * **study_design** (*str* *,* *optional*) – Study design description.
  * **subjects_count** (*int* *,* *optional*) – Number of subjects.
  * **ages** (*list* *[**int* *]* *,* *optional*) – Subject ages.
  * **age_mean** (*float* *,* *optional*) – Mean age of subjects.
  * **species** (*str* *,* *optional*) – Species (e.g., “Human”).
  * **sex_distribution** (*dict* *[**str* *,* *int* *]* *,* *optional*) – Sex distribution (e.g., {“m”: 50, “f”: 45}).
  * **handedness_distribution** (*dict* *[**str* *,* *int* *]* *,* *optional*) – Handedness distribution (e.g., {“r”: 80, “l”: 15}).
  * **contributing_labs** (*list* *[**str* *]* *,* *optional*) – Labs that contributed data (for multi-site studies).
  * **is_clinical** (*bool* *,* *optional*) – Whether this is clinical data.
  * **clinical_purpose** (*str* *,* *optional*) – Clinical purpose (e.g., “epilepsy”, “depression”).
  * **paradigm_modality** (*str* *,* *optional*) – Experimental modality (e.g., “visual”, “auditory”, “text”, “multisensory”, “resting_state”).
  * **cognitive_domain** (*str* *,* *optional*) – Cognitive domain (e.g., “attention”, “memory”, “motor”).
  * **is_10_20_system** (*bool* *,* *optional*) – Whether electrodes follow the 10-20 system.
  * **source_url** (*str* *,* *optional*) – Primary URL to the dataset source.
  * **osf_url** (*str* *,* *optional*) – Open Science Framework URL.
  * **github_url** (*str* *,* *optional*) – GitHub repository URL.
  * **paper_url** (*str* *,* *optional*) – URL to associated paper.
  * **stars** (*int* *,* *optional*) – Repository stars count (for git-based sources).
  * **forks** (*int* *,* *optional*) – Repository forks count.
  * **watchers** (*int* *,* *optional*) – Repository watchers count.
  * **digested_at** (*str* *,* *optional*) – ISO 8601 timestamp. If not provided, no timestamp is set (for deterministic output).
  * **dataset_modified_at** (*str* *,* *optional*) – Last modification timestamp.
* **Returns:**
  A fully populated Dataset document.
* **Return type:**
  Dataset

<!-- !! processed by numpydoc !! -->

### eegdash.schemas.create_record(, dataset: str, storage_base: str, bids_relpath: str, subject: str | None = None, session: str | None = None, task: str | None = None, run: str | None = None, acquisition: str | None = None, dep_keys: list[str] | None = None, datatype: str = 'eeg', suffix: str = 'eeg', storage_backend: Literal['s3', 'https', 'local'] = 's3', recording_modality: list[str] | None = None, ch_names: list[str] | None = None, sampling_frequency: float | None = None, nchans: int | None = None, ntimes: int | None = None, digested_at: str | None = None) → Record

Create an EEGDash record.

Helper to construct a valid `Record` TypedDict.

* **Parameters:**
  * **dataset** (*str*) – Dataset identifier (e.g., “ds000001”).
  * **storage_base** (*str*) – Remote storage base URI (e.g., “s3://openneuro.org/ds000001”).
  * **bids_relpath** (*str*) – BIDS-relative path to the raw file (e.g., “sub-01/eeg/sub-01_task-rest_eeg.vhdr”).
  * **subject** (*str* *,* *optional*) – BIDS entities.
  * **session** (*str* *,* *optional*) – BIDS entities.
  * **task** (*str* *,* *optional*) – BIDS entities.
  * **run** (*str* *,* *optional*) – BIDS entities.
  * **acquisition** (*str* *,* *optional*) – BIDS entities.
  * **dep_keys** (*list* *[**str* *]* *,* *optional*) – Dependency paths relative to storage_base.
  * **datatype** (*str* *,* *default "eeg"*) – BIDS datatype.
  * **suffix** (*str* *,* *default "eeg"*) – BIDS suffix.
  * **storage_backend** ( *{"s3"* *,*  *"https"* *,*  *"local"}* *,* *default "s3"*) – Storage backend type.
  * **recording_modality** (*list* *[**str* *]* *,* *optional*) – Recording modalities (e.g., [“eeg”, “meg”, “ieeg”]).
  * **digested_at** (*str* *,* *optional*) – ISO 8601 timestamp. Defaults to current time.
* **Returns:**
  A slim EEGDash record optimized for loading.
* **Return type:**
  Record

### Notes

Clinical and paradigm info is stored at the Dataset level, not per-file.

### Examples

```pycon
>>> record = create_record(
...     dataset="ds000001",
...     storage_base="s3://openneuro.org/ds000001",
...     bids_relpath="sub-01/eeg/sub-01_task-rest_eeg.vhdr",
...     subject="01",
...     task="rest",
... )
```

<!-- !! processed by numpydoc !! -->

### eegdash.schemas.validate_dataset(dataset: dict[str, Any]) → list[str]

Validate a dataset has required fields. Returns list of errors.

<!-- !! processed by numpydoc !! -->

### eegdash.schemas.validate_record(record: dict[str, Any]) → list[str]

Validate a record has required fields. Returns list of errors.

### Notes

- bids_relpath is the canonical unique identifier for records
- bidspath is a computed field (dataset + “/” + bids_relpath) and not strictly required
- storage.raw_key always equals bids_relpath when created via create_record

<!-- !! processed by numpydoc !! -->
