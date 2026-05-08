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

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

Clinical classification metadata (dataset-level).

#### Deprecated
Deprecated since version Use: the `tags` field with `pathology` key instead.

#### is_clinical

True if the dataset contains clinical population data.

* **Type:**
  [bool](https://docs.python.org/3/library/functions.html#bool)

#### purpose

The clinical condition or purpose (e.g., “epilepsy”, “depression”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

<!-- !! processed by numpydoc !! -->

#### is_clinical *: [bool](https://docs.python.org/3/library/functions.html#bool)*

#### purpose *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

### *class* eegdash.schemas.Dataset

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

TypedDict schema for a full Dataset document.

This Dictionary represents all metadata available for a study/dataset.

#### dataset_id

Unique identifier (e.g., “ds001785”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### name

Descriptive title of the dataset.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### canonical_name

Canonical / community-recognised name(s) for the dataset, each a valid
Python identifier (e.g. `["BrainTreeBank"]`, `["SleepEDF",
"SleepEDFPlus"]`). Used to register importable class aliases alongside
the `DS…`-style ID. Empty list or `None` means no alias is
registered.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | None

#### source

Origin source (e.g., “openneuro”, “nemar”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### readme

Content of the dataset’s README file.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### recording_modality

List of recording modalities (e.g., [“eeg”, “meg”]).

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### datatypes

BIDS datatypes present (e.g., [“eeg”, “anat”]).

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### experimental_modalities

Stimulus types used (e.g., [“visual”, “auditory”]).

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | None

#### bids_version

Version of the BIDS standard used.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### license

License string (e.g., “CC0”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### authors

List of author names.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### funding

List of funding sources.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### dataset_doi

Digital Object Identifier for the dataset.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### associated_paper_doi

DOI of the paper associated with the dataset.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### tasks

List of task names found in the dataset.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### sessions

List of session names.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### total_files

Total file count.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int) | None

#### size_bytes

Total dataset size in bytes.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int) | None

#### data_processed

Indicates if the data has been pre-processed.

* **Type:**
  [bool](https://docs.python.org/3/library/functions.html#bool) | None

#### study_domain

General domain of the study.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### study_design

Description of the study design.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### contributing_labs

List of labs contributing to the dataset.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | None

#### n_contributing_labs

Count of contributing labs.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int) | None

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
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### contact_info

Contact emails or names.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | None

#### timestamps

Timestamps for data processing and creation.

* **Type:**
  Timestamps

#### nemar_citation_count

Number of papers citing this dataset (from NEMAR citations repository).

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int) | None

<!-- !! processed by numpydoc !! -->

#### associated_paper_doi *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### authors *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### bids_version *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### canonical_name *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None)*

#### clinical *: Clinical*

#### contact_info *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None)*

#### contributing_labs *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None)*

#### data_processed *: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None)*

#### dataset_doi *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### dataset_id *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### datatypes *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### demographics *: Demographics*

#### experimental_modalities *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None)*

#### external_links *: ExternalLinks*

#### funding *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### ingestion_fingerprint *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### license *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### n_contributing_labs *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

#### name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### nemar_citation_count *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

#### readme *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### recording_modality *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### repository_stats *: RepositoryStats | [None](https://docs.python.org/3/library/constants.html#None)*

#### senior_author *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### sessions *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### size_bytes *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

#### source *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### storage *: Storage | [None](https://docs.python.org/3/library/constants.html#None)*

#### study_design *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### study_domain *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### tags *: Tags*

#### tasks *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### timestamps *: Timestamps*

#### total_files *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

### *class* eegdash.schemas.DatasetModel(, dataset_id: [Annotated](https://docs.python.org/3/library/typing.html#typing.Annotated)[[str](https://docs.python.org/3/library/stdtypes.html#str), MinLen(min_length=1)], source: [Annotated](https://docs.python.org/3/library/typing.html#typing.Annotated)[[str](https://docs.python.org/3/library/stdtypes.html#str), MinLen(min_length=1)], recording_modality: [Annotated](https://docs.python.org/3/library/typing.html#typing.Annotated)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], MinLen(min_length=1)], ingestion_fingerprint: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, senior_author: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, contact_info: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, timestamps: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, storage: StorageModel | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*extra_data: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: `BaseModel`

Pydantic model for dataset-level metadata.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### contact_info *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None)*

#### dataset_id *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### ingestion_fingerprint *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### recording_modality *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### senior_author *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### source *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### storage *: StorageModel | [None](https://docs.python.org/3/library/constants.html#None)*

#### timestamps *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Any] | [None](https://docs.python.org/3/library/constants.html#None)*

### *class* eegdash.schemas.Demographics

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

Subject demographics summary for a dataset.

#### subjects_count

Total number of subjects.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int)

#### ages

List of all subject ages (if available).

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]

#### age_min

Minimum age in the cohort.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int) | None

#### age_max

Maximum age in the cohort.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int) | None

#### age_mean

Mean age of subjects.

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float) | None

#### species

Species of subjects (e.g., “Human”, “Mouse”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### sex_distribution

Count of subjects by sex (e.g., {“m”: 50, “f”: 45}).

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] | None

#### handedness_distribution

Count of subjects by handedness (e.g., {“r”: 80, “l”: 15}).

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] | None

<!-- !! processed by numpydoc !! -->

#### age_max *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

#### age_mean *: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

#### age_min *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

#### ages *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)]*

#### handedness_distribution *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None)*

#### sex_distribution *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None)*

#### species *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### subjects_count *: [int](https://docs.python.org/3/library/functions.html#int)*

### *class* eegdash.schemas.Entities

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

BIDS entities parsed from the file path.

#### subject

Subject label (e.g., “01”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### session

Session label (e.g., “pre”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### task

Task label (e.g., “rest”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### run

Run label (e.g., “1” or “01”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### acquisition

Acquisition label (e.g., “bipolar”, “PSG”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

<!-- !! processed by numpydoc !! -->

#### acquisition *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### run *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### session *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### subject *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### task *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

### *class* eegdash.schemas.EntitiesModel(, subject: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, session: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, task: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, run: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, acquisition: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*extra_data: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: `BaseModel`

Pydantic model for BIDS entities.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### acquisition *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### run *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### session *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### subject *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### task *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

### *class* eegdash.schemas.ExternalLinks

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

Relevant external hyperlinks for the dataset.

#### source_url

URL to the primary data source (e.g. OpenNeuro page).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### osf_url

URL to the Open Science Framework project.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### github_url

URL to the associated GitHub repository.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### paper_url

URL to the primary publication.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

<!-- !! processed by numpydoc !! -->

#### github_url *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### osf_url *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### paper_url *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### source_url *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

### *class* eegdash.schemas.ManifestFileModel(, path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*extra_data: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

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

#### name *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### path *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### path_or_name() → [str](https://docs.python.org/3/library/stdtypes.html#str)

Return the path or name of the file.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.schemas.ManifestModel(, source: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, files: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str) | ManifestFileModel], \*\*extra_data: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: `BaseModel`

Pydantic model for a dataset file manifest.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### files *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str) | ManifestFileModel]*

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### source *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

### *class* eegdash.schemas.Record

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

TypedDict schema for a Record document.

Represents a single data file and its metadata. This structure is kept flat
and minimal to ensure fast loading times when querying millions of records.

#### dataset

Foreign key matching `Dataset.dataset_id`.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### data_name

Unique name for the data item (e.g., “ds001_sub-01_task-rest”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### bidspath

Legacy path identifier (e.g., “ds001/sub-01/eeg/…”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### bids_relpath

Standard BIDS relative path (e.g., “sub-01/eeg/…”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### datatype

BIDS datatype (e.g., “eeg”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### suffix

Filename suffix (e.g., “eeg”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### extension

File extension (e.g., “.vhdr”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### recording_modality

Modality of the recording.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | None

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
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | None

#### sampling_frequency

Sampling rate in Hz.

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float) | None

#### nchans

Channel count.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int) | None

#### ntimes

Number of time points.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int) | None

#### digested_at

Timestamp of when this record was processed.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### montage_hash

Foreign key into the `montages` collection, pointing at the BIDS
`*_electrodes.tsv` layout this record was recorded with. `None`
when the dataset publishes no scalp electrode positions (e.g.
iEEG depth-electrode datasets or MEG-only recordings).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

<!-- !! processed by numpydoc !! -->

#### bids_relpath *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### bidspath *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### ch_names *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None)*

#### data_name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### dataset *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### datatype *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### digested_at *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### entities *: Entities*

#### entities_mne *: Entities*

#### extension *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### montage_hash *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### nchans *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

#### ntimes *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)*

#### recording_modality *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None)*

#### sampling_frequency *: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None)*

#### storage *: Storage*

#### suffix *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

### *class* eegdash.schemas.RecordModel(, dataset: [Annotated](https://docs.python.org/3/library/typing.html#typing.Annotated)[[str](https://docs.python.org/3/library/stdtypes.html#str), MinLen(min_length=1)], bids_relpath: [Annotated](https://docs.python.org/3/library/typing.html#typing.Annotated)[[str](https://docs.python.org/3/library/stdtypes.html#str), MinLen(min_length=1)], storage: StorageModel, recording_modality: [Annotated](https://docs.python.org/3/library/typing.html#typing.Annotated)[[list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)], MinLen(min_length=1)], datatype: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, suffix: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, extension: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, entities: EntitiesModel | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*extra_data: [Any](https://docs.python.org/3/library/typing.html#typing.Any))

Bases: `BaseModel`

Pydantic model for a single recording file.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### bids_relpath *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### dataset *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### datatype *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### entities *: EntitiesModel | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Any] | [None](https://docs.python.org/3/library/constants.html#None)*

#### extension *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### recording_modality *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### storage *: StorageModel*

#### suffix *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

### *class* eegdash.schemas.RepositoryStats

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

Statistics for git-based repositories (e.g. GIN).

#### stars

Number of stars.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int)

#### forks

Number of forks.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int)

#### watchers

Number of watchers.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int)

<!-- !! processed by numpydoc !! -->

#### forks *: [int](https://docs.python.org/3/library/functions.html#int)*

#### stars *: [int](https://docs.python.org/3/library/functions.html#int)*

#### watchers *: [int](https://docs.python.org/3/library/functions.html#int)*

### *class* eegdash.schemas.Storage

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

Remote storage location details.

#### backend

Storage backend protocol. `"nemar"` is a non-fetchable marker
for NEMAR-hosted datasets — see `StorageAccessError` for the
out-of-band access paths (git-annex / nemar CLI / NEMAR API).

* **Type:**
  {‘s3’, ‘https’, ‘local’, ‘nemar’}

#### base

Base URI (e.g., “s3://openneuro.org/ds000001”).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### raw_key

Path relative to base to reach the file.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### dep_keys

Paths relative to base for sidecar files (e.g., .json, .vhdr).

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### annex_keys

Sparse map `relpath -> SHA-key` populated at digest time for
NEMAR records so the runtime can build the SHA-addressed S3 URI
directly without a GitHub-pointer round-trip. Mutually exclusive
with `sidecar_inline` for any given relpath.

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], optional

#### sidecar_inline

Sparse map `relpath -> UTF-8 text` populated at digest time
for small git-tracked NEMAR sidecars (TSV/JSON/README) so the
runtime can write them directly to disk without a GitHub fetch.
Mutually exclusive with `annex_keys` for any given relpath.

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], optional

<!-- !! processed by numpydoc !! -->

#### annex_keys *: [NotRequired](https://docs.python.org/3/library/typing.html#typing.NotRequired)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]]*

#### backend *: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['s3', 'https', 'local', 'nemar']*

#### base *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### dep_keys *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### raw_key *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### sidecar_inline *: [NotRequired](https://docs.python.org/3/library/typing.html#typing.NotRequired)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]]*

### *class* eegdash.schemas.StorageModel(\*, backend: ~typing.Annotated[str, ~annotated_types.MinLen(min_length=1)], base: ~typing.Annotated[str, ~annotated_types.MinLen(min_length=1)], raw_key: ~typing.Annotated[str, ~annotated_types.MinLen(min_length=1)], dep_keys: list[str] = <factory>, annex_keys: dict[str, str] | None = None, sidecar_inline: dict[str, str] | None = None, \*\*extra_data: ~typing.Any)

Bases: `BaseModel`

Pydantic model for storage location details.

<!-- !! processed by numpydoc !! -->

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

<!-- !! processed by numpydoc !! -->

#### annex_keys *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None)*

#### backend *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### base *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### dep_keys *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

#### model_config *= {'extra': 'allow'}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

<!-- !! processed by numpydoc !! -->

#### raw_key *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### sidecar_inline *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None)*

### *class* eegdash.schemas.Timestamps

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

Processing and lifecycle timestamps.

#### digested_at

ISO 8601 timestamp of when the data was processed by EEGDash.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### dataset_created_at

ISO 8601 timestamp of when the dataset was originally created.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### dataset_modified_at

ISO 8601 timestamp of when the dataset was last updated.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

<!-- !! processed by numpydoc !! -->

#### dataset_created_at *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### dataset_modified_at *: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None)*

#### digested_at *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

### eegdash.schemas.create_dataset(, dataset_id: [str](https://docs.python.org/3/library/stdtypes.html#str), name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, canonical_name: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, source: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'openneuro', readme: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, recording_modality: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, datatypes: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, modalities: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, experimental_modalities: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, bids_version: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, license: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, authors: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, funding: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, dataset_doi: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, associated_paper_doi: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, tasks: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, sessions: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, total_files: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, size_bytes: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, data_processed: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = None, study_domain: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, study_design: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, subjects_count: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, ages: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, age_mean: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, species: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, sex_distribution: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, handedness_distribution: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, contributing_labs: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, tags_pathology: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, tags_modality: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, tags_type: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, is_clinical: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = None, clinical_purpose: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, source_url: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, osf_url: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, github_url: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, paper_url: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, stars: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, forks: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, watchers: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, senior_author: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, contact_info: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, digested_at: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, dataset_created_at: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, dataset_modified_at: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, storage: Storage | [None](https://docs.python.org/3/library/constants.html#None) = None) → Dataset

Create a Dataset document.

This helper function constructs a `Dataset` TypedDict with default values
and logic to handle nested structures like demographics, clinical info, and
external links.

* **Parameters:**
  * **dataset_id** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Dataset identifier (e.g., “ds001785”).
  * **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Dataset title/name.
  * **canonical_name** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Canonical / community-recognised name(s) for the dataset (each a valid
    Python identifier, e.g. `["BrainTreeBank"]` or `["SleepEDF",
    "SleepEDFPlus"]`). Used by the dataset class registry to expose
    importable aliases. Empty list or `None` registers no aliases.
  * **source** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *default "openneuro"*) – Data source (“openneuro”, “nemar”, “gin”).
  * **recording_modality** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Recording types (e.g., [“eeg”, “meg”, “ieeg”]).
  * **datatypes** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – BIDS datatypes present in the dataset (e.g., [“eeg”, “anat”, “beh”]).
  * **experimental_modalities** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Stimulus/experimental modalities (e.g., [“visual”, “auditory”, “tactile”]).
  * **bids_version** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – BIDS version of the dataset.
  * **license** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Dataset license (e.g., “CC0”, “CC-BY-4.0”).
  * **authors** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Dataset authors.
  * **funding** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Funding sources.
  * **dataset_doi** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Dataset DOI.
  * **associated_paper_doi** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – DOI of associated publication.
  * **tasks** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Tasks in the dataset.
  * **sessions** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Sessions in the dataset.
  * **total_files** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Total number of files.
  * **size_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Total size in bytes.
  * **data_processed** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *optional*) – Whether data is processed.
  * **study_domain** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Study domain/topic.
  * **study_design** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Study design description.
  * **subjects_count** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Number of subjects.
  * **ages** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3/library/functions.html#int) *]* *,* *optional*) – Subject ages.
  * **age_mean** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – Mean age of subjects.
  * **species** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Species (e.g., “Human”).
  * **sex_distribution** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3/library/functions.html#int) *]* *,* *optional*) – Sex distribution (e.g., {“m”: 50, “f”: 45}).
  * **handedness_distribution** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3/library/functions.html#int) *]* *,* *optional*) – Handedness distribution (e.g., {“r”: 80, “l”: 15}).
  * **contributing_labs** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Labs that contributed data (for multi-site studies).
  * **is_clinical** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *optional*) – Whether this is clinical data.
  * **clinical_purpose** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Clinical purpose (e.g., “epilepsy”, “depression”).
  * **paradigm_modality** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Experimental modality (e.g., “visual”, “auditory”, “text”, “multisensory”, “resting_state”).
  * **cognitive_domain** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Cognitive domain (e.g., “attention”, “memory”, “motor”).
  * **is_10_20_system** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *optional*) – Whether electrodes follow the 10-20 system.
  * **source_url** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Primary URL to the dataset source.
  * **osf_url** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Open Science Framework URL.
  * **github_url** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – GitHub repository URL.
  * **paper_url** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – URL to associated paper.
  * **stars** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Repository stars count (for git-based sources).
  * **forks** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Repository forks count.
  * **watchers** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Repository watchers count.
  * **digested_at** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – ISO 8601 timestamp. If not provided, no timestamp is set (for deterministic output).
  * **dataset_modified_at** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Last modification timestamp.
* **Returns:**
  A fully populated Dataset document.
* **Return type:**
  Dataset

<!-- !! processed by numpydoc !! -->

### eegdash.schemas.create_record(, dataset: [str](https://docs.python.org/3/library/stdtypes.html#str), storage_base: [str](https://docs.python.org/3/library/stdtypes.html#str), bids_relpath: [str](https://docs.python.org/3/library/stdtypes.html#str), subject: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, session: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, task: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, run: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, acquisition: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, dep_keys: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, datatype: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'eeg', suffix: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'eeg', storage_backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['s3', 'https', 'local', 'nemar'] = 's3', recording_modality: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, ch_names: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, sampling_frequency: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, nchans: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, ntimes: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, digested_at: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, annex_keys: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, sidecar_inline: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → Record

Create an EEGDash record.

Helper to construct a valid `Record` TypedDict.

* **Parameters:**
  * **dataset** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Dataset identifier (e.g., “ds000001”).
  * **storage_base** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Remote storage base URI (e.g., “s3://openneuro.org/ds000001”).
  * **bids_relpath** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – BIDS-relative path to the raw file (e.g., “sub-01/eeg/sub-01_task-rest_eeg.vhdr”).
  * **subject** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – BIDS entities.
  * **session** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – BIDS entities.
  * **task** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – BIDS entities.
  * **run** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – BIDS entities.
  * **acquisition** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – BIDS entities.
  * **dep_keys** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Dependency paths relative to storage_base.
  * **datatype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *default "eeg"*) – BIDS datatype.
  * **suffix** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *default "eeg"*) – BIDS suffix.
  * **storage_backend** ( *{"s3"* *,*  *"https"* *,*  *"local"}* *,* *default "s3"*) – Storage backend type.
  * **recording_modality** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Recording modalities (e.g., [“eeg”, “meg”, “ieeg”]).
  * **digested_at** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – ISO 8601 timestamp. Defaults to current time.
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

### eegdash.schemas.validate_dataset(dataset: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

Validate a dataset has required fields. Returns list of errors.

<!-- !! processed by numpydoc !! -->

### eegdash.schemas.validate_record(record: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

Validate a record has required fields. Returns list of errors.

### Notes

- bids_relpath is the canonical unique identifier for records
- bidspath is a computed field (dataset + “/” + bids_relpath) and not strictly required
- storage.raw_key always equals bids_relpath when created via create_record

<!-- !! processed by numpydoc !! -->
