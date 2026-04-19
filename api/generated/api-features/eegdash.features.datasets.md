# eegdash.features.datasets

Datasets for Feature Management.

This module defines the core data structures for storing, manipulating, and
serializing extracted features.

Provides the base classes:

- `FeaturesDataset` — Represents features from a single recording.
- `FeaturesConcatDataset` — Manages multiple `FeaturesDataset`
  objects as a unified dataset.

<!-- !! processed by numpydoc !! -->

### Classes

| `FeaturesDataset`(features[, metadata, ...])   | A dataset of features extracted from a single recording.               |
|------------------------------------------------|------------------------------------------------------------------------|
| `FeaturesConcatDataset`([list_of_ds, ...])     | A concatenated dataset composed of multiple `FeaturesDataset` objects. |

### *class* eegdash.features.datasets.FeaturesDataset(features: DataFrame, metadata: DataFrame | None = None, description: dict | Series | None = None, transform: Callable | None = None, raw_info: Dict | None = None, raw_preproc_kwargs: Dict | None = None, window_kwargs: Dict | None = None, window_preproc_kwargs: Dict | None = None, features_kwargs: Dict | None = None)

Bases: `EEGWindowsDataset`

A dataset of features extracted from a single recording.

This class holds features in a `pandas.DataFrame` and provides an interface
compatible with braindecode’s dataset structure. A single object corresponds
to one recording.

* **Parameters:**
  * **features** (*pandas.DataFrame*) – A DataFrame where each row is a sample (e.g, EEG window)
    and each column is a feature.
  * **metadata** (*pandas.DataFrame* *,* *optional*) – A DataFrame containing metadata for each sample, indexed consistently
    with features. Must include columns ‘i_window_in_trial’,
    ‘i_start_in_trial’, ‘i_stop_in_trial’, and ‘target’.
  * **description** (*dict* *or* *pandas.Series* *,* *optional*) – Additional high-level information about the dataset.
  * **transform** (*callable* *,* *optional*) – A function or transform to apply to the feature data.
  * **raw_info** (*dict* *,* *optional*) – Information about the original raw recording (e.g., sampling rate,
    montage, channel names).
  * **raw_preproc_kwargs** (*dict* *,* *optional*) – Keyword arguments used for preprocessing the raw data.
  * **window_kwargs** (*dict* *,* *optional*) – Keyword arguments used for windowing the data.
  * **window_preproc_kwargs** (*dict* *,* *optional*) – Keyword arguments used for preprocessing the windowed data.
  * **features_kwargs** (*dict* *,* *optional*) – Keyword arguments used for feature extraction.

#### features

Table of extracted features.

* **Type:**
  pandas.DataFrame

#### n_features

Number of feature columns in the dataset.

* **Type:**
  int

#### metadata

Metadata describing each window.

* **Type:**
  pandas.DataFrame

#### transform

The transform applied to each sample.

* **Type:**
  callable or None

#### raw_info

Information about the raw recording.

* **Type:**
  dict or None

#### raw_preproc_kwargs

Parameters used during raw data preprocessing.

* **Type:**
  dict or None

#### window_kwargs

Parameters used during window segmentation.

* **Type:**
  dict or None

#### window_preproc_kwargs

Parameters used during window-level preprocessing.

* **Type:**
  dict or None

#### features_kwargs

Parameters used during feature extraction.

* **Type:**
  dict or None

#### crop_inds

Indices specifying window position within each trial:
(i_window_in_trial, i_start_in_trial, i_stop_in_trial).

* **Type:**
  numpy.ndarray of shape (n_samples, 3)

#### y

Target labels corresponding to each window.

* **Type:**
  list of int

<!-- !! processed by numpydoc !! -->

### *class* eegdash.features.datasets.FeaturesConcatDataset(list_of_ds: list[TypeAliasForwardRef('eegdash.features.datasets.FeaturesDataset')] | None = None, target_transform: Callable | None = None)

Bases: `BaseConcatDataset`

A concatenated dataset composed of multiple `FeaturesDataset` objects.

This class manages a collection of `FeaturesDataset` instances and
provides an interface for treating them as a single, unified dataset.
Supports concatenation, splitting, saving, and performing DataFrame-like
operations across all contained datasets.

* **Parameters:**
  * **list_of_ds** (*list* *of* *FeaturesDataset* *or* *None* *,* *optional*) – A list of `FeaturesDataset` objects to concatenate.
    If a list of `FeaturesConcatDataset` objects is provided,
    all contained datasets are automatically flattened into a single list.
  * **target_transform** (*callable* *or* *None* *,* *optional*) – A function to apply to target values before they are returned.

#### datasets

The list of individual datasets contained in this object.

* **Type:**
  list of FeaturesDataset

#### target_transform

Optional transform applied to target labels.

* **Type:**
  callable or None

<!-- !! processed by numpydoc !! -->

#### split(by: str | list[int] | list[list[int]] | dict[str, list[int]]) → dict[str, TypeAliasForwardRef('eegdash.features.datasets.FeaturesConcatDataset')]

Split the concatenated dataset into multiple subsets.

This method allows flexible splitting of the concatenated dataset into
several `FeaturesConcatDataset` objects based on a metadata field,
explicit indices, or custom grouping definitions.

* **Parameters:**
  **by** (*str* *or* *list* *of* *int* *or* *list* *of* *list* *of* *int* *or* *dict* *of*  *{str: list* *of* *int}*) – 

  Defines how the dataset is split:
  * **str** — Name of a column in the dataset description.
    Each unique value in that column defines a separate split.
  * **list of int** — Indices of datasets to include in one split.
  * **list of list of int** — A list of groups of indices, where each sub-list
    defines one split.
  * **dict of {str: list of int}** — Explicit mapping of split names to
    lists of dataset indices.
* **Returns:**
  A dictionary where each key is the split name (or index)
  and each value is a `FeaturesConcatDataset` containing
  the corresponding subset of datasets.
* **Return type:**
  dict[str, FeaturesConcatDataset]

### Examples

```pycon
>>> # Split by a metadata column (str)
>>> splits = concat_ds.split(by='subject_id')
>>> list(splits.keys())
['subj_01', 'subj_02', 'subj_03']
>>> splits['subj_01']
<FeaturesConcatDataset>
```

```pycon
>>> # Split by explicit indices (list of int)
>>> splits = concat_ds.split(by=[0, 2, 4])
>>> splits["0"]
<FeaturesConcatDataset>
```

```pycon
>>> # Split by groups of indices (list of list of int)
>>> splits = concat_ds.split(by=[[0, 1], [2, 3], [4, 5]])
>>> list(splits.keys())
['0', '1', '2']
```

```pycon
>>> # Split by custom mapping (dict)
>>> splits = concat_ds.split(by={'train': [0, 1, 2], 'test': [3, 4]})
>>> splits["train"], splits["test"]
(<FeaturesConcatDataset>, <FeaturesConcatDataset>)
```

### Notes

The resulting splits inherit the same `target_transform` as the original
dataset. Splitting by a string requires that `self.description` contains
the specified column.

<!-- !! processed by numpydoc !! -->

#### get_metadata() → DataFrame

Return a concatenated metadata DataFrame from all contained datasets.

Collects the metadata of each `FeaturesDataset` contained in
the `FeaturesConcatDataset` and concatenates them into a single
pandas DataFrame, adding each dataset’s description entries as
additional columns in the resulting DataFrame.

* **Returns:**
  Combined metadata from all contained datasets.
  Each row corresponds to a single sample from one of the underlying
  `FeaturesDataset` objects.
  Columns include both window-level metadata (e.g., `target`,
  `i_window_in_trial`, `i_start_in_trial`, `i_stop_in_trial`)
  and dataset-level description fields (e.g., `subject_id`,
  `session`, etc.).
* **Return type:**
  pandas.DataFrame
* **Raises:**
  **TypeError** – If one or more contained datasets are not instances
  of `FeaturesDataset`.

<!-- !! processed by numpydoc !! -->

#### save(path: str, overwrite: bool = False, offset: int = 0) → None

Save the concatenated dataset to a directory.

Each contained `FeaturesDataset` is saved in its own
numbered subdirectory within the specified `path`. The resulting
structure is compatible with later reloading using
`serialization.load_features_concat_dataset()`.

**Directory structure example**:

```default
path/
    0/
        0-feat.safetensors
        metadata_df.pkl
        description.json
        ...
    1/
        1-feat.safetensors
        ...
```

* **Parameters:**
  * **path** (*str*) – Path to the parent directory where the dataset should be saved.
    The directory will be created if it does not exist.
  * **overwrite** (*bool* *,* *default=False*) – If True, existing subdirectories that conflict with the new ones
    are removed before saving.
  * **offset** (*int* *,* *default=0*) – Integer offset added to subdirectory names. Useful when saving
    datasets in chunks or continuing a previous save session.
* **Raises:**
  * **ValueError** – If the concatenated dataset is empty.
  * **FileExistsError** – If a subdirectory already exists and `overwrite` is False.
* **Warns:**
  **UserWarning** – If the number of saved subdirectories does not match the number
  of existing ones, or if unrelated files remain in the directory.

### Notes

Each subdirectory contains:

- `*-feat.safetensors` — feature DataFrame for that dataset.
- `metadata_df.pkl` — corresponding metadata.
- `description.json` — dataset-level metadata.
- `raw_info.pkl` — recording information (optional).
- `*_kwargs.json` — preprocessing parameters.

<!-- !! processed by numpydoc !! -->

#### to_dataframe(include_metadata: bool | str | List[str] = False, include_target: bool = False, include_crop_inds: bool = False) → DataFrame

Convert the concatenated dataset into a single unified pandas DataFrame.

This method flattens the collection of individual recording datasets into
one table, allowing for the selective inclusion of metadata, target
labels, and window-cropping indices alongside features.

* **Parameters:**
  * **include_metadata** (*bool* *,* *str* *, or* *list* *of* *str* *,* *default=False*) – 

    Controls the inclusion of window-level metadata:
    - If **True** — includes all metadata columns available in the
      : underlying datasets.
    - If **str** or **list of str** — includes only the specified
      : metadata column(s).
    - If **False** — excludes metadata (unless overridden by other
      : flags).
  * **include_target** (*bool* *,* *default=False*) – If True, ensures the ‘target’ column is included in the resulting
    DataFrame.
  * **include_crop_inds** (*bool* *,* *default=False*) – If True, includes the internal windowing indices: ‘i_dataset’,
    ‘i_window_in_trial’, ‘i_start_in_trial’, and ‘i_stop_in_trial’.
* **Returns:**
  A concatenated DataFrame where each row represents a sample (window)
  and columns contain features and requested metadata.
* **Return type:**
  pd.DataFrame

### Notes

When metadata columns and feature columns share the same name, the
metadata columns are suffixed with `_metadata` to avoid name
collisions.

### Examples

```pycon
>>> # Get only features
>>> df = concat_ds.to_dataframe()
```

```pycon
>>> # Get features with target labels and specific metadata
>>> df = concat_ds.to_dataframe(
...     include_metadata=['subject_id'],
...     include_target=True
... )
```

<!-- !! processed by numpydoc !! -->

#### count(numeric_only: bool = False, n_jobs: int = 1) → Series

Count non-NA cells for each feature column across all datasets.

* **Parameters:**
  * **numeric_only** (*bool* *,* *default=False*) – If True, only includes columns with float, int, or boolean data
    types.
  * **n_jobs** (*int* *,* *default=1*) – The number of CPU cores to use for parallel processing of
    individual datasets.
* **Returns:**
  A Series containing the total count of non-missing values for
  each feature column, indexed by feature names.
* **Return type:**
  pd.Series

<!-- !! processed by numpydoc !! -->

#### mean(numeric_only: bool = False, n_jobs: int = 1) → Series

Compute the mean for each feature column across all datasets.

This method calculates the mean of each feature by aggregating the
individual means of each dataset, weighted by their respective
sample counts.

* **Parameters:**
  * **numeric_only** (*bool* *,* *default=False*) – If True, only includes columns with float, int, or boolean data
    types.
  * **n_jobs** (*int* *,* *default=1*) – The number of CPU cores to use for parallel processing of
    individual datasets.
* **Returns:**
  A Series containing the weighted mean of each feature column,
  indexed by feature names.
* **Return type:**
  pd.Series

<!-- !! processed by numpydoc !! -->

#### var(ddof: int = 1, numeric_only: bool = False, n_jobs: int = 1) → Series

Compute the variance for each feature column across all datasets.

This method calculates the total variance by combining within-dataset
variability and between-dataset mean differences.

* **Parameters:**
  * **ddof** (*int* *,* *default=1*) – Delta Degrees of Freedom.
  * **numeric_only** (*bool* *,* *default=False*) – If True, only includes columns with float, int, or boolean data
    types.
  * **n_jobs** (*int* *,* *default=1*) – The number of CPU cores to use for parallel processing of
    individual datasets.
* **Returns:**
  A Series containing the pooled variance of each feature column,
  indexed by feature names.
* **Return type:**
  pd.Series

<!-- !! processed by numpydoc !! -->

#### std(ddof: int = 1, numeric_only: bool = False, eps: float = 0, n_jobs: int = 1) → Series

Compute the standard deviation for each feature column across all datasets.

* **Parameters:**
  * **ddof** (*int* *,* *default=1*) – Delta Degrees of Freedom for the variance calculation.
  * **numeric_only** (*bool* *,* *default=False*) – If True, only includes numeric data types.
  * **eps** (*float* *,* *default=0*) – Small constant added to variance for numerical stability.
  * **n_jobs** (*int* *,* *default=1*) – Number of CPU cores for parallel processing.
* **Returns:**
  Standard deviation of each feature column. Indexed by feature names.
* **Return type:**
  pd.Series

<!-- !! processed by numpydoc !! -->

#### zscore(ddof: int = 1, numeric_only: bool = False, eps: float = 0, n_jobs: int = 1) → None

Apply z-score normalization to numeric columns in-place.

This method scales features to a mean of 0 and a standard deviation
of 1 based on statistics pooled across all contained datasets.

* **Parameters:**
  * **ddof** (*int* *,* *default=1*) – Delta Degrees of Freedom for the pooled variance.
  * **numeric_only** (*bool* *,* *default=False*) – If True, only includes numeric data types.
  * **eps** (*float* *,* *default=0*) – Small constant added to variance for numerical stability.
  * **n_jobs** (*int* *,* *default=1*) – Number of CPU cores for parallel statistics computation.

<!-- !! processed by numpydoc !! -->

#### fillna(\*args, \*\*kwargs) → None

Fill NA/NaN values in-place across all datasets.

* **Parameters:**
  * **\*args** – Arguments passed to `pandas.DataFrame.fillna()`.
  * **\*\*kwargs** – Arguments passed to `pandas.DataFrame.fillna()`.

### Notes

`inplace` is enforced as True.

#### SEE ALSO
`pandas.DataFrame.fillna`
: The underlying pandas method.

<!-- !! processed by numpydoc !! -->

#### replace(\*args, \*\*kwargs) → None

Replace values in-place across all datasets.

* **Parameters:**
  * **\*args** – Arguments passed to `pandas.DataFrame.replace()`.
  * **\*\*kwargs** – Arguments passed to `pandas.DataFrame.replace()`.

### Notes

`inplace` is enforced as True.

#### SEE ALSO
`pandas.DataFrame.replace`
: The underlying pandas method.

<!-- !! processed by numpydoc !! -->

#### interpolate(\*args, \*\*kwargs) → None

Interpolate values in-place across all datasets.

* **Parameters:**
  * **\*args** – Arguments passed to `pandas.DataFrame.interpolate()`.
  * **\*\*kwargs** – Arguments passed to `pandas.DataFrame.interpolate()`.

### Notes

`inplace` is enforced as True.

#### SEE ALSO
`pandas.DataFrame.interpolate`
: The underlying pandas method.

<!-- !! processed by numpydoc !! -->

#### dropna(\*args, \*\*kwargs) → None

Remove missing values in-place across all datasets.

* **Parameters:**
  * **\*args** – Arguments passed to `pandas.DataFrame.dropna()`.
  * **\*\*kwargs** – Arguments passed to `pandas.DataFrame.dropna()`.

### Notes

`inplace` is enforced as True.

#### SEE ALSO
`pandas.DataFrame.dropna`
: The underlying pandas method.

<!-- !! processed by numpydoc !! -->

#### drop(\*args, \*\*kwargs) → None

Drop specified labels from rows or columns in-place across all datasets.

This method removes features (columns) or samples (rows) from every
underlying dataset in the collection.

* **Parameters:**
  * **\*args** – Arguments passed to `pandas.DataFrame.drop()`.
  * **\*\*kwargs** – Arguments passed to `pandas.DataFrame.drop()`.

### Notes

`inplace` is enforced as True.

#### SEE ALSO
`pandas.DataFrame.drop`
: The underlying pandas method.

### Examples

```pycon
>>> # Remove specific feature columns by name from all datasets
>>> concat_ds.drop(columns=['Alpha_Power', 'Beta_Power'])
```

```pycon
>>> # Remove the first and third window (rows) from every dataset
>>> concat_ds.drop(index=[0, 2])
```

<!-- !! processed by numpydoc !! -->

#### join(concat_dataset: eegdash.features.datasets.FeaturesConcatDataset, \*\*kwargs) → None

Join columns with another FeaturesConcatDataset in-place.

This method merges the feature columns of another dataset into the
current one. Both collections must contain the same number of
individual datasets, and corresponding datasets must have matching
lengths.

* **Parameters:**
  * **concat_dataset** (*FeaturesConcatDataset*) – The dataset containing the new columns to be joined.
  * **\*\*kwargs** – Keyword arguments passed to `pandas.DataFrame.join()`.
* **Raises:**
  **AssertionError** – If the number of datasets or the lengths of corresponding
  datasets do not match.

### Notes

This operation is performed in-place. The `ds.features` attribute
of each underlying dataset is updated with the new columns.

<!-- !! processed by numpydoc !! -->
