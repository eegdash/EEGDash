# eegdash.dataset.registry module

<!-- !! processed by numpydoc !! -->

### eegdash.dataset.registry.fetch_chart_data_from_api(api_url: str = 'https://data.eegdash.org/api', database: str = 'eegdash', limit: int = 1000) → tuple[DataFrame, dict[str, Any]]

Fetch pre-aggregated chart data from API.

This uses the optimized /datasets/chart-data endpoint which returns
only chart-relevant fields and pre-computed aggregations.

Falls back to /datasets/summary if chart-data endpoint is unavailable.

* **Parameters:**
  * **api_url** (*str*) – Base API URL
  * **database** (*str*) – Database name
  * **limit** (*int*) – Maximum datasets to fetch
* **Returns:**
  DataFrame with dataset records and dict with pre-computed aggregations
* **Return type:**
  tuple[pd.DataFrame, dict]

<!-- !! processed by numpydoc !! -->

### eegdash.dataset.registry.fetch_datasets_from_api(api_url: str = 'https://data.eegdash.org/api', database: str = 'eegdash', force_refresh: bool = False) → DataFrame

Fetch dataset summaries from API and return as DataFrame matching CSV structure.

Note: This function makes a single API call to /datasets/summary.
Stats (nchans_counts, sfreq_counts) are already embedded in dataset documents
via the compute-stats endpoint, so no separate stats call is needed.

* **Parameters:**
  * **api_url** (*str*) – Base API URL.
  * **database** (*str*) – Database name.
  * **force_refresh** (*bool*) – If True, bypass the local cache and always fetch from the API.

<!-- !! processed by numpydoc !! -->

### eegdash.dataset.registry.register_openneuro_datasets(summary_file: str | Path | None = None, , base_class=None, namespace: Dict[str, Any] | None = None, add_to_all: bool = True, from_api: bool = False, api_url: str = 'https://data.eegdash.org/api', database: str = 'eegdash') → Dict[str, type]

Dynamically create and register dataset classes from a summary file or API.

This function reads a CSV file or queries the API containing summaries of
datasets and dynamically creates a Python class for each dataset. These
classes inherit from a specified base class and are pre-configured with the
dataset’s ID.

* **Parameters:**
  * **summary_file** (*str* *or* *pathlib.Path*) – The path to the CSV file containing the dataset summaries.
  * **base_class** (*type* *,* *optional*) – The base class from which the new dataset classes will inherit. If not
    provided, `eegdash.api.EEGDashDataset` is used.
  * **namespace** (*dict* *,* *optional*) – The namespace (e.g., globals()) into which the newly created classes
    will be injected. Defaults to the local globals() of this module.
  * **add_to_all** (*bool* *,* *default True*) – If True, the names of the newly created classes are added to the
    \_\_all_\_ list of the target namespace, making them importable with
    from … import \*.
* **Returns:**
  A dictionary mapping the names of the registered classes to the class
  types themselves.
* **Return type:**
  dict[str, type]

<!-- !! processed by numpydoc !! -->
