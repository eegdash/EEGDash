# eegdash.dataset.registry module

<!-- !! processed by numpydoc !! -->

### eegdash.dataset.registry.fetch_chart_data_from_api(api_url: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'https://data.eegdash.org/api', database: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'eegdash', limit: [int](https://docs.python.org/3/library/functions.html#int) = 1000) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame), [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

Fetch pre-aggregated chart data from API.

This uses the optimized /datasets/chart-data endpoint which returns
only chart-relevant fields and pre-computed aggregations.

Falls back to /datasets/summary if chart-data endpoint is unavailable.

* **Parameters:**
  * **api_url** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Base API URL
  * **database** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Database name
  * **limit** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Maximum datasets to fetch
* **Returns:**
  DataFrame with dataset records and dict with pre-computed aggregations
* **Return type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[pd.DataFrame, [dict](https://docs.python.org/3/library/stdtypes.html#dict)]

<!-- !! processed by numpydoc !! -->

### eegdash.dataset.registry.fetch_datasets_from_api(api_url: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'https://data.eegdash.org/api', database: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'eegdash', force_refresh: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame)

Fetch dataset summaries from API and return as DataFrame matching CSV structure.

Note: This function makes a single API call to /datasets/summary.
Stats (nchans_counts, sfreq_counts) are already embedded in dataset documents
via the compute-stats endpoint, so no separate stats call is needed.

* **Parameters:**
  * **api_url** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Base API URL.
  * **database** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Database name.
  * **force_refresh** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, bypass the local cache and always fetch from the API.

<!-- !! processed by numpydoc !! -->

### eegdash.dataset.registry.register_openneuro_datasets(summary_file: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [None](https://docs.python.org/3/library/constants.html#None) = None, , base_class=None, namespace: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, add_to_all: [bool](https://docs.python.org/3/library/functions.html#bool) = True, from_api: [bool](https://docs.python.org/3/library/functions.html#bool) = False, api_url: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'https://data.eegdash.org/api', database: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'eegdash') → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)]

Dynamically create and register dataset classes from a summary file or API.

This function reads a CSV file or queries the API containing summaries of
datasets and dynamically creates a Python class for each dataset. These
classes inherit from a specified base class and are pre-configured with the
dataset’s ID.

* **Parameters:**
  * **summary_file** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*pathlib.Path*](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – The path to the CSV file containing the dataset summaries.
  * **base_class** ([*type*](https://docs.python.org/3/library/functions.html#type) *,* *optional*) – The base class from which the new dataset classes will inherit. If not
    provided, `eegdash.api.EEGDashDataset` is used.
  * **namespace** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *,* *optional*) – The namespace (e.g., globals()) into which the newly created classes
    will be injected. Defaults to the local globals() of this module.
  * **add_to_all** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default True*) – If True, the names of the newly created classes are added to the
    \_\_all_\_ list of the target namespace, making them importable with
    from … import \*.
* **Returns:**
  A dictionary mapping the names of the registered classes to the class
  types themselves.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)]

<!-- !! processed by numpydoc !! -->
