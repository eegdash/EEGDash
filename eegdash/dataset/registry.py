from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tabulate import tabulate


def register_openneuro_datasets(
    summary_file: str | Path,
    *,
    base_class=None,
    namespace: Dict[str, Any] | None = None,
    add_to_all: bool = True,
) -> Dict[str, type]:
    """Dynamically create dataset classes from a summary file."""
    if base_class is None:
        from ..api import EEGDashDataset as base_class  # lazy import

    summary_path = Path(summary_file)
    namespace = namespace if namespace is not None else globals()
    module_name = namespace.get("__name__", __name__)
    registered: Dict[str, type] = {}

    df = pd.read_csv(summary_path, comment="#", skip_blank_lines=True)
    for _, row_series in df.iterrows():
        # Use the explicit 'dataset' column, not the CSV index.
        dataset_id = str(row_series.get("dataset", "")).strip()
        if not dataset_id:
            continue

        class_name = dataset_id.upper()

        # avoid zero-arg super() here
        def make_init(_dataset: str):
            def __init__(
                self,
                cache_dir: str,
                query: dict | None = None,
                s3_bucket: str | None = None,
                **kwargs,
            ):
                q = {"dataset": _dataset}
                if query:
                    q.update(query)
                # call base_class.__init__ directly
                base_class.__init__(
                    self,
                    query=q,
                    cache_dir=cache_dir,
                    s3_bucket=s3_bucket,
                    **kwargs,
                )

            return __init__

        init = make_init(dataset_id)

        # Generate rich docstring with dataset metadata
        doc = _generate_rich_docstring(dataset_id, row_series, base_class)

        # init.__doc__ = doc

        cls = type(
            class_name,
            (base_class,),
            {
                "_dataset": dataset_id,
                "__init__": init,
                "__doc__": doc,
                "__module__": module_name,  #
            },
        )

        namespace[class_name] = cls
        registered[class_name] = cls

        if add_to_all:
            ns_all = namespace.setdefault("__all__", [])
            if isinstance(ns_all, list) and class_name not in ns_all:
                ns_all.append(class_name)

    return registered


def _generate_rich_docstring(dataset_id: str, row_series: pd.Series, base_class) -> str:
    """Generate a comprehensive docstring for a dataset class."""
    # Extract metadata with safe defaults
    n_subjects = row_series.get("n_subjects", "Unknown")
    n_records = row_series.get("n_records", "Unknown")
    n_tasks = row_series.get("n_tasks", "Unknown")
    modality = row_series.get("modality of exp", "")
    exp_type = row_series.get("type of exp", "")
    subject_type = row_series.get("Type Subject", "")
    duration = row_series.get("duration_hours_total", "Unknown")
    size = row_series.get("size", "Unknown")

    # Create description based on available metadata
    description_parts = []
    if modality and str(modality).strip():
        description_parts.append(f"**Modality**: {modality}")
    if exp_type and str(exp_type).strip():
        description_parts.append(f"**Type**: {exp_type}")
    if subject_type and str(subject_type).strip():
        description_parts.append(f"**Subjects**: {subject_type}")

    description = (
        " | ".join(description_parts)
        if description_parts
        else "EEG dataset from OpenNeuro"
    )

    # Generate the docstring
    docstring = f"""OpenNeuro dataset ``{dataset_id}``.

{description}

This dataset contains {n_subjects} subjects with {n_records} recordings across {n_tasks} tasks.
Total duration: {duration} hours. Dataset size: {size}.

{_markdown_table(row_series)}

This dataset class provides convenient access to the ``{dataset_id}`` dataset through the EEGDash interface.
It inherits all functionality from :class:`~{base_class.__module__}.{base_class.__name__}` with the dataset filter pre-configured.

Parameters
----------
cache_dir : str
    Directory to cache downloaded data.
query : dict, optional
    Additional MongoDB-style filters to AND with the dataset selection.
    Must not contain the key ``dataset``.
s3_bucket : str, optional
    Base S3 bucket used to locate the data.
**kwargs
    Additional arguments passed to the base dataset class.

Examples
--------
Basic usage:

>>> from eegdash.dataset import {dataset_id.upper()}
>>> dataset = {dataset_id.upper()}(cache_dir="./data")
>>> print(f"Number of recordings: {{len(dataset)}}")

Load a specific recording:

>>> if len(dataset) > 0:
...     recording = dataset[0]
...     raw = recording.load()
...     print(f"Sampling rate: {{raw.info['sfreq']}} Hz")
...     print(f"Number of channels: {{len(raw.ch_names)}}")

Filter by additional criteria:

>>> # Get subset with specific task or subject
>>> filtered_dataset = {dataset_id.upper()}(
...     cache_dir="./data",
...     query={{"task": "RestingState"}}  # if applicable
... )

Notes
-----
More details available in the `NEMAR documentation <https://nemar.org/dataexplorer/detail?dataset_id={dataset_id}>`__.

See Also
--------
{base_class.__name__} : Base dataset class with full API documentation
"""

    return docstring


def _markdown_table(row_series: pd.Series) -> str:
    """Create a reStructuredText grid table from a pandas Series."""
    if row_series.empty:
        return ""
    dataset_id = row_series["dataset"]

    # Prepare the dataframe with user's suggested logic
    df = (
        row_series.to_frame()
        .T.rename(
            columns={
                "n_subjects": "#Subj",
                "nchans_set": "#Chan",
                "n_tasks": "#Classes",
                "sampling_freqs": "Freq(Hz)",
                "duration_hours_total": "Duration(H)",
                "size": "Size",
            }
        )
        .reindex(
            columns=[
                "dataset",
                "#Subj",
                "#Chan",
                "#Classes",
                "Freq(Hz)",
                "Duration(H)",
                "Size",
            ]
        )
        .infer_objects(copy=False)
        .fillna("")
    )

    # Use tabulate for the final rst formatting
    table = tabulate(df, headers="keys", tablefmt="rst", showindex=False)

    # Add a caption for the table
    # Use an anonymous external link (double underscore) to avoid duplicate
    # target warnings when this docstring is repeated across many classes.
    caption = (
        f"Short overview of dataset {dataset_id} more details in the "
        f"`NeMAR documentation <https://nemar.org/dataexplorer/detail?dataset_id={dataset_id}>`__."
    )
    # adding caption below the table
    # Indent the table to fit within the admonition block
    indented_table = "\n".join("    " + line for line in table.split("\n"))
    return f"\n\n{indented_table}\n\n{caption}"
