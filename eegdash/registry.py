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
        from .api import EEGDashDataset as base_class  # lazy import

    summary_path = Path(summary_file)
    namespace = namespace if namespace is not None else globals()
    module_name = namespace.get("__name__", __name__)
    registered: Dict[str, type] = {}

    df = pd.read_csv(summary_path, comment="#", skip_blank_lines=True)
    for _, row_series in df.iterrows():
        row = row_series.tolist()
        dataset_id = str(row[0]).strip()
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

        doc = f"""Create an instance for OpenNeuro dataset ``{dataset_id}``.

        {markdown_table(row_series)}

        Parameters
        ----------
        cache_dir : str
            Local cache directory.
        query : dict | None
            Extra Mongo query merged with ``{{'dataset': '{dataset_id}'}}``.
        s3_bucket : str | None
            Optional S3 bucket name.
        **kwargs
            Passed through to {base_class.__name__}.
        """

        init.__doc__ = doc

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


def markdown_table(row_series: pd.Series) -> str:
    """Create a reStructuredText grid table from a pandas Series."""
    if row_series.empty:
        return ""

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
            ]
        )
        .infer_objects(copy=False)
        .fillna("")
    )

    # Use tabulate for the final rst formatting
    table = tabulate(df, headers="keys", tablefmt="rst", showindex=False)

    # Indent the table to fit within the admonition block
    indented_table = "\n".join("    " + line for line in table.split("\n"))
    return f".. admonition:: Dataset Summary\n\n{indented_table}"
