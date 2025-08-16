from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict


def register_openneuro_datasets(
    summary_file: str | Path,
    *,
    base_class=None,
    namespace: Dict[str, Any] | None = None,
) -> Dict[str, type]:
    """Dynamically create dataset classes from a summary file.

    Parameters
    ----------
    summary_file : str | Path
        Path to a CSV file where each line starts with the dataset identifier.
    base_class : type | None
        Base class for the generated datasets. If ``None``, defaults to
        :class:`eegdash.api.EEGDashDataset`.
    namespace : dict | None
        Mapping where the new classes will be registered. Defaults to the
        module's global namespace.

    Returns
    -------
    dict
        Mapping from class names to the generated classes.

    """
    if base_class is None:
        from .api import EEGDashDataset as base_class  # lazy import

    summary_path = Path(summary_file)
    namespace = namespace if namespace is not None else globals()
    registered: Dict[str, type] = {}

    with summary_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            dataset_id = row[0].strip()
            if not dataset_id or dataset_id.startswith("#"):
                continue
            class_name = dataset_id.upper()

            def __init__(
                self,
                cache_dir: str,
                query: dict | None = None,
                s3_bucket: str | None = None,
                **kwargs,
            ):
                q = {"dataset": self._dataset}
                if query:
                    q.update(query)
                super().__init__(
                    query=q, cache_dir=cache_dir, s3_bucket=s3_bucket, **kwargs
                )

            cls = type(
                class_name,
                (base_class,),
                {"_dataset": dataset_id, "__init__": __init__},
            )
            namespace[class_name] = cls
            registered[class_name] = cls

    return registered
