from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict


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

    with summary_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            dataset_id = row[0].strip()
            if not dataset_id or dataset_id.startswith("#"):
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
            init.__doc__ = f"""Create an instance for OpenNeuro dataset ``{dataset_id}``.

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

            cls = type(
                class_name,
                (base_class,),
                {
                    "_dataset": dataset_id,
                    "__init__": init,
                    "__doc__": f"Dataset class for ``{dataset_id}``.",
                    "__module__": module_name,  # correct module for docs/import path
                },
            )

            namespace[class_name] = cls
            registered[class_name] = cls

            if add_to_all:
                ns_all = namespace.setdefault("__all__", [])
                if isinstance(ns_all, list) and class_name not in ns_all:
                    ns_all.append(class_name)

    return registered
