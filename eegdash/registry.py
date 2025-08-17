from __future__ import annotations

import csv
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterable, Mapping


def register_openneuro_datasets(
    summary_file: str | Path,
    *,
    base_class=None,
    namespace: Dict[str, Any] | None = None,
    add_to_all: bool = True,
) -> Dict[str, type]:
    """Dynamically create dataset classes from a summary file.

    The CSV can be either:
      1) a single column of dataset IDs
      2) a headered CSV with columns like: dataset_id,title,url,description,tasks,n_subjects
         (all optional except dataset_id)
    """
    if base_class is None:
        from .api import EEGDashDataset as base_class

    summary_path = Path(summary_file)
    namespace = namespace if namespace is not None else globals()
    module_name = namespace.get("__name__", __name__)
    registered: Dict[str, type] = {}

    # --- read rows (with or without header) -----------------------------------
    def _rows() -> Iterable[Mapping[str, str]]:
        with summary_path.open(newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            has_header = False
            try:
                has_header = csv.Sniffer().has_header(sample)
            except csv.Error:
                has_header = False

            if has_header:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row
            else:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    yield {"dataset_id": row[0]}

    # --- helpers --------------------------------------------------------------
    def _build_doc(class_name: str, meta: Mapping[str, str]) -> str:
        ds = meta.get("dataset_id") or meta.get("dataset") or class_name
        title = meta.get("title") or ds
        url = meta.get("url") or meta.get("homepage") or ""
        desc = meta.get("description", "").strip()
        tasks = meta.get("tasks") or meta.get("task") or ""
        n_subj = meta.get("n_subjects") or ""
        # Compose a friendly docstring
        parts = [
            f"{title}",
            "",
            f"OpenNeuro dataset identifier: ``{ds}``.",
        ]
        if desc:
            parts += ["", desc]
        if tasks or n_subj or url:
            bullets = []
            if tasks:
                bullets.append(f"- **Tasks:** {tasks}")
            if n_subj:
                bullets.append(f"- **Subjects:** {n_subj}")
            if url:
                bullets.append(f"- **URL:** {url}")
            parts += ["", *bullets]
        parts += [
            "",
            "Examples",
            "--------",
            ".. code-block:: python",
            "",
            f"   from {module_name} import {class_name}",
            f"   ds = {class_name}(cache_dir='/path/to/cache')",
            f"   # ds behaves like {base_class.__name__}",
        ]
        return dedent("\n".join(parts)).strip() + "\n"

    # --- create classes -------------------------------------------------------
    for meta in _rows():
        dataset_id = (meta.get("dataset_id") or "").strip()
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
            super(self.__class__, self).__init__(
                query=q, cache_dir=cache_dir, s3_bucket=s3_bucket, **kwargs
            )

        cls_dict = {
            "_dataset": dataset_id,
            "__init__": __init__,
            "__module__": module_name,  # important for Sphinx
            "__doc__": _build_doc(class_name, meta),  # register the docstring
        }
        cls = type(class_name, (base_class,), cls_dict)

        namespace[class_name] = cls
        registered[class_name] = cls

        if add_to_all:
            if "__all__" not in namespace or not isinstance(namespace["__all__"], list):
                namespace["__all__"] = []
            if class_name not in namespace["__all__"]:
                namespace["__all__"].append(class_name)

    return registered
