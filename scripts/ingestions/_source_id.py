"""Source-inference Seam: map an ingestion dataset id to its repository source.

Tiny, pure helpers extracted from ``3_digest.py`` so that both the orchestrator
and the manifest digest (``_manifest_digest.py``) can reconcile a dataset's source
without an import cycle. Depends only on the standard library.
"""

from __future__ import annotations

import sys

__all__ = ["_reconcile_source", "_source_from_dataset_id"]


def _source_from_dataset_id(dataset_id: str) -> str:
    """Infer source from dataset_id prefix pattern (ds* → openneuro, nm* → nemar)."""
    if dataset_id.startswith("ds") and dataset_id[2:].isdigit():
        return "openneuro"
    if dataset_id.startswith("nm") and dataset_id[2:].isdigit():
        return "nemar"
    if "EEGManyLabs" in dataset_id:
        return "gin"
    if dataset_id.startswith("EEG2025"):
        return "nemar"
    return "unknown"


def _reconcile_source(
    manifest_src: str | None, dataset_id: str, *, context: str
) -> str:
    """Trust dataset_id pattern over manifest source to prevent S3 bucket misrouting."""
    pattern_src = _source_from_dataset_id(dataset_id)
    if (
        manifest_src
        and pattern_src not in (None, "unknown")
        and manifest_src != pattern_src
    ):
        print(
            f"WARNING [{context}]: {dataset_id} manifest source={manifest_src!r} "
            f"disagrees with id-pattern source={pattern_src!r}; using pattern.",
            file=sys.stderr,
        )
        return pattern_src
    if manifest_src:
        return manifest_src
    return pattern_src
