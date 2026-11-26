# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Bring a dataset that is not in BIDS to a BIDS tree, one adapter per dataset.

A dataset each user obtains from its data holder under the holder's terms (a
gated archive that may not be redistributed, so not on NEMAR) becomes a BIDS
tree at ``<cache_dir>/<name>-<version>/``: the folder a local dataset is
read from (see :mod:`eegdash.corpus`), so ``files``, ``verify``, ``shards``,
``convert`` and ``merge`` take it over unchanged::

    from eegdash.to_bids import to_bids
    report = to_bids("tueg-2.0.1", "/data/tuh_eeg/v2.0.1", dry_run=True)
    report = to_bids("tueg-2.0.1", "/data/tuh_eeg/v2.0.1")

Command line::

    python -m eegdash.corpus bids tueg-2.0.1 --src DIR [--cache-dir C]
        [--link symlink|hard|copy] [--exclude CSV] [--keep-held-out] [--dry-run]
        [--n-jobs N] [--shard K --n-shards N] [--adapter module:Class]
        [--json REPORT.json]

Headers are read, never the signal; data files are linked (symbolic links by
default), never rewritten when the source format is a BIDS data file; nothing
is written under the source. Held-out recordings are left out and reported.
The adapters are in :data:`ADAPTERS`: the TUH EEG Corpus (``tueg``, from
NEDC). One outside the package is named ``module:Class`` (a subclass of
:class:`Adapter`).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Sequence

from .adapter import (
    Adapter,
    Channel,
    ChannelTable,
    Event,
    Exclusion,
    Header,
    Recording,
    read_exclusions,
)
from .core import LINK_MODES, BidsReport, run, split_id

# <name>: "module:Class", imported when used
ADAPTERS = {"tueg": "eegdash.to_bids.tueg:TUEG"}


def get_adapter(spec: str | type[Adapter]) -> type[Adapter]:
    """The adapter class of a dataset name (``tueg``), a ``module:Class`` path, or a class."""
    if isinstance(spec, type):
        cls = spec
    else:
        path = ADAPTERS.get(spec, spec)
        if ":" not in path:
            raise ValueError(f"no adapter for {spec!r}; known: {sorted(ADAPTERS)}")
        module, _, name = path.partition(":")
        cls = getattr(importlib.import_module(module), name)
    if not (isinstance(cls, type) and issubclass(cls, Adapter)):
        raise ValueError(f"{spec!r} is not an eegdash.to_bids.Adapter")
    return cls


def to_bids(
    dataset_id: str,
    src: str | Path,
    *,
    cache_dir: str | Path | None = None,
    link: str = "symlink",
    exclude: str | Path | Sequence[Exclusion] | None = None,
    keep_held_out: bool = False,
    dry_run: bool = False,
    n_jobs: int = 1,
    shard: int | None = None,
    n_shards: int | None = None,
    adapter: str | type[Adapter] | None = None,
) -> BidsReport:
    """Write the BIDS tree of *dataset_id* at ``<cache_dir>/<dataset_id>/`` from the source *src*.

    Parameters
    ----------
    dataset_id : str
        ``<name>-<version>`` (``tueg-2.0.1``); the adapter is the one
        registered for the name, and must know the version.
    src : str | Path
        The source tree; never written to, and the output may not be in it.
    cache_dir : str | Path | None
        The EEGDash cache (see :func:`eegdash.paths.get_default_cache_dir`).
    link : {"symlink", "hard", "copy"}
        How a data file enters the tree. A symbolic link (absolute) shows a
        moved source at once in ``verify``; a hard link or a copy makes a tree
        that stands alone.
    exclude : str | Path | sequence of Exclusion | None
        More recordings or subjects to leave out: a CSV ``key,match,reason``
        (``match`` is ``recording``, matched against a recording's key, the
        source's stem by default, or ``subject``).
    keep_held_out : bool
        Do not apply the adapter's held-out table (recorded in the report).
    dry_run : bool
        Plan and read every header; write nothing.
    n_jobs : int
        Processes, over subjects. They are spawned: a script that asks for
        more than one calls :func:`to_bids` under
        ``if __name__ == "__main__":``.
    shard, n_shards : int | None
        This job-array task's subjects: subjects sorted, every *n_shards*-th
        from *shard*. The root files and the report are written by a final
        call without a shard, which reuses what the tasks wrote.
    adapter : str | type[Adapter] | None
        Another adapter for the name: ``module:Class`` or a class.

    Returns
    -------
    BidsReport
        One row per recording; ``ok`` when nothing was unreadable, invalid or
        refused.

    """
    from ..paths import get_default_cache_dir  # noqa: PLC0415 (imports mne)

    name, _ = split_id(dataset_id)
    cls = get_adapter(adapter or name)
    out = Path(cache_dir or get_default_cache_dir()) / dataset_id
    return run(
        cls,
        dataset_id,
        src,
        out,
        link=link,
        exclude=exclude,
        keep_held_out=keep_held_out,
        dry_run=dry_run,
        n_jobs=n_jobs,
        shard=shard,
        n_shards=n_shards,
    )


__all__ = [
    "ADAPTERS",
    "LINK_MODES",
    "Adapter",
    "BidsReport",
    "Channel",
    "ChannelTable",
    "Event",
    "Exclusion",
    "Header",
    "Recording",
    "get_adapter",
    "read_exclusions",
    "to_bids",
]
