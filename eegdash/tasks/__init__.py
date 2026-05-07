# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""High-level task entry points for EEGDash.

Public surface: :class:`EEGTask` (base class), :class:`EyesOpenClosed`
(concrete HBN resting-state task), :data:`TASK_REGISTRY`,
:func:`get_task`, and :func:`list_tasks`.
"""

from __future__ import annotations

from typing import Any

from .base import EEGTask
from .eoec import EyesOpenClosed
from .registry import TASK_REGISTRY


def get_task(name: str, **kwargs: Any) -> EEGTask:
    """Look up a task by name and instantiate it.

    Parameters
    ----------
    name : str
        Registry key. Use :func:`list_tasks` (or inspect
        :data:`registry.TASK_REGISTRY`) to enumerate the available names.
    **kwargs : Any
        Forwarded to the task constructor. Stub entries ignore the kwargs and
        raise :class:`NotImplementedError` instead.

    Returns
    -------
    EEGTask
        An :class:`EEGTask` instance. For stub entries this call raises
        :class:`NotImplementedError` with a message naming the deferred
        tutorial.

    """
    try:
        factory = TASK_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover -- exercised in tests
        available = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"Unknown task {name!r}. Available tasks: {available}.") from exc
    return factory(**kwargs)


def list_tasks() -> list[str]:
    """Return the sorted list of registered task names."""
    return sorted(TASK_REGISTRY)


__all__ = [
    "EEGTask",
    "EyesOpenClosed",
    "TASK_REGISTRY",
    "get_task",
    "list_tasks",
]
