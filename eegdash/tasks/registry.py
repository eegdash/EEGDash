# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Registry of named tasks for :func:`eegdash.tasks.get_task`."""

from __future__ import annotations

from typing import Callable

from .base import EEGTask
from .eoec import EyesOpenClosed

# Mapping of task name -> factory that produces an :class:`EEGTask` instance.
TASK_REGISTRY: dict[str, Callable[..., EEGTask]] = {
    "eyes-open-closed": EyesOpenClosed,
}


__all__ = ["TASK_REGISTRY"]
