# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Registry of named tasks for :func:`eegdash.tasks.get_task`.

Only the eyes-open-closed task ships with a concrete implementation today.
The plan (``docs/tutorial_restructure_plan.md`` Workstream 2) explicitly
prescribes hard-coding a few excellent task manifests before extracting
abstractions, so the other registry entries are deliberate stubs that raise
``NotImplementedError`` when instantiated. They name the tutorial that will
provide the implementation so contributors can find the right hook.

TODO -- deferred tutorials that should populate the stub entries:

* ``visual-p300``      -> ``tutorial_50_visual_p300.py``
* ``auditory-oddball`` -> ``tutorial_55_auditory_oddball.py``
* ``age-regression``   -> ``tutorial_60_age_regression.py``
* ``eeg2025-pfactor``  -> ``tutorial_70_eeg2025_pfactor.py``
"""

from __future__ import annotations

from typing import Callable

from .base import EEGTask
from .eoec import EyesOpenClosed


def _stub_factory(name: str, tutorial: str) -> Callable[..., EEGTask]:
    """Return a callable that raises ``NotImplementedError`` when invoked.

    The callable is registered in :data:`TASK_REGISTRY` for tasks that are
    listed in the plan but have not been implemented yet. The error message
    names the tutorial that is scheduled to provide the implementation so
    that contributors can find the right hook.
    """

    def _missing_task(*args: object, **kwargs: object) -> EEGTask:
        raise NotImplementedError(
            f"Task {name!r} is not implemented yet; scheduled for {tutorial}."
        )

    _missing_task.__name__ = f"_missing_{name.replace('-', '_')}"
    _missing_task.__doc__ = (
        f"Placeholder factory for the {name!r} task. Scheduled for {tutorial}."
    )
    return _missing_task


# ---------------------------------------------------------------------- #
# Registry                                                                #
# ---------------------------------------------------------------------- #
#
# Mapping of task name -> zero-argument factory that produces an ``EEGTask``
# instance (or raises ``NotImplementedError`` for stub entries). Concrete
# subclasses are listed first; deferred tasks follow with stub factories.

TASK_REGISTRY: dict[str, Callable[..., EEGTask]] = {
    "eyes-open-closed": EyesOpenClosed,
    "visual-p300": _stub_factory("visual-p300", "plot_50_visual_p300.py"),
    "auditory-oddball": _stub_factory(
        "auditory-oddball", "plot_55_auditory_oddball.py"
    ),
    "age-regression": _stub_factory("age-regression", "plot_60_age_regression.py"),
    "eeg2025-pfactor": _stub_factory("eeg2025-pfactor", "plot_70_eeg2025_pfactor.py"),
}


__all__ = ["TASK_REGISTRY"]
