"""EEGDash tutorial audit infrastructure.

The tutorial audit package implements the validator pipeline described in
``docs/tutorial_implementation_strategy.md``. It provides a deterministic,
spec-driven evidence dossier for every tutorial in ``examples/tutorials/`` so
that compliance with the 49-rule rubric and the 12-dimension scorecard becomes
a machine-verifiable claim rather than an opinion.

Sub-packages:

* ``static``  -- cheap rule checks that operate on tutorial source files.
* ``runtime`` -- checks that require executing the tutorial via sphinx-gallery.
* ``reviewer``-- LLM/human-driven judgment on rules that resist automation.

The orchestrator entrypoint lives in :mod:`scripts.tutorial_audit.api`.
"""

from __future__ import annotations

__version__ = "0.1.0"
