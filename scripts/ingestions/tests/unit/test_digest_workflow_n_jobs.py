"""The NEMAR digest must read record headers in parallel (nm000281, 25,253 BDF).

Nightly 2026-10-25: nm000281 timed out at the 1800 s per-dataset guard after
9,680 of 25,253 header reads (one at a time, ~5.4 files/s). On a GitHub-hosted
runner the existing ``--n-jobs`` scales linearly (request latency flat at
~0.09 s): full read-only digest of nm000281 took 1,220 s at n_jobs 4 and 614 s
at n_jobs 8, identical records, strict validation clean. The reusable workflow
never passed ``--n-jobs``; it now takes an ``n_jobs`` input (default 1, so other
sources are unchanged) and both NEMAR callers set 8.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_WF = Path(__file__).resolve().parents[4] / ".github" / "workflows"


def _load(name):
    return yaml.safe_load((_WF / name).read_text())


def test_reusable_digest_passes_n_jobs_with_default_one():
    wf = _load("2-clone-digest.yml")
    inputs = wf[True]["workflow_call"]["inputs"]  # PyYAML reads `on:` as True
    assert inputs["n_jobs"]["default"] == 1
    steps = wf["jobs"]["digest"]["steps"]
    digest = next(s for s in steps if s.get("name") == "Digest datasets")
    assert "--n-jobs ${{ inputs.n_jobs }}" in digest["run"]


def test_nemar_callers_digest_in_parallel():
    nightly = _load("full-pipeline.yml")["jobs"]["digest-nemar"]["with"]
    manual = _load("2-digest-nemar.yml")["jobs"]["digest"]["with"]
    assert nightly["n_jobs"] == 8
    assert manual["n_jobs"] == 8


def test_other_sources_keep_serial_default():
    for name in ("2-digest-openneuro.yml", "2-digest-osf.yml", "2-digest-zenodo.yml"):
        assert "n_jobs" not in _load(name)["jobs"]["digest"]["with"]
