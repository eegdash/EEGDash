"""Workflow inputs must reach shell scripts through env, never ``${{ }}``.

GitHub expands ``${{ inputs.* }}`` into the script text before bash parses it,
so a crafted input value runs as shell code (security finding F8). Every step
must pass inputs through ``env:`` and reference them as ``"$NAME"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_WORKFLOWS = Path(__file__).resolve().parents[4] / ".github" / "workflows"
_FORBIDDEN = ("${{ inputs.", "${{ github.event.inputs.")


def _workflows() -> list[Path]:
    return sorted([*_WORKFLOWS.glob("*.yml"), *_WORKFLOWS.glob("*.yaml")])


def _interpolating_steps(workflow: Path) -> list[str]:
    jobs = yaml.safe_load(workflow.read_text()).get("jobs") or {}
    return [
        f"{job_id}: {step.get('name', '<unnamed>')}"
        for job_id, job in jobs.items()
        for step in job.get("steps") or []
        if any(token in str(step.get("run", "")) for token in _FORBIDDEN)
    ]


def test_workflows_found():
    assert _workflows()


@pytest.mark.parametrize("workflow", _workflows(), ids=lambda p: p.name)
def test_no_input_interpolation_in_run(workflow):
    assert _interpolating_steps(workflow) == []
