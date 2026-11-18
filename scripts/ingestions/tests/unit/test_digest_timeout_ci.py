"""CI per-dataset digest watchdog must cover the slowest datasets.

On 2026-10-17 a fresh digest of the whole catalogue (Expanse, 4 workers) took
longer than 120 s for 7 datasets (nm000181, nm000281, nm000289, nm000359,
nm000453, on004395, on007358); nm000359 took over 900 s, and nm000281 takes
~600 s once its 2,785 BDF headers are read. Every workflow that calls the
reusable digest must give a dataset at least 1800 s.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_WORKFLOWS = Path(__file__).resolve().parents[4] / ".github" / "workflows"
_MIN_SECONDS = 1800


def _default_timeout() -> int:
    text = (_WORKFLOWS / "2-clone-digest.yml").read_text()
    block = text[text.index("      dataset_timeout:") :]
    return int(re.search(r"\n\s+default:\s*(\d+)", block).group(1))


def _callers() -> list[Path]:
    return sorted(
        p
        for p in _WORKFLOWS.glob("*.yml")
        if "uses: ./.github/workflows/2-clone-digest.yml" in p.read_text()
    )


def test_reusable_default_is_long_enough():
    assert _default_timeout() >= _MIN_SECONDS


@pytest.mark.parametrize("workflow", _callers(), ids=lambda p: p.name)
def test_every_caller_gets_enough_time(workflow):
    explicit = re.findall(r"dataset_timeout:\s*(\d+)", workflow.read_text())
    timeouts = [int(t) for t in explicit] or [_default_timeout()]
    assert min(timeouts) >= _MIN_SECONDS
