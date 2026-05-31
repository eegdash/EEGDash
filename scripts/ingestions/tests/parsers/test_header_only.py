"""Guard: cheap parser paths produce n_times on real fixtures.

The strict byte-budget assertions live in the perf phase; this file is the
always-on smoke guard that the cheap n_times paths actually fire on the real
corpus fixtures. Marked ``network`` since it uses the testing-data corpus.
"""

from __future__ import annotations

import pytest

from eegdash.testing import data_file


@pytest.mark.network
def test_vhdr_parser_produces_n_times():
    from _vhdr_parser import parse_vhdr_metadata  # noqa: PLC0415

    meta = parse_vhdr_metadata(data_file("eeg/sub-xp101_task-motorloc_eeg.vhdr"))
    assert meta is not None
    # The .eeg companion is present in the corpus -> file-size arithmetic applies.
    assert meta.get("n_times", 0) > 0


@pytest.mark.network
def test_snirf_parser_produces_n_times():
    from _snirf_parser import parse_snirf_metadata  # noqa: PLC0415

    meta = parse_snirf_metadata(data_file("fnirs/openneuro_real.snirf"))
    assert meta is not None
    assert meta.get("n_times", 0) > 0
