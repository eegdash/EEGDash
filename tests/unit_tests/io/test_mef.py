"""eegdash.io.mef: MEF3 sessions that would crash pymef are refused first.

on004624 sub-c04 Longterm Passive run-10: channel CH20's segment has only its
``.tmet``; pymef 1.4.8 segfaults on it (SIGSEGV killed the read-check job twice).
pymef is replaced by a stub here, so the test can never crash the test process.
"""

import mne
import pymef.mef_session
import pytest

from eegdash.io import get_reader, tolerant


class _Reached(Exception):
    pass


@pytest.fixture
def pymef_stub(monkeypatch):
    def stub(*_args, **_kwargs):
        raise _Reached("pymef opened the session")

    monkeypatch.setattr(pymef.mef_session, "MefSession", stub)


def _session(tmp_path, ch20_files=(".tmet", ".tidx", ".tdat")):
    root = tmp_path / "sub-c04_ses-Longterm_task-Passive_acq-rec_run-10_ieeg.mefd"
    for ch, files in (("CH0", (".tmet", ".tidx", ".tdat")), ("CH20", ch20_files)):
        seg = root / f"{ch}.timd" / f"{ch}-000000.segd"
        seg.mkdir(parents=True)
        for ext in files:
            (seg / f"{ch}-000000{ext}").write_bytes(b"x")
    return root


def test_segment_without_data_is_refused_before_pymef(tmp_path, pymef_stub):
    root = _session(tmp_path, ch20_files=(".tmet",))
    with (
        tolerant(),
        pytest.raises(ValueError, match="CH20-000000.segd lacks .tidx, .tdat"),
    ):
        mne.io.read_raw_mef(root)
    with pytest.raises(ValueError, match="Incomplete MEF3 session"):
        get_reader(".mefd")(root)


def test_complete_session_reaches_pymef(tmp_path, pymef_stub):
    with tolerant(), pytest.raises(_Reached):
        mne.io.read_raw_mef(_session(tmp_path))
