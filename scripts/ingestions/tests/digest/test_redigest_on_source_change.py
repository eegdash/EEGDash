"""A digest is redone when its source moved to another commit (stale paths after a re-curation)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from _helpers import load_digest

from eegdash.testing import data_file

DS = "ds_snapshot_vhdr"


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(repo),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )
    return out.stdout.strip()


@pytest.fixture
def cloned(tmp_path: Path) -> Path:
    src = data_file(f"digest_snapshots/inputs/{DS}")
    if not src.exists() or shutil.which("git") is None:
        pytest.skip("snapshot fixture or git not available")
    inputs = tmp_path / "inputs"
    shutil.copytree(src, inputs / DS)
    repo = inputs / DS
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v1")
    return inputs


def _summary(out: Path) -> dict:
    return json.loads((out / DS / f"{DS}_summary.json").read_text())


def test_summary_records_source_commit(cloned: Path, tmp_path: Path):
    digest = load_digest()
    out = tmp_path / "out"
    summary = digest.digest_dataset(DS, cloned, out)
    assert summary["status"] == "success", summary
    head = _git(cloned / DS, "rev-parse", "HEAD")
    assert _summary(out)["source_commit"] == head


def test_same_commit_is_skipped(cloned: Path, tmp_path: Path):
    digest = load_digest()
    out = tmp_path / "out"
    digest.digest_dataset(DS, cloned, out)
    again = digest.digest_dataset(DS, cloned, out)
    assert again == {
        "status": "skipped",
        "dataset_id": DS,
        "reason": "already digested",
    }


def test_new_commit_is_redigested(cloned: Path, tmp_path: Path):
    digest = load_digest()
    out = tmp_path / "out"
    digest.digest_dataset(DS, cloned, out)
    old = _summary(out)["source_commit"]

    repo = cloned / DS
    (repo / "CHANGES").write_text("1.0.1 renamed runs\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v2")

    again = digest.digest_dataset(DS, cloned, out)
    assert again["status"] == "success", again
    new = _summary(out)["source_commit"]
    assert new != old
    assert new == _git(repo, "rev-parse", "HEAD")


def test_sidecar_only_change_changes_the_fingerprint(cloned: Path, tmp_path: Path):
    """Same recordings, new sidecar: inject must see a different fingerprint."""
    digest = load_digest()
    out = tmp_path / "out"
    digest.digest_dataset(DS, cloned, out)
    fp = lambda: json.loads((out / DS / f"{DS}_dataset.json").read_text())[  # noqa: E731
        "ingestion_fingerprint"
    ]
    before = fp()
    repo = cloned / DS
    (repo / "task-x_eeg.json").write_text('{"RecordingDuration": 1}')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "sidecar")
    digest.digest_dataset(DS, cloned, out)
    assert fp() != before


def test_digest_without_recorded_commit_is_kept(cloned: Path, tmp_path: Path):
    """Digests made before source_commit existed are not redone wholesale."""
    digest = load_digest()
    out = tmp_path / "out"
    digest.digest_dataset(DS, cloned, out)
    summary_path = out / DS / f"{DS}_summary.json"
    legacy = _summary(out)
    legacy.pop("source_commit")
    summary_path.write_text(json.dumps(legacy))

    repo = cloned / DS
    (repo / "CHANGES").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v2")

    again = digest.digest_dataset(DS, cloned, out)
    assert again["status"] == "skipped"
    assert "source_commit" not in _summary(out)
