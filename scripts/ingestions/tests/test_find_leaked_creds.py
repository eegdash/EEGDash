"""Tests for the find_leaked_creds.sh scanner (Task 5)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
SCANNER = _INGEST_DIR / "scripts" / "find_leaked_creds.sh"


# Synthesised token used ONLY in a temp git repo for scanner verification.
# Split into fragments so this source file itself does not trip the
# scanner's EEGDASH_ADMIN_TOKEN / ADMIN_TOKEN regex.
_TOKEN_VAR = "EEGDASH_" + "ADMIN_TOKEN"
_TOKEN_VAL = "AdminWrite2025" + "SecureTokenABC123"  # 30 alnum chars total
_PLANTED_SECRET = f"{_TOKEN_VAR}={_TOKEN_VAL}"


@pytest.fixture
def fake_repo(tmp_path):
    """Create a tiny git repo with a planted secret in a commit message."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    f = tmp_path / "f.txt"
    f.write_text("hello\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "f.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "commit",
            "-q",
            "-m",
            f"test commit\n\n{_PLANTED_SECRET}",
        ],
        check=True,
        env={
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "t@t.t",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "t@t.t",
            "PATH": "/usr/bin:/bin",
        },
    )
    return tmp_path


def test_scanner_detects_token_in_commit_message(fake_repo):
    result = subprocess.run(
        ["bash", str(SCANNER)],
        cwd=fake_repo,
        capture_output=True,
        text=True,
    )
    assert "EEGDASH_ADMIN_TOKEN" in result.stdout
    assert result.returncode == 1  # found leaks -> exit 1


def test_scanner_clean_repo_exits_0(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "x.txt").write_text("clean\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "x.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "harmless"],
        check=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t.t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t.t",
            "PATH": "/usr/bin:/bin",
        },
    )

    result = subprocess.run(
        ["bash", str(SCANNER)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
