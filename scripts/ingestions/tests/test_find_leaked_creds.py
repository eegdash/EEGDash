"""Tests for the find_leaked_creds.sh scanner (Task 5)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
SCANNER = _INGEST_DIR / "scripts" / "find_leaked_creds.sh"


# Synthesised token used ONLY in a temp git repo for scanner verification.
# DO NOT CONCATENATE THESE STRINGS — split intentionally so this test
# fixture file itself doesn't trip the find-leaked-creds scanner under
# pre-commit. See find_leaked_creds.sh PATTERNS.
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


def test_scanner_detects_yaml_form_token(tmp_path):
    """The pattern set must also catch ``KEY: value`` (YAML / docker-compose).

    Real-world driver: the cluster's ``docker-compose.yml`` writes
    credentials in YAML form, not env-shell form. A scanner that only
    matches ``=`` would miss them.
    """
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    f = tmp_path / "compose-like.yml"
    # Use split literal so this test file itself doesn't trip the scanner
    # under pre-commit (see also the existing _TOKEN_VAR pattern).
    yaml_secret = "ADMIN_TOKEN" + ": AdminWrite2025" + "SecureTokenABC123"
    f.write_text(f"environment:\n  {yaml_secret}\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "compose-like.yml"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "add config"],
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
    assert "ADMIN_TOKEN" in result.stdout
    assert result.returncode == 1


def test_scanner_outside_git_repo_exits_2(tmp_path):
    """In a non-git directory, scanner must refuse to claim clean."""
    import subprocess

    # tmp_path is NOT a git repo. Confirm:
    result = subprocess.run(
        ["bash", str(SCANNER)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Not in a git repo" in result.stderr
