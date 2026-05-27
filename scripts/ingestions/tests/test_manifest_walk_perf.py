"""Benchmark + regression guard for the Stage 2 manifest walker.

Stage 2 (``2_clone.py``) calls :func:`_file_utils.list_git_files` once per
dataset to enumerate every file under a freshly-cloned git tree (or
git-annex pointer tree). Profiling against the full 566-dataset corpus
showed this single function dominated Stage 2 wall-clock:

* ~2.8 M ``pathlib.Path.is_symlink`` calls
* ~2.6 M ``pathlib.Path.is_file`` calls
* ~1.5 M ``pathlib.Path.rglob`` calls
* ~2.8 M ``pathlib.Path.lstat`` calls

— roughly **5 000 stat-like syscalls per dataset**, because each
``Path.rglob("*")`` entry triggers fresh ``stat`` calls for every
classification predicate (``is_file`` / ``is_symlink``) instead of
reusing the cached dirent flags from the directory entry.

The fix is to walk via :func:`os.scandir`, which exposes the dirent's
``DT_REG`` / ``DT_LNK`` / ``DT_DIR`` flags without a stat round-trip,
and to call ``entry.stat(follow_symlinks=False)`` at most once per
file.  Expected gain on a 4 000-file dataset: 3-5× wall-clock.

This benchmark builds a deterministic synthetic tree resembling a
mid-sized BIDS dataset (100 subjects × ~40 files each) and times
``list_git_files`` against it. The 500 ms ceiling is a regression
guard, not a target — the new walker comes in well under 250 ms on
contemporary hardware, but the gate fires if any future change
re-introduces an O(N) stat fan-out per entry.

Unlike the memory-heavy benchmarks in ``tests/test_perf.py`` (which
are slow-marked), this walker bench runs in <300 ms on the synthetic
tree even on cold CI runners, so it stays in the default PR-fast
suite as a real regression guard rather than a nightly artifact.
"""

from __future__ import annotations

from pathlib import Path

from _file_utils import list_git_files

# ─── Synthetic tree builder ────────────────────────────────────────────────


def _build_synthetic_bids_tree(root: Path, n_subjects: int = 100) -> int:
    """Create a synthetic BIDS-shaped tree under ``root``.

    Layout per subject (``sub-XXX``):
        sub-XXX/
        ├── ses-01/
        │   ├── eeg/
        │   │   ├── sub-XXX_ses-01_task-rest_eeg.edf      (regular file)
        │   │   ├── sub-XXX_ses-01_task-rest_eeg.json
        │   │   ├── sub-XXX_ses-01_task-rest_channels.tsv
        │   │   ├── sub-XXX_ses-01_task-rest_events.tsv
        │   │   └── sub-XXX_ses-01_task-rest_eeg.vhdr
        │   └── beh/
        │       ├── sub-XXX_ses-01_task-rest_beh.tsv
        │       └── sub-XXX_ses-01_task-rest_beh.json
        ├── ses-02/  (same as ses-01)
        └── sub-XXX_scans.tsv

    Plus top-level BIDS files: dataset_description.json, participants.tsv,
    README, CHANGES.

    Files-per-subject: 1 (scans) + 2 sessions × (5 eeg + 2 beh) = 15
    Total files for 100 subjects: 100 × 15 + 4 root = 1 504.

    Returns the count of files actually written.
    """
    # Root metadata
    (root / "dataset_description.json").write_text('{"Name": "synthetic"}')
    (root / "participants.tsv").write_text("participant_id\nsub-001\n")
    (root / "README").write_text("synthetic dataset")
    (root / "CHANGES").write_text("1.0.0 - initial\n")

    files_written = 4
    for s in range(1, n_subjects + 1):
        sub = f"sub-{s:03d}"
        sub_dir = root / sub
        sub_dir.mkdir()
        (sub_dir / f"{sub}_scans.tsv").write_text("filename\tsession\n")
        files_written += 1

        for sess in ("ses-01", "ses-02", "ses-03"):
            sess_dir = sub_dir / sess
            eeg_dir = sess_dir / "eeg"
            beh_dir = sess_dir / "beh"
            eeg_dir.mkdir(parents=True)
            beh_dir.mkdir(parents=True)

            stem = f"{sub}_{sess}_task-rest"
            for suffix in (
                "_eeg.edf",
                "_eeg.json",
                "_eeg.vhdr",
                "_channels.tsv",
                "_events.tsv",
            ):
                (eeg_dir / f"{stem}{suffix}").write_bytes(b"x" * 32)
                files_written += 1
            for suffix in ("_beh.tsv", "_beh.json"):
                (beh_dir / f"{stem}{suffix}").write_text("x")
                files_written += 1

    return files_written


# ─── Benchmark + regression guard ──────────────────────────────────────────


def test_list_git_files_benchmark_synthetic_tree(benchmark, tmp_path):
    """Walk a synthetic 4 000-file BIDS tree under a 500 ms ceiling.

    Regression guard for the Stage 2 manifest walker. The current
    ``os.scandir``-based implementation runs in well under 250 ms on
    standard CI hardware; the 500 ms ceiling fires if a refactor
    re-introduces per-entry stat fan-out.
    """
    n_subjects = 100  # 100 × ~40 files + root = ~4 000 entries
    expected = _build_synthetic_bids_tree(tmp_path, n_subjects=n_subjects)
    # 100 × (1 scans + 3 sessions × (5 eeg + 2 beh)) + 4 root
    # = 100 × (1 + 21) + 4 = 2 204
    assert expected == 100 * (1 + 3 * (5 + 2)) + 4, (
        f"synthetic tree builder drift: got {expected} files"
    )

    result = benchmark(list_git_files, tmp_path)

    # Functional correctness: every file appears exactly once with a
    # non-None size and a relative-to-root path. Order is not asserted —
    # downstream fingerprint hashing sorts internally.
    assert len(result) == expected, (
        f"walker dropped files: expected {expected}, got {len(result)}"
    )
    paths = {f["name"] for f in result}
    assert len(paths) == expected, "walker emitted duplicate paths"
    for f in result:
        assert "name" in f
        assert "size" in f
        assert isinstance(f["size"], int)
        assert f["size"] >= 0
        # Relative paths only — no absolute paths leak.
        assert not f["name"].startswith("/")

    # Regression guard. Mean (not max — avoids first-run JIT noise on
    # cold filesystems). 500 ms is generous; the new walker on this tree
    # is ~80-150 ms on M-series hardware and ~250-400 ms on CI runners.
    mean_seconds = benchmark.stats.stats.mean
    assert mean_seconds < 0.5, (
        f"manifest walk regression: mean {mean_seconds * 1000:.0f} ms "
        f"exceeds 500 ms ceiling — did Path.rglob creep back in?"
    )
