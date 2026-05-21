"""Performance gates — memory ceilings + throughput floors.

These tests catch regressions BEFORE they OOM the CI worker or slow
the nightly digest below an acceptable budget. The numbers are
calibrated against the actual CC0 fixtures in ``tests/fixtures/``.

Marked ``@pytest.mark.slow`` so they are excluded from the default
PR-fast suite. Run explicitly with::

    pytest -m slow

The pytest-benchmark numbers below also feed the github-action-benchmark
artifact when the bench CI workflow lands.
"""

from __future__ import annotations

import gc
import tracemalloc
from pathlib import Path

import pytest

from _fingerprint import fingerprint_from_manifest
from _vhdr_parser import parse_vhdr_metadata

EEG_VHDR = (
    Path(__file__).parent / "fixtures" / "eeg" / "sub-xp101_task-motorloc_eeg.vhdr"
)


# ─── Memory ceilings ───────────────────────────────────────────────────────


@pytest.mark.slow
def test_parse_vhdr_peak_memory_under_2mb():
    """Parsing one .vhdr fixture must not allocate more than 2 MB.

    Caps a regression where the parser would inadvertently materialise
    the full channel-list across all (potentially 256+) channels into
    a large intermediate structure.
    """
    gc.collect()
    tracemalloc.start()
    try:
        meta = parse_vhdr_metadata(EEG_VHDR)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert meta is not None
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 2.0, (
        f"parse_vhdr_metadata peaked at {peak_mb:.2f} MB on a 64-channel "
        f"fixture; ceiling is 2 MB. Suspect a quadratic accumulator or "
        f"forgotten copy. See ROBUSTNESS/05-EVALUATION.md § Phase 7."
    )


@pytest.mark.slow
def test_fingerprint_1000_files_peak_memory_under_5mb():
    """Fingerprinting a 1000-file manifest must stay under 5 MB peak.

    A regression that built the entries list as O(N²) (e.g., re-sorting
    on every push) would fail this gate before it could ship.
    """
    manifest = {
        "files": [
            {"path": f"sub-{i:04d}/eeg/sub-{i:04d}_task-rest_eeg.edf", "size": i * 100}
            for i in range(1000)
        ]
    }

    gc.collect()
    tracemalloc.start()
    try:
        fp = fingerprint_from_manifest("ds001", "openneuro", manifest)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(fp) == 64
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 5.0, (
        f"fingerprint_from_manifest(1000 files) peaked at {peak_mb:.2f} MB; "
        f"ceiling is 5 MB."
    )


# ─── Throughput floors (pytest-benchmark integration) ─────────────────────


def test_parse_vhdr_median_under_5ms(benchmark):
    """The 64-channel VHDR parser must process under 5 ms at the median.

    Gates the *typical* per-call cost — the one the nightly fuzz layer
    pays thousands of times. Median (not max) is the right metric: a
    single GC-pause / IO-interrupt outlier shouldn't fail the gate
    because it's not a real regression. github-action-benchmark
    captures the full distribution for trend tracking.

    The viewer-side fuzz layer triggers thousands of header parses per
    nightly run; doubling the median time cascades.
    """
    meta = benchmark(parse_vhdr_metadata, EEG_VHDR)
    assert meta is not None
    # Median across all rounds. Resistant to single-round outliers from
    # GC / IO interrupts. A *systemic* slowdown moves the median; a
    # one-off pause doesn't. Earlier version used .max which is
    # dominated by outliers and was flaky on local + CI runs.
    median_s = benchmark.stats.stats.median
    assert median_s < 0.005, (
        f"parse_vhdr_metadata median = {median_s * 1000:.2f} ms; ceiling is 5 ms."
    )


def test_fingerprint_throughput_1000_files(benchmark):
    """Fingerprinting 1000 files must run in under 5 ms (mean).

    This guards the cold-start cost of the digest step's
    "did this dataset change?" check.
    """
    manifest = {"files": [{"path": f"f{i}.edf", "size": i * 100} for i in range(1000)]}

    fp = benchmark(fingerprint_from_manifest, "ds001", "openneuro", manifest)
    assert len(fp) == 64
    assert benchmark.stats.stats.mean < 0.005, (
        f"fingerprint(1000 files) mean = {benchmark.stats.stats.mean * 1000:.2f} ms; "
        f"ceiling is 5 ms."
    )
