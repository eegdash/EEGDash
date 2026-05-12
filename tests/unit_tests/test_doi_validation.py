"""DOI validation tests for EEGDash datasets.

Offline tests validate DOI format for all datasets in dataset_summary.csv.
Network tests (marked ``@pytest.mark.network``) resolve DOIs via doi.org
and verify author metadata.  A persistent JSON cache avoids redundant
API calls across runs.

Run offline tests only::

    pytest tests/unit_tests/test_doi_validation.py -m "not network"

Run all (including network)::

    pytest tests/unit_tests/test_doi_validation.py -m network

Inspired by MOABB ``moabb/tests/test_doi_validation.py``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from scripts.citations.doi_utils import (
    extract_surnames,
    is_valid_doi,
    load_cache,
    load_dataset_dois,
    normalize_doi,
    resolve_doi_cached,
    save_cache,
    surnames_overlap,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_PATH = Path(__file__).resolve().parent.parent / "doi_cache.json"
CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "eegdash"
    / "dataset"
    / "dataset_summary.csv"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_dois() -> list[tuple[str, str]]:
    """All (dataset_id, raw_doi) pairs from the summary CSV."""
    if not CSV_PATH.exists():
        pytest.skip(f"dataset_summary.csv not found at {CSV_PATH}")
    return load_dataset_dois(CSV_PATH)


@pytest.fixture(scope="module")
def doi_cache() -> dict:
    """Persistent DOI cache loaded once per module."""
    return load_cache(CACHE_PATH)


# ===================================================================
# Offline tests — no network required
# ===================================================================


class TestDOIFormat:
    """Validate DOI syntax for every dataset in the CSV."""

    # DOIs known to be invalid upstream (e.g. OpenNeuro placeholder values).
    KNOWN_BAD_DOIS: set[str] = {"mockDOI"}
    STRICT = os.getenv("EEGDASH_STRICT_DOI_CSV", "").strip() == "1"

    def test_all_dois_have_valid_format(self, dataset_dois):
        """Every non-empty DOI should match ``10.XXXX/...``."""
        bad: list[tuple[str, str]] = []
        for dataset_id, raw_doi in dataset_dois:
            norm = normalize_doi(raw_doi)
            if norm in self.KNOWN_BAD_DOIS:
                continue
            if not is_valid_doi(norm):
                bad.append((dataset_id, raw_doi))

        if bad:
            msg = "\n".join(f"  {did}: {doi}" for did, doi in bad[:20])
            extra = f"\n  ... and {len(bad) - 20} more" if len(bad) > 20 else ""
            pytest.fail(f"{len(bad)} dataset(s) have malformed DOIs:\n{msg}{extra}")

    def test_known_bad_dois_are_flagged(self, dataset_dois):
        """Datasets with known-bad DOIs should be listed for upstream fix."""
        flagged: list[tuple[str, str]] = []
        for dataset_id, raw_doi in dataset_dois:
            norm = normalize_doi(raw_doi)
            if norm in self.KNOWN_BAD_DOIS:
                flagged.append((dataset_id, raw_doi))

        if flagged:
            msg = "\n".join(f"  {did}: {doi}" for did, doi in flagged)
            print(
                f"\n{len(flagged)} dataset(s) with known-bad DOIs "
                f"(need upstream fix):\n{msg}"
            )

    def test_no_url_prefix_in_doi_column(self, dataset_dois):
        """DOIs should be stored without ``https://doi.org/`` prefix."""
        prefixed: list[tuple[str, str]] = []
        for dataset_id, raw_doi in dataset_dois:
            stripped = raw_doi.strip()
            if stripped.startswith("http"):
                prefixed.append((dataset_id, raw_doi))

        if prefixed:
            msg = "\n".join(f"  {did}: {doi}" for did, doi in prefixed[:10])
            if self.STRICT:
                pytest.fail(
                    f"{len(prefixed)} DOIs stored with URL prefix (should be bare):\n{msg}"
                )
            print(
                f"\n{len(prefixed)} DOI(s) stored with URL prefix (normalization will handle it):\n{msg}"
            )

    def test_no_trailing_whitespace(self, dataset_dois):
        """DOI strings should not have leading/trailing whitespace."""
        bad = [(did, doi) for did, doi in dataset_dois if doi != doi.strip()]
        if bad:
            msg = "\n".join(f"  {did}: repr={doi!r}" for did, doi in bad[:10])
            pytest.fail(f"{len(bad)} DOIs with whitespace:\n{msg}")

    def test_doi_coverage(self, dataset_dois):
        """Report how many datasets have DOIs (informational)."""
        if not CSV_PATH.exists():
            pytest.skip("CSV not found")
        with open(CSV_PATH, newline="") as fh:
            total = sum(1 for _ in csv.DictReader(fh))
        with_doi = len(dataset_dois)
        coverage = with_doi / total * 100 if total else 0
        print(f"\nDOI coverage: {with_doi}/{total} ({coverage:.1f}%)")
        # We expect high coverage but don't hard-fail on it.
        assert coverage > 80, f"DOI coverage dropped below 80%: {coverage:.1f}%"


class TestDOICacheCompleteness:
    """Verify the local cache covers all DOIs in the CSV."""

    STRICT = os.getenv("EEGDASH_STRICT_DOI_CSV", "").strip() == "1"

    def test_cache_exists(self):
        """The cache file should exist after the initial build."""
        if not CACHE_PATH.exists():
            pytest.skip(
                "doi_cache.json not built yet — run: "
                "python -m pytest tests/unit_tests/test_doi_validation.py "
                "-m network -k test_resolve_all_dois"
            )

    def test_cache_covers_dataset_dois(self, dataset_dois, doi_cache):
        """Every DOI in the CSV should be in the cache."""
        if not doi_cache:
            pytest.skip("Cache is empty — build it with network tests first")

        missing: list[str] = []
        for _, raw_doi in dataset_dois:
            norm = normalize_doi(raw_doi)
            if norm and norm not in doi_cache:
                missing.append(norm)

        if missing:
            msg = "\n".join(f"  {d}" for d in missing[:20])
            extra = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
            if self.STRICT:
                pytest.fail(
                    f"{len(missing)} DOI(s) missing from cache "
                    f"(run network tests to update):\n{msg}{extra}"
                )
            pytest.skip(
                f"Cache incomplete for {len(missing)} DOI(s); run network DOI sync to refresh cache."
            )


# ===================================================================
# Network tests — require doi.org access
# ===================================================================

network = pytest.mark.network


@network
class TestDOIResolution:
    """Resolve DOIs via doi.org and validate metadata."""

    def test_resolve_sample_dois(self, dataset_dois, doi_cache):
        """Resolve a sample of DOIs to verify they're live."""
        sample = dataset_dois[:10]  # small sample for quick CI
        failed: list[tuple[str, str]] = []

        for dataset_id, raw_doi in sample:
            norm = normalize_doi(raw_doi)
            if not norm:
                continue
            result = resolve_doi_cached(norm, doi_cache, delay=0.2)
            if result is None:
                failed.append((dataset_id, raw_doi))

        # Persist updated cache
        save_cache(doi_cache, CACHE_PATH)

        if failed:
            msg = "\n".join(f"  {did}: {doi}" for did, doi in failed)
            pytest.fail(f"{len(failed)} DOI(s) failed to resolve:\n{msg}")

    def test_resolve_all_dois(self, dataset_dois, doi_cache):
        """Resolve every DOI and build/update the persistent cache.

        This is slow (~2 min for 700 DOIs) — run explicitly::

            pytest -m network -k test_resolve_all_dois -s
        """
        failed: list[tuple[str, str, str]] = []
        resolved = 0

        for i, (dataset_id, raw_doi) in enumerate(dataset_dois):
            norm = normalize_doi(raw_doi)
            if not norm:
                continue

            # Skip if already cached
            if norm in doi_cache:
                resolved += 1
                continue

            result = resolve_doi_cached(norm, doi_cache, delay=0.2)
            if result is None:
                failed.append((dataset_id, raw_doi, "resolution failed"))
            else:
                resolved += 1

            # Save periodically
            if (i + 1) % 50 == 0:
                save_cache(doi_cache, CACHE_PATH)
                print(
                    f"  Progress: {i + 1}/{len(dataset_dois)}, "
                    f"resolved={resolved}, failed={len(failed)}"
                )

        save_cache(doi_cache, CACHE_PATH)

        print(
            f"\nFinal: resolved={resolved}, failed={len(failed)}, "
            f"cache size={len(doi_cache)}"
        )

        if failed:
            msg = "\n".join(
                f"  {did}: {doi} ({reason})" for did, doi, reason in failed[:20]
            )
            extra = f"\n  ... and {len(failed) - 20} more" if len(failed) > 20 else ""
            pytest.fail(f"{len(failed)} DOI(s) failed to resolve:\n{msg}{extra}")

    def test_resolved_dois_have_authors(self, dataset_dois, doi_cache):
        """Resolved DOIs should have at least one author."""
        no_authors: list[str] = []
        for _, raw_doi in dataset_dois:
            norm = normalize_doi(raw_doi)
            if not norm or norm not in doi_cache:
                continue
            entry = doi_cache[norm]
            authors = entry.get("authors", [])
            if not authors:
                no_authors.append(norm)

        if no_authors:
            msg = "\n".join(f"  {d}" for d in no_authors[:10])
            print(
                f"\nWarning: {len(no_authors)} DOI(s) resolved without "
                f"authors (may be datasets, not papers):\n{msg}"
            )


# ===================================================================
# Unit tests for doi_utils helpers
# ===================================================================


class TestNormalizeDOI:
    def test_bare(self):
        assert normalize_doi("10.1234/foo") == "10.1234/foo"

    def test_doi_prefix(self):
        assert normalize_doi("doi:10.1234/foo") == "10.1234/foo"

    def test_https_url(self):
        assert normalize_doi("https://doi.org/10.1234/foo") == "10.1234/foo"

    def test_http_url(self):
        assert normalize_doi("http://doi.org/10.1234/foo") == "10.1234/foo"

    def test_dx_url(self):
        assert normalize_doi("https://dx.doi.org/10.1234/foo") == "10.1234/foo"

    def test_whitespace(self):
        assert normalize_doi("  10.1234/foo  ") == "10.1234/foo"

    def test_none(self):
        assert normalize_doi(None) is None

    def test_empty(self):
        assert normalize_doi("") is None


class TestIsValidDOI:
    def test_valid(self):
        assert is_valid_doi("10.1234/foo") is True
        assert is_valid_doi("10.18112/openneuro.ds001785.v1.1.1") is True

    def test_invalid_prefix(self):
        assert is_valid_doi("11.1234/foo") is False

    def test_short_registrant(self):
        assert is_valid_doi("10.12/foo") is False

    def test_no_suffix(self):
        assert is_valid_doi("10.1234/") is False

    def test_none(self):
        assert is_valid_doi(None) is False

    def test_with_url_prefix(self):
        assert is_valid_doi("doi:10.1234/foo") is True


class TestExtractSurnames:
    def test_basic(self):
        authors = [
            {"family": "Smith", "given": "John"},
            {"family": "Doe", "given": "Jane"},
        ]
        assert extract_surnames(authors) == ["smith", "doe"]

    def test_empty(self):
        assert extract_surnames([]) == []

    def test_missing_family(self):
        assert extract_surnames([{"given": "John"}]) == []


class TestSurnamesOverlap:
    def test_overlap(self):
        a = [{"family": "Smith"}, {"family": "Doe"}]
        b = [{"family": "Doe"}, {"family": "Roe"}]
        assert surnames_overlap(a, b) == {"doe"}

    def test_no_overlap(self):
        a = [{"family": "Smith"}]
        b = [{"family": "Doe"}]
        assert surnames_overlap(a, b) == set()
