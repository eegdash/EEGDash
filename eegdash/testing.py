"""Lazy fetch of the eegdash binary test corpus.

The raw signal fixtures (BDF, EDF, SET, VHDR, SNIRF, FIF, MEF3 ...) live in
the separate `eegdash/eegdash-testing-data
<https://github.com/eegdash/eegdash-testing-data>`__ repository, modeled
after ``mne-testing-data``. The first time a test asks for one we download
the pinned tarball, verify its SHA-256, and unpack into a per-user cache.

Pin (bump both lines when re-tagging the upstream repo):

* :data:`VERSION` — git tag on ``eegdash-testing-data``
* :data:`SHA256` — sha256 of the codeload tarball for that tag

Environment overrides
---------------------
``EEGDASH_TESTING_DATA_DIR``
    Cache directory (default: ``~/.cache/eegdash/testing-data``).
``EEGDASH_SKIP_TESTING_DATA=true``
    Skip every ``@requires_testing_data`` test; used by air-gapped CI.

Examples
--------
>>> from eegdash.testing import data_path  # doctest: +SKIP
>>> bdf = data_path() / "eeg" / "sub-001_ses-01_task-meditation_eeg.bdf"

"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

VERSION = "0.1.0"
SHA256 = "bf7303148bf4acab2b51ad69adc76574360ac096bf29b5a76b510ad77981bbae"
URL = (
    "https://codeload.github.com/eegdash/eegdash-testing-data/"
    f"tar.gz/refs/tags/v{VERSION}"
)

_DEFAULT_CACHE = Path.home() / ".cache" / "eegdash" / "testing-data"


def _cache_dir() -> Path:
    return Path(os.environ.get("EEGDASH_TESTING_DATA_DIR", _DEFAULT_CACHE))


def _root_dir() -> Path:
    # GitHub's codeload tarball unpacks into ``<name>-<tag>/`` at the top.
    return _cache_dir() / f"eegdash-testing-data-{VERSION}"


def _skip_requested() -> bool:
    return os.environ.get("EEGDASH_SKIP_TESTING_DATA", "").lower() in (
        "1",
        "true",
        "yes",
    )


def has_testing_data() -> bool:
    """Return True if the corpus is already unpacked in the cache."""
    return _root_dir().is_dir() and any(_root_dir().iterdir())


@lru_cache(maxsize=1)
def data_path() -> Path:
    """Return the root of the test corpus, fetching on first use.

    Returns
    -------
    Path
        The unpacked ``eegdash-testing-data-{VERSION}/`` directory.

    Raises
    ------
    RuntimeError
        If the download is required but ``EEGDASH_SKIP_TESTING_DATA=true``
        is set, or if pooch fails to retrieve the tarball.

    """
    if has_testing_data():
        return _root_dir()

    if _skip_requested():
        raise RuntimeError("EEGDASH_SKIP_TESTING_DATA=true — refusing to fetch corpus")

    import pooch

    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    pooch.retrieve(
        url=URL,
        known_hash=f"sha256:{SHA256}",
        fname=f"eegdash-testing-data-{VERSION}.tar.gz",
        path=str(cache),
        processor=pooch.Untar(extract_dir=str(cache)),
    )
    if not has_testing_data():
        raise RuntimeError(
            f"pooch reported success but {_root_dir()} is empty; "
            "tarball layout may have changed upstream"
        )
    return _root_dir()


def data_file(relpath: str) -> Path:
    """Convenience: ``data_path() / relpath`` as a single call."""
    return data_path() / relpath


def requires_testing_data(func):
    """Pytest decorator: skip if the corpus is unavailable.

    Skips when ``EEGDASH_SKIP_TESTING_DATA=true`` or the corpus cannot
    be fetched (e.g. offline CI without a cache hit). The decorator
    triggers the fetch at collection time so tests that depend on the
    corpus all share a single download.
    """
    import pytest

    reason: str | None
    if _skip_requested():
        reason = "EEGDASH_SKIP_TESTING_DATA=true"
    else:
        try:
            data_path()
            reason = None
        except Exception as exc:  # noqa: BLE001 — any fetch failure = skip
            reason = f"eegdash-testing-data unavailable: {exc}"

    return pytest.mark.skipif(reason is not None, reason=reason or "")(func)


def _main() -> int:
    """``python -m eegdash.testing`` — eagerly download the corpus."""
    try:
        root = data_path()
    except Exception as exc:  # noqa: BLE001
        print(f"failed: {exc}")
        return 1
    print(f"ok: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
