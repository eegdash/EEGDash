"""Hypothesis property-based tests for the format parsers.

These tests run 200-500 generated inputs against each parser. They
catch the class of bug that example-based tests miss: an assumption
about the input shape that holds for the test fixtures but breaks on
adversarial input. The viewer's session caught a real FIFF parser
bug this way — synthetic random bytes never accidentally collided
with the expected magic, so the parser appeared to "work".

The headline property is **no-crash**: a parser may raise a
*documented* exception class, but it must never SegFault, raise
BaseException (KeyboardInterrupt, SystemExit), or hang.

Each parser also gets a stability property if applicable:
- Re-parsing the same bytes twice returns the same value.
- The output type is bounded (None or dict, never anything else).
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from _set_parser import parse_set_metadata
from _snirf_parser import parse_snirf_metadata
from _vhdr_parser import parse_vhdr_metadata

# Cap the inputs at 8 KB — the parsers handle real files, but generating
# arbitrarily-large random buffers makes the test suite slow without
# finding extra bugs (the parsing logic at byte 8000 is the same as at
# byte 8 million for these formats).
ARBITRARY_BUFFER = st.binary(min_size=0, max_size=8192)


def _write_and_call(tmp_path: Path, suffix: str, buf: bytes, parser):
    """Helper: write ``buf`` to a tmp file with ``suffix``, call ``parser``."""
    f = tmp_path / f"fuzz{suffix}"
    f.write_bytes(buf)
    try:
        result = parser(f)
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        UnicodeDecodeError,
        UnicodeError,
        TypeError,
    ):
        return None  # acceptable failure
    # Output type contract: every parser returns None or a dict.
    assert result is None or isinstance(result, dict), (
        f"parser returned {type(result).__name__}; expected None or dict"
    )
    return result


# ─── No-crash properties ───────────────────────────────────────────────────


@given(buf=ARBITRARY_BUFFER)
@settings(max_examples=200, deadline=None)
def test_parse_vhdr_never_crashes(buf: bytes, tmp_path_factory):
    """parse_vhdr_metadata may raise ValueError/UnicodeDecodeError/OSError
    but never BaseException or non-Exception types. Output is None|dict."""
    tmp = tmp_path_factory.mktemp("vhdr_fuzz")
    _write_and_call(tmp, ".vhdr", buf, parse_vhdr_metadata)


@given(buf=ARBITRARY_BUFFER)
@settings(max_examples=200, deadline=None)
def test_parse_set_never_crashes(buf: bytes, tmp_path_factory):
    """parse_set_metadata robust to random bytes."""
    tmp = tmp_path_factory.mktemp("set_fuzz")
    _write_and_call(tmp, ".set", buf, parse_set_metadata)


@given(buf=ARBITRARY_BUFFER)
@settings(max_examples=200, deadline=None)
def test_parse_snirf_never_crashes(buf: bytes, tmp_path_factory):
    """parse_snirf_metadata (HDF5 container) robust to random bytes."""
    tmp = tmp_path_factory.mktemp("snirf_fuzz")
    _write_and_call(tmp, ".snirf", buf, parse_snirf_metadata)


# ─── Determinism properties ────────────────────────────────────────────────


@given(buf=ARBITRARY_BUFFER)
@settings(max_examples=50, deadline=None)
def test_parse_vhdr_is_deterministic(buf: bytes, tmp_path_factory):
    """Parsing the same bytes twice returns the same value (no hidden state)."""
    tmp = tmp_path_factory.mktemp("vhdr_det")
    a = _write_and_call(tmp, ".vhdr", buf, parse_vhdr_metadata)
    b = _write_and_call(tmp, ".vhdr", buf, parse_vhdr_metadata)
    assert a == b


@given(buf=ARBITRARY_BUFFER)
@settings(max_examples=50, deadline=None)
def test_parse_set_is_deterministic(buf: bytes, tmp_path_factory):
    """parse_set_metadata is a pure function over the file bytes."""
    tmp = tmp_path_factory.mktemp("set_det")
    a = _write_and_call(tmp, ".set", buf, parse_set_metadata)
    b = _write_and_call(tmp, ".set", buf, parse_set_metadata)
    assert a == b


# ─── INI-shaped property for the VHDR parser ───────────────────────────────


# A more constrained generator that produces something LIKE a .vhdr —
# valid INI sections with arbitrary keys. The parser should always
# return *something* (dict, possibly with `nchans=None`) on well-formed
# INI even if the keys are wrong.
ini_value = st.text(
    alphabet=st.characters(blacklist_characters="\n\r=[]"),
    min_size=0,
    max_size=64,
)
ini_key = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "Nd"),
        whitelist_characters="_",
    ),
    min_size=1,
    max_size=32,
)
ini_section = st.tuples(ini_key, st.dictionaries(ini_key, ini_value, max_size=5))


@given(sections=st.lists(ini_section, min_size=0, max_size=4))
@settings(max_examples=100, deadline=None)
def test_parse_vhdr_accepts_arbitrary_ini(sections, tmp_path_factory):
    """A well-formed INI file (with arbitrary keys) doesn't crash parse_vhdr."""
    tmp = tmp_path_factory.mktemp("vhdr_ini")
    lines = []
    for section, kvs in sections:
        lines.append(f"[{section}]")
        for k, v in kvs.items():
            lines.append(f"{k}={v}")
    text = "\n".join(lines) + "\n"
    f = tmp / "fuzz.vhdr"
    f.write_text(text, encoding="utf-8")
    try:
        result = parse_vhdr_metadata(f)
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError, RuntimeError, UnicodeDecodeError):
        pass  # documented failure modes


# ─── Fingerprint stability under property generation ───────────────────────


@given(
    dataset_id=st.text(min_size=1, max_size=32),
    source=st.sampled_from(["openneuro", "nemar", "figshare", "zenodo", "osf"]),
    n_files=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=100, deadline=None)
def test_fingerprint_property_deterministic(
    dataset_id: str, source: str, n_files: int
) -> None:
    """For any (dataset_id, source, manifest), the fingerprint is stable."""
    from _fingerprint import fingerprint_from_manifest

    manifest = {
        "files": [{"path": f"f{i}.edf", "size": i * 100} for i in range(n_files)],
    }
    f1 = fingerprint_from_manifest(dataset_id, source, manifest)
    f2 = fingerprint_from_manifest(dataset_id, source, manifest)
    assert f1 == f2
    assert len(f1) == 64
