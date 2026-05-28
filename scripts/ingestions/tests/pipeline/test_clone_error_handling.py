"""Regression tests for 2_clone.py error-handling contracts.

Addresses silent error masking around `future.result()` in the main
thread pool. Before this fix, any exception escaping `process_dataset`
was caught by a broad ``except Exception`` and reported as
``{"status": "error", "error": "<str>"}``. Recoverable failures and
programmer errors were indistinguishable.

These tests pin the new contract:

- **Recoverable failures** (network, OS, malformed metadata) →
  ``{"status": "error", ...}`` record; the batch continues.
- **Programmer errors** (AttributeError, TypeError, RuntimeError) →
  propagate so CI fails loudly with a stack trace.

The module is loaded via importlib because 2_clone.py starts with a
digit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import httpx
import pytest
from _helpers import INGEST_DIR as _INGEST_DIR

_CLONE_PATH = _INGEST_DIR / "2_clone.py"


@pytest.fixture(scope="module")
def clone() -> ModuleType:
    """Load ``2_clone.py`` as a module (digit-prefixed filename)."""
    spec = importlib.util.spec_from_file_location("clone_under_test", _CLONE_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Module-level handler stubs (avoid nested-function lint rule) ─────────


def _handler_raise_request_error(*args, **kwargs):
    raise httpx.RequestError("upstream down")


def _handler_raise_oserror_enospc(*args, **kwargs):
    raise OSError(28, "No space left on device")  # ENOSPC


def _handler_raise_value_error(*args, **kwargs):
    raise ValueError("manifest missing 'files' key")


def _handler_raise_attribute_error(*args, **kwargs):
    # Typical real-world shape: a renamed field surfaces as None, then a
    # chained attribute access trips an AttributeError.
    raise AttributeError("'NoneType' object has no attribute 'get'")


def _handler_raise_type_error(*args, **kwargs):
    raise TypeError("expected str, got int")


def _handler_raise_runtime_error(*args, **kwargs):
    raise RuntimeError("BUG: state should be 'idle' here")


# ─── Recoverable failures → structured error record (batch continues) ──────


@pytest.mark.parametrize(
    ("handler", "dataset_id", "expected_in_error_msg"),
    [
        pytest.param(
            _handler_raise_request_error,
            "ds999999",
            "upstream down",
            id="network",
        ),
        pytest.param(
            _handler_raise_oserror_enospc,
            "ds999999",
            "No space left on device",
            id="oserror_enospc",
        ),
        pytest.param(
            _handler_raise_value_error,
            "ds_malformed",
            "manifest missing",
            id="value_error",
        ),
    ],
)
def test_process_dataset_recoverable_failure_returns_error_record(
    clone: ModuleType,
    tmp_path: Path,
    handler,
    dataset_id: str,
    expected_in_error_msg: str,
) -> None:
    """Handler-raised RequestError / OSError / ValueError becomes a
    ``{'status': 'error', ...}`` record so the batch continues to the
    next dataset rather than crashing the whole run."""
    dataset = {"dataset_id": dataset_id, "source": "openneuro"}
    with patch.dict(clone.HANDLERS, {"openneuro": handler}):
        result = clone.process_dataset(dataset, tmp_path)
    assert result["status"] == "error"
    assert result["dataset_id"] == dataset_id
    assert expected_in_error_msg in result["error"]


# ─── Programmer errors → propagate (CI fails loudly with a stack trace) ────


@pytest.mark.parametrize(
    ("handler", "exc_type", "match"),
    [
        pytest.param(
            _handler_raise_attribute_error,
            AttributeError,
            "NoneType",
            id="attribute_error",
        ),
        pytest.param(
            _handler_raise_type_error, TypeError, "expected str", id="type_error"
        ),
        pytest.param(
            _handler_raise_runtime_error, RuntimeError, "BUG", id="runtime_error"
        ),
    ],
)
def test_process_dataset_programmer_error_propagates(
    clone: ModuleType,
    tmp_path: Path,
    handler,
    exc_type,
    match: str,
) -> None:
    """Programmer-bug exception classes are NOT swallowed.

    Pre-fix this would have returned a generic ``{"status": "error",
    ...}`` and the operator would see a generic error in the batch
    report without knowing the pipeline code itself was broken.
    Post-fix the exception propagates, CI fails loudly, the operator
    gets the stack trace.
    """
    dataset = {"dataset_id": "ds_buggy", "source": "openneuro"}
    with patch.dict(clone.HANDLERS, {"openneuro": handler}):
        with pytest.raises(exc_type, match=match):
            clone.process_dataset(dataset, tmp_path)
