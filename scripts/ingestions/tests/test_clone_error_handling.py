"""Regression tests for 2_clone.py error-handling contracts.

Addresses silent error masking around `future.result()` in the main
thread pool. Before
this fix, any exception escaping `process_dataset` was caught by a
broad `except Exception` and reported as `{"status": "error", "error":
"<str>"}`. Recoverable failures and programmer errors were
indistinguishable.

These tests pin the new contract:

- Recoverable failures (network, OS) → status="error" record (continue).
- Programmer errors (AttributeError, TypeError, etc.) → propagate.

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

_CLONE_PATH = Path(__file__).parent.parent / "2_clone.py"


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


# ─── process_dataset: recoverable failures become structured errors ──────


def test_process_dataset_network_error_returns_error_record(
    clone: ModuleType, tmp_path: Path
) -> None:
    """httpx.RequestError from a handler becomes {'status': 'error', ...}."""
    dataset = {"dataset_id": "ds999999", "source": "openneuro"}
    with patch.dict(clone.HANDLERS, {"openneuro": _handler_raise_request_error}):
        result = clone.process_dataset(dataset, tmp_path)
    assert result["status"] == "error"
    assert result["dataset_id"] == "ds999999"
    assert "upstream down" in result["error"]


def test_process_dataset_oserror_returns_error_record(
    clone: ModuleType, tmp_path: Path
) -> None:
    """OSError (disk full / permission) becomes a structured error, not a crash."""
    dataset = {"dataset_id": "ds999999", "source": "openneuro"}
    with patch.dict(clone.HANDLERS, {"openneuro": _handler_raise_oserror_enospc}):
        result = clone.process_dataset(dataset, tmp_path)
    assert result["status"] == "error"
    assert "No space left on device" in result["error"]


def test_process_dataset_value_error_is_treated_as_recoverable(
    clone: ModuleType, tmp_path: Path
) -> None:
    """ValueError (malformed dataset metadata) is recoverable.

    The handler might raise ValueError on, e.g., a missing dataset_id
    field. The pipeline records the failure and continues to the next
    dataset rather than crashing the whole batch.
    """
    dataset = {"dataset_id": "ds_malformed", "source": "openneuro"}
    with patch.dict(clone.HANDLERS, {"openneuro": _handler_raise_value_error}):
        result = clone.process_dataset(dataset, tmp_path)
    assert result["status"] == "error"
    assert "manifest missing" in result["error"]


# ─── process_dataset: programmer errors PROPAGATE (Phase 9 F1 fix) ───────


def test_process_dataset_attribute_error_propagates(
    clone: ModuleType, tmp_path: Path
) -> None:
    """AttributeError is a programmer bug; it must NOT be swallowed.

    Pre-Phase-9: this would have returned ``{"status": "error", "error":
    "'NoneType' has no attribute 'X'"}`` and the operator would have
    seen a generic "error" in the batch report without knowing the
    pipeline code itself was broken.

    Post-Phase-9: the AttributeError propagates, CI fails loudly, the
    operator gets the stack trace.
    """
    dataset = {"dataset_id": "ds_buggy", "source": "openneuro"}
    with patch.dict(clone.HANDLERS, {"openneuro": _handler_raise_attribute_error}):
        with pytest.raises(AttributeError, match="NoneType"):
            clone.process_dataset(dataset, tmp_path)


def test_process_dataset_type_error_propagates(
    clone: ModuleType, tmp_path: Path
) -> None:
    """TypeError (e.g. passing wrong arg type) is a programmer bug."""
    dataset = {"dataset_id": "ds_buggy", "source": "openneuro"}
    with patch.dict(clone.HANDLERS, {"openneuro": _handler_raise_type_error}):
        with pytest.raises(TypeError, match="expected str"):
            clone.process_dataset(dataset, tmp_path)


def test_process_dataset_runtime_error_propagates(
    clone: ModuleType, tmp_path: Path
) -> None:
    """RuntimeError indicates internal invariant violation; propagate."""
    dataset = {"dataset_id": "ds_buggy", "source": "openneuro"}
    with patch.dict(clone.HANDLERS, {"openneuro": _handler_raise_runtime_error}):
        with pytest.raises(RuntimeError, match="BUG"):
            clone.process_dataset(dataset, tmp_path)
