"""Characterization tests for the multiprocessing digest watchdog.

The watchdog (per-dataset subprocess isolation + per-operation timeouts) had
ZERO test coverage. These tests pin its current behavior so the extraction into
``_digest_runner.py`` is provably zero-behavior-change.

Refactor-agnostic by design: ``_RUNNER`` resolves to ``_digest_runner`` once it
exists and otherwise to the in-tree ``3_digest`` module, so the SAME assertions
run green before AND after the extraction.

Two layers:
  1. Helper-level unit tests using duck-typed fake process/queue objects — fast,
     deterministic, no real subprocesses.
  2. Real-subprocess integration tests of ``process_datasets_with_watchdog`` that
     exercise the actual spawn + pickle + dependency-injection path. These only
     run once ``digest_fn`` injection exists (skipped pre-extraction).
"""

from __future__ import annotations

import inspect
import queue as queue_mod
import subprocess
import sys

import pytest

from tests._helpers import INGEST_DIR, fake_workers, load_digest


def _resolve_runner():
    """Module that owns the watchdog functions (``_digest_runner`` post-extraction)."""
    try:
        import _digest_runner as mod  # noqa: PLC0415 — refactor-agnostic resolution

        return mod
    except ModuleNotFoundError:
        return load_digest()


_RUNNER = _resolve_runner()
_HAS_DIGEST_FN = (
    "digest_fn" in inspect.signature(_RUNNER.process_datasets_with_watchdog).parameters
)
_needs_di = pytest.mark.skipif(
    not _HAS_DIGEST_FN,
    reason="digest_fn injection not present yet (pre-extraction run)",
)


def _unused_digest_fn(dataset_id, input_dir, output_dir):
    """Injected when ``_start_digest_process`` is monkeypatched away — must never run."""
    raise AssertionError("digest_fn called despite patched _start_digest_process")


# ─── duck-typed fakes (no real multiprocessing) ──────────────────────────────


class _FakeProcess:
    """Minimal stand-in for ``multiprocessing.Process``.

    ``alive`` is consumed one value per ``is_alive()`` call; the final value
    repeats, so ``(True, False)`` means "alive on the first check, dead after".
    """

    def __init__(self, alive=(False,), exitcode=0):
        self._alive = list(alive)
        self.exitcode = exitcode
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_timeouts: list = []
        self.close_calls = 0

    def is_alive(self) -> bool:
        if len(self._alive) > 1:
            return self._alive.pop(0)
        return self._alive[0]

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def join(self, timeout=None) -> None:
        self.join_timeouts.append(timeout)

    def close(self) -> None:
        self.close_calls += 1


_UNSET = object()


class _FakeQueue:
    def __init__(
        self, *, get=_UNSET, get_exc=None, get_nowait=_UNSET, get_nowait_exc=None
    ):
        self._get = get
        self._get_exc = get_exc
        self._get_nowait = get_nowait
        self._get_nowait_exc = get_nowait_exc
        self.closed = False

    def get(self, timeout=None):
        if self._get_exc is not None:
            raise self._get_exc
        return self._get

    def get_nowait(self):
        if self._get_nowait_exc is not None:
            raise self._get_nowait_exc
        return self._get_nowait

    def close(self) -> None:
        self.closed = True


def _job(process: _FakeProcess, q: _FakeQueue, *, dataset_id="ds1", started_at=0.0):
    return {
        "dataset_id": dataset_id,
        "process": process,
        "queue": q,
        "started_at": started_at,
    }


class _BoomQueue(_FakeQueue):
    def close(self):
        raise OSError("nope")


class _BoomProc(_FakeProcess):
    def close(self):
        raise ValueError("nope")


# ─── _worker_error_result ─────────────────────────────────────────────────────


def test_worker_error_result_minimal():
    assert _RUNNER._worker_error_result("ds1", "oops") == {
        "status": "error",
        "dataset_id": "ds1",
        "error": "oops",
    }


def test_worker_error_result_with_elapsed_rounds():
    out = _RUNNER._worker_error_result("ds1", "oops", elapsed_seconds=1.23456)
    assert out["elapsed_seconds"] == round(1.23456, 3)


def test_worker_error_result_with_traceback():
    out = _RUNNER._worker_error_result("ds1", "oops", traceback_text="tb-here")
    assert out["traceback"] == "tb-here"


def test_worker_error_result_omits_optional_fields_when_absent():
    out = _RUNNER._worker_error_result("ds1", "oops")
    assert "elapsed_seconds" not in out
    assert "traceback" not in out


# ─── _terminate_active_process ────────────────────────────────────────────────


def test_terminate_when_alive_then_dead_sends_sigterm_only():
    proc = _FakeProcess(alive=(True, False))
    _RUNNER._terminate_active_process(_job(proc, _FakeQueue()), "reason")
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 0
    assert proc.join_timeouts == [_RUNNER.PROCESS_SHUTDOWN_TIMEOUT_SECONDS]


def test_terminate_escalates_to_sigkill_when_sigterm_ignored():
    proc = _FakeProcess(alive=(True, True, False))
    _RUNNER._terminate_active_process(_job(proc, _FakeQueue()), "reason")
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1
    assert proc.join_timeouts == [_RUNNER.PROCESS_SHUTDOWN_TIMEOUT_SECONDS] * 2


def test_terminate_noop_when_already_dead():
    proc = _FakeProcess(alive=(False,))
    _RUNNER._terminate_active_process(_job(proc, _FakeQueue()), "reason")
    assert proc.terminate_calls == 0
    assert proc.kill_calls == 0


def test_terminate_warns_when_alive_after_kill():
    # Alive on every check: SIGTERM ignored, SIGKILL ignored -> reaches the
    # final "still alive after kill" warning branch and returns without raising.
    proc = _FakeProcess(alive=(True, True, True))
    _RUNNER._terminate_active_process(_job(proc, _FakeQueue()), "reason")
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1


# ─── _collect_finished_process ────────────────────────────────────────────────


def test_collect_enriches_result_with_defaults():
    proc = _FakeProcess(alive=(False,))
    q = _FakeQueue(get={"status": "success"})
    out = _RUNNER._collect_finished_process(_job(proc, q, dataset_id="dsX"))
    assert out["status"] == "success"
    assert out["dataset_id"] == "dsX"
    assert "elapsed_seconds" in out


def test_collect_preserves_existing_dataset_id():
    proc = _FakeProcess(alive=(False,))
    q = _FakeQueue(get={"status": "success", "dataset_id": "already-set"})
    out = _RUNNER._collect_finished_process(_job(proc, q, dataset_id="dsX"))
    assert out["dataset_id"] == "already-set"


def test_collect_queue_empty_becomes_error():
    proc = _FakeProcess(alive=(False,), exitcode=7)
    q = _FakeQueue(get_exc=queue_mod.Empty())
    out = _RUNNER._collect_finished_process(_job(proc, q, dataset_id="dsX"))
    assert out["status"] == "error"
    assert "without returning a result" in out["error"]
    assert "exitcode=7" in out["error"]


def test_collect_queue_oserror_becomes_error():
    proc = _FakeProcess(alive=(False,))
    q = _FakeQueue(get_exc=OSError("pipe closed"))
    out = _RUNNER._collect_finished_process(_job(proc, q, dataset_id="dsX"))
    assert out["status"] == "error"
    assert "failed to collect worker result" in out["error"]
    assert "OSError" in out["error"]


def test_collect_nondict_result_becomes_error():
    proc = _FakeProcess(alive=(False,))
    q = _FakeQueue(get="i am a string")
    out = _RUNNER._collect_finished_process(_job(proc, q, dataset_id="dsX"))
    assert out["status"] == "error"
    assert "worker returned str, expected dict" in out["error"]


def test_collect_terminates_process_still_alive_after_join():
    # is_alive: call#1 (post-join check)=True -> terminate; #2 inside terminate=True
    # -> sigterm; #3=False -> no sigkill.
    proc = _FakeProcess(alive=(True, True, False))
    q = _FakeQueue(get={"status": "success"})
    out = _RUNNER._collect_finished_process(_job(proc, q))
    assert out["status"] == "success"
    assert proc.terminate_calls == 1


# ─── _timeout_active_process ──────────────────────────────────────────────────


def test_timeout_terminates_and_returns_error():
    proc = _FakeProcess(alive=(True, False))
    out = _RUNNER._timeout_active_process(_job(proc, _FakeQueue()), 2.0)
    assert out["status"] == "error"
    assert "exceeded 2.0s timeout" in out["error"]
    assert proc.terminate_calls == 1
    assert "elapsed_seconds" in out


# ─── _close_active_resources ──────────────────────────────────────────────────


def test_close_closes_queue_and_process():
    proc = _FakeProcess()
    q = _FakeQueue()
    _RUNNER._close_active_resources(_job(proc, q))
    assert q.closed is True
    assert proc.close_calls == 1


def test_close_swallows_errors():
    # Must not raise even when queue.close()/process.close() themselves error.
    _RUNNER._close_active_resources(_job(_BoomProc(), _BoomQueue()))


def test_close_handles_missing_keys():
    # Must not raise even when the active dict lacks process/queue.
    _RUNNER._close_active_resources({})


# ─── process_datasets_with_watchdog (orchestration loop, fake scheduler) ──────


class _FakeStart:
    """Module-level stand-in for ``_start_digest_process`` (a callable class, not a
    nested function, per the project's no-nested-functions hook). Records started
    jobs in ``started`` and returns fake active-job dicts modelling each scenario."""

    def __init__(self, mode: str = "success", *, error_id: str | None = None):
        self.mode = mode
        self.error_id = error_id
        self.started: list = []

    def __call__(
        self, ctx, dataset_id, input_dir, output_dir, *, position, total, digest_fn
    ):
        if self.mode == "mixed":
            status = "error" if dataset_id == self.error_id else "success"
            q = _FakeQueue(get_nowait={"status": status, "dataset_id": dataset_id})
            proc = _FakeProcess()
        elif self.mode == "slow":
            # Empty fast-path drain + dead process -> slow _collect_finished_process.
            q = _FakeQueue(
                get_nowait_exc=queue_mod.Empty(),
                get={"status": "success", "dataset_id": dataset_id},
            )
            proc = _FakeProcess(alive=(False,))
        elif self.mode == "kbd":
            # KeyboardInterrupt escapes the inner handler -> outer cleanup block.
            q = _FakeQueue(get_nowait_exc=KeyboardInterrupt())
            proc = _FakeProcess(alive=(True, False))
        else:  # "success"
            q = _FakeQueue(get_nowait={"status": "success", "dataset_id": dataset_id})
            proc = _FakeProcess()
        job = _job(proc, q, dataset_id=dataset_id)
        self.started.append(job)
        return job


def test_watchdog_loop_collects_all_success(monkeypatch, tmp_path):
    """Patch the spawner so the loop drives fake jobs that immediately succeed."""
    monkeypatch.setattr(_RUNNER, "_start_digest_process", _FakeStart("success"))

    ids = ["dsA", "dsB", "dsC"]
    results, stats = _RUNNER.process_datasets_with_watchdog(
        ids,
        tmp_path,
        tmp_path,
        workers=2,
        dataset_timeout=60.0,
        digest_fn=_unused_digest_fn,
    )

    assert {r["dataset_id"] for r in results} == set(ids)
    assert stats["success"] == 3
    assert stats["error"] == 0
    assert len(results) == 3


def test_watchdog_loop_accounts_mixed_statuses(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _RUNNER, "_start_digest_process", _FakeStart("mixed", error_id="bad")
    )

    _results, stats = _RUNNER.process_datasets_with_watchdog(
        ["ok1", "bad", "ok2"],
        tmp_path,
        tmp_path,
        workers=3,
        dataset_timeout=60.0,
        digest_fn=_unused_digest_fn,
    )
    assert stats["success"] == 2
    assert stats["error"] == 1


def test_watchdog_loop_dispatches_dead_worker_to_slow_collect(monkeypatch, tmp_path):
    """Empty fast-path drain + a dead process routes through _collect_finished_process,
    whose slow path enriches the result with ``elapsed_seconds`` (unlike the fast path)."""
    monkeypatch.setattr(_RUNNER, "_start_digest_process", _FakeStart("slow"))

    results, stats = _RUNNER.process_datasets_with_watchdog(
        ["dsZ"],
        tmp_path,
        tmp_path,
        workers=1,
        dataset_timeout=60.0,
        digest_fn=_unused_digest_fn,
    )
    assert stats["success"] == 1
    assert results[0]["dataset_id"] == "dsZ"
    assert "elapsed_seconds" in results[0]


def test_watchdog_loop_keyboardinterrupt_cleans_up_active_jobs(monkeypatch, tmp_path):
    """A KeyboardInterrupt mid-loop terminates + closes every still-active job, then re-raises."""
    fake = _FakeStart("kbd")
    monkeypatch.setattr(_RUNNER, "_start_digest_process", fake)

    with pytest.raises(KeyboardInterrupt):
        _RUNNER.process_datasets_with_watchdog(
            ["k1", "k2"],
            tmp_path,
            tmp_path,
            workers=2,
            dataset_timeout=60.0,
            digest_fn=_unused_digest_fn,
        )

    assert len(fake.started) == 2
    for job in fake.started:
        assert job["process"].terminate_calls == 1
        assert job["queue"].closed is True


# ─── real subprocess: spawn + pickle + dependency injection ───────────────────


@_needs_di
def test_watchdog_real_spawn_success_and_error(tmp_path):
    results, stats = _RUNNER.process_datasets_with_watchdog(
        ["ok-1", "err-1"],
        tmp_path,
        tmp_path,
        workers=2,
        dataset_timeout=60.0,
        digest_fn=fake_workers.digest_dispatch,
    )
    by_id = {r["dataset_id"]: r for r in results}
    # The worker's own dict (ok-1) and the worker-built error result (err-1) flow
    # through the fast get_nowait() drain verbatim — that path does not add
    # ``elapsed_seconds`` (only the slow _collect/_timeout paths do).
    assert by_id["ok-1"]["status"] == "success"
    assert by_id["ok-1"]["marker"] == "from-fake"
    assert by_id["err-1"]["status"] == "error"
    assert "boom-err-1" in by_id["err-1"]["error"]
    assert stats["success"] == 1
    assert stats["error"] == 1


@_needs_di
def test_watchdog_real_spawn_nondict_result_becomes_error(tmp_path):
    results, stats = _RUNNER.process_datasets_with_watchdog(
        ["nondict-1"],
        tmp_path,
        tmp_path,
        workers=1,
        dataset_timeout=60.0,
        digest_fn=fake_workers.digest_dispatch,
    )
    assert stats["error"] == 1
    # Exact worker-boundary message — distinguishes the in-worker isinstance check
    # from the orchestrator-side coercion (both contain the loose "expected dict").
    assert "digest_dataset returned list, expected dict" in results[0]["error"]


@pytest.mark.slow
@_needs_di
def test_watchdog_real_spawn_timeout_kills_worker(tmp_path):
    results, stats = _RUNNER.process_datasets_with_watchdog(
        ["slow-1"],
        tmp_path,
        tmp_path,
        workers=1,
        dataset_timeout=2.0,
        digest_fn=fake_workers.digest_dispatch,
    )
    assert stats["error"] == 1
    assert "timeout" in results[0]["error"]


@pytest.mark.slow
@_needs_di
def test_watchdog_production_main_module_pickle_path():
    """Exercise the exact production shape: a ``digest_fn`` defined in a
    run-as-``__main__`` script, pickled and resolved across the spawn boundary
    (this is how ``python 3_digest.py`` injects ``digest_dataset``)."""
    probe = INGEST_DIR / "tests" / "_helpers" / "watchdog_main_probe.py"
    proc = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(INGEST_DIR),
    )
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "PROBE_RESULT: OK" in proc.stdout
