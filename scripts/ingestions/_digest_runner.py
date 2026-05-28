"""Digest runner Seam: per-dataset subprocess isolation + per-operation timeouts.

Extracted from ``3_digest.py``. Owns the multiprocessing watchdog that runs each
dataset's digest in its own child process and enforces a per-dataset timeout,
escalating SIGTERM -> SIGKILL on a stalled worker.

The per-dataset work is dependency-injected as ``digest_fn`` so this module stays
free of any BIDS/digest coupling and of an import cycle with ``3_digest.py``.
(Under the ``spawn`` start method the child still re-runs ``__main__`` =
``3_digest.py`` to bootstrap and to unpickle ``digest_fn`` — the win here is a
clean, acyclic module boundary, not a cheaper re-import.) ``3_digest.py`` passes
its ``digest_dataset`` as ``digest_fn``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm import tqdm

__all__ = ["process_datasets_with_watchdog"]

WORKER_POLL_INTERVAL_SECONDS = 1.0
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5.0
RESULT_QUEUE_TIMEOUT_SECONDS = 5.0

# A per-dataset digest callable: (dataset_id, input_dir, output_dir) -> result dict.
DigestFn = Callable[[str, Path, Path], Any]


def _worker_error_result(
    dataset_id: str,
    error: str,
    *,
    elapsed_seconds: float | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    """Build a standardised error result dict for batch summary accounting."""
    result: dict[str, Any] = {
        "status": "error",
        "dataset_id": dataset_id,
        "error": error,
    }
    if elapsed_seconds is not None:
        result["elapsed_seconds"] = round(elapsed_seconds, 3)
    if traceback_text:
        result["traceback"] = traceback_text
    return result


def _digest_dataset_worker(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
    result_queue: Any,
    digest_fn: DigestFn,
) -> None:
    """Subprocess entry: digest one dataset and put the result dict (or error) on the queue."""
    try:
        try:
            result = digest_fn(dataset_id, input_dir, output_dir)
            if not isinstance(result, dict):
                result = _worker_error_result(
                    dataset_id,
                    f"digest_dataset returned {type(result).__name__}, expected dict",
                )
        except BaseException as exc:  # noqa: BLE001 — worker boundary: parent decides retry/shutdown
            result = _worker_error_result(
                dataset_id,
                f"{type(exc).__name__}: {exc}",
                traceback_text=traceback.format_exc(),
            )
        result_queue.put(result, timeout=RESULT_QUEUE_TIMEOUT_SECONDS)
    finally:
        logging.shutdown()


def _start_digest_process(
    ctx: mp.context.BaseContext,
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
    *,
    position: int,
    total: int,
    digest_fn: DigestFn,
) -> dict[str, Any]:
    """Spawn a child process for one dataset and return the active-job dict."""
    result_queue = ctx.Queue(maxsize=100)
    process = ctx.Process(
        target=_digest_dataset_worker,
        args=(dataset_id, input_dir, output_dir, result_queue, digest_fn),
        name=f"digest-{position}-{dataset_id}",
    )
    process.start()
    return {
        "dataset_id": dataset_id,
        "position": position,
        "total": total,
        "process": process,
        "queue": result_queue,
        "started_at": time.monotonic(),
    }


def _close_active_resources(active: dict[str, Any]) -> None:
    """Release process and queue handles (best-effort, non-blocking)."""
    result_queue = active.get("queue")
    if result_queue is not None:
        try:
            result_queue.close()
        except (OSError, ValueError, AttributeError):
            pass

    process = active.get("process")
    if process is not None:
        try:
            process.close()
        except (OSError, ValueError, AttributeError):
            pass


def _terminate_active_process(active: dict[str, Any], reason: str) -> None:
    """Terminate a child process, escalating to SIGKILL if SIGTERM is ignored."""
    process = active["process"]
    dataset_id = active["dataset_id"]

    if process.is_alive():
        tqdm.write(f"[digest] terminating {dataset_id}: {reason}")
        process.terminate()
        process.join(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)

    if process.is_alive():
        tqdm.write(f"[digest] killing {dataset_id}: terminate did not exit")
        process.kill()
        process.join(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)

    if process.is_alive():
        tqdm.write(f"[digest] warning {dataset_id}: process still alive after kill")


def _collect_finished_process(active: dict[str, Any]) -> dict[str, Any]:
    """Collect one completed child result from the queue with an explicit timeout."""
    process = active["process"]
    dataset_id = active["dataset_id"]
    elapsed = time.monotonic() - active["started_at"]

    try:
        result = active["queue"].get(timeout=RESULT_QUEUE_TIMEOUT_SECONDS)
    except queue.Empty:
        result = _worker_error_result(
            dataset_id,
            f"worker exited without returning a result (exitcode={process.exitcode})",
            elapsed_seconds=elapsed,
        )
    except (OSError, EOFError, ValueError) as exc:
        result = _worker_error_result(
            dataset_id,
            f"failed to collect worker result: {type(exc).__name__}: {exc}",
            elapsed_seconds=elapsed,
        )

    process.join(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
    if process.is_alive():
        _terminate_active_process(active, "process still alive after result collection")

    if isinstance(result, dict):
        result.setdefault("dataset_id", dataset_id)
        result.setdefault("elapsed_seconds", round(elapsed, 3))
        return result

    return _worker_error_result(
        dataset_id,
        f"worker returned {type(result).__name__}, expected dict",
        elapsed_seconds=elapsed,
    )


def _timeout_active_process(
    active: dict[str, Any],
    dataset_timeout: float,
) -> dict[str, Any]:
    """Kill a stalled dataset worker and return an error summary."""
    elapsed = time.monotonic() - active["started_at"]
    dataset_id = active["dataset_id"]
    _terminate_active_process(
        active,
        f"dataset exceeded {dataset_timeout:.1f}s timeout",
    )
    return _worker_error_result(
        dataset_id,
        f"dataset exceeded {dataset_timeout:.1f}s timeout",
        elapsed_seconds=elapsed,
    )


def process_datasets_with_watchdog(
    dataset_ids: list[str],
    input_dir: Path,
    output_dir: Path,
    *,
    workers: int,
    dataset_timeout: float,
    digest_fn: DigestFn,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Process datasets with per-dataset subprocess isolation and per-operation timeouts."""
    total = len(dataset_ids)
    max_workers = max(1, workers)
    ctx = mp.get_context()
    active: dict[int, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    stats = {"success": 0, "error": 0, "skipped": 0, "empty": 0}
    next_index = 0

    with tqdm(total=total, desc="Digesting") as progress:
        try:
            while next_index < total or active:
                while next_index < total and len(active) < max_workers:
                    dataset_id = dataset_ids[next_index]
                    active_job = _start_digest_process(
                        ctx,
                        dataset_id,
                        input_dir,
                        output_dir,
                        position=next_index + 1,
                        total=total,
                        digest_fn=digest_fn,
                    )
                    active[id(active_job["process"])] = active_job
                    next_index += 1

                finished: list[tuple[int, dict[str, Any]]] = []
                now = time.monotonic()
                for key, active_job in list(active.items()):
                    dataset_id = active_job["dataset_id"]
                    process = active_job["process"]
                    elapsed = now - active_job["started_at"]

                    # Drain queue non-blocking first to prevent deadlock on a full queue.
                    result_queue = active_job["queue"]
                    try:
                        result = result_queue.get_nowait()
                        print(
                            f"[QUEUE] Got result from {dataset_id} after {elapsed:.1f}s",
                            flush=True,
                        )
                        finished.append((key, result))
                        process.join(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
                        continue
                    except (queue.Empty, OSError, EOFError, ValueError) as e:
                        print(
                            f"[QUEUE] No result yet for {dataset_id}: {type(e).__name__}",
                            flush=True,
                        )

                    if process.is_alive():
                        if elapsed > dataset_timeout:
                            result = _timeout_active_process(
                                active_job, dataset_timeout
                            )
                            finished.append((key, result))
                        continue

                    result = _collect_finished_process(active_job)
                    finished.append((key, result))

                if not finished:
                    time.sleep(WORKER_POLL_INTERVAL_SECONDS)
                    continue

                for key, result in finished:
                    active_job = active.pop(key, None)
                    if active_job is not None:
                        _close_active_resources(active_job)
                    results.append(result)
                    status = result.get("status", "error")
                    stats[status] = stats.get(status, 0) + 1
                    if status == "error":
                        tqdm.write(
                            f"[digest] error {result.get('dataset_id')}: "
                            f"{result.get('error', 'unknown error')}"
                        )
                    progress.update(1)
        except KeyboardInterrupt:
            for active_job in list(active.values()):
                _terminate_active_process(active_job, "interrupted")
                _close_active_resources(active_job)
            raise

    return results, stats
