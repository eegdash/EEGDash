# Audit 1 — Concurrency in `concurrent.futures` paths

**Auditor**: Phase 9 of the ingestion robustness programme (autonomous run, 2026-05-21)
**Scope**: `2_clone.py`, `3_digest.py`, `5_inject.py` ThreadPoolExecutor blocks.
**Pattern**: Read every block that submits to `ThreadPoolExecutor`, look for shared mutable state, missing timeouts, missing cancellation, broad exception handling around `future.result()`.

## TL;DR

| Severity | Count |
|---|---:|
| P1 (silent data loss risk) | 1 |
| P2 (operational footgun) | 2 |
| P3 (cosmetic / hygiene) | 1 |

## F1 — [P2] `_stats` is mutated from worker threads without a lock

**Location**: `2_clone.py:580-609`, `_stats` is a module-level dict updated by `process_dataset` inside each worker thread.

**Evidence**: 
```python
# 2_clone.py:585-609
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {
        executor.submit(process_dataset, ds, ...): ds for ds in datasets
    }
    pbar = tqdm(as_completed(futures), ..., postfix=_stats)
    for future in pbar:
        try:
            result = future.result()
            results.append(result)
            pbar.set_postfix(_stats)
        except Exception as e:
            results.append({"status": "error", "error": str(e)})
```

`process_dataset` writes to `_stats` (a shared dict) from each worker. CPython's GIL makes individual `dict[key] = value` atomic, but compound updates like `_stats["downloaded"] += 1` are NOT atomic — they're a `LOAD-INC-STORE` sequence that the GIL can interrupt mid-way.

**Trigger**: Two workers completing the same `+= 1` operation within the same GIL tick. Rare but documented. Loss is silent — the progress bar shows a smaller total than actually completed.

**Severity**: P2. The actual data write (the cloned dataset on disk) is durable; only the *report counter* drifts. But operators rely on the counter to know "is the pipeline making progress?" and a confused counter delays incident response.

**Suggested fix**: Replace `_stats["k"] += 1` with `_stats.setdefault("k", AtomicInt()).inc()` (Python lacks AtomicInt natively; `threading.Lock` around the increment is acceptable). The cleanest is to have `process_dataset` return its delta and aggregate single-threaded on the main loop, mirroring the `inserted_count + updated_count` pattern in `5_inject.py:404-407`.

**Regression test**: `tests/test_concurrency.py::test_clone_stats_under_concurrent_load` — run `process_dataset` 100 times concurrently, assert final counter equals 100 deterministically.

## F2 — [P1] Bare `except Exception` around `future.result()` in `2_clone.py`

**Location**: `2_clone.py:608-609`.

```python
except Exception as e:
    results.append({"status": "error", "error": str(e)})
```

**Evidence**: This block catches *any* exception from `future.result()`. Two failure modes are silently merged:

1. `process_dataset` raised a recoverable error (network blip, dataset missing) — fine.
2. `process_dataset` raised an *unrecoverable* programmer error (TypeError, AttributeError on a misshapen response) — should crash + alert.

Currently both produce the same `{"status": "error", "error": "<str>"}` record. The unrecoverable case looks identical to a transient failure in the final report.

**Severity**: P1. A misshapen response from a renamed OpenNeuro API endpoint would be silently swallowed and reported as "error" without the operator knowing the digest pipeline is now broken for *every* dataset (not just one).

**Suggested fix**: Narrow to `except (httpx.RequestError, httpx.HTTPStatusError, BIDSParseError, FileNotFoundError, PermissionError)` and let everything else propagate. This is the same lesson Phase 3 applied to the parsers.

**Regression test**: `tests/test_concurrency.py::test_clone_propagates_programmer_errors` — patch `process_dataset` to raise `AttributeError`, assert the pipeline exits non-zero rather than logging "error".

## F3 — [P2] No timeout on the futures themselves

**Location**: `2_clone.py:585`, `3_digest.py:2966`, `5_inject.py:392`, `5_inject.py:443`.

**Evidence**: Every `ThreadPoolExecutor` block uses `as_completed(futures)` without a `timeout=` argument. If one worker hangs (TCP keepalive never trips, deadlock in mne-bids, etc.), `as_completed` blocks indefinitely. The CI workflow's `timeout-minutes:` is the only protection — a CI worker reports "timed out" without saying *which* dataset hung.

**Severity**: P2 — operational, not correctness. Pipeline can be restarted, but every restart is a 30-minute incident.

**Suggested fix**: Pass `timeout=` to `as_completed()` (sum of per-task timeouts plus a safety margin) and pass `timeout=` to `future.result()`. If a worker times out, cancel it explicitly and log which dataset was responsible.

**Regression test**: `tests/test_concurrency.py::test_hanging_worker_does_not_deadlock_main_thread` — submit a `time.sleep(60)` job to a 1-worker pool with `timeout=2`, assert main thread escapes within 3 seconds.

## F4 — [P3] `max_workers=8` is hard-coded in `5_inject.py`

**Location**: `5_inject.py:392`, `5_inject.py:443`.

**Evidence**: `max_workers=8` is a magic number. CI workers run on 2 vCPUs (GitHub free tier) and 16 vCPUs (self-hosted runners). The same constant is wrong for both.

**Severity**: P3 — cosmetic. The pipeline works at 8 workers everywhere; the loss is a degree of efficiency.

**Suggested fix**: Add a `--workers` CLI flag (default `min(8, (os.cpu_count() or 2) * 2)`). `2_clone.py` already exposes this; mirror the pattern.

**Regression test**: None needed; manual verification.

## What was NOT found

- No `multiprocessing` usage that would have introduced inter-process state.
- No use of `asyncio` event loops that could surface ordering bugs.
- No global locks held across blocking I/O (which would deadlock).
- No usage of `Queue` from the standard library — the futures-based pattern is consistent.

The audit covered ~80 LOC of concurrent code across 3 files. Of the 4
findings, F2 is the one to fix immediately (silent error masking is a
known incident category in the audit's risk ranking).
