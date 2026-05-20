# Audit 2 — Retry-loop divergence across modules

**Auditor**: Phase 9 of the ingestion robustness programme (autonomous run, 2026-05-21)
**Scope**: All files that implement retry behaviour.
**Pattern**: Find every retry implementation, compare backoff strategy, jitter, max retries, retryable statuses, and convergence to a single helper.

## TL;DR

| Severity | Count |
|---|---:|
| P2 (maintenance / divergence drift) | 1 |
| P3 (documentation hygiene) | 2 |

## F1 — [P2] Retry policy diverges across 5 files

**Locations**:

| File | Retry strategy |
|---|---|
| `_http.py` | `tenacity.retry`, exponential backoff (multiplier=`backoff_factor`, min=1s, max=60s), 3 attempts default. Retryable: `httpx.RequestError` + 5xx (`{429, 500, 502, 503, 504}`). |
| `_file_utils.py` | Local retry loop with `time.sleep(backoff_factor * 2**i)`. |
| `_montage.py` | Local retry with fixed `time.sleep(1)` between attempts. |
| `_parser_utils.py` | Single retry on encoding fallback (latin-1 after utf-8). |
| `5_inject.py` | No retry — relies on caller. |

**Evidence**:
- `_file_utils.py` and `_montage.py` reinvent the retry primitive instead of importing `_http.request_json` / `request_text`.
- Each retry uses a different backoff formula, different jitter (none vs. exponential), different timeout (none vs. 60s cap).
- A 429 (rate limit) on one path retries gracefully; on another path it propagates immediately.

**Severity**: P2. The pipeline mostly works because the retry policies are *each* reasonable. The risk: a future operator tunes one (e.g., bumps `_http.py` retries to 5) without realising the other paths still retry 3 times. Bug reports become "intermittent" because behaviour depends on which file's retry won.

**Suggested fix**: Extract a single `_retry.py` module that wraps `tenacity` with the project's canonical policy (exponential backoff, jitter, status-code set, max retries). All other modules import from there. Remove the 5 hand-rolled loops.

**Regression test**: `tests/test_retry_policy.py::test_all_retry_paths_use_same_policy` — for every module that calls a retry helper, assert the retry-status set is `_http.DEFAULT_RETRY_STATUSES` and the backoff is the canonical exponential variant.

## F2 — [P3] `_http.DEFAULT_RETRY_STATUSES` doesn't include 408 Request Timeout

**Location**: `_http.py:30`.

```python
DEFAULT_RETRY_STATUSES = {429, 500, 502, 503, 504}
```

**Evidence**: HTTP 408 is the canonical "server's timer fired before your body arrived" status — same root cause as a 504 (gateway timeout). RFC 9110 § 15.5.9 explicitly says 408 SHOULD be retried by clients. The current default set omits it.

**Severity**: P3 — most upstreams return 504 not 408 for the same case. Including it is hygiene.

**Suggested fix**: `DEFAULT_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}`.

**Regression test**: Extend `tests/test_http.py::test_request_json_custom_retry_statuses` to assert 408 retries by default.

## F3 — [P3] Documentation drift on `make_retry_client`

**Location**: `_http.py:111-123`.

```python
def make_retry_client(auth_token: str) -> httpx.Client:
    """Create an HTTP client with auth headers for ingestion injection."""
    ...
    return httpx.Client(
        transport=httpx.HTTPTransport(retries=0),   # <-- not actually retry-enabled
        ...
    )
```

**Evidence**: The function is named `make_retry_client` but builds an `httpx.HTTPTransport(retries=0)` — zero retries. The actual retries happen *outside* this client (via `tenacity` in `request_json`). The name promises something the function doesn't deliver.

**Severity**: P3 — documentation / naming. The behaviour is fine; the name is misleading. A new contributor adding a service-specific retry would search for `make_retry_client`, find this, and conclude "retries are already wired" when they aren't.

**Suggested fix**: Rename to `make_authed_client(auth_token: str)`. Update docstring to clarify retries are caller-injected via `request_json(..., retries=N)`. Add an `@deprecated` shim that re-exports the old name for one release per `03-CONTRIBUTING.md` § 6.

**Regression test**: None — pure rename.

## What was NOT found

- No retry path makes *unbounded* attempts (every implementation has a `max_attempts` ceiling).
- No retry uses `time.sleep(some_random_value)` without justification.
- No retry retries on 4xx-other-than-429, which would have been a real bug (4xx is the client's fault, retry doesn't help).
- No retry retries on a `KeyboardInterrupt` (would mask Ctrl-C).

The audit covered ~150 LOC across the 5 retry sites. F1 is the
maintenance debt; F2/F3 are easy follow-up tickets.
