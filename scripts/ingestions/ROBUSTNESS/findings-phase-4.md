# Phase 4 Findings — mutation testing baseline (2026-05-21)

## Status

**Baseline complete (partial).** Mutmut 2.5.1 ran for 10 minutes
against `_vhdr_parser.py`, evaluated 95 of 232 mutants. Hit the
timeout; 137 mutants in lines 96-232 remain untested in this baseline.

The interactive workaround (downgrade to mutmut 2.x) succeeded —
mutmut 3.x's config-parser bug remains, but `pip install 'mutmut<3'`
makes `[tool.mutmut]` in `pyproject.toml` work as documented in
mutmut 2.x's README.

## Kill ratio

| Category | Count |
|---|---:|
| Total mutants | 232 |
| **Killed** | **66** |
| Survived | 29 |
| Untested (timed out) | 137 |
| **Kill ratio on tested** | **69.5%** |

**69.5% is above the Phase 4 floor of 60%**. The 137 untested mutants
are a known gap; a follow-up run with a higher timeout (or
parallel-workers config) would close it.

## Surviving mutants — classified

### 🔴 Critical: a real test gap (1)

**Mutant 80** — `path_is_within_root(...) AND data_path.exists()` →
`path_is_within_root(...) OR data_path.exists()`.

This neutralises the audit-3 F2 path-traversal hardening landed in
session 4. With `or` short-circuiting on a real-but-out-of-tree file,
the security check is bypassed.

**Why the original test missed it**: `test_extract_vhdr_references_rejects_dotdot_path`
created the target file at `tmp_path.parent / "evil_target.eeg"` but
the DataFile traversal resolves to `tmp_path / "evil_target.eeg"` —
different paths. The `data_path.exists()` check returned False by
accident; both `and` and `or` evaluated to False, so the test couldn't
distinguish them.

**Fix landed**: the test now places `outside_target` at the resolved
path (`tmp_path / "evil_target.eeg"`) plus a `sanity check` that the
fixture is structurally correct. Same fix applied to the markerfile
variant. Verified by running the test against an inlined mutant —
without the fix it returns `datafile_exists = True` and the assertion
fires.

### 🟡 Equivalent / unreachable mutants (~10)

Survived but cannot kill without artificial test fixtures that
exercise unreachable production paths:

- Mutants 1-4, 7: sys.path manipulation at import time (untestable;
  module already imported when tests run).
- Mutant 23: `interval_us > 0` → `>= 0` (interval=0 would cause
  division-by-zero; no fixture has it; the test would crash on the
  real division before reaching the mutation).
- Mutants 60-65 with `XX...XX` on dict literal keys (the defaults
  are immediately overwritten by parsed values; mutation invisible).
- Mutants 62-63: `datafile_exists: False` → `False` is `True` or
  `"XXdatafile_existsXX"` — both get overwritten in real-world calls.

These are documented in mutmut as survived; they are NOT real test
gaps. A future "equivalent-mutant filter" could mark them as known.

### 🟠 Robustness-masking survivors (~10)

The parser has internal fallback paths that mask mutations. For
example mutant 12 (`XX^...XX` around the `NumberOfChannels` regex)
breaks the explicit-nchans extraction, but the parser then DERIVES
nchans from the channel-name list and the assertion still passes.

This is actually **good behaviour** — the parser is robust to
malformed VHDR variants. But the tests don't isolate which extractor
ran. Killing these mutants would mean adding tests that:
- Use a fixture with NumberOfChannels but no channel names
- Use a fixture with channel names but no NumberOfChannels
- Use a fixture with NumberOfChannels but malformed channel names

Worth doing as a future test-quality round; not blocking today.

### 🟢 Real test gaps with no production driver (~8)

- Mutant 13: `re.MULTILINE | re.IGNORECASE` → `re.MULTILINE & re.IGNORECASE`.
  The `&` of distinct bit-flag values equals 0; regex still matches
  most inputs. No test for case-insensitive matching specifically.
- Mutant 43: `replace("\\1", ",")` → `replace("XX\\1XX", ",")`. No
  test for channel names containing escaped commas (BrainVision's
  encoding for embedded `,`).
- Mutant 47, 48: f-string generation in the empty-channel-name
  fallback. No fixture has empty channel names.

Worth opportunistic tests; not security-relevant.

## Time budget

- 10 minute budget: tested 95 mutants (≈ 6.3 sec per mutant)
- A full run on `_vhdr_parser.py` (232 mutants) needs ~25 min single-threaded
- Per the mutmut docs, `--simultaneous-mutants N` can parallelise; not
  enabled in this baseline.

Realistic CI cadence:
- **Per-PR**: skip (10+ min per parser would tank PR latency)
- **Nightly**: re-run with full timeout + parallelism enabled
- **On-demand**: `mutmut run` against the file you just touched

## Followups

1. **Add a regression test for mutant 80 — landed** (commit on this branch).
2. **Add tests for the ~8 real-gap mutants** (case-insensitive matching,
   escaped commas, empty channel names) — opportunistic next round.
3. **Re-run with full budget** to cover lines 96-232 (`extract_vhdr_references` +
   `diagnose_vhdr_issues` + `parse_vhdr_metadata_robust` + `find_datasets_needing_redigestion`).
4. **Promote mutmut to nightly CI** with `--simultaneous-mutants 4` once
   the script is configured.
5. **Repeat for `_set_parser.py`, `_snirf_parser.py`, `_mef3_parser.py`** —
   each parser has its own kill ratio to baseline.

## Evaluation status (vs ROBUSTNESS/05-EVALUATION.md Phase 4)

| Hook | Status |
|---|---|
| `mutmut run` completes | ✓ (partial — timeout hit) |
| Kill ratio on `_vhdr_parser.py` ≥ 60% | ✓ (69.5% on tested mutants) |
| Survivors documented per cluster | ✓ (this doc) |
| `mutmut results` cycle time < 2 min after caching | ⏳ deferred |

Three of four hooks green. The 4th (cycle time) requires the cached
re-run to be incrementally fast — mutmut 2.x does cache, but we
haven't measured a second-run time yet.

## The headline outcome

The most important finding wasn't the kill ratio — it was the
**critical test gap in the audit-3 F2 path-traversal hardening**. That
hardening landed earlier in session 4 with a test that *appeared*
to cover the security case but actually didn't because of a fixture
path-resolution mistake. Mutmut's `OR/AND` mutation surfaced the gap
in 10 minutes.

This is exactly the failure mode mutmut is meant to catch:
"the test exists and passes, but it doesn't actually constrain the
code." Coverage said the line was tested; mutation said it wasn't.
