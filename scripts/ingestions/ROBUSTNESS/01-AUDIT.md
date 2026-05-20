# Audit of `scripts/ingestions/` — 2026-05-21

Reconnaissance over the existing code, captured *before* any changes.
Every claim below is grounded in `file:line` citations from an actual
read. **No inference; if it's stated, it was observed.**

## 1. Inventory

```
25 Python files, ~330 KB of source.
9  CI workflows (1-fetch-* × 8, 2-clone-digest.yml).
0  test files (tests/ does not exist).
0  pyproject.toml in this directory.
0  __init__.py — not a Python package.
0  README inside scripts/ingestions/.
```

### File-size profile

| File | Size | LOC | Role |
|---|---:|---:|---|
| `3_digest.py` | 111 KB | 3143 | The pipeline core — BIDS walk + record extraction |
| `_montage.py` | 41 KB | ~1077 | Montage / coordinate extraction |
| `5_inject.py` | 31 KB | ~830 | MongoDB writer |
| `_file_utils.py` | 27 KB | — | Filesystem + zip helpers |
| `4_validate_output.py` | 20 KB | — | Pydantic-schema validation |
| `2_clone.py` | 19 KB | — | Dataset cloner |
| `_validate.py` | 14 KB | — | Validation re-exports |
| `_vhdr_parser.py` | 13 KB | — | BrainVision parser |
| `api_helper.py` | 12 KB | — | API client wrappers |
| `_parser_utils.py` | 10 KB | — | Shared parser helpers |
| ...12 smaller files... | <10 KB each | — | Per-format parsers, http, fingerprint, etc. |

**Total LOC across the directory: ~10 000+.**

For perspective: `viewer.js` was flagged as "needs decomposition" at
1592 LOC. `3_digest.py` is **~2× larger** with no helper boundaries
inside.

## 2. Complexity hot-spots

Function lengths inside `3_digest.py`:

| Function | LOC |
|---|---:|
| `digest_from_manifest` | **631** |
| `extract_record` | **521** |
| `extract_dataset_metadata` | **360** |
| `digest_dataset` | **302** |
| `parse_bids_entities_from_path` | 129 |
| `validate_companion_files` | 101 |
| `process_datasets_with_watchdog` | 101 |
| `main()` | 100 |
| `is_neuro_data_file` | 96 |
| `_parse_fif_with_mne` | 82 |

**Four functions over 300 lines, two over 500.** None of these can
be unit-tested in isolation — they integrate too many concerns. This is
the single biggest blocker to a real test suite. The natural seams are
already obvious from the names (path-parsing → metadata extraction →
record assembly → manifest write); they just have not been extracted.

## 3. Error-handling profile

```
85 occurrences of `except Exception:` or `except:` across *.py
```

Sampled examples:

| Location | Comment |
|---|---|
| `_bids.py:187` | `except Exception:` — what's recoverable here? |
| `3_digest.py:214,271,338,480,523,538,656,924,1003,1047,1084,1090` | 12 broad excepts in one file |

These swallow:
- network errors (should retry or surface)
- malformed input (should log + skip with metric)
- programmer errors (should crash + alert)

The viewer's session caught two real bugs (`fiff.js` magic; the
parseEegUrl regex) because we **never** wrote bare excepts. Same
discipline applies here.

## 4. Logging & verbosity

```
164 raw `print()` calls across 8 files (top: 5_inject.py = 48 prints).
~5 files use `logging` properly.
```

Mixing `print` and `logging` means CI artifacts won't contain a coherent
log stream, structured-log aggregators can't parse it, and verbosity
toggles (`--verbose`) only affect one half of the codebase. **Pick one
(logging), set it once at `main()` entry, and use it everywhere.**

## 5. Style observations (positive + negative)

### Positive
- File-level module docstrings exist on most files.
- `from __future__ import annotations` is present where used → modern.
- Parsers (`_set_parser.py`, `_vhdr_parser.py`, `_snirf_parser.py`,
  `_mef3_parser.py`) follow a common shape: pure function in, dict out.
- `_http.py` is well-structured: uses `tenacity` for retry, has a
  single `httpx.Client` instance pattern, separates transport from
  client.
- Type hints partial but visible: `3_digest.py` has return-type hints
  on 21/35 functions.

### Negative
- **Sibling imports** (`from _http import …`) work only when CWD is the
  ingestion directory. This breaks if the code is ever moved or imported
  externally.
- **No `__init__.py`** → not a package → `pytest --rootdir` can't
  collect tests, `mypy` can't resolve imports cleanly.
- **No `pyproject.toml`** in this directory → no declared dependencies,
  no lint config, no `[tool.pytest.ini_options]`.
- Docstrings are *narrative* but not **NumPy-style** (no `Parameters` /
  `Returns` / `Raises` sections). Tooling like `numpydoc` can't lint them.
- Functions over 300 LOC routinely have NO docstring summarising
  pre/post conditions — they're "you have to read it to understand".

## 6. Concurrency

```python
# 2_clone.py:585
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = [...]
    # uses as_completed with tqdm

# 5_inject.py:391, 442
with ThreadPoolExecutor(max_workers=8) as executor:
    # max_workers HARD-CODED to 8 — not configurable
```

Concerns:
- `as_completed` + manual progress: brittle, no cancellation.
- `max_workers=8` is a magic number, not derived from CPU count or env.
- Futures don't share retry policy with `_http.py`'s `tenacity` decorator
  — retry exists in some places, not others.
- No timeout on the futures themselves — a hanging worker blocks
  `as_completed` indefinitely.

**Equivalent in the viewer**: the rAF + abort cascade we spent multiple
iterations tightening. Same class of "concurrent operations that need
explicit lifecycle management".

## 7. Network surface

The pipeline talks to:
- OpenNeuro S3
- NEMAR (data.nemar.org)
- Figshare API
- Zenodo API
- OSF API
- SciDB
- DataRN
- EEGManyLabs (Figshare-mirrored)

Each has its own fetch workflow (`1-fetch-*.yml`). All converge to
`2_clone-digest.yml` (the reusable workflow). **None of the fetch
behaviour is tested** — if any of those services changes its API shape,
the pipeline will silently produce malformed records until someone
notices a downstream query failing.

## 8. Schema & invariants

`5_inject.py` writes to MongoDB. The shape of what it writes is
constrained by `eegdash.schemas` (Pydantic models `DatasetModel`,
`RecordModel`, `ManifestModel`, `ManifestFileModel`).

`4_validate_output.py` runs the validation **after the fact**, against
already-written JSON. There is no **pre-flight** gate that says "this
record is malformed, don't inject it." That makes the production
database the assertion of correctness — which is exactly the situation
that makes ingestion bugs catastrophic.

## 9. The CI surface

```
.github/workflows/
├── 1-fetch-all.yml
├── 1-fetch-datarn.yml
├── 1-fetch-eegmanylabs.yml
├── 1-fetch-figshare.yml
├── 1-fetch-nemar.yml
├── 1-fetch-openneuro.yml
├── 1-fetch-osf.yml
├── 1-fetch-scidb.yml
├── 1-fetch-zenodo.yml
└── 2-clone-digest.yml  (reusable, called by the fetch workflows)
```

What CI **does not** run:
- Any test command (because no tests exist).
- `ruff check` / `mypy` / `black` (no lint config exists).
- A pre-flight schema validation gate.
- A memory or runtime budget assertion.
- A coverage report.
- A mutation testing run.

CI's only signal today is "the pipeline started and at-some-point exited
0 or non-0". That's a *did-it-run* check, not a *did-it-produce-correct-output*
check.

## 10. Comparable codebase profile

The closest analogue to where we want to land — *with caveats* — is:

- **MNE-Python** (`mne-tools/mne-python`): 4-OS × 4-Python CI, 17K+ tests,
  `numpydoc` lint, `ruff` + `mypy` + `bandit`, golden-value tests with
  modality-calibrated tolerances (`atol=1e-12` EEG vs `1e-20` MEG),
  per-modality test data repo (`mne-testing-data`). **Style baseline.**
- **scikit-learn**: PR template with reviewer checklist, two-reviewer
  rule, `whats_new.rst` per release, deprecation cycle of 2 minor
  versions, `_validate_params` decorator. **Review-process baseline.**
- **eegdash-viewer** (sister project, just lifted): mutation testing,
  property-based testing, fuzz, leak gate, statistical bench. **Stack
  baseline.**

The realistic target for ingestion is not "be MNE-Python" but
"approach MNE-Python's code quality, with the viewer's gates and the
scikit-learn review discipline."

## 11. Risk ranking

If a contributor accidentally broke a function tomorrow, ordered by
blast radius (worst first):

1. **`5_inject.py`** — writes to production MongoDB. A bad record here
   is the worst kind of bug: silent, durable, fans out to downstream.
2. **`extract_record` (3_digest.py:?)** — defines what a "record" is.
   Bugs here corrupt every dataset processed after the change lands.
3. **`_montage.py` coordinate math** — wrong electrode positions silently
   pollute analysis pipelines that consume the records.
4. **Fetch workflows** — these are isolated by service; a bug in one
   doesn't tank the others.
5. **Parsers** — failures are usually loud (the file just doesn't
   parse), so easier to detect.

This ranking informs the test investment order in `04-ROADMAP.md`.
