# Pipeline config pattern (C6.5 / C7 / C8)

All four main-stage scripts use the same Pydantic-settings + argparse
pattern. This doc records why, how, and where future config work
should slot in.

## The four configs

| Stage | Script | Config module | Fields | Validators |
|---|---|---|---:|---|
| 2 | `2_clone.py` | `_clone_config.py` | 9 | bounds + sources |
| 3 | `3_digest.py` | `_digest_config.py` | 6 | bounds |
| 4 | `4_validate_output.py` | `_validate_config.py` | 5 | input-must-exist |
| 5 | `5_inject.py` | `_inject_config.py` | 14 | only-* mutex + skip/only contradict + bounds |

Stage 1 (`1_fetch_sources/*`) is NOT consolidated — each per-source
script has its own argparse layout that varies meaningfully (e.g.
figshare needs an API key; zenodo doesn't). A shared base would
either compose with per-source mixins or stay an open question.
Documented as future work below.

## The pattern, abstracted

```
_<stage>_config.py
├─ <Stage>Config(BaseSettings)
│  model_config = SettingsConfigDict(env_prefix=EEGDASH_<STAGE>_, extra="ignore")
│  ├─ Fields with descriptions + bounds (ge / le / gt / lt)
│  ├─ Literal types for closed-set fields (database, source names)
│  ├─ @field_validator for per-field invariants
│  ├─ @model_validator for cross-field invariants
│  │  (mutually-exclusive flags, contradiction checks,
│  │   input-must-exist gates)
│  └─ @property accessors for derived booleans
│     (want_records / want_montages — readable instead of
│      ``not args.only_records and not args.only_montages``)
└─ load_<stage>_config_from_argv()
   └─ thin argparse → kwargs bridge
```

The argparse layer is intentionally small — it owns:

- `--help` / `-h` ergonomics
- Short-flag aliases (`-i`, `-v`)
- Per-source choices that need to match the HANDLERS dict
- Bool flags via `action="store_true"`

Everything else (bounds, env var fallback, cross-field validation)
lives in the Pydantic model. The argparse Namespace is filtered to
drop None/False values so the Pydantic Field defaults apply.

## Tests, abstracted

Each stage gets a `tests/test_<stage>_config.py` (or the consolidated
`test_stage_configs.py` for C7 stages 3+4):

```
def test_<stage>_config_defaults(...)
def test_<stage>_config_rejects_<invariant>(...)   # for each @model_validator
def test_<stage>_config_<field>_bounds(...)        # for each ge/le bound
def test_<stage>_argv_parses_all_flags(...)
def test_<stage>_argv_env_var_picked_up(...)
def test_<stage>_argv_validation_error_surfaces(...)
```

**No subprocess harness.** Construct the model directly. The argparse
layer is a thin bridge; subprocess testing would have been the wrong
shape (slow, opaque error messages, hard to assert against).

## Wins this pattern delivered

Across the 4 stages:

| Win | Concrete example |
|---|---|
| Bounds caught at boot | `--workers 0` rejected before any handler runs |
| Type errors caught at boot | `--timeout 1.5` (float) rejected for int field |
| Mutually-exclusive flags as validators | `--only-datasets --only-records` raises ValidationError |
| Env var fallbacks native | `EEGDASH_ADMIN_TOKEN` via `AliasChoices` |
| Source/database typos rejected | `--sources made_up_source` fails at config |
| Tests don't need subprocess | 28 + 18 + 11 = 57 tests run in milliseconds |

## Caveats + future work

### Database list drift risk

`_inject_config.py` declares the valid databases as a `Literal`:

```python
ValidDatabase = Literal[
    "eegdash", "eegdash_dev", "eegdash_archive",
    "eegdash_staging", "eegdash_v1",
]
```

The cluster API has the same list in `api/main.py:Settings.valid_databases`.
**These two lists can drift.** If the API adds `eegdash_v2`, our
`InjectConfig` rejects it. If we add a database in our config, the API
rejects requests to it.

Mitigation options (future work):
1. Fetch valid databases from the API at boot (1 extra HTTP call;
   makes config construction async or blocking-network).
2. Generate the Literal from a shared YAML / JSON in the deploy repo
   that both sides read at build time.
3. Accept the drift and rely on the C6.3 / C6.4 integration tests
   to catch it (current state — works but takes one full PR-fast +
   integration cycle to detect).

The C6.4 stress tests would catch a database-add drift on the cluster
side; the PR-fast tests would catch a database-remove drift on our
side. So drift IS observable; just not at config-construction time.

### Stage 1 fetch scripts (open)

Each per-source fetch script has its own argparse. Consolidating
would need:

- A `FetchConfigBase` with common fields (output_dir, retries, timeout)
- Per-source extensions (`FigshareConfig(FetchConfigBase)` with
  api_key, `ZenodoConfig` without, etc.)
- An entry-point script that dispatches by `--source`

Worth doing if a 10th source lands. Until then, the per-script
argparse blocks are stable and the consolidation cost > benefit.

### Shared base for the 4 done stages

Looking at the 4 configs, there's repetition:

- `input: Path` (clone, digest, validate, inject)
- `output: Path` (clone, digest)
- `datasets: list[str] | None` filter (clone, digest, inject)
- `@model_validator` input-must-exist (clone, digest, validate)

A `PipelineConfigBase(BaseSettings)` with these fields + the
input-must-exist validator would DRY ~15 lines per stage. But:

- Pydantic inheritance + `SettingsConfigDict` has subtle gotchas
  (env_prefix must be overridden in subclass; `extra` settings don't
  always inherit cleanly).
- The 4 stages have slightly different defaults for `input` (each
  stage's stage-1 input), so the base would need either no default
  or a property override.
- Today's 4 modules average ~80 LOC each. A base would shave ~5-10
  LOC per module but add 1 more file to navigate.

**Verdict: skip the base for now.** Three test modules duplicate ~10
lines of boilerplate (sys.path append + `_load_*` helper); DRYing
that has a clearer payoff if a 5th stage joins. Until then, four
flat configs are more readable than a base + four extensions.

### What gets refactored next

In priority order:

1. **Database list source-of-truth** — biggest production risk
   (config drift). Concrete fix: API exposes `/admin/valid-databases`
   read endpoint; `InjectConfig.__init__` calls it once. Or:
   move to shared YAML.
2. **Stage 1 consolidation** — only if a 10th source lands. Real
   driver, not speculative.
3. **CONFIG_BASE** — only if a 5th stage with a main() shows up.

## Cross-references

- `_inject_config.py` — original (C6.5)
- `_validate_config.py` — C7.1
- `_digest_config.py` — C7.2
- `_clone_config.py` — C8 (this commit)
- `ROBUSTNESS/INTEGRATION-TESTING.md` — how the cluster-level test
  catches drift the config-level tests can't
- `ROBUSTNESS/BIDS-GAP-AUDIT.md` — the parallel doc for the Record
  schema enrichment
