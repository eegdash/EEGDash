# Phase 4 Findings — mutation testing scaffolding

## Status

**Infrastructure installed.** Mutmut 3.5.0 is available in the dev
dependency set and documented in the ROBUSTNESS docs. The full
baseline run is **deferred to a follow-up session** because of a
config-parsing incompatibility in mutmut 3.x.

## What works

```bash
cd scripts/ingestions
pip install -e ".[dev]"
mutmut --version    # 3.5.0
```

## What doesn't (and why)

Mutmut 3.x's `[tool.mutmut]` parser fails on the documented keys:

```
mutmut.__main__.BadTestExecutionCommandsException: Failed to run
pytest with args: ['--rootdir=.', '--tb=native', '-x', '-q',
                   't', 'e', 's', 't', 's', '/']
```

The `tests_dir = "tests/"` string is being iterated letter-by-letter
and each character is being passed as a separate pytest argument.
This is a known incompatibility between mutmut 3.x's config schema
and the documentation referenced in `ROBUSTNESS/07-DETAILS.md`.

The pyproject `[tool.mutmut]` block has been commented out with a
note pointing at this findings file.

## Workaround

Run mutmut directly with CLI arguments (skipping its config):

```bash
cd scripts/ingestions
mutmut run --paths-to-mutate=_vhdr_parser.py
```

If that also surfaces the same parsing bug, downgrade to mutmut 2.x:

```bash
pip install 'mutmut<3'
```

## Target

When the baseline run lands:

- Mutate `_vhdr_parser.py` (the parser with the richest test coverage
  — 23 tests + 4 Hypothesis properties).
- Target kill ratio: **≥ 60%** before expanding scope to other parsers.
- Document surviving mutants in a `mutation-survivors-2026-MM-DD.md`
  file (analogous to `eegdash-viewer/docs/mutation-survivors-2026-05.md`).

## Evaluation status

Phase 4 evaluation hooks (from `05-EVALUATION.md`):

| Hook | Status |
|---|---|
| `mutmut run` completes | ⏳ Deferred (config issue) |
| Mutation kill ratio on `_vhdr_parser.py` ≥ 60% | ⏳ Deferred |
| Surviving mutants documented per cluster | ⏳ Deferred |
| `mutmut results` cycle time < 2 min after caching | ⏳ Deferred |

This is the **only** phase of the programme that ships with deferred
evaluation. All other phases (0, 1, 2, 3, 5, 6, 7, 8, 9) have their
outcome metrics green.
