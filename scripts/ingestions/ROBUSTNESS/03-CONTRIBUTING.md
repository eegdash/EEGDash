# Contributing to `scripts/ingestions/` — scikit-learn-style review

This is the review process. It is deliberately stricter than the
unwritten one that exists today, because every record this pipeline
writes flows into a production database that downstream consumers
(including `eegdash-viewer`) trust.

Cross-reference:
- [scikit-learn CONTRIBUTING](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md)
- [scikit-learn PR template](https://github.com/scikit-learn/scikit-learn/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
- [scikit-learn dev guide](https://scikit-learn.org/dev/developers/contributing.html)

## 1. Before opening a PR

Run locally:

```bash
# Lint + format (fast, fix-mostly automatic)
ruff check ingestions/
ruff format ingestions/

# Type-check (slow, no auto-fix)
mypy --strict ingestions/

# Tests
pytest ingestions/tests/ -x --tb=short

# Coverage (warning if below threshold)
pytest ingestions/tests/ --cov=ingestions --cov-fail-under=70
```

If any of those fail, **fix or explicitly document** before opening
the PR. A reviewer's time is the scarcest resource.

## 2. PR title

Format: `<type>(<scope>): <subject>` — same as Conventional Commits.

| Type | Use for |
|---|---|
| `feat` | New behaviour (a new parser, a new fetch source) |
| `fix` | Bug fix (must include a regression test) |
| `refactor` | No behaviour change, but a meaningful structural shift |
| `test` | Test-only changes |
| `chore` | Tooling, deps, CI, docs |
| `perf` | Measurable speedup (must include a benchmark) |

Subject < 70 chars, imperative mood ("add X", not "added X"), lowercase
after the colon. Examples:

```
feat(bids): walk inheritance for task-level sidecars
fix(http): retry on httpx.TimeoutException, not just RequestError
refactor(digest): extract extract_record from digest_dataset
```

## 3. PR description template

```markdown
## What this PR does

<One-sentence summary. The thing a reviewer would tell their colleague.>

## Why

<The motivation — a user-facing bug, a measured perf regression, a
schema change. Link to the issue if one exists.>

## Reference issues / PRs

Fixes #NNN
Related: #MMM
Builds on: #LLL

## How to verify

```bash
pytest ingestions/tests/test_<file>.py -k <test_name> -v
```

<Or a manual repro if the test alone doesn't tell the story.>

## What I did NOT do (out of scope)

<Preempt reviewer questions: list things that LOOK related but aren't
addressed here.>

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation only
- [ ] Refactor (no functional change, no API change)
- [ ] CI / tooling

## Checklist

- [ ] My code follows `02-STYLE_GUIDE.md`
- [ ] I have added tests that prove my fix is effective or my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] `ruff check` and `mypy --strict` are clean for the changed files
- [ ] Coverage on the changed lines is ≥ 80% (run `pytest --cov`)
- [ ] If this changes a schema, I have updated `whats_new.rst`
- [ ] If this changes a public API, I have added a deprecation warning
      for the old form (see § 6)
```

## 4. Reviewer checklist

The reviewer scans the diff with this checklist in mind:

| Check | Pass means |
|---|---|
| **Correctness** | The code does what the PR description says it does |
| **Tests** | At least one new test exists; it would fail without this change |
| **Naming** | Function/variable names match `02-STYLE_GUIDE.md` § 2 |
| **Docstrings** | New public functions have NumPy-style docstrings |
| **Error handling** | No `except Exception:`; specific catches with logged context |
| **Type hints** | New functions have full annotations; `mypy --strict` is clean |
| **Length** | No function added exceeds 80 LOC (or it has explicit reviewer ack) |
| **Scope** | The PR does ONE thing; if it does two, it's two PRs |
| **CI** | All gates green: lint, types, tests, coverage |
| **whats_new** | If user-visible, an entry in `whats_new.rst` exists |

If any check fails, **the reviewer writes a comment, not a "request
changes"**, unless the issue is severe. The goal is dialogue, not
gate-keeping theatre.

## 5. Two-reviewer rule (for high-blast-radius files)

PRs touching the following files need **two** reviewer approvals:

- `5_inject.py` (writes to MongoDB)
- `3_digest.py` — `extract_record`, `extract_dataset_metadata`
- `4_validate_output.py` (the safety net)
- `eegdash.schemas` references (cross-package contract)

PRs touching everything else need **one** approval.

The two reviewers should NOT both be the original PR author and their
direct collaborator on the same feature. Diversity catches bugs.

## 6. Deprecation policy

Inspired by scikit-learn's 2-minor-version cycle.

If you change a public function signature or schema field:

1. **Keep the old form working**, marked deprecated, for at least one
   release.
   ```python
   def parse_record(path, sidecars=None, **deprecated_kwargs):
       if "sidecar" in deprecated_kwargs:  # old singular form
           warnings.warn(
               "parse_record(..., sidecar=...) is deprecated; use "
               "sidecars=... (a dict). Will be removed in v0.5.",
               DeprecationWarning,
               stacklevel=2,
           )
           sidecars = {"eeg": deprecated_kwargs.pop("sidecar")}
       ...
   ```

2. **Document in `whats_new.rst`**:
   ```rst
   .. deprecated:: 0.4
       The ``sidecar`` argument is deprecated in favor of ``sidecars``.
       Will be removed in 0.5.
   ```

3. **Remove in the planned release**, with a migration note in the
   `whats_new` entry for that release.

## 7. `whats_new.rst` discipline

Every PR that affects user-visible behaviour adds one entry:

```rst
Version 0.4
===========

Enhancements
------------

- :func:`ingestions.bids.walk_inheritance` now follows BIDS 1.7.0
  task-level sidecars (#PR_NUMBER, by :user:`ghuser`)

Bug fixes
---------

- :func:`ingestions.digest.extract_record` no longer crashes on
  malformed ``channels.tsv`` with CRLF line endings (#PR_NUMBER)

API changes
-----------

- The ``sidecar`` parameter of :func:`parse_record` is deprecated in
  favor of ``sidecars`` (#PR_NUMBER). See § Deprecations below.
```

This file is the README for what changed between versions. It is
NOT a CHANGELOG — those are auto-generated from commit messages.
`whats_new.rst` is curated, has user-facing prose, and is what we read
when upgrading.

## 8. CI gates

The PR cannot merge unless ALL of these pass:

| Gate | Where defined | What it checks |
|---|---|---|
| `lint` | `.github/workflows/ci.yml` | `ruff check` + `ruff format --check` |
| `types` | same | `mypy --strict ingestions/` |
| `tests` | same | `pytest ingestions/tests/` |
| `coverage` | same | Line coverage ≥ floor (initial: 70%) |
| `schema-dryrun` | new workflow | `5_inject.py --dry-run` validates fixture records |
| `mutation-nightly` | new workflow | Stryker / mutmut score above floor (initial: 50%) — only blocks the *nightly* run, not per-PR |

## 9. Commit hygiene

Inside a PR:

- Each commit should be a small, coherent change.
- Commit messages follow the same `<type>(<scope>): <subject>` format
  as the PR title.
- **No** "fix typo" / "address review" commits in the final history.
  Squash or rebase before merge.
- **No** `git push --force` to a branch with multiple authors without
  coordination. Force-push to your own branch is fine.

## 10. When you disagree with a reviewer

This happens. The escalation ladder:

1. Reply in the comment thread with your reasoning. Cite a docstring,
   a test, a benchmark, or a section of `02-STYLE_GUIDE.md`.
2. If unresolved, request a third reviewer (any maintainer). Their
   reading breaks the tie.
3. If the disagreement is about the style guide itself, open a separate
   PR proposing the style change. Don't litigate it in a feature PR.

scikit-learn calls this "consensus, with a tie-breaker." It works
because (1) most disagreements are about misunderstood requirements,
not values; (2) when they ARE about values, the style guide is the
canonical reference. Make the canonical reference good and the rest
follows.
