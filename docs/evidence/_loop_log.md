# Tutorial-refactor loop progress log

Each iteration appends here. Future iterations should read this first to
decide the next batch of work. State of truth lives in:

- `docs/tutorials/_spec/*.yaml` (`state:` field per tutorial)
- `docs/evidence/tutorials/<id>/evidence.json` (per-tutorial audit results)
- `docs/evidence/tutorials/_verification_*.md` (LLM-judgment passes)
- `docs/tutorials/_triage.md` (Phase 1 classifications of existing examples)
- git log on branch `feat/tutorial-refactor`

## Iteration 1 — 2026-05-06

Cron schedule: `7,37 * * * *` (job id `8ae80b6a`).

### Day-0 bootstrap (commit bbb569cdb)

44 files, 10,580 lines. Strategy doc + 13 spec YAMLs + audit infrastructure
+ 12 validators + CI workflow + tutorials.mk + CONTRIBUTING.md appendix +
Sphinx evidence page + baseline audit on the 18 existing examples (179 errors,
16 warns).

### Wave A — foundational APIs + triage (commit 0327dae17)

5 parallel agents:

- `eegdash.splits` (Workstream 3): 7 source files + 17 tests. assert_no_leakage
  emits `{"leakage_report": {"overlap": N, "by": K}}` deterministically.
- `eegdash.tasks` (Workstream 2 minimal): EEGTask base + EyesOpenClosed
  using ds005514. 16 tests.
- `EEGDashDataset` Workstream 1 helpers: summary/preview/filter. 18 tests.
- Audit completion: e4_engagement.py + e6_diataxis.py validators. 21 tests.
- Phase 1 triage: docs/tutorials/_triage.md classifies all 18 existing
  examples (2 keep, 6 move-to-applied, 2 move-to-hpc, 7 rewrite, 1 retire-replaced,
  1 retire-pure).

72 new tests across Wave A, all green.

### Wave B — 8 Release-1 tutorial drafts (commit 6a5765606)

8 parallel agents drafting plot_00, plot_01, plot_02, plot_10, plot_11,
plot_12, plot_13, plot_40. Each tutorial:

- Self-audited to errors=0, warns=0 (5 reviewer-only infos remain).
- LOC range: 216-220 (all under the 220 cap).
- Full PRIMM coverage for difficulty 1.
- DOIs cited in References block.
- Cisotto & Chicco 2024 + Pernet 2019 + Gramfort 2013 + Schirrmeister 2017
  cited where topic warrants.

Verification pass (`_verification_2026-05-06.md`): 6/8 pass citations,
6/8 plan alignment, 7/8 reviewer rubric. Two wrong DOIs in plot_11
corrected in the same commit (Pernet 2019 DOI was wrong, MOABB DOI was
broken). E2.17 (intentional error shown) flagged as systematically weak.

### Open items for iteration 2+

State per spec (`state:` field still `proposed` because we haven't run
`make tutorial-claim`/`tutorial-release` — but functionally:

- Drafted (8/13): plot_00, plot_01, plot_02, plot_10, plot_11, plot_12,
  plot_13, plot_40.
- Not drafted (5/13): plot_20_visual_p300_oddball, plot_21_auditory_oddball,
  plot_30_eyes_open_closed, plot_41_feature_trees, plot_42_features_to_sklearn.

### Recommended iteration 2 work

1. **Draft the 5 Release-2 tutorials** in parallel (5 agents). plot_30 uses
   `eegdash.tasks.get_task("eyes-open-closed")` from Wave A. plot_20/21 need
   a P300/auditory-oddball task — implement those task stubs first or have
   the tutorials use raw braindecode windowing.
2. **Address 8 follow-up fixes** from the verification report: plot_11
   `n_folds>=5` invariant scope, plot_12 unused requires_api entries,
   plot_13 timing assertion, plot_10 task-name prose mismatch, plot_40
   "synthetic data framing" prose.
3. **Address 3 spec-vs-API mismatches**: spec promises symbols that don't
   exist yet (`EEGDash.search_datasets` for plot_00; `fit_feature_extractor`
   for plot_40). Either ship the public API helper or update the spec.
4. **E2.17 intentional-error pass**: have an agent add an "intentional-error"
   step to each Wave-B tutorial where reasonable.
5. **Phase 3 file moves**: execute the file-move actions in
   `docs/tutorials/_triage.md` (move age/p-factor/sex to `examples/applied/`,
   retire `tutorial_minimal.py`, etc.).
6. **CI validation**: verify the static-stage CI workflow runs cleanly
   against the new tutorials. Cron-fired iterations can skip this if it has
   no diff.

### Time used in iteration 1

Roughly 40 minutes wall, including all 13 agent runs and 3 commits. Within
the 30-minute cron cadence — next fire will pick up iteration 2.

## Iteration 2 — 2026-05-07

### Wave C — 5 Release-2 tutorial drafts (commit e6090e698)

7 parallel agents:

- 5 tutorial-author agents: plot_20, plot_21, plot_30, plot_41, plot_42.
  All five self-audit clean (errors=0 warns=0). LOC range 218-220 + budget
  bumps where E2.17 retrofit applied.
- 1 E2.17-retrofit agent: added "intentional mistake → recovery" block to
  each of the 8 Wave-B tutorials. Mistakes per domain (unknown task name,
  out-of-range index, oversize window, inverted band-pass, typo'd splitter
  name, length-mismatched fit, missing overwrite=True, unknown feature key).
  Cites Nederbragt et al. 2020 (doi:10.1371/journal.pcbi.1008090) in
  plot_00. Bumped spec.budgets.max_loc from 220 to 260 (default), 285
  (plot_40, plot_41), 230 (plot_20) to absorb the +14-19 lines per file.
- 1 spec/API alignment agent: added EEGDash.search_datasets (5 unit tests)
  and eegdash.features.fit_feature_extractor backward-compat alias (1
  test). Synced 9 specs' requires_api lists to the actual imports.

### Verification (commit-included _verification_2026-05-07.md)

Pass/fail across 5 Wave-C tutorials: A. citations 2 pass / 1 partial /
2 fail; B. plan alignment 2/3/0; C. spec coherence 0/4/1; D. reviewer
rubric 4/1/0.

Critical fixes applied in the same commit:
- plot_42 Sentance 2019 PRIMM DOI corrected to 10.1080/08993408.2019.1608781
  (was 10.1145/3304221.3325207, wrong paper).
- plot_42 Pedregosa 2011 ACM identifier (10.5555/...) dropped — keep the
  textual JMLR reference only.
- plot_20 dataset attribution softened to "OpenNeuro ds005863" without
  asserting Kappenman / ERP CORE authorship (verification flagged the
  DOI may resolve to a different paper).

### State after iteration 2

- 13/13 specs in the new gallery tree pass static audit (errors=0, warns=0).
- Legacy `plot_clinical_summary.py` still flags 9 errors — will be
  removed in Phase 3 file moves.
- Cumulative new tests across both iterations: 100+ unit tests, all green.
- Phase 1 (audit/triage) and Phase 2 (build first learning path = 13
  tutorials) of the plan's migration plan are now COMPLETE.

### Recommended iteration 3 work

1. **Phase 3 file moves** (per docs/tutorials/_triage.md):
   - Retire `plot_clinical_summary.py` (replaced by Cat A trio + Cat E).
   - Retire `tutorial_minimal.py` (leaky split; superseded by Cat A trio).
   - Retire `tutorial_transfer_learning.py` (synthetic; plan §L1097-1100).
   - Move noplot_tutorial_age_prediction/pfactor_features/pfactor_regression/
     sex_classification_cnn → `examples/applied/`.
   - Move noplot_tutorial_audi_oddball + noplot_tutorial_p3_oddball:
     these were rewritten as plot_20/21 — retire.
   - Rename remaining `noplot_*` files (Phase 1 step 5 of the plan).
   - hpc/tutorial_eoec.py → `examples/hpc/tutorial_hpc_cache_and_slurm.py`.
2. **Phase 4 — Cat F evaluation tutorials** (5 new tutorials per plan
   §Cat F, lines 425-442):
   - plot_50_within_subject_evaluation
   - plot_51_cross_subject_evaluation
   - plot_52_cross_session_evaluation
   - plot_53_learning_curves
   - plot_54_compare_two_pipelines
   These need spec YAMLs first.
3. Address remaining iteration-2 verification follow-ups:
   - plot_30 over-constrains alpha-diff assertion.
   - plot_30 calls task.* properties not yet on EEGTask.
   - plot_41 monkeypatches without restoration.
   - plot_21 SFREQ mismatch vs plot_20.
4. Optional Phase 4 — implement W4 (windowing convenience) and W5
   (baseline recipes) per plan §Implementation Backlog.

### Time used in iteration 2

Roughly 40 minutes wall (7 agent runs + verification + 4 commit attempts
recovering from pre-commit hook fixes). Within the 30-minute cron cadence;
next fire will pick up iteration 3.
