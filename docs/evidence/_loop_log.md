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
