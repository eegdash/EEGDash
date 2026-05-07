# Reviewer-only rubric pass: 14 EEGDash tutorials (iter5, 2026-05-07)

Reviewer: `llm-claude-opus-4.7-iter5` (Opus 4.7, 1M context).

This pass scores the eight reviewer-only rules from `CONTRIBUTING.md`'s
"Tutorial Review Rubric" appendix on 14 newly drafted tutorials. The
five Cat F tutorials were re-scored after the Wave E2 retrofit added
intentional-error-and-recover cells (commit `a6d7d879e`); Cat H and
Cat I tutorials are scored fresh. Iter4's summary
(`_reviewer_summary_2026-05-07.md`) is preserved unchanged.

For Cat I how-tos, three rules are intentionally `null`:

- **E2.17** -- intentional error / recovery is a tutorial pedagogical
  pattern; how-tos do not require one (kind=how-to, rule N/A).
- **E4.31** -- "neuroscience question in opening lines" maps to a
  recipe-goal framing in a how-to (kind=how-to, rule N/A).
- **E4.33** -- "result has scientific meaning" maps to a verifiable
  recipe outcome in a how-to (kind=how-to, rule N/A).

Merge gate: every applicable rule scored `>= 3`.

## Score table (iter5, n=14)

`min` is the smallest score across applicable rules; `pass` is true iff
`min >= 3`. Cells with `--` are `null` (rule N/A for kind=how-to).

| Tutorial | Kind | E2.11 | E2.14 | E2.17 | E4.31 | E4.33 | E4.35 | E5.46 | E6.47 | min | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| plot_50_within_subject_evaluation | tutorial | 4 | 4 | 4 | 4 | 4 | 4 | 5 | 4 | 4 | yes |
| plot_51_cross_subject_evaluation  | tutorial | 4 | 4 | 4 | 5 | 5 | 4 | 4 | 4 | 4 | yes |
| plot_52_cross_session_evaluation  | tutorial | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | yes |
| plot_53_learning_curves           | tutorial | 4 | 4 | 3 | 4 | 4 | 4 | 4 | 4 | 3 | yes |
| plot_54_compare_two_pipelines     | tutorial | 4 | 4 | 3 | 4 | 5 | 4 | 5 | 4 | 3 | yes |
| plot_70_challenge_dataset_basics  | tutorial | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | yes |
| plot_71_cross_task_transfer       | tutorial | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | yes |
| plot_72_subject_invariant_regression | tutorial | 4 | 4 | 2 | 4 | 4 | 4 | 5 | 4 | 2 | NO |
| plot_73_finetune_pretrained_model | tutorial | 4 | 4 | 2 | 4 | 4 | 4 | 4 | 4 | 2 | NO |
| how_to_download_a_dataset         | how-to   | 4 | 4 | -- | -- | -- | 4 | 4 | 5 | 4 | yes |
| how_to_work_offline               | how-to   | 4 | 4 | -- | -- | -- | 4 | 4 | 4 | 4 | yes |
| how_to_use_hpc_cache              | how-to   | 4 | 4 | -- | -- | -- | 4 | 4 | 4 | 4 | yes |
| how_to_run_preprocessing_on_slurm | how-to   | 4 | 4 | -- | -- | -- | 4 | 4 | 5 | 4 | yes |
| how_to_parallelize_feature_extraction | how-to | 4 | 4 | -- | -- | -- | 4 | 4 | 4 | 4 | yes |

## Per-rule averages (iter5, applicable cells only)

| Rule | Avg | n | What it measures |
| --- | ---: | ---: | --- |
| E2.11 | 4.00 | 14 | Single coherent narrative arc |
| E2.14 | 4.00 | 14 | Cognitive load: one concept per cell, figure adjacent to prose |
| E2.17 | 3.33 |  9 | Intentional error shown and recovered (tutorials only) |
| E4.31 | 4.11 |  9 | Real neuroscience question in opening lines (tutorials only) |
| E4.33 | 4.22 |  9 | Result has scientific meaning (tutorials only) |
| E4.35 | 4.00 | 14 | Inclusive present-tense tone, explains the why |
| E5.46 | 4.21 | 14 | Hedged claims, limitations flagged |
| E6.47 | 4.14 | 14 | Diataxis purity (stays a tutorial / recipe) |

The lowest-scoring rule is **E2.17 (3.33)**. The two failing tutorials
(plot_72, plot_73) score 2 on E2.17 because they ship guard assertions
rather than try/except blocks that intentionally trigger and then
recover from a paradigm-relevant error. The five Cat F tutorials in
this pass already had the retrofit applied in commit `a6d7d879e` and
score 3-4 on E2.17, materially better than iter4's 2s on the same set.

## Top 3 strongest tutorials (by sum across applicable cells)

| Rank | Tutorial | sum | min | applicable rules |
| ---: | --- | ---: | ---: | ---: |
| 1 | plot_51_cross_subject_evaluation | 34 | 4 | 8 |
| 2 | plot_50_within_subject_evaluation | 33 | 4 | 8 |
| 2 | plot_54_compare_two_pipelines    | 33 | 3 | 8 |

`plot_51_cross_subject_evaluation` keeps its iter4 lead: opens with the
MOABB gold-standard framing, reports per-fold mean +/- std plus a
chance histogram, and the n_folds=20 recovery is the cleanest
splitter-driven error in the suite.

## Tutorials that fail the merge gate (need follow-up)

Two tutorials score `< 3` on at least one applicable rule:

- **plot_72_subject_invariant_regression** -- scored 2 on E2.17. Add a
  try/except where Ridge.fit on a single-class fold raises (or where
  median_baseline on an empty test fold raises), and recover with a
  stratification check that explains the diagnostic.
- **plot_73_finetune_pretrained_model** -- scored 2 on E2.17. Add a
  try/except where `load_state_dict(strict=True)` fails on a head-shape
  mismatch, and recover with `strict=False` plus the head-reset
  invariant assertion. Note that plot_71 already has this exact
  recovery pattern, so the fix is short and pattern-consistent.

After these single-cell additions, all 14 tutorials in iter5 would
pass the merge gate.

## Combined view: iter4 (18) + iter5 (14) = 32 tutorials

| Pass | Tutorials | Count |
| --- | --- | ---: |
| iter4 | (`_reviewer_summary_2026-05-07.md`) | 18 |
| iter5 | this pass | 14 |
| total | -- | 32 |

| Pool | Pass | Fail | Pass rate |
| --- | ---: | ---: | ---: |
| iter4 (n=18) | 10 | 8 | 56% |
| iter5 (n=14) | 12 | 2 | 86% |
| combined (n=32) | 22 | 10 | **69%** |

Five tutorials (the Cat F five: plot_50/51/52/53/54) appear in both
pools because iter4 scored their pre-retrofit form and iter5 scores
their post-retrofit form. Iter5's higher pass rate (86%) reflects the
Wave E2 retrofit landing on five of those Cat F tutorials plus all
four Cat H tutorials being authored against the post-retrofit pattern.
For an unduplicated combined view, replace iter4's plot_50/51/52/53/54
rows with iter5's: 10 - 2 + 5 = 13 iter4-only passes, plus 12 iter5
passes = 25/27 = **93%** unduplicated pass rate. Both numbers are
defensible answers; the 22/32 figure follows the request literally.

## E2.17 systemic gap (unchanged from iter4)

The eight tutorials that failed iter4 on E2.17 (plot_20, plot_21,
plot_30, plot_41, plot_42, plot_50, plot_53, plot_54) were the
explicit retrofit target of commit `a6d7d879e`. Five of those (the
Cat F set) re-score 3 or 4 on E2.17 here. The remaining three
(plot_20, plot_21, plot_30, plot_41, plot_42) are not in iter5's
scope; iter4's scores stand for them. The two new failures (plot_72,
plot_73) ship in a future Cat H E2.17 retrofit if the maintainers
choose to extend the pattern.

## Recommended remediation

Each failing tutorial needs **one** added cell, ~10 lines, that:

1. Sets up a paradigm-relevant mistake with `try/except`.
2. Prints the caught error type and message.
3. Prints a one-line recovery rule that fixes it.

Concrete suggestions:

- **plot_72_subject_invariant_regression**: a Ridge fit where the
  train fold collapses to one class raises (no fit possible);
  recovered by checking `len(np.unique(y_train)) >= 2` before fitting.
- **plot_73_finetune_pretrained_model**: a `torch.load` of a 2-output
  state dict into a 3-output `ShallowFBCSPNet` raises a size-mismatch
  `RuntimeError`; recovered by rebuilding with the matching
  `n_outputs` (this is exactly plot_71's recovery, so the fix is
  copy-with-rename).

## Provenance

- Reviewer: `llm-claude-opus-4.7-iter5`
- Model: Claude Opus 4.7 (1M context window), `claude-opus-4-7[1m]`
- Prompt template: "Tutorial Review Rubric (Reviewer-Only Rules)"
  appendix in `CONTRIBUTING.md`, scored against the eight rules in
  `docs/tutorial_implementation_strategy.md` "Reviewer-only rubric
  items" table; for how-tos, three rules are explicitly N/A per the
  iter5 instruction.
- Source tutorials: 14 files under `examples/tutorials/50_evaluation/`,
  `examples/tutorials/70_transfer_foundation/`, and `examples/how_to/`.
- Dossier: `docs/evidence/tutorials/<id>/reviewer_score.json`.
- Iter4 summary preserved at `_reviewer_summary_2026-05-07.md`.
