# Reviewer-only rubric pass: 18 EEGDash tutorials (2026-05-07)

Reviewer: `llm-claude-opus-4.7-iter4` (Opus 4.7, 1M context).

This pass scores the eight reviewer-only rules from `CONTRIBUTING.md`'s
"Tutorial Review Rubric" appendix. Merge gate: every rule scored `>= 3`.

## Score table

Columns are 1-5 scores per rule. `min` is the smallest score across the
eight rules; `pass` is true iff `min >= 3`. `sum` is the rule-sum used
to rank "strongest" tutorials.

| Tutorial | E2.11 | E2.14 | E2.17 | E4.31 | E4.33 | E4.35 | E5.46 | E6.47 | min | sum | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| plot_00_first_search | 4 | 4 | 3 | 4 | 4 | 4 | 4 | 5 | 3 | 32 | yes |
| plot_01_first_recording | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 32 | yes |
| plot_02_dataset_to_dataloader | 4 | 4 | 4 | 4 | 4 | 4 | 5 | 4 | 4 | 33 | yes |
| plot_10_preprocess_and_window | 4 | 4 | 4 | 4 | 4 | 5 | 4 | 4 | 4 | 33 | yes |
| plot_11_leakage_safe_split | 5 | 4 | 5 | 5 | 5 | 4 | 4 | 4 | 4 | 36 | yes |
| plot_12_train_a_baseline | 4 | 4 | 3 | 4 | 4 | 4 | 4 | 4 | 3 | 31 | yes |
| plot_13_save_and_reuse_prepared_data | 4 | 4 | 4 | 3 | 3 | 4 | 4 | 4 | 3 | 30 | yes |
| plot_20_visual_p300_oddball | 4 | 4 | 2 | 5 | 5 | 4 | 4 | 4 | 2 | 32 | NO |
| plot_21_auditory_oddball | 4 | 4 | 2 | 4 | 4 | 4 | 4 | 4 | 2 | 30 | NO |
| plot_30_eyes_open_closed | 5 | 4 | 2 | 5 | 5 | 5 | 4 | 4 | 2 | 34 | NO |
| plot_40_first_features | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 32 | yes |
| plot_41_feature_trees | 4 | 4 | 2 | 3 | 4 | 4 | 4 | 4 | 2 | 29 | NO |
| plot_42_features_to_sklearn | 4 | 4 | 2 | 4 | 4 | 4 | 4 | 4 | 2 | 30 | NO |
| plot_50_within_subject_evaluation | 4 | 4 | 2 | 4 | 4 | 4 | 5 | 4 | 2 | 31 | NO |
| plot_51_cross_subject_evaluation | 4 | 4 | 4 | 5 | 5 | 4 | 4 | 4 | 4 | 34 | yes |
| plot_52_cross_session_evaluation | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 32 | yes |
| plot_53_learning_curves | 4 | 4 | 2 | 4 | 4 | 4 | 4 | 4 | 2 | 30 | NO |
| plot_54_compare_two_pipelines | 4 | 4 | 2 | 4 | 5 | 4 | 5 | 4 | 2 | 32 | NO |

## Per-rule averages (n=18)

| Rule | Avg | What it measures |
| --- | ---: | --- |
| E2.11 | 4.11 | Single coherent narrative arc |
| E2.14 | 4.00 | Cognitive load: one concept per cell, figure adjacent to prose |
| E2.17 | 3.06 | Intentional error shown and recovered |
| E4.31 | 4.11 | Real neuroscience question in opening lines |
| E4.33 | 4.22 | Result has scientific meaning, not just code-runs |
| E4.35 | 4.11 | Inclusive present-tense tone, explains the why |
| E5.46 | 4.17 | Hedged claims, limitations flagged |
| E6.47 | 4.06 | Diataxis purity (stays a tutorial) |

The lowest-scoring rule on average is **E2.17** (3.06) — exactly the
systemic gap the prior verification reports flagged. Eight tutorials
score 2 on this rule because they rely on assertions or guard clauses
rather than a try/except block that intentionally triggers and then
recovers from an error.

## Top 5 strongest tutorials (by sum of scores)

| Rank | Tutorial | sum | min |
| ---: | --- | ---: | ---: |
| 1 | plot_11_leakage_safe_split | 36 | 4 |
| 2 | plot_30_eyes_open_closed | 34 | 2 |
| 2 | plot_51_cross_subject_evaluation | 34 | 4 |
| 4 | plot_02_dataset_to_dataloader | 33 | 4 |
| 4 | plot_10_preprocess_and_window | 33 | 4 |

`plot_11_leakage_safe_split` is the cleanest exemplar — its leaky-vs-safe
arc is exactly the rubric model: a real failure mode, an in-prose
diagnosis, and a hedged recovery.

`plot_30_eyes_open_closed` has the highest non-E2.17 profile in the suite
(four 5s, three 4s, one 2), which is why it lands second on raw sum but
fails the merge gate.

## Top 5 with the most-frequent low scores (rules consistently rated <= 3)

The systematic weakness across the suite is **E2.17 (intentional error
shown and recovered)**. Eight tutorials score 2 on E2.17. Of those, the
five with the lowest overall sums are:

| Rank | Tutorial | min | sum | Failing rules (score <= 3) |
| ---: | --- | ---: | ---: | --- |
| 1 | plot_41_feature_trees | 2 | 29 | E2.17, E4.31 |
| 2 | plot_21_auditory_oddball | 2 | 30 | E2.17 |
| 2 | plot_42_features_to_sklearn | 2 | 30 | E2.17 |
| 2 | plot_53_learning_curves | 2 | 30 | E2.17 |
| 5 | plot_50_within_subject_evaluation | 2 | 31 | E2.17 |

Several otherwise-strong tutorials (`plot_30`, `plot_54`) sit above this
floor by raw sum but still fail the gate because of the same E2.17 gap.

## Tutorials that fail the merge gate (need follow-up)

Eight tutorials score `< 3` on at least one reviewer-only rule. All
eight failures are on **E2.17**:

- plot_20_visual_p300_oddball
- plot_21_auditory_oddball
- plot_30_eyes_open_closed
- plot_41_feature_trees
- plot_42_features_to_sklearn
- plot_50_within_subject_evaluation
- plot_53_learning_curves
- plot_54_compare_two_pipelines

The `plot_00`, `plot_11`, `plot_12`, `plot_13`, `plot_40`, `plot_51`,
`plot_52` set show the pattern that works: a small `try/except` block
that triggers a paradigm-relevant mistake (wrong task name, leaky split,
filter cutoff swap, oversize window, n_folds > n_subjects, wrong
`by=...` argument) and recovers in prose.

## Recommended remediation

Each failing tutorial needs **one** added cell, ~10 lines, that:

1. Sets up a paradigm-relevant mistake with `try/except`.
2. Prints the caught error type and message.
3. Prints a one-line recovery rule that fixes it.

Concrete suggestions:

- **plot_20 / plot_21**: a wrong event-mapping that yields zero epochs,
  recovered by listing `event_id` keys.
- **plot_30**: an explicit "swap alpha for delta+theta" mistake before
  the modify cell, framed as a debug demo (the contrast collapse already
  exists in the tutorial — frame it as the recovered error).
- **plot_41**: a missing `feature_predecessor` decorator on a custom
  feature, recovered by adding it.
- **plot_42**: a non-grouped split (e.g. `train_test_split` shuffled on
  feature_table) that leaks subjects, recovered by reapplying the
  manifest from `plot_11`.
- **plot_50**: `assert_no_leakage(by="subject")` on a within-subject
  manifest fails as expected, recovered by `by="trial"`.
- **plot_53**: a `data_size` fraction larger than the available cohort
  that fails, recovered by clamping `FRACTIONS`.
- **plot_54**: shuffle the manifest folds for Pipeline B so
  `fold_ids_a != fold_ids_b`, the assert fires, recovered by reusing the
  shared manifest.

After these single-cell additions, every tutorial would pass the merge
gate without any other rubric-side change.

## Provenance

- Reviewer: `llm-claude-opus-4.7-iter4`
- Model: Claude Opus 4.7 (1M context window), `claude-opus-4-7[1m]`
- Prompt template: "Tutorial Review Rubric (Reviewer-Only Rules)"
  appendix in `CONTRIBUTING.md`, scored against the eight rules listed
  in `docs/tutorial_implementation_strategy.md` "Reviewer-only rubric
  items" table.
- Source tutorials: 18 files under `examples/tutorials/`.
- Dossier: `docs/evidence/tutorials/<id>/reviewer_score.json`.
