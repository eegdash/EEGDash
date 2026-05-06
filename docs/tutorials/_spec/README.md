# Tutorial specs

This directory holds the spec-first contract for each tutorial in the EEGDash
refactor. Every spec is a YAML file that declares rubric expectations, budgets,
prerequisites, and required public API symbols *before* any tutorial code is
written. CI rejects authoring without a matching spec entry
(`check_E1_spec_present`).

The canonical schema and validator pipeline live in
[`docs/tutorial_implementation_strategy.md`](../../tutorial_implementation_strategy.md)
under "Spec contract". The 49-rule rubric (E1.* through E6.*) is sourced from
[`new_tutorials/compass_artifact_wf-96e3d362-5e7a-4e1f-82f3-9095193459df_text_markdown.md`](../../../new_tutorials/compass_artifact_wf-96e3d362-5e7a-4e1f-82f3-9095193459df_text_markdown.md).
The tutorial roster, file layout, and quality bar come from
[`docs/tutorial_restructure_plan.md`](../../tutorial_restructure_plan.md).

## Release 1: Core learning path

| Order | Spec | Category | Difficulty |
| ---: | --- | --- | :---: |
| 0 | [`plot_00_first_search.yaml`](plot_00_first_search.yaml) | A. Start Here | 1 |
| 1 | [`plot_01_first_recording.yaml`](plot_01_first_recording.yaml) | A. Start Here | 1 |
| 2 | [`plot_02_dataset_to_dataloader.yaml`](plot_02_dataset_to_dataloader.yaml) | A. Start Here | 1 |
| 3 | [`plot_10_preprocess_and_window.yaml`](plot_10_preprocess_and_window.yaml) | B. Core Workflow | 1 |
| 4 | [`plot_11_leakage_safe_split.yaml`](plot_11_leakage_safe_split.yaml) | B. Core Workflow | 1 |
| 5 | [`plot_12_train_a_baseline.yaml`](plot_12_train_a_baseline.yaml) | B. Core Workflow | 1 |
| 6 | [`plot_13_save_and_reuse_prepared_data.yaml`](plot_13_save_and_reuse_prepared_data.yaml) | B. Core Workflow | 1 |
| 7 | [`plot_40_first_features.yaml`](plot_40_first_features.yaml) | E. Feature Engineering | 1 |

## Release 2: Topical extensions

| Spec | Category | Difficulty |
| --- | --- | :---: |
| [`plot_20_visual_p300_oddball.yaml`](plot_20_visual_p300_oddball.yaml) | C. Event-Related | 2 |
| [`plot_21_auditory_oddball.yaml`](plot_21_auditory_oddball.yaml) | C. Event-Related | 2 |
| [`plot_30_eyes_open_closed.yaml`](plot_30_eyes_open_closed.yaml) | D. Resting State | 1 |
| [`plot_41_feature_trees.yaml`](plot_41_feature_trees.yaml) | E. Feature Engineering | 2 |
| [`plot_42_features_to_sklearn.yaml`](plot_42_features_to_sklearn.yaml) | E. Feature Engineering | 2 |

## Spec lifecycle

```
proposed -> drafted -> static-pass -> runtime-pass -> reviewed -> merged
```

All 13 specs in this directory start at `state: proposed`. CI guards each
transition; only the spec author (recorded in `assignee`) may advance state,
and a regression in any validator drops state back to `drafted`.

## Conventions

- Filenames match `plot_<NN>_<short>.yaml` so they pair 1:1 with
  `examples/tutorials/<bucket>/plot_<NN>_<short>.py`.
- Plan citations use the format `tutorial_restructure_plan.md#L<a>-L<b>` and
  must reference real, current line ranges. `validate_spec.py` re-resolves and
  hashes them on every PR.
- Rubric citations use the format `compass_artifact.md#E<group>.<rule>`
  (e.g., `compass_artifact.md#E5.42`).
- `requires_api` lists the public symbols the tutorial depends on (e.g.,
  `eegdash.splits.assert_no_leakage`). When an upstream Workstream blocks an
  API, the orchestrator surfaces the blocked tutorials.
