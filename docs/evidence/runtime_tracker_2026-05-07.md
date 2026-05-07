# Tutorial runtime and data-size tracker

Generated: 2026-05-06T23:19:55.728428+00:00

Source of truth: `docs/tutorials/_spec/*.yaml` -- `budgets.max_runtime_seconds` and `budgets.max_network_mb`. Per `docs/tutorial_restructure_plan.md` Phase 5 governance, every tutorial declares an upper bound for runtime and on-the-wire data so the CI matrix and the docs gallery stay green.

## Per-tutorial estimates

| ID | Category | HW | Network | Est. runtime | Est. data |
| --- | --- | --- | --- | --- | --- |
| `plot_00_first_search` | A-start-here | C | metadata_only | 1m00s | 5 MB |
| `plot_01_first_recording` | A-start-here | C | cached_first_run | 1m00s | 100 MB |
| `plot_02_dataset_to_dataloader` | A-start-here | C | cached_first_run | 1m00s | 100 MB |
| `plot_10_preprocess_and_window` | B-core-workflow | C | cached_first_run | 3m00s | 100 MB |
| `plot_11_leakage_safe_split` | B-core-workflow | C | cached_first_run | 3m00s | 50 MB |
| `plot_12_train_a_baseline` | B-core-workflow | C | cached_first_run | 3m00s | 50 MB |
| `plot_13_save_and_reuse_prepared_data` | B-core-workflow | C | cached_first_run | 3m00s | 50 MB |
| `plot_20_visual_p300_oddball` | C-event-related | C | cached_first_run | 5m00s | 200 MB |
| `plot_21_auditory_oddball` | C-event-related | C | cached_first_run | 5m00s | 200 MB |
| `plot_30_eyes_open_closed` | D-resting-state | C | cached_first_run | 3m00s | 200 MB |
| `plot_40_first_features` | E-feature-engineering | C | cached_first_run | 5m00s | 100 MB |
| `plot_41_feature_trees` | E-feature-engineering | C | cached_first_run | 5m00s | 100 MB |
| `plot_42_features_to_sklearn` | E-feature-engineering | C | cached_first_run | 5m00s | 50 MB |
| `plot_50_within_subject_evaluation` | F-evaluation | C | cached_first_run | 4m00s | 50 MB |
| `plot_51_cross_subject_evaluation` | F-evaluation | C | cached_first_run | 4m00s | 50 MB |
| `plot_52_cross_session_evaluation` | F-evaluation | C | cached_first_run | 4m00s | 50 MB |
| `plot_53_learning_curves` | F-evaluation | C | cached_first_run | 5m00s | 50 MB |
| `plot_54_compare_two_pipelines` | F-evaluation | C | cached_first_run | 5m00s | 50 MB |
| `plot_70_challenge_dataset_basics` | H-transfer-foundation | C | cached_first_run | 4m00s | 200 MB |
| `plot_71_cross_task_transfer` | H-transfer-foundation | G | cached_first_run | 25m00s | 1.5 GB |
| `plot_72_subject_invariant_regression` | H-transfer-foundation | G | cached_first_run | 25m00s | 1.5 GB |
| `plot_73_finetune_pretrained_model` | H-transfer-foundation | G | cached_first_run | 30m00s | 2.0 GB |
| `how_to_download_a_dataset` | I-scaling-hpc | C | required_first_run | 10m00s | 2.0 GB |
| `how_to_parallelize_feature_extraction` | I-scaling-hpc | C | cached_first_run | 10m00s | 200 MB |
| `how_to_run_preprocessing_on_slurm` | I-scaling-hpc | C | cached_first_run | 10m00s | 0 MB |
| `how_to_use_hpc_cache` | I-scaling-hpc | C | cached_first_run | 5m00s | 200 MB |
| `how_to_work_offline` | I-scaling-hpc | C | cached_first_run | 4m00s | 0 MB |

## Per-category totals

| Category | Count | Total runtime | Total data |
| --- | --- | --- | --- |
| A-start-here | 3 | 3m00s | 205 MB |
| B-core-workflow | 4 | 12m00s | 250 MB |
| C-event-related | 2 | 10m00s | 400 MB |
| D-resting-state | 1 | 3m00s | 200 MB |
| E-feature-engineering | 3 | 15m00s | 250 MB |
| F-evaluation | 5 | 22m00s | 250 MB |
| H-transfer-foundation | 4 | 1h24m | 5.1 GB |
| I-scaling-hpc | 5 | 39m00s | 2.3 GB |

## Overall total

- Tutorials tracked: **27**
- Sum of declared runtimes: **3h08m**
- Sum of declared on-the-wire data: **8.9 GB**

If a single PR pushes the overall runtime past 240 minutes or the data total past 12 GB, escalate to the docs maintainers and consider demoting one tutorial to the nightly-only stage of `.github/workflows/tutorial-audit.yml`.
