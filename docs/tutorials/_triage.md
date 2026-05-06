# Phase 1 Triage Report

This document is the Phase 1 triage product mandated by
`docs/tutorial_restructure_plan.md`, §"Migration Plan: Phase 1 — Audit And
Triage" (lines 1181-1196). Each existing example file under `examples/` is
classified against the recommendations in §"Current Tutorial Inventory"
(lines 190-213) and §"Applied Examples To Keep But Reframe"
(lines 1052-1101) of the same plan. The new spec ids referenced in the
"New path" and "Replaces" columns originate from §"Proposed Tutorial
Catalogue" (lines 360-470) and §"Target Layout" (lines 590-634) of the
plan. The 18-row scope is calibrated against
`docs/evidence/tutorials/_baseline_2026-05-06/_aggregate.md`.

This is a triage product: it records intent, not actions. No files are moved
or modified by writing this report. Phase 1 step 1 of the plan calls for
labelling each gallery file as "true tutorial", "how-to guide", "applied
example", "challenge material", "developer-only", or "retire"; the
`Triage` column below is a finer-grained restatement of that taxonomy
matched to the target layout.

## Triage Table

| File | Current path | Triage | New path (if move) | Replaces (if retire) | Plan citation | Rationale (1 sentence) |
| --- | --- | --- | --- | --- | --- | --- |
| `tutorial_minimal.py` | `examples/core/tutorial_minimal.py` | `retire-replaced-by-tutorial_00_first_search` | — | `tutorial_00_first_search.py`, `tutorial_01_first_recording.py`, `tutorial_02_dataset_to_dataloader.py` | plan L194 ("Replace with a safer first tutorial. It currently notes obvious leakage and takes a long time in gallery execution."), plan L367-370 (Category A spec ids), plan L1200-1201 (Phase 2 step 1-2) | The current minimal example self-confesses leakage and is too heavy for a first run, so it is replaced by the new "Start Here" trio rather than rewritten in place. |
| `tutorial_eoec.py` (core) | `examples/core/tutorial_eoec.py` | `rewrite-as-tutorial_30_eyes_open_closed` | — | plan L195 ("Keep as a core learning tutorial after shortening and making the split/evaluation story explicit."), plan L405-406 (Category D spec id), plan L960-975 (move-from list explicitly cites `examples/core/tutorial_eoec.py`) | Source material for the new resting-state state-decoding tutorial; this is the canonical EO/EC keeper, distinct from the HPC duplicate of the same stem at `examples/hpc/tutorial_eoec.py` which goes to the HPC track. |
| `tutorial_feature_extractor_open_close_eye.py` | `examples/core/tutorial_feature_extractor_open_close_eye.py` | `rewrite-as-tutorial_40_first_features` | — | plan L196 ("Merge concepts into a feature-engineering tutorial; avoid duplicating the EO/EC decoding tutorial."), plan L417 (Category E spec id), plan L977-987 (`tutorial_40_first_features.py` rewrite scope) | EO/EC + feature-extraction concepts are merged into the new `tutorial_40_first_features.py` so the EO/EC decoding tutorial is not duplicated. |
| `p300_transfer_learning.py` | `examples/core/p300_transfer_learning.py` | `move-to-applied` | `examples/applied/` (advanced/transfer-learning track, exact filename TBD by Phase 3) | — | plan L197 ("Move to advanced examples or transfer learning track."), plan L1208-1210 (Phase 3 step 1: move age/p-factor/sex to `examples/applied`) | AS-MMD cross-dataset P300 is paper-scale advanced material, not a first-week tutorial, and the plan routes such items into `examples/applied`. |
| `tutorial_api.py` | `examples/tutorials/tutorial_api.py` | `rewrite-as-tutorial_00_first_search` | — | plan L198 ("Keep, but promote earlier as 'Find datasets and records'. Add richer query patterns and expected outputs."), plan L367 (Category A spec id), plan L1200 (Phase 2 step 1: "Rewrite `tutorial_api.py` as `tutorial_00_first_search.py`.") | This is the only direct, name-level rewrite call-out in the plan; existing query content is the seed for the new "Start Here" tutorial. |
| `noplot_tutorial_age_prediction.py` | `examples/tutorials/noplot_tutorial_age_prediction.py` | `move-to-applied` | `examples/applied/project_age_regression.py` | — | plan L199 ("Move to applied clinical/regression track. Make it a project-style example, not onboarding."), plan L450 (Category G spec id `project_age_regression.py`), plan L623, plan L1057-1066 (required reframing) | Continuous-target Conformer regression is a realistic project, not onboarding, and the plan gives it a named slot under `examples/applied/`. |
| `noplot_tutorial_audi_oddball.py` | `examples/tutorials/noplot_tutorial_audi_oddball.py` | `rewrite-as-tutorial_21_auditory_oddball` | — | plan L200 ("Keep as event-related decoding how-to or applied tutorial."), plan L394-395 (Category C spec id `tutorial_21_auditory_oddball.py`, "ideally framed as a contrast with the visual P300 tutorial rather than a duplicate") | Kept as an event-related tutorial, but reframed as the contrast partner to the visual P3 tutorial so the two are not duplicates. |
| `noplot_tutorial_p3_oddball.py` | `examples/tutorials/noplot_tutorial_p3_oddball.py` | `rewrite-as-tutorial_20_visual_p300_oddball` | — | plan L201 ("Keep one P3/oddball tutorial; either merge with auditory oddball comparison or turn one into a how-to."), plan L393 (Category C spec id `tutorial_20_visual_p300_oddball.py`), plan L941-958 (rewrite scope) | This becomes the canonical visual-P3 tutorial that the auditory oddball tutorial contrasts against, satisfying the "keep one P3/oddball tutorial" rule. |
| `noplot_tutorial_feature_extraction.py` | `examples/tutorials/noplot_tutorial_feature_extraction.py` | `rewrite-as-tutorial_42_features_to_sklearn` | — | plan L202 ("Split into 'Extract features' and 'Train classical baseline'. Remove hidden TODOs."), plan L417-421 (Category E spec ids `tutorial_40_first_features.py` and `tutorial_42_features_to_sklearn.py`), plan L1009-1019 (rewrite scope) | Sex+features is the source material for the classical-baseline half of the split (the extract-features half draws on the EO/EC feature file); hidden TODOs are removed in the rewrite. |
| `noplot_tutorial_features_eoec.py` | `examples/tutorials/noplot_tutorial_features_eoec.py` | `rewrite-as-tutorial_41_feature_trees` | — | plan L203 ("Use as source material for a better feature extractor lesson; currently too long for a beginner tutorial."), plan L418-419 (Category E spec id `tutorial_41_feature_trees.py`), plan L1005-1007 (move-from list explicitly cites this file), plan L972-975 (also feeds `tutorial_30_eyes_open_closed.py`) | The plan explicitly names this as the source for `tutorial_41_feature_trees.py`, with the EO/EC framing instead absorbed by `tutorial_30_eyes_open_closed.py`. |
| `noplot_tutorial_pfactor_features.py` | `examples/tutorials/noplot_tutorial_pfactor_features.py` | `move-to-applied` | `examples/applied/project_pfactor_features.py` | — | plan L204 ("Move to applied clinical/regression examples."), plan L451 (Category G spec id `project_pfactor_regression_features.py`), plan L624 (target layout filename `project_pfactor_features.py`), plan L1068-1079 (required reframing) | Feature-based p-factor regression is a project example; the plan keeps it under `examples/applied/` with explicit privacy/baseline framing. |
| `noplot_tutorial_pfactor_regression.py` | `examples/tutorials/noplot_tutorial_pfactor_regression.py` | `move-to-applied` | `examples/applied/project_pfactor_deep.py` | — | plan L205 ("Move to applied clinical/regression examples. Avoid duplicating p-factor feature tutorial."), plan L452 (Category G spec id `project_pfactor_regression_deep.py`), plan L625 (target layout filename `project_pfactor_deep.py`), plan L1068-1079 (required framing, single project with two pipelines acceptable) | Deep p-factor regression is the second pipeline alongside the feature version, kept only because the two answer different questions. |
| `noplot_tutorial_sex_classification_cnn.py` | `examples/tutorials/noplot_tutorial_sex_classification_cnn.py` | `move-to-applied` | `examples/applied/project_sex_classification.py` | — | plan L206 ("Move to applied examples; keep only if framed carefully around labels, leakage, and limitations."), plan L453-454 (Category G spec id `project_sex_classification.py`), plan L626 (target layout filename), plan L1081-1090 (required framing on labels and confounds) | Kept as a workflow demonstration only with the mandatory caveats on label semantics and site/dataset confounds. |
| `plot_clinical_summary.py` | `examples/tutorials/plot_clinical_summary.py` | `move-to-applied` | `examples/applied/project_clinical_dataset_summary.py` (or how-to per plan L1094-1095) | — | plan L207 ("Move to 'How to explore the catalog' or 'Dataset catalog examples'."), plan L455-456 (Category G spec id `project_clinical_dataset_summary.py`), plan L1092-1095 ("Move out of the model-training tutorial path. This is a catalog exploration how-to or applied visualization example.") | Catalog exploration that does not train a model belongs alongside applied/how-to material, not in the model-training tutorial sequence. |
| `tutorial_transfer_learning.py` | `examples/tutorials/tutorial_transfer_learning.py` | `retire` | — | — | plan L208 ("Retire or rewrite with real EEGDash data. A simulated example does not teach EEGDash enough."), plan L1097-1100 ("Retire unless rewritten with real EEGDash data. It does not teach the library well enough to occupy a public tutorial slot.") | Pure synthetic data with `torch.randn` does not exercise EEGDash; retired outright in the absence of a concrete real-data rewrite plan. |
| `tutorial_challenge_1.py` | `examples/eeg2025/tutorial_challenge_1.py` | `keep` | — | — | plan L209 ("Keep under a separate 'Competition and foundation challenge' track."), plan L466 (Category H spec id `tutorial_71_cross_task_transfer.py` — same content under new id), plan L627-630 (target layout keeps file under `examples/eeg2025/`) | Stays put under the EEG2025 challenge track, which the target layout preserves verbatim. |
| `tutorial_challenge_2.py` | `examples/eeg2025/tutorial_challenge_2.py` | `keep` | — | — | plan L210 ("Keep under separate challenge track."), plan L467 (Category H spec id `tutorial_72_subject_invariant_regression.py`), plan L627-630 (target layout) | Stays put under the EEG2025 challenge track. |
| `tutorial_eegdash_offline.py` | `examples/eeg2025/tutorial_eegdash_offline.py` | `move-to-hpc` | `examples/how_to/how_to_work_offline.py` (cross-linked from `examples/hpc/`) | — | plan L211 ("Move to how-to guide; also cross-link from HPC docs."), plan L478 (`how_to_work_offline.py`), plan L615-621 (target `how_to/` directory), plan L1021-1037 (rewrite scope), plan L1202 (Phase 2 step 3: "Rewrite cache/offline basics from EEG2025 material into a general how-to.") | The offline workflow is generic, not challenge-specific; the plan promotes it to a how-to with HPC cross-links. |
| `tutorial_eoec.py` (hpc) | `examples/hpc/tutorial_eoec.py` | `move-to-hpc` | `examples/hpc/tutorial_hpc_cache_and_slurm.py` | — | plan L212 ("Move to 'Scaling and HPC' how-to. It should not appear as a duplicate beginner tutorial."), plan L631-633 (target layout spec id `tutorial_hpc_cache_and_slurm.py`) | Duplication note: this file shares its filename with `examples/core/tutorial_eoec.py`, but its content is HPC-flavoured (multi-subject, NUMBA disable, fake MNE home, file plot). The two share the stem but the plan splits them: the core copy seeds `tutorial_30_eyes_open_closed.py` while this HPC copy is reframed under the HPC track and not republished as a beginner duplicate. |

Row count: 18, matching the audit aggregate scope.

## Phase 1 Action Items

These are intent-only entries for later phases. No files are moved by this
report. Citations are to lines in
`docs/tutorial_restructure_plan.md`.

- `examples/tutorials/tutorial_api.py` -> rewrite as
  `examples/tutorials/00_start_here/tutorial_00_first_search.py` in Phase 2
  (plan L1200, L594).
- `examples/core/tutorial_minimal.py` -> retire; superseded by the new
  Category A trio
  (`tutorial_00_first_search.py`, `tutorial_01_first_recording.py`,
  `tutorial_02_dataset_to_dataloader.py`)
  (plan L194, L367-370, L592-596).
- `examples/core/tutorial_eoec.py` -> rewrite as
  `examples/tutorials/30_resting_state/tutorial_30_eyes_open_closed.py` in
  Phase 2 step 4 (plan L195, L405-406, L607-609, L972-975, L1203).
- `examples/tutorials/noplot_tutorial_features_eoec.py` -> rewrite as
  `examples/tutorials/40_features/tutorial_41_feature_trees.py` (also a
  secondary source for `tutorial_30_eyes_open_closed.py`)
  (plan L203, L418-419, L972-975, L1005-1007).
- `examples/core/tutorial_feature_extractor_open_close_eye.py` -> rewrite as
  `examples/tutorials/40_features/tutorial_40_first_features.py`
  (plan L196, L417, L610-614, L977-987, L1205).
- `examples/tutorials/noplot_tutorial_feature_extraction.py` -> rewrite as
  `examples/tutorials/40_features/tutorial_42_features_to_sklearn.py`
  (plan L202, L420-421, L1009-1019, L1205).
- `examples/tutorials/noplot_tutorial_p3_oddball.py` -> rewrite as
  `examples/tutorials/20_event_related/tutorial_20_visual_p300_oddball.py`
  (plan L201, L393, L605, L941-958).
- `examples/tutorials/noplot_tutorial_audi_oddball.py` -> rewrite as
  `examples/tutorials/20_event_related/tutorial_21_auditory_oddball.py`
  (plan L200, L394-395, L606).
- `examples/core/p300_transfer_learning.py` ->
  `examples/applied/` (advanced/transfer-learning track; exact filename
  resolved in Phase 3)
  (plan L197, L621-626, L1208-1210).
- `examples/tutorials/noplot_tutorial_age_prediction.py` ->
  `examples/applied/project_age_regression.py`
  (plan L199, L450, L623, L1057-1066, L1210).
- `examples/tutorials/noplot_tutorial_pfactor_features.py` ->
  `examples/applied/project_pfactor_features.py`
  (plan L204, L451, L624, L1068-1079, L1210).
- `examples/tutorials/noplot_tutorial_pfactor_regression.py` ->
  `examples/applied/project_pfactor_deep.py`
  (plan L205, L452, L625, L1068-1079, L1210).
- `examples/tutorials/noplot_tutorial_sex_classification_cnn.py` ->
  `examples/applied/project_sex_classification.py`
  (plan L206, L453-454, L626, L1081-1090, L1210).
- `examples/tutorials/plot_clinical_summary.py` ->
  `examples/applied/project_clinical_dataset_summary.py` (or under
  `examples/how_to/` per plan L1094-1095)
  (plan L207, L455-456, L1092-1095).
- `examples/tutorials/tutorial_transfer_learning.py` -> retire; do not
  republish as a public tutorial unless rewritten with real EEGDash data
  (plan L208, L1097-1100).
- `examples/eeg2025/tutorial_eegdash_offline.py` ->
  `examples/how_to/how_to_work_offline.py`, with cross-link from the HPC
  track (plan L211, L478, L615-621, L1021-1037, L1202).
- `examples/hpc/tutorial_eoec.py` ->
  `examples/hpc/tutorial_hpc_cache_and_slurm.py` (do not republish as a
  beginner duplicate of EO/EC) (plan L212, L631-633).
- `examples/eeg2025/tutorial_challenge_1.py` and
  `examples/eeg2025/tutorial_challenge_2.py` -> keep in place; surface under
  the separate "Competition and foundation challenge" track
  (plan L209-210, L466-467, L627-630).

## Public-gallery cleanup

Per plan L1190 ("Remove dev scripts from public gallery"), the following
file(s) under `examples/dev_scripts/` must not be rendered into the public
Sphinx-Gallery output (move to developer notes if needed, per plan L213):

- `examples/dev_scripts/debug_pybids_braindecode.py`

This is the only file currently present under `examples/dev_scripts/` other
than `README.txt`. The plan's inventory line L213 names this file
explicitly.

## Naming hygiene

Per plan L1193-1196 ("Fix naming: no `noplot_tutorial_*` as public names;
consistent title casing; consistent `braindecode` spelling"), the
following `noplot_tutorial_*.py` filenames need renaming as part of the
rewrites listed above. Each is shown alongside the target spec id so that
no `noplot_tutorial_*` filename survives as a public gallery entry.

- `examples/tutorials/noplot_tutorial_age_prediction.py`
  -> `project_age_regression.py`
- `examples/tutorials/noplot_tutorial_audi_oddball.py`
  -> `tutorial_21_auditory_oddball.py`
- `examples/tutorials/noplot_tutorial_feature_extraction.py`
  -> `tutorial_42_features_to_sklearn.py`
- `examples/tutorials/noplot_tutorial_features_eoec.py`
  -> `tutorial_41_feature_trees.py`
- `examples/tutorials/noplot_tutorial_p3_oddball.py`
  -> `tutorial_20_visual_p300_oddball.py`
- `examples/tutorials/noplot_tutorial_pfactor_features.py`
  -> `project_pfactor_features.py`
- `examples/tutorials/noplot_tutorial_pfactor_regression.py`
  -> `project_pfactor_deep.py`
- `examples/tutorials/noplot_tutorial_sex_classification_cnn.py`
  -> `project_sex_classification.py`
