# EEGDash Tutorial Refactor — Implementation Strategy and Evidence System

Date: 2026-05-06

## Purpose

This document is the implementation contract for the tutorial refactor described
in `docs/tutorial_restructure_plan.md`. It defines:

1. The fixed list of tutorials to build (verbatim from the plan).
2. A literature-anchored 49-rule validation rubric, sourced from
   `new_tutorials/compass_artifact_wf-96e3d362-5e7a-4e1f-82f3-9095193459df_text_markdown.md`.
3. An operational 12-dimension scorecard, sourced from
   `new_tutorials/validation_documentation.md`.
4. A three-role coordination pipeline (author / evaluator / CI gate) with
   concrete Python validators and a GitHub Actions workflow.
5. An evidence dossier format that lets every tutorial PR show, before merge,
   exactly which rubric rules it satisfies and which it does not.

The point of the system is that "we are following the plan" must be a
reproducible, machine-verifiable claim, not an opinion.

## Source documents (binding)

| Document | Role | Path |
| --- | --- | --- |
| Tutorial restructure plan | Prescribes which tutorials exist, naming, file layout, design template, quality bar, migration phases. | `docs/tutorial_restructure_plan.md` |
| Data-viz design | Prescribes palette, figure rules, card layout, identity helpers. | `docs/design/data-viz-design.md` |
| Compass artifact (rubric) | Provides the 49-rule rubric (E.1-E.6) anchored to cognitive science, Diataxis, sphinx-gallery, MNE, Braindecode, neuroscience-specific best practice. | `new_tutorials/compass_artifact_wf-96e3d362-5e7a-4e1f-82f3-9095193459df_text_markdown.md` |
| Validation documentation | Provides the 12-dimension operational checklist, the 10-method automated validation matrix, working Python validators for structure/execution/budget/engagement, and the author/evaluator/CI agent pipeline. | `new_tutorials/validation_documentation.md` |

Every validator in this document carries `cite_rubric` (back to the compass
artifact rule) and `cite_plan` (back to the plan section) so the chain from
tutorial to evidence to literature is auditable.

## Design principles

1. **Spec-first**. No tutorial gets written without a YAML spec. The spec
   declares its rubric expectations before code exists.
2. **Validators cite literature**. Each rule references either an empirical
   paper or an authoritative tool (Diataxis, sphinx-gallery, Cisotto & Chicco
   2024, Cepeda 2006, Sweller & Cooper 1985, Rule et al. 2019, Sentance & Waite
   2017, Wilkinson et al. 2016). Goodharting a validator is harder when the
   rule has a citation.
3. **Difficulty-aware rubric**. The 49 rules do not all apply uniformly. A
   beginner tutorial needs full PRIMM scaffolding; an advanced project tutorial
   relaxes the same rules (Kalyuga et al. 2003 — expertise-reversal effect).
   The spec's `difficulty` field gates which rules are required.
4. **Static cheap, runtime gated**. Run static checks on every PR. Run runtime
   checks (clean kernel, budgets, visual regression) on a smaller matrix and
   nightly. Reserve full multi-version matrix for `develop`.
5. **Evidence is committed**. Every audit run writes a deterministic JSON file
   into `docs/evidence/tutorials/<id>/`. PR diffs show which rules flipped.
6. **Three roles, no role-mixing on the same tutorial**. Author writes;
   Evaluator audits; Reviewer (human or LLM) handles judgment-only rules.
7. **Loops are bounded**. Inner = a few hours per tutorial. Middle = a week per
   category. Outer = a phase per the plan's migration plan.
8. **Baseline before change**. Run the validators on the existing 20-file
   gallery first; commit the snapshot as `_baseline_2026-05-06/`. Every later
   improvement claim is referenced against that snapshot.

## The 13 tutorials, taken verbatim from the plan

Source: `docs/tutorial_restructure_plan.md`, section "Recommended Initial
Tutorial Set" and "Proposed Tutorial Categories". Filenames, categories, and
ordering match the plan's `Proposed File Layout`.

### Release 1: Core learning path

| Order | File | Category | Plan source |
| ---: | --- | --- | --- |
| 0 | `examples/tutorials/00_start_here/plot_00_first_search.py`              | A | plan §Cat A item 1, §Detailed L819 |
| 1 | `examples/tutorials/00_start_here/plot_01_first_recording.py`           | A | plan §Cat A item 2, §Detailed L846 |
| 2 | `examples/tutorials/00_start_here/plot_02_dataset_to_dataloader.py`     | A | plan §Cat A item 3, §Detailed L865 |
| 3 | `examples/tutorials/10_core_workflow/plot_10_preprocess_and_window.py`  | B | plan §Cat B item 1, §Detailed L883 |
| 4 | `examples/tutorials/10_core_workflow/plot_11_leakage_safe_split.py`     | B | plan §Cat B item 2, §Detailed L902 |
| 5 | `examples/tutorials/10_core_workflow/plot_12_train_a_baseline.py`       | B | plan §Cat B item 3, §Detailed L921 |
| 6 | `examples/tutorials/10_core_workflow/plot_13_save_and_reuse_prepared_data.py` | B | plan §Cat B item 4 |
| 7 | `examples/tutorials/40_features/plot_40_first_features.py`              | E | plan §Cat E item 1, §Detailed L977 |

### Release 2: Topical extensions

| File | Category | Plan source |
| --- | --- | --- |
| `examples/tutorials/20_event_related/plot_20_visual_p300_oddball.py`    | C | plan §Cat C item 1, §Detailed L941 |
| `examples/tutorials/20_event_related/plot_21_auditory_oddball.py`       | C | plan §Cat C item 2 |
| `examples/tutorials/30_resting_state/plot_30_eyes_open_closed.py`       | D | plan §Cat D item 1, §Detailed L960 |
| `examples/tutorials/40_features/plot_41_feature_trees.py`               | E | plan §Cat E item 2, §Detailed L994 |
| `examples/tutorials/40_features/plot_42_features_to_sklearn.py`         | E | plan §Cat E item 3, §Detailed L1009 |

### Out-of-scope for the first two releases

- Categories F (Evaluation/Benchmarking), G (Applied projects), H (Transfer/foundation), I (HPC) are tracked in the plan but ship in later phases.
- Existing `examples/applied/`, `examples/eeg2025/`, `examples/hpc/` keep their content, audited but not blocked on the rubric's beginner-mode rules.

## Spec contract

Single source of truth: `docs/tutorials/_spec/<tutorial_id>.yaml`. Authoring
without a spec entry is rejected by `check_E1_spec_present`.

```yaml
id: plot_11_leakage_safe_split
category: B-core-workflow
title: "Split EEG data without subject leakage"
state: proposed                  # proposed | drafted | static-pass | runtime-pass | reviewed | merged
assignee: null
difficulty: 1                    # 1-star (beginner) | 2-star (intermediate) | 3-star (advanced)
estimated_runtime_minutes: 3
hardware: cpu
network: cached_first_run
prerequisites:
  - plot_10_preprocess_and_window
cites:
  plan:
    - "tutorial_restructure_plan.md#L380-L384"     # Category B placement
    - "tutorial_restructure_plan.md#L497"          # Release 1 row 4
    - "tutorial_restructure_plan.md#L902-L920"     # Detailed proposal
    - "tutorial_restructure_plan.md#L1158-L1178"   # Quality Bar
  rubric:
    - "compass_artifact.md#E5.42"                  # subject-aware split
    - "compass_artifact.md#E3.21-E3.30"            # reproducibility rules
learning_objectives:                # rubric E2.20: 3-5 bullets, "After this tutorial you will be able to..."
  - "Build a leakage-safe train/test split on EEG records using `eegdash.splits.get_splitter`"
  - "Run `assert_no_leakage` and read its report"
  - "Compare a leaky random split against a subject-aware split on the same data"
audience: "Beginner ML user with one prior EEGDash tutorial"
budgets:
  max_loc: 220                      # plan §Quality Bar L1174
  max_runtime_seconds: 180          # plan §Quality Bar L1165
  max_network_mb: 50
  cpu_only: true
  gpu_required: false
artifacts:
  thumbnail: thumbnails/plot_11.png
  figures:
    - figure_split_audit.png
    - figure_leaky_vs_safe_confusion.png
asserted_invariants:                # rubric E3.27, plan §Tutorial Design Template L535-541
  - "n_subjects_total > 10"
  - "subject_overlap(train, test) == 0"
  - "n_folds >= 5"
  - "abs(class_balance_train - class_balance_test) < 0.10"
sections_required:                  # rubric E1.4, plan §Tutorial Design Template
  - title
  - opening
  - learning_objectives
  - requirements
  - setup
  - lesson_steps:
      min: 5
      labelled_subgoals: true       # rubric E2 — Sweller & Cooper 1985 + sub-goal labels
  - checkpoints:
      min: 4
  - result
  - wrapup
  - links
primm_required:                     # rubric E2.13 — Sentance & Waite 2017
  predict: 1
  run: ">=2"
  investigate: 1
  modify: 1
  make: 1
faded_scaffolding:                  # rubric E2.16 — Weinman et al. 2021
  required: true
  pattern: "completion"             # completion | parsons | full
viz_compliance:                     # data-viz-design.md
  palette: eegdash-rail
  required_elements:
    - chance_level_line
    - subject_count_in_subtitle
    - source_provenance_footer
links:
  concept: docs/source/concepts/leakage_and_evaluation.rst
  api: eegdash.splits
  related_how_to:
    - how_to_handle_bad_records
  related_papers:
    - "Cisotto & Chicco 2024, PeerJ CS — clinical EEG ten quick tips"
requires_api:                       # plan §Implementation Backlog Workstream 3
  - "eegdash.splits.assert_no_leakage"
  - "eegdash.splits.make_split_manifest"
review_rubric_overrides: {}         # difficulty-aware exemptions, see Appendix A
```

## The 49-rule validation rubric

Sourced from the compass artifact §E. Each rule maps to one or more concrete
validators. Rule IDs are stable: `E<group>.<rule>` (e.g., `E5.42`).

### E.1 Structural / sphinx-gallery conformity (10 rules)

| ID | Rule | Tool / validator | Level |
| --- | --- | --- | --- |
| E1.1 | File named `plot_*.py` so sphinx-gallery executes it. | static `check_filename` | error |
| E1.2 | First lines are a triple-quoted module docstring with reST H1 title and 2-4 sentence motivating problem statement naming dataset and scientific question. | static `check_docstring_header` | error |
| E1.3 | Lives in a gallery sub-folder with `README.txt` (sphinx-gallery convention) defining the section. | static `check_gallery_section` | error |
| E1.4 | Code blocks separated from prose using consistent block delimiters (`# %%` or `# %% [markdown]`). | static `check_block_delimiters` | error |
| E1.5 | Imports grouped (stdlib → third-party → eegdash/braindecode/mne) at the top, no hidden later imports. | static AST `check_imports_clean` | warn |
| E1.6 | Every figure has axis labels, units, title, legend if multiple traces. | runtime `check_figure_labels` | error |
| E1.7 | Ends with "References" / "See also" / "Next steps" linking to user guide, related examples, primary literature (DOI). | static `check_footer_links` | warn |
| E1.8 | CPU runtime under spec budget; longer ones marked and use `mini=True` releases. | runtime `check_runtime_budget` | error |
| E1.9 | Runs top-to-bottom from a clean kernel. | runtime `check_clean_kernel_run` | error |
| E1.10 | Sphinx-gallery generates downloadable `.ipynb` and `.py`. | build `check_gallery_artifacts` | warn |

### E.2 Pedagogical (cognitive load + PRIMM + scaffolding) (10 rules)

| ID | Rule | Tool / validator | Level |
| --- | --- | --- | --- |
| E2.11 | One narrative: beginning ("why we care") → middle ("how") → end ("what we found, next"). | reviewer (LLM/human) `score_narrative` | warn |
| E2.12 | Worked-example structure for novices: complete code shown then explained before learner is asked to modify (Sweller & Cooper 1985). | static `check_worked_example_order` | warn |
| E2.13 | PRIMM elements visible: at least one Predict, Run, Investigate, Modify, Make (Sentance & Waite 2017). | static `check_primm_blocks` | error (1-star), warn (2-star), exempt (3-star) |
| E2.14 | Cognitive load managed: each cell does one conceptual thing; long expressions broken into named intermediates; figure adjacent to its prose. | static `check_cell_complexity` + reviewer | warn |
| E2.15 | Tagged for level (★/★★/★★★); does not over-explain at wrong level (Kalyuga et al. 2003). | static `check_difficulty_tag` | error |
| E2.16 | At least one faded or completion exercise (Weinman et al. 2021). | static `check_completion_exercise` | error (1-star), warn (2-star), exempt (3-star) |
| E2.17 | Errors shown intentionally at least once and recovered from in prose (Nederbragt et al. 2020). | static `check_intentional_error` | warn |
| E2.18 | Active-learning prompts every ~10-15 minutes of estimated reading (Brown & Wilson 2018). | static `check_prompt_density` | warn |
| E2.19 | Vocabulary introduced before use (BIDS, montage, epoch, ICA, ERP, ...) with one-sentence operational definitions. | static `check_glossary_intro` | warn |
| E2.20 | Opening sets explicit learning objectives (3-5 bullets). | static `check_learning_objectives` | error |

### E.3 Technical (reproducibility, correctness, runnability) (10 rules)

| ID | Rule | Tool / validator | Level |
| --- | --- | --- | --- |
| E3.21 | All RNGs seeded (`np.random.seed`, `torch.manual_seed`, `random_state=`). | static AST `check_seeds` | error |
| E3.22 | Versions pinned or printed (`mne.sys_info()`, `print(eegdash.__version__)`). | static `check_versions_printed` | warn |
| E3.23 | Small deterministic data subset (one subject, one task, `mini=True`) so it runs in CI. | spec + runtime `check_data_minimality` | error |
| E3.24 | Data downloaded via `eegdash` caching API; cache directory parametrised, not hard-coded. | static `check_no_hardcoded_paths` | error |
| E3.25 | Restart-and-run-all without warnings other than expected scientific ones. | runtime `check_clean_warnings` | warn |
| E3.26 | Outputs (figures, prints) committed as artefacts. | runtime `check_outputs_committed` | warn |
| E3.27 | Computation split so learner can stop at any cell with a meaningful intermediate result. | static `check_intermediate_results` + reviewer | warn |
| E3.28 | Handles offline/airgapped case (`download=False`); documents network requirements. | static `check_offline_mention` | warn |
| E3.29 | Heavy training loops use tiny epochs/small model and say "increase this for real work". | static `check_tiny_training_disclosure` | warn |
| E3.30 | Notebook does not silently `pip install` packages mid-execution. | static `check_no_inline_pip` | error |

### E.4 Engagement (narrative, authentic data, motivation) (6 rules)

| ID | Rule | Tool / validator | Level |
| --- | --- | --- | --- |
| E4.31 | First 5 lines name a real neuroscience question, not a generic "this example shows class X". | reviewer `score_motivating_question` | warn |
| E4.32 | Dataset is real, citable, has DOI; eegdash auto-citation surfaced. | static `check_dataset_citation` | error |
| E4.33 | Result has scientific meaning (chance vs above-chance, interpretable spectrum, ERP that resembles literature). | runtime `check_result_meaningful` + reviewer | warn |
| E4.34 | Conclusion has "Try it yourself / Extensions" with at least 3 graded modifications. | static `check_extensions_section` | warn |
| E4.35 | Tone "we"-inclusive, present tense, explains the *why* of choices. | reviewer `score_tone` | info |
| E4.36 | Includes recognisable EEG canon figures (topomap, PSD, ERP butterfly, confusion matrix). | runtime `check_canon_figure_present` | warn |

### E.5 Domain-correctness (EEG / neuroscience accuracy) (10 rules)

Anchored to Cisotto & Chicco (2024) ten quick tips for clinical EEG.

| ID | Rule | Tool / validator | Level |
| --- | --- | --- | --- |
| E5.37 | Filtering choices justified; pass-band, stop-band, filter type reported; causal/non-causal flag set appropriately. | static `check_filter_disclosed` | error |
| E5.38 | Reference scheme and montage explicit (`set_montage`, `set_eeg_reference`). | static `check_reference_montage` | error |
| E5.39 | Bad-channel handling at least mentioned. | static `check_bad_channel_mention` | warn |
| E5.40 | Artefact strategy named (ICA, autoreject, threshold) with citation. | static `check_artefact_strategy` | warn |
| E5.41 | Epochs use baseline correction; sign convention/units stated (µV). | static `check_epoch_baseline_units` | warn |
| E5.42 | If classifier is trained, train/test split is **subject-aware**. The single most common EEG-ML mistake. | runtime `check_no_subject_leakage` (uses `eegdash.splits.assert_no_leakage`) | error |
| E5.43 | Class balance and chance level explicitly computed and displayed alongside accuracy. | runtime `check_chance_level_reported` | error |
| E5.44 | BIDS entities (`dataset`, `subject`, `task`, `session`, `run`) surfaced in prose, not hidden. | static `check_bids_entities_surfaced` | warn |
| E5.45 | Cites dataset paper (DOI), eegdash entry, MNE-Python (Gramfort et al. 2013), Braindecode (Schirrmeister et al. 2017) where appropriate. | static `check_canonical_citations` | warn |
| E5.46 | Does not over-claim; phrasing hedged; limitations flagged. | reviewer `score_hedging` | warn |

### E.6 Diataxis purity (3 rules)

| ID | Rule | Tool / validator | Level |
| --- | --- | --- | --- |
| E6.47 | Document is unambiguously a tutorial — one rail, one outcome; reference material in API docs, not embedded. | reviewer `score_diataxis_purity` | warn |
| E6.48 | Where deeper conceptual material would help, links to an explanation page rather than inlining. | static `check_concept_link_present` | warn |
| E6.49 | Where competent user would want a quick recipe, splits out a separate how-to. | reviewer | info |

### Difficulty-aware rule activation

Mapping is encoded in `scripts/tutorial_audit/rule_matrix.yaml`:

| Rule group | 1-star | 2-star | 3-star |
| --- | --- | --- | --- |
| E.1 (structural) | all error | all error | all error |
| E.2.13 (PRIMM) | error | warn | exempt |
| E.2.16 (faded) | error | warn | exempt |
| E.2.17 (intentional error) | warn | warn | info |
| E.2.18 (prompt density) | warn | info | info |
| E.3 (reproducibility) | all error/warn | all error/warn | all error/warn |
| E.4 (engagement) | warn | warn | info |
| E.5 (domain) | full set | full set | full set |
| E.6 (Diataxis) | warn | warn | warn |

This encodes Kalyuga et al. 2003: heavier scaffolding for novice tutorials,
lighter for advanced. Project-tier examples don't need PRIMM blocks.

## Operational checklist (12 dimensions)

Sourced from `validation_documentation.md`. Used as a per-PR scorecard,
emitted as a markdown table inside the evidence dossier.

| Dimension | Minimum | Excellence | Validators that contribute |
| --- | --- | --- | --- |
| Audience | Persona, prerequisites, exit objective explicit | Measurable objectives aligned with final assessment | E2.20, spec.audience |
| Structure | H1, objectives, setup, dataset, exercises, recap | Navigable Sphinx TOC, estimated time per block | E1.2, E1.4, E2.20 |
| Examples | At least one worked example per key concept | Sub-goal labels and faded scaffolding through the series | E2.12, E2.16 |
| Retrieval | Short check every 10-15 min | Predictions before execution, final memory recap | E2.13, E2.18 |
| Spacing | Core concepts reappear later | Reappear in new context with active production | cross-tutorial: `check_concept_revisit` |
| Interleaving | Similar variants appear adjacent | Contrasting cases force strategic discrimination | reviewer + cross-tutorial check |
| Feedback | `assert` or visible criteria in exercises | Specific feedback, gradient hints, task-oriented messages | E3.27, runtime asserts |
| Data | Small, licensed, versioned, stable dataset | Local cache, checksum, "tiny" sample for CI | E3.23, E3.24, E4.32 |
| Reproducibility | Specified env, clean kernel, top-down execution | Binder by commit, paired notebooks, auto publishing | E3.21-E3.30, E1.9 |
| Accessibility | Correct headings, alt text, textual graph summaries | Alternative tables, clear language, contrast | E1.6, custom `check_alt_text` |
| Community | Inclusive language, code-of-conduct adherence | Culturally neutral examples | reviewer |
| Reuse | Public repo, exec instructions | Versioned artefacts, releases, changelog, bibliography | E1.7, evidence dossier itself |

## The pipeline

Three roles, mirrored from `validation_documentation.md` §"Agent pipeline and CI".

```
spec.yaml ──► author writes plot_NN_*.py ──► local self-audit (static)
                                                       │
                                                       ▼
                                                  PR opens
                                                       │
                                                       ▼
                              ┌────────────  CI: tutorial-audit.yml  ────────────┐
                              │  1. structural rubric        (static, fast)     │
                              │  2. notebook lint            (nbqa+ruff+black)  │
                              │  3. clean execution          (nbclient)         │
                              │  4. parameterised tiny run   (papermill)        │
                              │  5. unit + visual regression (pytest+mpl)       │
                              │  6. time/memory budget       (psutil)           │
                              │  7. accessibility checks     (custom)           │
                              │  8. PRIMM/E.2 rubric         (static)           │
                              │  9. EEG domain checks        (static+runtime)   │
                              │ 10. evidence.json written    (deterministic)    │
                              └─────────────────────────────────────────────────┘
                                                       │
                                            errors > 0 ──► fail; comment summary
                                            warnings > 0 ──► comment, allow merge
                                            zero errors  ──► reviewer gate
                                                                │
                                                                ▼
                                                        merge + dossier committed
```

Stages 1-2 and 8-10 run on every PR. Stages 3-7 run on PRs that touch the
tutorial source or its spec, and nightly on `develop`.

## Per-tutorial state machine

State persisted in the spec YAML:

```
proposed → drafted → static-pass → runtime-pass → reviewed → merged
                          ↑              │             │
                          └─── any fail ─┴── rolls back to drafted
```

CI guards transitions: state can only advance when the matching evidence
artifacts are present and have zero error-level findings. State can only
regress if a validator now reports errors.

Concurrent claims rejected: spec YAML has `assignee` field; CI rejects YAML
diffs that change `assignee` without an accompanying state transition.

Make targets (defined in `Makefile`):

```
make tutorial-claim   TUTORIAL=plot_11_leakage_safe_split BY=author-A
make tutorial-audit   TUTORIAL=plot_11_leakage_safe_split
make tutorial-baseline                                      # audit all current files, write _baseline_<date>/
make tutorial-dossier TUTORIAL=plot_11_leakage_safe_split   # render evidence.json into report.md
make tutorial-release TUTORIAL=plot_11_leakage_safe_split   # only if state=reviewed
make tutorial-phase-report PHASE=2                          # delta vs baseline
```

## Validator implementations

Layout: `scripts/tutorial_audit/`. Each validator is a pure function
`(tutorial_path, spec) -> List[Finding]`. Findings are accumulated and serialised
into the evidence dossier.

```
scripts/tutorial_audit/
  __init__.py
  api.py                  # Finding dataclass, run_audit() entrypoint
  rule_matrix.yaml        # difficulty-aware rule activation
  static/
    e1_structural.py      # E1.1-E1.5, E1.7
    e2_pedagogical.py     # E2.12-E2.20 except E2.11/E2.14/E2.17 reviewer-only parts
    e3_technical.py       # E3.21-E3.24, E3.28-E3.30
    e4_engagement.py      # E4.32, E4.34
    e5_domain.py          # E5.37-E5.41, E5.44-E5.45
    e6_diataxis.py        # E6.48
  runtime/
    e1_runtime.py         # E1.6, E1.8, E1.9, E1.10
    e3_runtime.py         # E3.25, E3.26
    e4_runtime.py         # E4.33, E4.36
    e5_runtime.py         # E5.42 (no_subject_leakage), E5.43 (chance_level_reported)
    budgets.py            # runtime/memory budgets, network bytes
    visual.py             # matplotlib.testing baseline diffs, palette compliance
  reviewer/
    rubric_judge.py       # LLM/human-driven E2.11/E4.31/E4.35/E5.46/E6.47 stubs
  pipeline.py             # orchestrates static then runtime then reviewer
  report.py               # renders evidence.json into report.md + dashboard rows
```

### Finding dataclass

```python
# scripts/tutorial_audit/api.py
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Level = Literal["error", "warn", "info"]

@dataclass(frozen=True)
class Finding:
    rule_id: str                    # e.g., "E5.42"
    level: Level
    message: str
    cite_rubric: str                # e.g., "compass_artifact.md#E5"
    cite_plan: str                  # e.g., "tutorial_restructure_plan.md#L1158-L1178"
    evidence: dict[str, Any] = field(default_factory=dict)
    tool: str = ""                  # e.g., "nbclient", "ruff", "matplotlib.testing"

    def to_dict(self) -> dict:
        return asdict(self)
```

### Static structural checks (E.1)

Adapted directly from `validation_documentation.md` §"Structural and pedagogical
rubric", lifted from notebooks to sphinx-gallery `plot_*.py`:

```python
# scripts/tutorial_audit/static/e1_structural.py
from __future__ import annotations
import ast
import re
from pathlib import Path
from ..api import Finding

REST_TITLE_RE = re.compile(r"^={3,}\s*$")
BLOCK_DELIM_RE = re.compile(r"^# %%(\s*\[markdown\])?\s*$")

def check_filename(path: Path, spec: dict) -> list[Finding]:
    name = path.name
    if not name.startswith("plot_") or not name.endswith(".py"):
        return [Finding(
            rule_id="E1.1", level="error",
            message=f"Tutorial filename must match plot_*.py for sphinx-gallery; got {name}",
            cite_rubric="compass_artifact.md#E1.1",
            cite_plan="tutorial_restructure_plan.md#L1194",
            tool="filename-check",
        )]
    return []

def check_docstring_header(path: Path, spec: dict) -> list[Finding]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    docstring = ast.get_docstring(tree)
    findings: list[Finding] = []
    if not docstring:
        findings.append(Finding(
            rule_id="E1.2", level="error",
            message="Module-level docstring missing",
            cite_rubric="compass_artifact.md#E1.2",
            cite_plan="tutorial_restructure_plan.md#L516-L517",
            tool="ast",
        ))
        return findings
    lines = docstring.strip().splitlines()
    title_line = lines[0] if lines else ""
    if len(lines) < 2 or not REST_TITLE_RE.match(lines[1] if len(lines) > 1 else ""):
        findings.append(Finding(
            rule_id="E1.2", level="error",
            message="Docstring must open with a reST H1 title (line of '=' under the title)",
            cite_rubric="compass_artifact.md#E1.2",
            cite_plan="tutorial_restructure_plan.md#L516",
            tool="regex",
        ))
    paras = [p for p in docstring.split("\n\n") if p.strip()]
    if len(paras) < 2 or len(paras[1].split()) < 30:
        findings.append(Finding(
            rule_id="E1.2", level="warn",
            message="Motivating paragraph after title should be 2-4 sentences naming dataset and scientific question",
            cite_rubric="compass_artifact.md#E1.2",
            cite_plan="tutorial_restructure_plan.md#L517-L518",
            tool="regex",
        ))
    return findings

def check_block_delimiters(path: Path, spec: dict) -> list[Finding]:
    src = path.read_text(encoding="utf-8")
    has_delim = any(BLOCK_DELIM_RE.match(line) for line in src.splitlines())
    if not has_delim:
        return [Finding(
            rule_id="E1.4", level="error",
            message="Tutorial must use '# %%' or '# %% [markdown]' block delimiters",
            cite_rubric="compass_artifact.md#E1.4",
            cite_plan="tutorial_restructure_plan.md#L530",
            tool="regex",
        )]
    return []

def check_loc_budget(path: Path, spec: dict) -> list[Finding]:
    n = sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
    cap = spec.get("budgets", {}).get("max_loc", 220)
    if n > cap:
        return [Finding(
            rule_id="E1.8", level="error",
            message=f"LOC {n} exceeds budget {cap}",
            cite_rubric="compass_artifact.md#E1.8",
            cite_plan="tutorial_restructure_plan.md#L1174",
            evidence={"loc": n, "budget": cap},
            tool="wc",
        )]
    return []
```

### PRIMM detector (E.2.13)

```python
# scripts/tutorial_audit/static/e2_pedagogical.py
import re
from pathlib import Path
from ..api import Finding

PRIMM_HEADERS = {
    "predict":   re.compile(r"^\s*\*?\*?(predict|prediction)\b", re.I),
    "run":       re.compile(r"^\s*\*?\*?(run|let'?s run)\b", re.I),
    "investigate": re.compile(r"^\s*\*?\*?(investigate|why this output|what we see)\b", re.I),
    "modify":    re.compile(r"^\s*\*?\*?(modify|your turn|change|try changing)\b", re.I),
    "make":      re.compile(r"^\s*\*?\*?(make|try it yourself|mini-project|extension)\b", re.I),
}

def check_primm_blocks(path: Path, spec: dict) -> list[Finding]:
    difficulty = spec.get("difficulty", 1)
    required = spec.get("primm_required") or {"predict": 1, "run": 2, "investigate": 1, "modify": 1, "make": 1}
    if difficulty == 3:
        return []  # exempt for advanced/project tutorials (Kalyuga 2003)
    src = path.read_text(encoding="utf-8")
    md_blocks = re.split(r"^# %% \[markdown\]\s*$", src, flags=re.M)[1:]
    counts = {k: 0 for k in PRIMM_HEADERS}
    for block in md_blocks:
        for line in block.splitlines():
            stripped = line.lstrip("# ").strip()
            for kind, pat in PRIMM_HEADERS.items():
                if pat.search(stripped):
                    counts[kind] += 1
                    break
    findings = []
    for kind, target in required.items():
        target_n = int(str(target).lstrip(">=").strip())
        if counts[kind] < target_n:
            level = "error" if difficulty == 1 else "warn"
            findings.append(Finding(
                rule_id="E2.13", level=level,
                message=f"PRIMM '{kind}' block: found {counts[kind]}, need {target}",
                cite_rubric="compass_artifact.md#E2.13",
                cite_plan="tutorial_restructure_plan.md#L516-L550",
                evidence={"counts": counts, "required": required, "difficulty": difficulty},
                tool="regex",
            ))
    return findings
```

### Subject-leakage runtime check (E.5.42)

```python
# scripts/tutorial_audit/runtime/e5_runtime.py
import json
from pathlib import Path
from nbclient import NotebookClient
import nbformat
from ..api import Finding

def check_no_subject_leakage(executed_nb_path: Path, spec: dict) -> list[Finding]:
    """
    Inspect notebook outputs for an emitted leakage report from
    `eegdash.splits.assert_no_leakage`. The tutorial is required to print a
    JSON line: {"leakage_report": {"overlap": <int>, "by": "subject"}}.
    """
    nb = nbformat.read(executed_nb_path, as_version=4)
    overlap = None
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", "")) if out.get("output_type") == "stream" else str(out.get("data", {}).get("text/plain", ""))
            if "leakage_report" in text:
                try:
                    data = json.loads(text.splitlines()[-1])
                    overlap = data["leakage_report"]["overlap"]
                except Exception:
                    continue
    if overlap is None:
        return [Finding(
            rule_id="E5.42", level="error",
            message="Tutorial trains a classifier without printing a leakage_report",
            cite_rubric="compass_artifact.md#E5.42",
            cite_plan="tutorial_restructure_plan.md#L902-L920",
            tool="nbclient+json",
        )]
    if overlap > 0:
        return [Finding(
            rule_id="E5.42", level="error",
            message=f"Subject overlap between train and test: {overlap} subjects",
            cite_rubric="compass_artifact.md#E5.42",
            cite_plan="tutorial_restructure_plan.md#L902-L920",
            evidence={"overlap": overlap},
            tool="eegdash.splits.assert_no_leakage",
        )]
    return []
```

### Clean-execution + reproducibility (E.1.9, E.3.25, E.3.26)

```python
# scripts/tutorial_audit/runtime/e1_runtime.py
import hashlib, json, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from ..api import Finding

def execute_to_artifact(plot_py: Path, dst_ipynb: Path, timeout: int = 300) -> tuple[float, list[str]]:
    # convert plot_*.py to a notebook via sphinx-gallery's converter, then execute
    import sphinx_gallery.notebook  # type: ignore
    nb = sphinx_gallery.notebook.python_to_jupyter_cli([str(plot_py), "--out-file", str(dst_ipynb)])
    start = time.perf_counter()
    nb_obj = nbformat.read(dst_ipynb, as_version=4)
    client = NotebookClient(nb_obj, timeout=timeout, kernel_name="python3", record_timing=True)
    client.execute()
    nbformat.write(nb_obj, dst_ipynb)
    elapsed = time.perf_counter() - start
    warnings_seen = [
        "".join(o.get("text", ""))
        for cell in nb_obj.cells if cell.cell_type == "code"
        for o in cell.get("outputs", []) if o.get("name") == "stderr"
    ]
    return elapsed, warnings_seen

def stable_output_hash(nb_path: Path) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    payload = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        outs = []
        for out in cell.get("outputs", []):
            o = dict(out); o.pop("execution_count", None)
            if "metadata" in o: o["metadata"] = {}
            outs.append(o)
        payload.append(outs)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
```

### Visual identity check (data-viz-design.md)

```python
# scripts/tutorial_audit/runtime/visual.py
from PIL import Image
from collections import Counter
from pathlib import Path
from ..api import Finding

EEGDASH_PALETTE = {
    (0x00, 0x6C, 0xA3),  # primary
    (0x00, 0x4A, 0x76),  # primary dark
    (0xF7, 0x94, 0x1D),  # accent (orange)
    (0x4F, 0x8C, 0xFF),  # sky
    (0x22, 0xD3, 0xEE),  # mint
    (0x10, 0x2A, 0x43),  # ink
    (0x64, 0x74, 0x8B),  # muted
    (0x7A, 0x8C, 0xA0),  # grid
    (0xF7, 0xFB, 0xFE),  # surface
    (0xFF, 0xFF, 0xFF),  # white
    (0x00, 0x00, 0x00),  # black
}

def check_palette_compliance(figure_path: Path, spec: dict) -> list[Finding]:
    img = Image.open(figure_path).convert("RGB")
    pixels = img.getdata()
    counts = Counter(pixels)
    offending = []
    total = 0
    for color, count in counts.most_common(50):
        total += count
        if min(abs(c - p) for c, p in zip(color, pal) for pal in EEGDASH_PALETTE) > 24:
            offending.append({"rgb": color, "frac": count / len(pixels)})
    if offending and sum(o["frac"] for o in offending) > 0.05:
        return [Finding(
            rule_id="DV.palette", level="warn",
            message=f"Figure uses non-palette colors covering >5% of pixels",
            cite_rubric="data-viz-design.md#palette",
            cite_plan="data-viz-design.md#L34-L52",
            evidence={"offending": offending[:5]},
            tool="PIL",
        )]
    return []
```

## Evidence dossier

Each tutorial: `docs/evidence/tutorials/<id>/`.

```
docs/evidence/tutorials/plot_11_leakage_safe_split/
  evidence.json           # full Findings list, deterministic
  report.md               # human-readable, embedded in PR comment + Sphinx
  thumbnail.png           # the gallery thumbnail
  timing.csv              # cell-level wall time (record_timing=True)
  outputs/                # asserted-invariant snapshots, hashes
  figures/                # all rendered figures, used for palette + canon checks
  baseline_thumbnail.png  # for matplotlib.testing diff
```

`evidence.json` structure:

```json
{
  "tutorial_id": "plot_11_leakage_safe_split",
  "spec_hash": "sha256:9f3c...",
  "spec_path": "docs/tutorials/_spec/plot_11_leakage_safe_split.yaml",
  "git_sha": "a63d165",
  "ran_at": "2026-05-06T18:34:11Z",
  "host": {"python": "3.11.7", "platform": "linux", "ci": "github-actions"},
  "totals": {"errors": 0, "warns": 1, "infos": 12, "passed": 47, "applicable": 48},
  "rule_results": [
    {"rule_id": "E1.1", "level": "info", "result": "pass"},
    {"rule_id": "E1.8", "level": "info", "result": "pass", "evidence": {"loc": 198, "budget": 220, "runtime_s": 142}},
    {"rule_id": "E2.13", "level": "info", "result": "pass", "evidence": {"counts": {"predict": 1, "run": 4, "investigate": 2, "modify": 1, "make": 1}}},
    {"rule_id": "E5.42", "level": "info", "result": "pass", "evidence": {"overlap": 0, "by": "subject", "n_subjects_train": 18, "n_subjects_test": 6}},
    {"rule_id": "DV.palette", "level": "warn", "result": "fail", "evidence": {"offending": [{"rgb": [31, 119, 180]}]}}
  ],
  "scorecard": {
    "audience": "pass", "structure": "pass", "examples": "pass",
    "retrieval": "pass", "spacing": "pass", "interleaving": "n/a",
    "feedback": "pass", "data": "pass", "reproducibility": "pass",
    "accessibility": "pass", "community": "pass", "reuse": "pass"
  }
}
```

`report.md` is rendered from `evidence.json` and committed alongside it. The
docs CI builds it into a "Behind the lesson" admonition appended to the
gallery page so readers can see the audit trail.

## CI workflow

Adapted from `validation_documentation.md` §"Example of a workflow in GitHub
Actions", with sphinx-gallery format and EEG-specific stages.

```yaml
# .github/workflows/tutorial-audit.yml
name: tutorial-audit
on:
  pull_request:
    paths:
      - "examples/tutorials/**"
      - "docs/tutorials/_spec/**"
      - "scripts/tutorial_audit/**"
  push:
    branches: [develop]
  schedule:
    - cron: "23 3 * * *"   # nightly regression

jobs:
  static:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install audit deps
        run: |
          pip install -e ".[docs,audit]"
          pip install ruff black nbqa pyyaml pillow
      - name: E.1 structural
        run: python -m scripts.tutorial_audit.pipeline --stage static --pattern "examples/tutorials/**/plot_*.py"
      - name: Notebook lint (nbqa wraps ruff/black on plot_*.py)
        run: |
          nbqa ruff examples/tutorials/**/plot_*.py
          nbqa black --check examples/tutorials/**/plot_*.py
      - name: Spec coherence
        run: python -m scripts.tutorial_audit.validate_spec
      - uses: actions/upload-artifact@v4
        with: { name: evidence-static, path: docs/evidence/tutorials/ }

  runtime:
    runs-on: ubuntu-latest
    needs: static
    if: github.event_name != 'pull_request' || contains(github.event.pull_request.changed_files, 'examples/tutorials')
    strategy:
      matrix:
        tutorial:
          - plot_00_first_search
          - plot_01_first_recording
          - plot_02_dataset_to_dataloader
          - plot_10_preprocess_and_window
          - plot_11_leakage_safe_split
          - plot_12_train_a_baseline
          - plot_13_save_and_reuse_prepared_data
          - plot_40_first_features
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: pip install -e ".[docs,audit]"
      - name: Clean-kernel execution + budgets + leakage + palette
        run: python -m scripts.tutorial_audit.pipeline --stage runtime --tutorial ${{ matrix.tutorial }}
      - uses: actions/upload-artifact@v4
        with:
          name: evidence-${{ matrix.tutorial }}
          path: docs/evidence/tutorials/${{ matrix.tutorial }}/

  gate:
    runs-on: ubuntu-latest
    needs: [static, runtime]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with: { path: docs/evidence/tutorials }
      - name: Aggregate + comment PR
        run: python -m scripts.tutorial_audit.report --aggregate --comment-pr
      - name: Fail on any error-level finding
        run: python -m scripts.tutorial_audit.report --gate
```

## Three-loop coordination

### Inner loop — one tutorial

Bounded by author session (a few hours). The author claims the tutorial
(`make tutorial-claim`), drafts following the spec, runs `make tutorial-audit`
locally, fixes findings until `state: static-pass`, opens the PR. The
runtime/reviewer loop continues in CI.

### Middle loop — one category

A category ships when every member tutorial is `merged` and the
category-aggregate report passes:

- Sum of category runtimes within budget (e.g., Start Here ≤ 8 min total).
- No tutorial in the category fails E.6.47 (Diataxis purity).
- Cross-tutorial coverage of `plan §New User Questions` for that category
  is ≥ 90% (mapped via `coverage.yaml`).
- The "spacing" dimension is satisfied: at least 2 core concepts reappear
  across category members in different contexts.

Output: `docs/evidence/categories/<category>_<date>.md`.

### Outer loop — one phase

The plan defines 5 phases; each ends with a delta report comparing pre/post
against the `_baseline_2026-05-06/` snapshot.

```
docs/evidence/phases/phase_2_2026-06-15.md
```

Required deltas:

| Metric | Phase 2 target | Source |
| --- | --- | --- |
| Public gallery files | ≤ 15 | plan §Migration Phase 2 |
| Total smoke-build runtime | ≤ 25 min | plan §Quality Bar L1165 + E1.8 budgets |
| `noplot_*` files in public path | 0 | plan §Phase 1 step 5 |
| Tutorials with subject-leakage flag | 0 | rubric E5.42 |
| Tutorials with full PRIMM scaffolding (1-star only) | 100% of 1-star | rubric E2.13 |
| Tutorials with palette compliance | ≥ 90% | data-viz-design.md |
| Spec-coverage of plan §New User Questions | ≥ 80% | coverage.yaml |

### Continuous regression loop

Nightly job re-runs the full audit on `develop`. If a previously merged
tutorial now reports an error, an issue is auto-opened with the failing
finding and the diff in evidence.json. Catches API drift before users do.

## Bootstrap (Day-0)

The minimum to commit before any tutorial gets touched:

1. `docs/tutorials/_spec/` — 13 YAML stubs, all `state: proposed`, all rubric
   fields filled per the plan.
2. `scripts/tutorial_audit/` skeleton with the 12 highest-leverage validators:
   - E1.1, E1.2, E1.4, E1.8 (filename, header, delimiters, LOC)
   - E2.13, E2.20 (PRIMM, learning objectives)
   - E3.21, E3.24, E3.30 (seeds, no hardcoded paths, no inline pip)
   - E5.37, E5.42, E5.43 (filtering disclosed, no subject leakage, chance reported)
3. `Makefile` targets: `tutorial-audit`, `tutorial-baseline`, `tutorial-claim`,
   `tutorial-release`.
4. `.github/workflows/tutorial-audit.yml` (static stage only at first; runtime
   added once first 3 tutorials draft).
5. **Run baseline**: audit all 20 current files; commit
   `docs/evidence/tutorials/_baseline_2026-05-06/`. This is the "before" we
   measure all gains against.
6. `CONTRIBUTING.md` appendix: the reviewer rubric (E2.11, E4.31, E5.46, E6.47)
   with a yes/partial/no scoring grid.
7. `docs/source/contributing/tutorial_evidence.rst` — Sphinx page that ingests
   the dossier folder and renders a dashboard.

Estimated: 2-3 days of focused work, blocking nothing else.

## Reviewer-only rubric items

A small set of rules require human or LLM judgment because they assess
narrative quality, scientific framing, or hedging. These are documented in
`CONTRIBUTING.md` and gate merge separately:

| Rule | What the reviewer scores |
| --- | --- |
| E2.11 | Does the tutorial have a clear beginning/middle/end narrative? |
| E2.14 | Is each cell doing one conceptual thing? Are figures adjacent to their explanation? |
| E2.17 | Is at least one error shown intentionally and recovered? |
| E4.31 | Does the opening name a real neuroscience question? |
| E4.33 | Is the result scientifically interpretable, not just "code runs"? |
| E4.35 | Tone: inclusive, present, explains *why*? |
| E5.46 | Are claims hedged and limitations flagged? |
| E6.47 | Diataxis purity: stays a tutorial, doesn't drift into reference/how-to/explanation? |

The reviewer files `reviewer_score.json` into the dossier with a 1-5 score
per rule plus a 1-paragraph rationale. The CI gate requires every reviewer-only
rule scored ≥ 3.

## Acceptance criteria for the system itself

The system is "done" when:

1. All 13 Release-1+2 specs exist, every tutorial's spec validates against
   `tutorial_restructure_plan.md` via `validate_spec.py`.
2. `make tutorial-audit` exits clean (zero error-level findings) on all 13
   tutorials.
3. `_baseline_2026-05-06/` snapshot shows ≥ 5 tutorials with E5.42 (leakage)
   findings and ≥ 7 with `noplot_*` naming; the post-Phase-2 snapshot shows
   zero of each.
4. Smoke-build runtime under 25 min; Start-Here cluster under 8 min.
5. Every merged tutorial has a dossier folder with `evidence.json`,
   `report.md`, `thumbnail.png`, `timing.csv`, and `reviewer_score.json`.
6. The Sphinx dashboard page is linked from `docs/source/index.rst` and
   updates nightly.
7. CI rejects any tutorial PR without a spec or with `errors > 0`.
8. `CONTRIBUTING.md` has the reviewer rubric appendix.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Authors goodhart numerical budgets (e.g., trim a real cell to fit LOC). | Reviewer-only rules (E2.11, E4.33) are non-negotiable; rubric stays mixed-method. |
| Validators bikeshed at the wrong threshold (warn → error too aggressively). | Every new validator starts at `warn` for two weeks; promotion to `error` requires green dashboard. |
| Spec drifts from plan when plan is edited. | `validate_spec.py` re-resolves cited line ranges and hashes; flags stale cites. |
| API workstreams (Workstream 1-5 in plan) stall the tutorial pipeline. | Spec `requires_api:` field; orchestrator surfaces blocked tutorials. Reorder API work, don't fake it. |
| Visual identity violations slip in via copy-pasted matplotlib. | `check_palette_compliance` runs on every saved figure; offending colors over 5% of pixels are warns. |
| One agent overwrites another's tutorial. | `assignee` lock + CI rejects YAML diffs that change `assignee` without a state transition. |
| Evidence files become noisy diffs. | Deterministic JSON (sorted keys, fixed precision). Commit only on real change. Nightly diffs not committed. |
| Reviewer fatigue. | Difficulty-aware rubric reduces reviewer load on advanced tutorials. LLM-reviewer scaffolding documented in `CONTRIBUTING.md` to suggest scores; human ratifies. |
| Over-reliance on PRIMM for adult learners. | PRIMM relaxed for 2-star; exempt for 3-star (per the plan's caveat about Sentance & Waite's K-12 origin). |

## Appendix A — Difficulty-aware rule matrix

`scripts/tutorial_audit/rule_matrix.yaml`:

```yaml
defaults:
  E1.*: error
  E2.*: warn
  E3.*: error
  E4.*: warn
  E5.*: error
  E6.*: warn
  DV.*: warn
overrides:
  difficulty_1:
    E2.13: error
    E2.16: error
    E2.20: error
    E5.42: error
    E5.43: error
  difficulty_2:
    E2.13: warn
    E2.16: warn
    E5.42: error
    E5.43: error
  difficulty_3:
    E2.13: exempt
    E2.16: exempt
    E2.18: info
    E5.42: error      # leakage is non-negotiable at any level
    E5.43: error
```

## Appendix B — Coverage map (plan §New User Questions ↔ tutorials)

`docs/tutorials/_spec/coverage.yaml`:

```yaml
installation:
  - tutorial: plot_00_first_search
    answers: ["Which Python versions are supported?"]
dataset_discovery:
  - tutorial: plot_00_first_search
    answers:
      - "What datasets exist for my task?"
      - "How do I find datasets by modality, task, subject count, clinical group, sampling rate?"
      - "Which fields can I query?"
loading_caching:
  - tutorial: plot_01_first_recording
    answers:
      - "How do I create an EEGDashDataset?"
      - "Where is data cached?"
  - tutorial: plot_02_dataset_to_dataloader
    answers:
      - "How do I get a PyTorch DataLoader from EEGDashDataset?"
preprocessing:
  - tutorial: plot_10_preprocess_and_window
    answers:
      - "How do I resample/filter/select channels?"
      - "How do I create fixed-length windows?"
windowing:
  - tutorial: plot_10_preprocess_and_window
  - tutorial: plot_20_visual_p300_oddball
    answers: ["How do I make event-related windows?"]
evaluation:
  - tutorial: plot_11_leakage_safe_split
    answers:
      - "How do I split by subject?"
      - "How do I verify no subject leakage?"
modeling:
  - tutorial: plot_12_train_a_baseline
feature_extraction:
  - tutorial: plot_40_first_features
  - tutorial: plot_41_feature_trees
  - tutorial: plot_42_features_to_sklearn
scaling_reproducibility:
  - tutorial: plot_13_save_and_reuse_prepared_data
```

A category passes its coverage gate when ≥ 90% of the plan's "New User
Questions" entries map to ≥ 1 tutorial with `state: merged`.

## Appendix C — Worked example: spec → audit → evidence flow for `plot_11`

1. Author claims: `make tutorial-claim TUTORIAL=plot_11_leakage_safe_split BY=author-A`. Spec YAML now has `assignee: author-A`, `state: drafted`.
2. Author writes `examples/tutorials/10_core_workflow/plot_11_leakage_safe_split.py`. Uses `# %% [markdown]` blocks for prose, `# %%` for code. Includes a Predict block, Run, Investigate, one Modify exercise, one Make extension. Calls `assert_no_leakage()` and prints a JSON line.
3. Local audit: `make tutorial-audit TUTORIAL=plot_11_leakage_safe_split`. Static stage runs in ~2s. Reports E2.13 PRIMM counts, E1.8 LOC=198/220, E3.21 seeds present.
4. PR opens. CI runs static + runtime stages. nbclient executes the .py via sphinx-gallery's converter, captures outputs, hashes for reproducibility, runs E5.42 leakage check by parsing the printed JSON.
5. CI bot comments PR with a 12-row scorecard (one row per dimension), a link to the dossier folder, and a list of any warns/errors.
6. Reviewer fills `reviewer_score.json` (E2.11, E4.31, E4.33, E4.35, E5.46, E6.47). Spec advances to `state: reviewed`.
7. Merge. CI commits the final `evidence.json` and `report.md`. State becomes `merged`. The tutorial appears in the Sphinx dashboard with full provenance.
8. Nightly regression: if next week the API of `eegdash.splits` changes and the tutorial's `assert_no_leakage()` call breaks, the nightly job opens an issue with the failing finding and the diff against the last green evidence.json.

## What to build first

The day-0 bootstrap above. Concrete deliverables, in order:

1. `docs/tutorials/_spec/` with the 13 YAMLs.
2. `scripts/tutorial_audit/` with the 12 highest-leverage validators.
3. The static-only `tutorial-audit.yml` workflow.
4. The baseline run, committed as `docs/evidence/tutorials/_baseline_2026-05-06/`.
5. `validate_spec.py` to keep specs aligned with the plan.

Then, and only then, start writing `plot_00_first_search.py`. Every line of
tutorial code that lands afterwards is checked against a rubric whose
provenance traces back to peer-reviewed work on instructional design and to
the prescriptive plan in this same `docs/` folder.
