# Audit dossier -- how_to_run_preprocessing_on_slurm

State: `proposed` | Difficulty: `2` | Errors: **0** | Warns: **0** | Infos: **7**

Spec: `docs/tutorials/_spec/how_to_run_preprocessing_on_slurm.yaml`
Git SHA: `765af12962d69745c1b81483b8203bddc2ec95b7`

## 12-dimension scorecard

| Dimension | Result | Validators |
| --- | :---: | --- |
| audience | unknown | E2.20, spec.audience |
| structure | unknown | E1.2, E1.4, E2.20 |
| examples | unknown | E2.12, E2.16 |
| retrieval | unknown | E2.13, E2.18 |
| spacing | unknown | cross-tutorial check_concept_revisit |
| interleaving | unknown | reviewer + cross-tutorial |
| feedback | unknown | E3.27, runtime asserts |
| data | unknown | E3.23, E3.24, E4.32 |
| reproducibility | unknown | E3.21-E3.30, E1.9 |
| accessibility | unknown | E1.6, custom check_alt_text |
| community | unknown | reviewer |
| reuse | unknown | E1.7, evidence dossier |

## Findings

| Rule | Level | Message | Tool |
| --- | :---: | --- | --- |
| E1.2 | info | Skipped: how-to source is Markdown (no module docstring). | filename |
| E4.31 | info | Skipped: how-to recipes open with a Goal statement, not a motivating question (E4.31 does not apply) | reviewer-stub |
| E4.33 | info | E4.33 'result has scientific meaning' is reviewer-only; see tutorial_implementation_strategy.md 'Reviewer-only rubric items' | reviewer-stub |
| E4.35 | info | E4.35 'tone: we-inclusive, present tense, explains why' is reviewer-only; see tutorial_implementation_strategy.md 'Reviewer-only rubric items' | reviewer-stub |
| E6.47 | info | E6.47 'Diataxis purity: stays a tutorial, doesn't drift into reference/how-to/explanation' is reviewer-only; see tutorial_implementation_strategy.md 'Reviewer-only rubric items' | reviewer-stub |
| E6.48 | info | Source does not link to any explanation/concept page; Diataxis purity asks tutorials to defer deeper theory to the explanation quadrant via a :doc: or markdown link | regex |
| E6.49 | info | E6.49 'where a competent user wants a quick recipe, split out a separate how-to' is reviewer-only; see tutorial_implementation_strategy.md 'Reviewer-only rubric items' | reviewer-stub |

