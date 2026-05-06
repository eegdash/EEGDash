# Wave C reviewer-pass verification, 2026-05-07

Reviewer-only audit of the 5 Release-2 tutorials in
`examples/tutorials/{20_event_related,30_resting_state,40_features}/`,
covering checks A (citations), B (plan alignment), C (spec coherence) and
D (8 reviewer-rubric scores). Static validators (`scripts/tutorial_audit/`)
ran clean before this pass; this is the human/LLM-judgment layer. The
shape and wording mirrors `_verification_2026-05-06.md`.

DOI resolution used Crossref / Datacite / OpenAlex direct fetches. Each
unique DOI was fetched once and cached; reuse from the 2026-05-06 cache
is noted in the citations table.

## Executive summary

| Section | Pass | Partial | Fail |
| --- | ---: | ---: | ---: |
| A. Citations resolve | 2 | 1 | 2 |
| B. Plan alignment | 2 | 3 | 0 |
| C. Spec coherence | 0 | 4 | 1 |
| D. Reviewer rubric (E2.11/E2.14/E2.17/E4.31/E4.33/E4.35/E5.46/E6.47) | 4 | 1 | 0 |

Anchor reference coverage (plan §"reviewer-only" rubric items — does each
anchor appear where its topic warrants?):

| Anchor reference | Tutorials where required | Tutorials cited correctly |
| --- | --- | --- |
| Polich 2007 P300 review (`10.1016/j.clinph.2007.04.019`) | plot_20, plot_21 | 2/2 |
| Cisotto & Chicco 2024 (filter / leakage / baseline) | plot_20, plot_21, plot_30, plot_41, plot_42 | 5/5 |
| Pernet et al. 2019 EEG-BIDS | plot_20, plot_21, plot_30, plot_42 | 3/4 *(plot_30 cites the DOI but never surfaces BIDS entities in prose)* |
| Gramfort et al. 2013 MNE | plot_20, plot_21, plot_30, plot_41 | 4/4 |
| Schirrmeister 2017 (CNN) | none in this wave (sklearn-only); spec for plot_20 lists it as a related paper but tutorial does not invoke a CNN | n/a (skipped per rubric guidance) |
| Dataset DOI matches code | plot_20 (ds005863), plot_21 (ds003061), plot_30 (ds005514), plot_41 (no live dataset), plot_42 (no live dataset) | 2/3 *(plot_20 mismatch — see top fix #1)* |

## DOI resolution cache (this pass + reuse from 2026-05-06)

Verified via Crossref / Datacite / OpenAlex during this audit (status 2026-05-07).
Rows marked "(reused)" were resolved on 2026-05-06 and reused here.

| DOI | Resolved as | Status |
| --- | --- | --- |
| 10.1016/j.clinph.2007.04.019 | Polich 2007, "Updating P300", *Clin. Neurophysiol.* | OK |
| 10.18112/openneuro.ds005863.v1.0.0 | Isbell et al. 2025, "Cognitive Electrophysiology in Socioeconomic Context in Adulthood" | OK *(does NOT match the "ERP CORE / Kappenman" cited in plot_20 line 218)* |
| 10.18112/openneuro.ds003061.v1.1.0 | Delorme 2021 OpenNeuro deposit, "EEG data from an auditory oddball task" | OK |
| 10.18112/openneuro.ds002718.v1.0.5 | Wakeman OpenNeuro deposit, "Face processing EEG dataset for EEGLAB" | OK (reused) |
| 10.18112/openneuro.ds005514.v1.0.0 | Shirazi 2024 OpenNeuro deposit, "Healthy Brain Network EEG- Release 9" | OK (reused) |
| 10.1038/sdata.2017.181 | Alexander et al. 2017, "An open resource for transdiagnostic research…" (HBN), *Sci. Data* | OK |
| 10.1007/BF01797193 | Berger 1929, "Über das Elektrenkephalogramm des Menschen" | OK (reused) |
| 10.1038/s41597-019-0104-8 | Pernet et al. 2019, EEG-BIDS, *Sci. Data* | OK (reused) |
| 10.3389/fnins.2013.00267 | Gramfort et al. 2013, MNE-Python, *Front. Neurosci.* | OK (reused) |
| 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024, *PeerJ CS* (Ten quick tips, clinical EEG) | OK (reused) |
| 10.1145/3304221.3325207 | 404 on Crossref/OpenAlex | **DOES NOT RESOLVE** *(plot_42 cites this for "Sentance et al. 2019 PRIMM" — the real DOI is `10.1080/08993408.2019.1608781`)* |
| 10.5555/1953048.2078195 | ACM DL identifier for Pedregosa 2011 scikit-learn (JMLR) | DOI does NOT resolve via doi.org *(returns 404 on Crossref). The underlying JMLR paper is real and canonical, but JMLR papers are not registered DOIs at Crossref. Use the JMLR URL or no DOI.* |

## Top fixes recommended (prioritised)

1. **plot_20_visual_p300_oddball.py:218** — `Dataset: Kappenman et al., ERP CORE / ds005863. https://doi.org/10.18112/openneuro.ds005863.v1.0.0` is wrong. The DOI resolves, but to **Isbell et al. 2025, "Cognitive Electrophysiology in Socioeconomic Context in Adulthood"** — not the ERP CORE / Kappenman P3 dataset. ERP CORE lives at `ds003061`-style P3 paradigms or at `ds002893` (the canonical Kappenman 2021 ERP CORE OpenNeuro release). Either correct the dataset name to "Isbell 2025 socioeconomic-cognition oddball" (and verify the task code `P3` actually exists in that BIDS deposit), or switch the dataset id to `ds002893` and re-cite `Kappenman et al. 2021, NeuroImage` (`10.1016/j.neuroimage.2020.117465`).
2. **plot_42_features_to_sklearn.py:90** — `(Sentance et al. 2019, doi:10.1145/3304221.3325207)` does not resolve. The canonical PRIMM paper is **Sentance, Waite & Kallia 2019, "Teaching computer programming with PRIMM: a sociocultural perspective", *Computer Science Education*, doi:10.1080/08993408.2019.1608781**. Replace the DOI inline.
3. **plot_42_features_to_sklearn.py:217** — `Pedregosa et al. 2011, *JMLR* 12:2825. https://doi.org/10.5555/1953048.2078195`. The `10.5555/...` is an ACM Digital Library identifier and does not resolve via doi.org. Either drop the DOI and use the JMLR URL `https://www.jmlr.org/papers/v12/pedregosa11a.html`, or replace with the (also unofficial) arXiv preprint `10.48550/arXiv.1201.0490`. The reference itself is correct; only the DOI string is broken.
4. **plot_30_eyes_open_closed.py:61** — `task = get_task("eyes-open-closed")` and the subsequent `task.metadata_query()`, `task.preprocessing_recipe()`, `task.windowing_recipe()`, `task.bandpass`, `task.dataset`, `task.subjects`, `task.label_definition()`, `task.name` are all called on the same object. The current public `eegdash.tasks` API exports `EyesOpenClosed`/`get_task`, but a single object exposing every one of those eight properties simultaneously is not part of the documented API surface (the live tutorial at `examples/eeg2025/` uses `EyesOpenClosed.make_windows` instead). The tutorial flags the gap textually ("`task.get_windows()` from the spec is not wired yet", line 78), but the eight downstream calls are still made — this will fail at runtime. Replace with the `make_windows` flow already used in `eeg2025/`, or keep the prose disclaimer and fall back to a synthesised dataset (the pattern used by plot_41/plot_42).
5. **plot_41_feature_trees.py:40** — `from braindecode.datasets import BaseConcatDataset, RawDataset`. Same defect flagged on plot_13 in Wave B: **`RawDataset` is not part of Braindecode's public API.** The public class is `braindecode.datasets.BaseDataset`. The static audit cannot catch this; runtime will. Replace with `BaseDataset` (and adjust the constructor: `BaseDataset(raw=..., description=..., target_name=...)`), or import `RawDataset` from wherever the project stubs it (it is not in the published Braindecode 0.8.x release).
6. **plot_30_eyes_open_closed.py:138** — `assert mean_alpha_diff > 0, "Expected closed > open in alpha; got the reverse."` on a real HBN child. Berger's effect is robust on group means, but on a single subject (the spec scopes "one subject" at line 84) the alpha bump is not guaranteed in every recording. This converts a hedged scientific claim (E5.46) into a hard runtime fail. Either soften to a warn-print, scope the assertion to a group mean across `>=10` subjects (which spec invariant `n_subjects_total >= 10` already requires), or annotate the assertion as "we expect this on this curated subject; flip your guess if the assertion fires".
7. **plot_41_feature_trees.py:54-55** — `_spec.welch = lambda *a, **k: ...` monkeypatches the public `eegdash.features.feature_bank.spectral.welch` symbol at import time and never restores it. Any subsequent tutorial in the same kernel session inherits the patched function. Wrap the patch in a try/finally restoring `_orig_welch` after the two `extract_features` calls, or use `unittest.mock.patch` as a context manager. Otherwise downstream `noplot_*` examples that rely on the original `scipy.signal.welch` semantics will silently produce inflated PSD-call counters.
8. **plot_21_auditory_oddball.py:165** — `visual_acc_from_plot20 = 0.78  # placeholder; rerun plot_20 to refresh`. Spec asserted_invariant line 56 says "metric_table contains both auditory and visual rows for direct comparison". A hard-coded placeholder is not a "direct comparison"; it will silently misreport whenever plot_20 changes. Either persist the visual accuracy from plot_20 to a JSON/Parquet artefact (the "save and reload" lesson plot_13 already establishes), read it here, and assert "fresh enough"; or drop the visual row from the table and reframe the result print as "auditory-only — see plot_20 for visual reference".
9. **plot_30_eyes_open_closed.py:97-99** — `X = np.stack([w[0] for w in windows_ds]).astype(np.float32); y = np.asarray([w[1] for w in windows_ds], ...)`. Iterating `windows_ds` twice is fine, but the indexing `w[0]/w[1]` returns the underlying `(data, target, ind)` triple from Braindecode's `WindowsDataset.__getitem__`; the standard pattern (and what the prior tutorials use) is `w[0], w[1], _ = windows_ds[i]`. With Braindecode 0.8.x the tuple is length-3 and the slice works, but the asymmetric indexing is fragile. Match the pattern used in plot_02 / plot_10.
10. **plot_42_features_to_sklearn.py:218** — `HBN ds005514 (Release 9). https://doi.org/10.18112/openneuro.ds005514.v1.0.0` appears in the references but the tutorial **does not load the HBN dataset**; it synthesises features locally (lines 64-77). This mirrors the systematic deviation flagged in Wave B (plot_12, plot_13, plot_40 silently re-synthesise) and is the same fix: either load the real artefact (preferred per the plan's "reuse upstream artifacts" rule) or move the dataset DOI from "References" to a "Dataset (would-be)" subsection so a reader does not believe a real HBN download has happened.

## Per-tutorial review

### 1. plot_20_visual_p300_oddball.py

**A. Citations. PARTIAL.**
- DOIs in references block (lines 213-218):
  - `10.1016/j.clinph.2007.04.019` (Polich 2007) — resolves OK; cited in-text at line 7 and line 81.
  - `10.1038/s41597-019-0104-8` (Pernet 2019) — resolves OK; cited in-text line 56-57.
  - `10.3389/fnins.2013.00267` (Gramfort 2013) — resolves OK; cited in-text line 68-69.
  - `10.7717/peerj-cs.2256` (Cisotto 2024) — resolves OK; cited in-text lines 87, 90.
  - `10.18112/openneuro.ds005863.v1.0.0` cited as **"Kappenman et al., ERP CORE / ds005863"** (line 218). The DOI resolves, but to **Isbell et al. 2025** — not the ERP CORE / Kappenman P3 deposit. **MISMATCH.** See top fix #1.
- Anchor coverage: Polich (P300 mechanism) — present, OK. Pernet (BIDS) — present and BIDS-entity language used at lines 16, 60-64. Gramfort (MNE) — present. Cisotto Tip 4/Tip 7 — present. Schirrmeister listed in spec.related_papers but not invoked here (sklearn-only) — OK to skip per rubric guidance. **PARTIAL** (one DOI mismatch).

**B. Plan alignment.** Plan §L941-958 prescribes 8 steps: (1) query P3 dataset → (2) inspect annotations → (3) map target/standard → (4) create event windows → (5) check class balance → (6) split safely → (7) train baseline → (8) plot ERP. Tutorial covers all 8 explicitly: line 60 (query), line 75 (annotations), line 106 (mapping), line 101 (windows), line 110-111 (class balance), line 121-126 (split + leakage), line 138-143 (train), line 165-173 (ERP plot). PASS.

**C. Spec coherence. PARTIAL.**
- Four learning_objectives map 1:1 to bullet list at lines 14-20.
- asserted_invariants:
  - `n_target_epochs > 0 and n_standard_epochs > 0` — line 110-111 prints; not asserted.
  - `n_standard / n_target >= 3` (oddball imbalance preserved) — printed but not asserted.
  - `epochs.tmin <= 0 and epochs.tmax >= 0.5` — line 100 sets `TMIN=-0.2, TMAX=0.8`; satisfies but not asserted.
  - `baseline applied and unit reported as µV` — baseline correction language is in prose (line 89); the unit µV appears in plot label (line 170) but not in any assert.
  - `leakage_report['overlap'] == 0` — line 126 asserts this.
  - `model_accuracy_test reported alongside chance level` — line 152.
- requires_api: lists `eegdash.EEGDashDataset`, `braindecode.preprocessing.create_windows_from_events`, `eegdash.splits.assert_no_leakage`, `eegdash.splits.get_splitter`. The first three are exercised; **`get_splitter` is NOT** — the tutorial uses `sklearn.model_selection.StratifiedKFold` instead (line 121). The spec calls `get_splitter`, so this is a drift.
- Spec rubric_overrides set `E2.16: warn`. Tutorial has a Modify (line 175) and Make (line 180) block; faded scaffolding pattern is mostly the "Modify" call, not a parsons exercise as the spec demands (line 81). PARTIAL.

**D. Reviewer rubric.**
- E2.11 narrative: 5. Strong opening hook (child watches letters, P300 anchor) → predict (line 80) → run (line 91) → investigate (line 156) → wrap-up + Try-it-yourself.
- E2.14 cognitive load: 4. One concept per cell. Step 5 (line 134-143) packs three operations: per-channel z-score, vectorise, LR fit. Could split.
- E2.17 intentional error: 3. The Modify block (line 177) "re-run with TMAX=0.4 and ROC-AUC drops" is an intentional-degradation exercise, not an error-and-recovery. Marked warn (consistent with other tutorials).
- E4.31 motivating question: 5. Lines 1-9 — "A child watches letters … can we decode whether a child saw a target?" — concrete, neurosci-grounded.
- E4.33 result meaningful: 4. ROC-AUC and accuracy alongside chance is the right framing for an imbalanced two-class task. Single-subject within-subject split limits the result's external validity (correctly hedged).
- E4.35 tone: 5. "We" throughout, present tense, explains *why* (line 132 "the simplest target-vs-standard decoder").
- E5.46 hedging: 4. Line 6-9 hedges, line 196 "the only honest summary of an imbalanced two-class decoder" is good. Could be sharper on within-subject split limitations.
- E6.47 Diataxis purity: 5. Stays a tutorial; concept page linked.

### 2. plot_21_auditory_oddball.py

**A. Citations. PASS.**
- DOIs in references (lines 210-214):
  - `10.1038/s41597-019-0104-8` (Pernet) — resolves OK; cited in-text line 47.
  - `10.3389/fnins.2013.00267` (Gramfort) — resolves OK; cited in-text line 49.
  - `10.7717/peerj-cs.2256` (Cisotto) — resolves OK; cited in-text line 142-143, 176.
  - `10.1016/j.clinph.2007.04.019` (Polich) — resolves OK; cited in-text lines 10, 122-123, 188-189.
  - `10.18112/openneuro.ds003061.v1.1.0` (Delorme) — resolves OK; cited in-text line 8.
- Anchor coverage: Polich, Pernet, Gramfort, Cisotto all present and warranted. Schirrmeister not invoked (no CNN) — OK. PASS.

**B. Plan alignment. PARTIAL.** No §Detailed Tutorial Proposal exists for plot_21 (the plan only lists it under Cat C item 2 at L394-L395 and the spec acknowledges this extrapolation in `notes.extrapolation`). Spec is the contract here. Spec calls for: load → inspect → map events → epoch → split → train → compare with plot_20. Tutorial covers all six **but** **synthesises** the data instead of loading `ds003061` (line 67-89). The opening text frames this as "we synthesise the same shape below" (line 50) — appropriate framing — but the spec line 11 lists `plot_20_visual_p300_oddball` as the prerequisite *and* asks for a metric_table comparison with plot_20 (spec invariant line 56), neither of which is honoured in code (the comparison row is a hard-coded placeholder, line 165). PARTIAL.

**C. Spec coherence. PARTIAL.**
- Four learning_objectives map to lines 18-21.
- asserted_invariants:
  - `n_target_epochs > 0 and n_standard_epochs > 0` — line 92.
  - `epochs.info['sfreq'] == reference_sfreq_from_plot_20` — `SFREQ=256.0` is set at line 40 with comment "matches plot_20 reference" — but plot_20 actually uses `SFREQ=128.0` (plot_20 line 94). **VIOLATED.**
  - `epochs.tmin == reference_tmin and epochs.tmax == reference_tmax` — `TMIN=-0.1, TMAX=0.5` (line 41) vs plot_20 `TMIN=-0.2, TMAX=0.8` (plot_20 line 100). **VIOLATED.** Spec wants the auditory tutorial to mirror plot_20's window; tutorial chose a different window. Either fix the tutorial or fix the spec wording (the auditory N100/P300 community uses tighter pre-stim baselines; tutorial's choice is defensible — but the spec asserts equality).
  - `leakage_report['overlap'] == 0` — line 137.
  - `auditory_p300_peak_latency_ms reported with hedged language` — line 108-109 ("hedged: single channel, simulated subject pool"). PASS.
  - `metric_table contains both auditory and visual rows for direct comparison` — line 165-169 has both rows but visual is a hard-coded `0.78`. Not "direct comparison" in any reproducible sense.
- requires_api: lists `EEGDashDataset`, `create_windows_from_events`, `get_splitter`, `assert_no_leakage`. **None** of the first three are imported or called — only `assert_no_leakage` and `majority_baseline` are imported at line 37. PARTIAL (4 of 6 invariants, 1 of 4 API entries).

**D. Reviewer rubric.**
- E2.11: 5. The "contrast not duplicate" framing (lines 4-13) is an effective narrative scaffold; predict (line 52) → run (line 59) → investigate (line 122) → modify → make.
- E2.14: 4. Step 3 (line 96-119) computes both ERPs, peaks, and the figure in one cell. Plot rendering and peak extraction could split.
- E2.17: 3. Modify cell (line 174) "shift baseline window" is a soft regression demo, not a recovered error. Warn.
- E4.31: 5. Line 11-13 — "what stays the same and what shifts between modalities" — is a real cross-paradigm question, not a generic re-run.
- E4.33: 3. The result is interpretable, but the synthesised data (n100/p300 injected at fixed channels) and a 4-subject pool with 64 windows each plus a 0.78 placeholder for visual make the contrast scientifically weaker than it should be. Marked warn.
- E4.35: 5. "We" throughout, explains *why* the auditory window differs (line 41), explains *why* class balance is preserved (line 75-76).
- E5.46: 4. Multiple hedge points (line 109, line 195-198). The placeholder visual row (line 165) is the one place the prose under-flags fragility.
- E6.47: 5. Stays a tutorial.

### 3. plot_30_eyes_open_closed.py

**A. Citations. PASS.**
- DOIs in references (lines 215-220):
  - `10.1038/s41597-019-0104-8` (Pernet) — resolves OK; cited in-text via doi link only (no surfaced BIDS entity in prose). Borderline anchor coverage.
  - `10.3389/fnins.2013.00267` (Gramfort) — resolves OK; cited in-text via doi link.
  - `10.7717/peerj-cs.2256` (Cisotto) — resolves OK; cited in-text line 146-147.
  - `10.1038/sdata.2017.181` (Alexander HBN) — resolves OK; cited in-text via doi link.
  - `10.18112/openneuro.ds005514.v1.0.0` — resolves OK (Shirazi 2024 HBN Release 9); cited in-text line 56.
  - Berger 1929 textually cited (line 215) without DOI — fine; the canonical DOI `10.1007/BF01797193` is not in the references but Berger 1929's exact bibliographic entry is given.
- Anchor coverage: Berger (alpha bump) — warranted, present (lines 4, 69-72). Cisotto Tip 9 — present. Gramfort (MNE) — present. Pernet — DOI present but no BIDS-entity language in prose (only in the references block). Mild gap; not flagged as fail. PASS.

**B. Plan alignment.** Plan §L960-975 prescribes 5 steps: (1) load HBN EO/EC → (2) reannotate intervals → (3) create balanced windows → (4) inspect one signal → (5) split + train. Tutorial covers (1) at line 85, (2) implicitly via `task.preprocessing_recipe()` (line 80-82 prose claims `hbn_ec_ec_reannotation` is in the recipe), (3) line 88, (4) line 110-124 (topomap; the spec asks for "one signal" but a topomap is a stronger artefact), (5) line 149-162. PASS in spirit.

**C. Spec coherence. FAIL (runtime correctness, not narrative).**
- Four learning_objectives map to bullet list at lines 17-21.
- asserted_invariants:
  - `n_subjects_total >= 10` — tutorial uses **one** subject (line 64 prints `subject={task.subjects[0]}`); spec wants ≥10. **VIOLATED.** Either widen to all HBN subjects or drop the invariant for difficulty-1 single-subject scope.
  - `abs(n_eo - n_ec) / max(...) < 0.20` — not asserted in code.
  - `leakage_report['overlap'] == 0` — line 157-158.
  - `psd_alpha_power_ec >= psd_alpha_power_eo` (textbook expectation, hedged in prose) — line 138 hard-asserts `mean_alpha_diff > 0`. Spec said "hedged in prose" but tutorial enforces a hard runtime fail. Top fix #6.
  - `model_accuracy_test reported with chance level` — line 162.
- requires_api lists 5 entries: `EEGDashDataset` (line 85), `Preprocessor` (imported but not directly used; `preprocess` consumes the recipe via `task.preprocessing_recipe()`), `create_fixed_length_windows` (NOT used; tutorial uses `create_windows_from_events`), `get_splitter` (line 153), `assert_no_leakage` (line 157). One missing, one substituted.
- The bigger issue: the entire `task = get_task("eyes-open-closed")` flow (line 61) and the eight downstream `task.<attribute>` calls are not part of the documented `eegdash.tasks` public API surface today. Top fix #4.

**D. Reviewer rubric.**
- E2.11: 5. Berger anchor (line 4) → predict (line 67-72) → run (line 92, line 105) → investigate (line 127-138) → modify → make. Strong arc.
- E2.14: 5. Each step does one thing. Topomap rendering is its own block.
- E2.17: 4. Modify block (line 165-176) is a real "swap the band, watch the contrast collapse" exercise — exactly the spec's faded-scaffolding completion pattern.
- E4.31: 5. Line 9-10 — "Can we tell from a 2-second EEG snippet whether a child has their eyes open or closed?" — canonical EEG-101 question.
- E4.33: 4. Topomap + per-channel ranking is interpretable. The hard `assert mean_alpha_diff > 0` (top fix #6) over-interprets a single-subject result.
- E4.35: 5. "We" throughout; explains *why* alpha (line 69-72), *why* topomap (line 106-109), *why* logistic regression on log-alpha-power (line 142-147).
- E5.46: 4. Hedges in prose ("textbook expectation", "single subject") but the line 138 assertion contradicts the hedge. Drop the hedge in code or drop the assert.
- E6.47: 5. Stays a tutorial; concept page linked.

### 4. plot_41_feature_trees.py

**A. Citations. PASS.**
- DOIs in references (lines 217-219):
  - `10.7717/peerj-cs.2256` (Cisotto) — resolves OK; cited in-text line 9, 23.
  - `10.3389/fnins.2013.00267` (Gramfort) — resolves OK; cited in-text via "Gramfort 2013" (line 33) without inline DOI but with reference entry.
  - `10.18112/openneuro.ds005514.v1.0.0` (HBN) — resolves OK; **but the tutorial does not load HBN**; the dataset is synthesised at lines 67-83. Anchor link is aspirational. Mark borderline.
- Anchor coverage: Cisotto Tip 5 (filter disclosure / pipeline reuse) — present. Gramfort (MNE under the hood) — present. Pernet/Schirrmeister not warranted (no BIDS, no CNN). PASS.

**B. Plan alignment.** Plan §L994-1007 prescribes 4 steps: (1) show repeated spectral computation problem → (2) add spectral preprocessor → (3) extract multiple downstream spectral features → (4) compare feature names and shape. Tutorial covers all four (lines 92-114, line 124-137, line 153-164, line 192-200). The plan also says "Move from `examples/tutorials/noplot_tutorial_features_eoec.py`"; the tutorial keeps the move spirit (HBN-themed alpha contrast). PASS.

**C. Spec coherence. PARTIAL.**
- Four learning_objectives map to lines 19-25.
- asserted_invariants:
  - `feature_table_tree.shape[0] == feature_table_flat.shape[0]` — line 147.
  - `feature_table_tree.shape[1] >= feature_table_flat.shape[1]` — line 148.
  - `runtime_seconds_tree < runtime_seconds_flat * 0.9` — **NOT asserted in this form.** Line 201 only asserts `speedup >= 1.0`. Spec wanted ≥10% speedup; tutorial accepts no slowdown.
  - `spectral preprocessor invoked once per window across all dependent features` — line 149 asserts `psds_tree * len(BANDS) == psds_flat`. PASS (more precise than the spec wording).
  - `feature column names disambiguate by channel and band` — implicit in the feature table; tutorial prints columns at line 187.
- requires_api lists 6 entries: `FeatureExtractor`, `extract_features`, `fit_feature_extractors`, `spectral_preprocessor`, `spectral_bands_power`, `preprocessor_as_feature`. Of these `FeatureExtractor`, `extract_features`, `spectral_preprocessor`, `spectral_bands_power` are used (lines 44-47). `fit_feature_extractors` and `preprocessor_as_feature` are imported (line 47 also imports `feature_predecessor`, `univariate_feature`) but `fit_feature_extractors` is NOT called and `preprocessor_as_feature` is NOT called or imported. Drift between spec and code.
- Imports at line 40 use `RawDataset` from `braindecode.datasets`, which is not Braindecode's published public API (top fix #5).
- Monkeypatching of `_spec.welch` at module scope without restoration (top fix #7) is a correctness bug.

**D. Reviewer rubric.**
- E2.11: 4. Build → predict → run flat → run tree → investigate speedup → modify (gamma) → make (custom feature). Slightly dense (7 steps in 220 LOC). Beginning is a bit terse — assumes the reader knows what Welch is.
- E2.14: 4. Steps 3 and 5 each fit a `FeatureExtractor`, time it, and tabulate — three concepts per cell. Could split the timing harness into a helper. The monkeypatch at line 53-55 is a clever PSD counter but is hidden from the narrative.
- E2.17: 3. Modify (gamma band) and Make (`relative_alpha`) are good extensions, not error-recovery. Warn.
- E4.31: 4. "How many PSDs run per batch?" — a real engineering question. Slightly software-flavoured rather than neuroscience-flavoured.
- E4.33: 4. Speedup table + dependency tree print + identical row counts is interpretable.
- E4.35: 5. "We" throughout; explains *why* sharing matters (line 1-13 opening); explains the API per call.
- E5.46: 4. Doesn't over-claim; speedup is "≥1.0x" not "10x". Mild gap: doesn't flag that on small data the tree may NOT be faster than flat (the synthesised tiny dataset can favour flat).
- E6.47: 5. Stays a tutorial; concept page linked at line 212.

### 5. plot_42_features_to_sklearn.py

**A. Citations. FAIL.**
- DOIs in references (lines 215-218):
  - `10.7717/peerj-cs.2256` (Cisotto) — resolves OK; cited in-text line 135-136.
  - `10.1038/s41597-019-0104-8` (Pernet) — resolves OK; cited in-text via reference block only (no in-text BIDS entity language).
  - `10.5555/1953048.2078195` cited as **Pedregosa 2011 scikit-learn JMLR** — DOI **does NOT resolve via doi.org** (404 on Crossref). The paper is real and canonical; the DOI is an ACM DL identifier, not a registered DOI. Top fix #3.
  - `10.18112/openneuro.ds005514.v1.0.0` (HBN) — resolves OK; **but the tutorial does not load HBN** (synthesises features locally; lines 64-77). Aspirational anchor.
  - In-text only (not in reference block but cited inline): `10.1145/3304221.3325207` for "Sentance et al. 2019 PRIMM" at line 90. **Does NOT resolve.** Real DOI is `10.1080/08993408.2019.1608781`. Top fix #2.
  - In-text only: `10.1007/BF01797193` for "Berger 1929" at line 58. Resolves OK (reused).
- Anchor coverage: Cisotto present. Pernet listed but no BIDS-entity language. Pedregosa cited with broken DOI. Sentance cited with broken DOI. **FAIL** (two broken DOIs in the same tutorial, both for canonical anchor papers).

**B. Plan alignment.** Plan §L1009-1019 prescribes 5 steps: (1) load saved features → (2) split by subject → (3) normalize using train only → (4) train logistic regression / random forest / LightGBM → (5) report metrics and feature importance. Tutorial covers all five (line 64-83, line 92-111, line 128-130, line 158-186, line 188-205). The "feature importance" step is in the Try-it-yourself (line 211) rather than the main flow — the spec invariant at line 54 ("top_feature_importances annotated with channel × band labels") is therefore not satisfied in code; only mentioned in the extension list. PASS in spirit, with one missing artefact (feature importance plot).

**C. Spec coherence. PARTIAL.**
- Four learning_objectives map to lines 14-21.
- asserted_invariants:
  - `scaler fitted only on X_train` — line 128 wraps in a `Pipeline` so the train-only fit is enforced by sklearn semantics.
  - `leakage_report['overlap'] == 0` — line 109.
  - `model_accuracy_test reported with chance level` — line 142.
  - `logistic_regression_accuracy and random_forest_accuracy both reported` — lines 141, 186 (logistic) and 186 falls back to RandomForest only if LightGBM is unavailable. On a CI runner without LightGBM, both are reported; on a runner with LightGBM, RandomForest is NOT reported.
  - `top_feature_importances annotated with channel × band labels` — NOT enforced; only suggested in Try-it-yourself.
- requires_api lists 7 entries; all imported at lines 41-44 except the spec's `eegdash.features.extract_features`, which is **not imported and not called** (the tutorial synthesises features rather than calling `extract_features`). PARTIAL.

**D. Reviewer rubric.**
- E2.11: 4. Synthesise → predict → split → train LR → score → confusion → modify (Ridge) → make (LightGBM/RF) → result. Clean arc; the synth in step 1 is well-flagged ("In production you would reload `plot_40_features.parquet`", line 55-56).
- E2.14: 5. Each cell does one thing. Pipeline construction is one cell; scoring is one cell; confusion matrix is one cell.
- E2.17: 4. Modify block (line 156-165) and Make block (line 168-186) are both PRIMM-clean. The LightGBM/RandomForest fallback is a nice "graceful degradation" pattern, not an error per se.
- E4.31: 5. Lines 1-11 — "Can a logistic-regression Pipeline on a handful of EEG features beat the majority-class chance level on a held-out subject?" — concrete, leakage-aware.
- E4.33: 4. Three Pipelines on the same held-out subject is a defensible artefact. The synthetic alpha contrast is too on-the-nose (same critique as plot_12 in Wave B).
- E4.35: 5. "We" throughout; explains *why* Pipeline (line 124-126), *why* RidgeClassifier (line 158-159), *why* LightGBM-or-RandomForest fallback (line 168-170).
- E5.46: 4. Hedges around the synthetic source but does not flag the per-fold variance (only one fold is scored despite 4 subjects → 4 folds available).
- E6.47: 5. Stays a tutorial; concept page linked.

## Citations status table

Status legend: `OK` = DOI resolves and matches the in-text claim; `MISMATCH` =
DOI resolves but to a different paper; `404` = DOI does not resolve.

| Tutorial | DOI | Cited as | Status |
| --- | --- | --- | --- |
| plot_20 | 10.1016/j.clinph.2007.04.019 | Polich 2007 P300 | OK |
| plot_20 | 10.1038/s41597-019-0104-8 | Pernet 2019 EEG-BIDS | OK |
| plot_20 | 10.3389/fnins.2013.00267 | Gramfort 2013 MNE | OK |
| plot_20 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 | OK |
| plot_20 | 10.18112/openneuro.ds005863.v1.0.0 | "Kappenman / ERP CORE / ds005863" | **MISMATCH** (resolves to Isbell 2025 socioeconomic-cognition) |
| plot_21 | 10.1038/s41597-019-0104-8 | Pernet 2019 EEG-BIDS | OK |
| plot_21 | 10.3389/fnins.2013.00267 | Gramfort 2013 MNE | OK |
| plot_21 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 | OK |
| plot_21 | 10.1016/j.clinph.2007.04.019 | Polich 2007 P300 | OK |
| plot_21 | 10.18112/openneuro.ds003061.v1.1.0 | Delorme 2021 auditory oddball | OK |
| plot_30 | 10.1038/s41597-019-0104-8 | Pernet 2019 EEG-BIDS | OK |
| plot_30 | 10.3389/fnins.2013.00267 | Gramfort 2013 MNE | OK |
| plot_30 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 | OK |
| plot_30 | 10.1038/sdata.2017.181 | Alexander 2017 HBN | OK |
| plot_30 | 10.18112/openneuro.ds005514.v1.0.0 | HBN ds005514 Release 9 | OK |
| plot_30 | 10.1007/BF01797193 (textual) | Berger 1929 | OK |
| plot_41 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 | OK |
| plot_41 | 10.3389/fnins.2013.00267 | Gramfort 2013 MNE | OK |
| plot_41 | 10.18112/openneuro.ds005514.v1.0.0 | HBN ds005514 (data not actually loaded) | OK |
| plot_42 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 | OK |
| plot_42 | 10.1038/s41597-019-0104-8 | Pernet 2019 EEG-BIDS | OK |
| plot_42 | 10.5555/1953048.2078195 | Pedregosa 2011 scikit-learn (JMLR) | **404** (ACM DL id, not a registered DOI; paper is real) |
| plot_42 | 10.18112/openneuro.ds005514.v1.0.0 | HBN ds005514 (data not actually loaded) | OK |
| plot_42 | 10.1145/3304221.3325207 (in-text only) | Sentance 2019 PRIMM | **404** (real DOI is 10.1080/08993408.2019.1608781) |
| plot_42 | 10.1007/BF01797193 (in-text) | Berger 1929 | OK |

## Closing note

Wave C tutorials are stronger on narrative and PRIMM scaffolding than the
average Wave B tutorial — every one of them lands beginning/middle/end
cleanly and reuses the rubric vocabulary the spec demands. The systematic
deviations are:

1. **Citations**: plot_20 has a Kappenman-vs-Isbell dataset mismatch on the
   exact DOI used, and plot_42 carries two non-resolving DOIs (Pedregosa
   ACM-DL identifier and a wrong Sentance PRIMM DOI). plot_42 in particular
   should not ship with broken anchor citations.
2. **Spec invariants**: plot_30 hard-asserts a single-subject Berger effect
   and lists a single subject under an `n_subjects_total >= 10` invariant;
   plot_21 advertises `sfreq == reference_sfreq_from_plot_20` and sets
   `SFREQ=256.0` against plot_20's `SFREQ=128.0`. Either fix the code or
   relax the spec.
3. **API drift**: plot_30 calls eight `task.*` properties that are not part
   of `eegdash.tasks` today (`make_windows` is the supported entry); plot_41
   imports `RawDataset` from `braindecode.datasets` (Wave B plot_13 had the
   same issue); plot_42 lists `eegdash.features.extract_features` in
   `requires_api` but never calls it. These will surface at the runtime
   gate, not at static lint.
4. **Reused-artifact rule**: plot_21, plot_41, plot_42 all silently
   re-synthesise data instead of consuming what the upstream tutorial saved
   — same pattern flagged in Wave B for plot_12/plot_13/plot_40. The plan's
   "reuse upstream artifacts" rule is the largest cross-cutting gap in the
   release.

plot_20 and plot_30 carry the strongest narrative arcs of this wave and
the most actionable Modify/Make exercises. plot_42 is the one that most
needs a citation pass before merging.
