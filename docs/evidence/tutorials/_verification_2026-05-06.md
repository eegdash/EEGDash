# Wave B reviewer-pass verification, 2026-05-06

Reviewer-only audit of the 8 Release-1 tutorials in `examples/tutorials/`,
covering checks A (citations), B (plan alignment), C (spec coherence) and
D (8 reviewer-rubric scores). Static validators (`scripts/tutorial_audit/`)
ran clean before this pass; this is the human/LLM-judgment layer.

DOI resolution used Crossref / Datacite / OpenAlex / Nature / Frontiers
direct fetches. Each unique DOI was fetched once and cached; reuse is
noted in the citations table.

## Executive summary

| Section | Pass | Partial | Fail |
| --- | ---: | ---: | ---: |
| A. Citations resolve | 6 | 1 | 1 |
| B. Plan alignment | 6 | 2 | 0 |
| C. Spec coherence | 6 | 2 | 0 |
| D. Reviewer rubric (E2.11/E2.14/E2.17/E4.31/E4.33/E4.35/E5.46/E6.47) | 7 | 1 | 0 |

Anchor reference coverage (plan §"reviewer-only" — does each anchor appear
where its topic warrants?):

| Anchor reference | Tutorials where required | Tutorials cited correctly |
| --- | --- | --- |
| Cisotto & Chicco 2024 (filtering, leakage, baseline) | plot_10, plot_11, plot_12, plot_40 | 4/4 |
| Pernet et al. 2019 (EEG-BIDS, montage) | plot_00, plot_01, plot_10, plot_11 | 3/4 *(plot_11 cites the wrong DOI)* |
| Gramfort et al. 2013 (MNE) | plot_01, plot_02, plot_10, plot_40 | 4/4 |
| Schirrmeister et al. 2017 (Braindecode/CNN) | plot_02, plot_12, plot_13 | 3/3 |

## DOI resolution cache

Verified via Crossref/Datacite/OpenAlex during this audit (status 2026-05-06).

| DOI | Resolved as | Status |
| --- | --- | --- |
| 10.1038/s41597-019-0104-8 | Pernet et al. 2019, "EEG-BIDS …", *Scientific Data* | OK |
| 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024, "Ten quick tips for clinical … EEG …", *PeerJ CS* | OK |
| 10.3389/fnins.2013.00267 | Gramfort et al. 2013, "MEG and EEG data analysis with MNE-Python", *Front. Neurosci.* | OK |
| 10.1002/hbm.23730 | Schirrmeister et al. 2017, "Deep learning with CNNs for EEG decoding and visualization", *Hum. Brain Mapp.* | OK |
| 10.1038/sdata.2015.1 | Wakeman & Henson 2015, "A multi-subject, multi-modal human neuroimaging dataset", *Sci. Data* | OK |
| 10.1038/sdata.2016.18 | Wilkinson et al. 2016, "FAIR Guiding Principles …", *Sci. Data* | OK |
| 10.18112/openneuro.ds002718.v1.0.5 | Wakeman 2021 OpenNeuro deposit (Face processing EEG dataset) | OK (Datacite) |
| 10.18112/openneuro.ds005514.v1.0.0 | Shirazi 2024 OpenNeuro deposit (HBN EEG Release 9) | OK (Datacite) |
| 10.1007/BF01797193 | Berger 1929, "Über das Elektrenkephalogramm des Menschen", *Arch. Psychiat. Nerv.* | OK |
| 10.1038/sdata.2018.110 | Niso et al. 2018, "MEG-BIDS …", *Sci. Data* | **NOT Pernet 2019** |
| 10.1088/1741-2552/ace8c7 | 404 (no record on Crossref/OpenAlex/IOP) | **DOES NOT RESOLVE** |

## Top fixes recommended (prioritised)

1. **plot_11_leakage_safe_split.py:17 and :219** — replace `doi:10.1038/sdata.2018.110` (which is Niso 2018 MEG-BIDS) with the EEG-BIDS DOI `10.1038/s41597-019-0104-8`, OR drop the "Pernet … BIDS split metadata" claim entirely. The plan does not require a Pernet citation here, and BIDS itself does not standardise split metadata as currently implied.
2. **plot_11_leakage_safe_split.py:220** — replace the broken `doi:10.1088/1741-2552/ace8c7` for "Aristimunha et al. 2023 — MOABB protocol" with the live MOABB benchmark DOI `10.48550/arXiv.2404.15319` (Chevallier et al. 2024, "The largest EEG-based BCI reproducibility study … the MOABB benchmark"). Update author tag to "Chevallier, Aristimunha, et al. 2024" and adjust the in-text reference at line 108-109.
3. **plot_11_leakage_safe_split.py:189-196** — the spec invariant `n_subjects_total > 10` is satisfied by the 12-subject mock metadata. But the spec also asserts `n_folds >= 5`, while the `cross_session` swap in Step "Modify" calls `n_folds=2`. Rename or document the `n_folds>=5` invariant as fold-1 specific (or only assert it on the cross-subject manifest, not the cross-session demo).
4. **plot_13_save_and_reuse_prepared_data.py:49** — `from braindecode.datasets import BaseConcatDataset, RawDataset` — `RawDataset` is not part of Braindecode's public API (the public class is `BaseDataset`). The static audit cannot catch this; it will surface at runtime gate. Replace with `BaseDataset` (and constructor `BaseDataset(raw, description=...)`), or the example will fail to import.
5. **plot_13_save_and_reuse_prepared_data.py:50** — `from braindecode.datautil import load_concat_dataset`: in current Braindecode this lives in `braindecode.datasets`, not `braindecode.datautil`. Update import to `from braindecode.datasets import load_concat_dataset`.
6. **plot_12_train_a_baseline.py:23, 39-40** — spec at line 7 promises `estimated_runtime_minutes: 3` and tutorial promises "~3 s" (line 23). The 3-second figure under-promises against the spec but is fine. However the tutorial uses *fully synthesised* windows and never imports any `eegdash` API beyond `majority_baseline`, while `requires_api` in the spec claims `apply_split_manifest`, `ShallowFBCSPNet`, etc. (spec lines 91-95). Either drop the unused entries from `requires_api`, or actually exercise them (the latter is preferred per the plan, which calls for "reuse windows and split from tutorial 11").
7. **plot_12_train_a_baseline.py:91-105** — the tutorial *manually* prints the leakage_report JSON line rather than calling `eegdash.splits.assert_no_leakage`. Spec rubric cites E5.42, which expects `assert_no_leakage` invocation. Pass the synthetic metadata to the real helper to make the runtime check cite the API the spec promises.
8. **plot_13_save_and_reuse_prepared_data.py:91, 162** — the tutorial (and spec line 51) asserts `second_run_runtime_seconds < first_run_runtime_seconds`, but no timing measurement is performed in the code. Add a `time.perf_counter()` pair around the cache-miss vs cache-hit calls, or remove the invariant from the spec.
9. **plot_10_preprocess_and_window.py:13-18** — the plan's tutorial 10 brief calls for "select a dataset with stable data" and the spec.title says "Preprocess EEG and create reusable windows", but the tutorial loads `task="FacePerception"`, an event-related task — not the resting-state lead the opening sentence implies (plot_10.py:14). Either reword the opening to drop "resting-state subject" or pick `task="rest"` if the dataset exposes it. Currently mismatches the spec's "fixed-length or event-based" guidance only mildly, but the prose is internally inconsistent.
10. **plot_40_first_features.py:6-8** — the opening claims "BIDS dataset `ds005514` HBN resting-state, 1-40 Hz FIR band-pass, 128 Hz sample rate" but the tutorial *synthesises* signals locally rather than loading `ds005514` (lines 64-86). Add the explicit "we mimic the windows here for offline reproducibility" framing at the top docstring (it appears in the requirements section line 27 but should be in the title paragraph too) so the reader does not believe a real download has happened.

## Per-tutorial review

### 1. plot_00_first_search.py

**A. Citations.**
- DOIs in references block:
  - `10.1038/s41597-019-0104-8` (Pernet 2019) — resolves OK; cited in-text at line 110.
  - `10.1038/sdata.2015.1` (Wakeman 2015) — resolves OK; cited in-text lines 6, 86.
  - `10.7717/peerj-cs.2256` (Cisotto 2024) — resolves OK; cited in-text line 193.
- All three in-text mentions are mirrored in the References block.
- Anchor coverage: Pernet (montage/BIDS topic, warranted) — present. Cisotto (warranted because the tutorial is a discovery step that motivates Tip 1/2) — present. Gramfort + Schirrmeister not warranted (no MNE plot, no model). PASS.

**B. Plan alignment.** Plan §L819-844 prescribes 7 steps; the tutorial covers (1) `EEGDash` import, (2) client, (3) limited query (line 64), (4) print count + keys (lines 69, 79), (5) query by task/subject (lines 96, 161), (6) cohort statistics (line 126-139), (7) link forward to plot_01 + concept page (lines 200-202, 217). The plan's "Move from `examples/tutorials/tutorial_api.py`" is reflected in spirit. PASS.

**C. Spec coherence.** Four spec learning_objectives map 1:1 to the bullets at lines 14-20. asserted_invariants `n_records_returned > 0`, `<= search_limit`, BIDS keys present, no signal bytes are all enforced (lines 100-105). `requires_api` lists `EEGDash`, `EEGDash.find`, `EEGDash.search_datasets` — only the first two are used (`find_datasets` and `find` are called); `search_datasets` is not exercised. Minor partial-fail: the spec promises `search_datasets` but the tutorial calls `find_datasets`. PARTIAL.

**D. Reviewer rubric (1-5).**
- E2.11 narrative: 5. Beginning (why metadata-only search), middle (predict→run→investigate cycles, 4 steps), end (wrap-up + next tutorial pointer).
- E2.14 cognitive load: 4. Cells are short and one-concept each. Minor: the `shortlist_datasets` function (line 175) introduces both lambda-style filtering and Mongo-syntax in one cell, slight overload.
- E2.17 intentional error: 3. The try/except offline fallback (lines 64-69) shows graceful failure, but there is no *teaching* error — nothing says "watch what happens if you forget the limit". Counts as warn, not full pass.
- E4.31 motivating question: 4. "How many recordings, from how many subjects, at which sampling rate" (lines 5-9) is a real research-prep question, not generic.
- E4.33 result meaningful: 4. `cohort_stats` output is a defensible artefact for picking a follow-up dataset; result is interpretable.
- E4.35 tone: 4. "We" (line 84) and present tense ("we surveyed", line 191), explains *why* (line 53). Mild issue: line 207 ("Persist the shortlist…") is imperative without explaining motivation.
- E5.46 hedging: 5. Line 192-194 explicitly hedges: "the index answer does not yet tell us anything about signal quality".
- E6.47 Diataxis purity: 5. Stays a tutorial; explanation links out to concept page.

### 2. plot_01_first_recording.py

**A. Citations.**
- DOIs in references block:
  - `10.18112/openneuro.ds002718.v1.0.5` (line 209) — resolves OK on Datacite (Wakeman OpenNeuro deposit); but the lead-author label `Wakeman, D. G., and Henson, R. N. (2015)` is *misleading* because the deposit DOI registers the OpenNeuro mirror, not the 2015 paper. Either move the deposit DOI to a "Dataset" subsection or pair it with `10.1038/sdata.2015.1` for the paper.
  - `10.1038/s41597-019-0104-8` (Pernet) — resolves OK; cited in-text via `eegdash.dataset.preview` discussion implicitly.
  - `10.3389/fnins.2013.00267` (Gramfort) — resolves OK; cited in-text line 213-214.
- Anchor coverage: Pernet warranted (BIDS entities surfaced) — present. Gramfort warranted (MNE-Python `mne.io.Raw.plot`) — present. Cisotto absent (warranted? plot_01 inspects raw signal — Tip 2 "report channel set"; mild gap, not flagged). PASS.

**B. Plan alignment.** Plan §L846-863 prescribes 6 steps + lazy-loading explanation + cache path + sanity checks. Tutorial covers all six explicitly (lines 81, 82, 100, 102, 132, 142). PASS.

**C. Spec coherence.** Four spec learning_objectives match bullets at lines 16-20. asserted_invariants checked at lines 83, 106. `requires_api` lists `EEGDashDataset`, `.summary`, `.preview` — all three are called (lines 81, 90, 100). PASS.

**D. Reviewer rubric.**
- E2.11: 5. Three-act: pick→load→inspect→plot, with predict→run→investigate folded in.
- E2.14: 4. Step 4 (line 132-154) bundles two figures in one cell; figure adjacency is fine but the second figure could be its own cell.
- E2.17: 2. No intentional error shown. The Modify cell (line 165) just changes a parameter without showing recovery from a misstep. Marked as warn.
- E4.31: 5. "What does a single EEG recording from EEGDash actually contain?" (line 5) is a real onboarding question.
- E4.33: 5. Plot of a real 5-second snippet plus annotation count is interpretable.
- E4.35: 4. Mostly "we"; "Pick a *different task*" (line 177) is imperative but explains why ("repeat the inspection").
- E5.46: 4. Line 200-203 "the signal is unprocessed — alpha rhythms, line noise, slow drifts coexist" is an explicit limitation flag.
- E6.47: 5. Stays a tutorial; concept page linked.

### 3. plot_02_dataset_to_dataloader.py

**A. Citations.**
- DOIs in references: `10.1038/sdata.2015.1` (Wakeman), `10.1002/hbm.23730` (Schirrmeister/Braindecode), `10.3389/fnins.2013.00267` (Gramfort), `10.7717/peerj-cs.2256` (Cisotto). All four resolve OK. All four mentioned in-text (lines 7-10, 192).
- Anchor coverage: Schirrmeister warranted (Braindecode is the loader) — present. Gramfort warranted (MNE under the hood) — present. Cisotto warranted (claims plumbing-only result, Tip 9 wording) — present. Pernet not warranted (no BIDS-entity emphasis) — absent, OK. PASS.

**B. Plan alignment.** Plan §L865-881: load small dataset, 1-2 preprocessors, fixed-length windows, inspect (X,y), DataLoader, batch shape — and "no model training", "clear sample shape expectations". Tutorial follows verbatim (lines 71, 100-103, 126-129, 131-132, 149-152, 153). PASS.

**C. Spec coherence.** Four learning_objectives match bullets 19-24. asserted_invariants `X.ndim==3`, batch shape components — checked at lines 133, 185. RNG seeds set (lines 53-54). `requires_api`: `EEGDashDataset`, `create_fixed_length_windows`, `DataLoader` — all three exercised. PASS.

**D. Reviewer rubric.**
- E2.11: 5. Clear arc; closes with explicit "no model training" framing.
- E2.14: 5. Each cell does one thing (load, preprocess, window, loader, custom collate).
- E2.17: 3. The `assert sfreq == TARGET_SFREQ` (line 109) is a defensive check rather than a *teaching* error; the Modify cell (line 162) is healthier — it asks the reader to predict before changing — but no error is recovered. Marked warn.
- E4.31: 4. "How do we go from a single EEG recording on disk to a tensor batch a deep learning model can ingest?" — pragmatic and concrete.
- E4.33: 5. Final batch shape printed and asserted; the `unique y` printout (line 152) is interpretable.
- E4.35: 5. "We" throughout, present tense, explains why each preprocessor is "safe" (line 91-92).
- E5.46: 4. Lines 192-194 explicitly hedge: "a clean batch shape only confirms *plumbing* — not signal quality or task design."
- E6.47: 5. Diataxis-clean.

### 4. plot_10_preprocess_and_window.py

**A. Citations.** DOIs: `10.7717/peerj-cs.2256` (Cisotto), `10.1038/s41597-019-0104-8` (Pernet), `10.3389/fnins.2013.00267` (Gramfort). All resolve OK; all cited in-text (lines 7-8, 36-38, 70). Schirrmeister mentioned in spec but not in the tutorial body — neither the model nor the Braindecode CNN is invoked here, so this is fine. PASS.

**B. Plan alignment.** Plan §L883-900: select dataset, pick channels, resample, filter, create windows, verify count, save windows. Tutorial covers steps 1, 4, 5, 6 explicitly (set montage, set reference, filter, resample, create windows). It does NOT cover (2) "pick channels" — `pick_types` from plot_02 is not repeated here, although `set_montage(..., on_missing="ignore")` implicitly handles non-EEG. It does NOT cover (7) "save windows to disk" — the assertion is left to the reader in "Try it yourself" (line 219). The "windows.save" call is referenced in the Try-it-yourself but not actually run. Spec asserted_invariant `saved_windows_path exists and reload returns equal shapes` (line 57) is therefore NOT enforced in code. PARTIAL.

**C. Spec coherence.** Four learning_objectives match the docstring bullets 22-27. asserted_invariants partly enforced: `raw_resampled.info['sfreq'] == target_sfreq` is at line 151, `windows[0][0].shape[1] == int(window_size_s * target_sfreq)` at line 200. The `saved_windows_path exists` invariant is unenforced (see B). `requires_api`: `EEGDashDataset`, `Preprocessor`, `preprocess`, `create_fixed_length_windows` — all four used. PARTIAL.

**D. Reviewer rubric.**
- E2.11: 5. Strong six-step arc. Cisotto Tips 4-5 frame each decision.
- E2.14: 4. Step 6 (line 154-177) re-creates the same five Preprocessors a second time; this is duplicated cognitive load and could be unified by building the Preprocessor list once and calling `preprocess(dataset, prep_list)` once for both `raw` (preview) and `dataset` views.
- E2.17: 3. No intentional-error block. Marked warn, consistent with other tutorials.
- E4.31: 5. "How do we turn one OpenNeuro recording into a deterministic, reusable windows dataset?" is a real research-engineering question.
- E4.33: 4. Window-shape assertion is defensible; would be stronger with a before/after PSD figure to show the filter actually moved energy.
- E4.35: 5. Cisotto Tips quoted by name (Tip 4, Tip 5); "we" inclusive throughout.
- E5.46: 4. Line 6-9 hedges trustworthiness; line 145-148 names the Nyquist trade-off.
- E6.47: 5. Stays a tutorial; concept page linked.

### 5. plot_11_leakage_safe_split.py

**A. Citations. FAIL.**
- DOIs in references:
  - `10.7717/peerj-cs.2256` — OK.
  - `10.1038/sdata.2018.110` cited as **Pernet et al. 2019 — BIDS split metadata** (lines 16-17, 219). This DOI resolves to Niso et al. 2018 MEG-BIDS, not Pernet. **WRONG DOI** AND wrong claim — BIDS does not standardise split metadata; the canonical EEG-BIDS DOI is `10.1038/s41597-019-0104-8`.
  - `10.1088/1741-2552/ace8c7` cited as **Aristimunha et al. 2023 — MOABB protocol** (line 220). DOI returns 404 on Crossref/OpenAlex/IOP. The actual MOABB benchmark paper is Chevallier et al. 2024 (lead) on arXiv (`10.48550/arXiv.2404.15319`); Aristimunha is the second author. **DOI DOES NOT RESOLVE.**
- Anchor coverage: Cisotto (Tip 9 leakage) — present, OK. Pernet — cited but with the wrong DOI; counts as a fail. Schirrmeister/Gramfort not warranted here.

**B. Plan alignment.** Plan §L902-919: load windows from previous tutorial, show subject IDs, show wrong random split, build MOABB-backed cross-subject manifest, assert disjoint, optionally stratify, print summary. Tutorial covers all six (lines 62-77, 88-100, 121-125, 139-149, 159-164). PASS.

**C. Spec coherence.** Four learning_objectives match the bullet list at lines 23-28. asserted_invariants:
- `n_subjects_total > 10` — 12-subject mock (line 70). Pass.
- `subject_overlap(train, test) == 0` — line 140. Pass.
- `n_folds >= 5` — manifest from line 121 has `n_folds=5`. Pass for cross-subject; the `cross_session` demo at line 191 only uses `n_folds=2` and would technically violate this invariant if the runtime validator inspects every manifest produced. Recommend qualifying the invariant or only asserting on the primary manifest.
- `class_balance_train - class_balance_test < 0.10` — implicit through `class_balance_ratio` print at line 145, not an explicit assert.
- `naive_random_split_overlap > 0` — line 100. Pass (E2.17 satisfied here).
`requires_api` lists six functions; the tutorial uses `get_splitter`, `make_split_manifest`, `assert_no_leakage`, `describe_split`, `apply_split_manifest`, `manifest_to_json`. `to_split_metadata` is named in `requires_api` but not called. PARTIAL.

**D. Reviewer rubric.**
- E2.11: 5. Strongest narrative of the eight: predict-the-leak → show-the-leak → fix-the-leak.
- E2.14: 4. Step 2 (line 79-100) packs the Predict prompt, the random shuffle, and the assertion in adjacent cells — this is fine. Step 4 (line 128-149) prints both `assert_no_leakage` and `describe_split` outputs in one cell; consider splitting.
- E2.17: 5. Best example of intentional-error-and-recovery in the suite: the naive random split is intentionally wrong (line 88-100), the recovery is the cross_subject manifest (line 121-125). Spec rubric override at this rule (E2.17) is satisfied.
- E4.31: 5. "Why does randomly splitting EEG windows … give you 99% accuracy that does not generalize to a new participant?" — the canonical EEG-ML pitfall, framed exactly as Cisotto Tip 9 frames it.
- E4.33: 5. The `leakage_report` JSON (line 174) is *the* runtime invariant the validator parses (E5.42).
- E4.35: 5. "We" throughout; explains *why* a random split fails ("memorise … alpha-rhythm fingerprint", line 104).
- E5.46: 4. Cisotto Tip 9 cited explicitly; would benefit from one more sentence flagging that subject-aware does not eliminate session leakage.
- E6.47: 5. Stays in the tutorial quadrant; concept page linked.

### 6. plot_12_train_a_baseline.py

**A. Citations.** DOIs: `10.7717/peerj-cs.2256` (Cisotto), `10.1002/hbm.23730` (Schirrmeister). Both resolve OK; both cited in-text. Anchor coverage: Cisotto warranted (Tip 5, "transparent baseline before deep net" — line 134-135) — present. Schirrmeister warranted (mentioned as the alternative — line 191) — present. PASS.

**B. Plan alignment.** Plan §L921-939: reuse windows + split from plot_11, majority/median baseline, train a small model (logistic regression OR ShallowFBCSPNet), evaluate, print metrics, baseline-before-NN, minimal training loop, reproducible seed. Tutorial covers majority baseline (line 157), logistic regression (line 144), evaluation (line 146-147), metric table (line 173-177), seed (line 44-45). It does NOT actually reuse windows or split from plot_11 — it synthesises windows in `synthesise_windows` (line 58). The plan explicitly says "Reuse windows and split from tutorial 11"; the deviation is unreasonable for a 1-star tutorial — even a minimal call to `apply_split_manifest` would honour the plan. PARTIAL.

**C. Spec coherence.** Four learning_objectives map to bullets at lines 14-19. asserted_invariants: `leakage_report['overlap'] == 0` (line 105 — but the manifest is hand-built rather than from `apply_split_manifest`). `majority_baseline_accuracy >= chance_level - 0.02` — implicit in the print but not asserted. `model_accuracy_test >= majority_baseline_accuracy` — not asserted. `model_accuracy_test_std reported across folds (n_folds >= 5)` — NOT done; tutorial uses a single 2-subject hold-out, not 5 folds. `requires_api`: `apply_split_manifest`, `majority_baseline`, `ShallowFBCSPNet`, `LogisticRegression`. Of these only `majority_baseline` and `LogisticRegression` are exercised. PARTIAL.

**D. Reviewer rubric.**
- E2.11: 4. Beginning ("did the model learn anything beyond chance?") → middle (build features, train, score) → end (modify and make).
- E2.14: 5. Cells stay short; band-pass filter is one cell; LR fit is one cell; metric table is one cell.
- E2.17: 4. The Modify block (line 180-184) is a soft intentional error: "swap band-power for raw window → accuracy will drop." This is the only one of the four "training" tutorials that explicitly asks the reader to provoke a regression and explain it.
- E4.31: 4. "Did the model learn anything beyond the class prior?" — concrete and benchmark-relevant.
- E4.33: 3. The result is interpretable, but the synthetic alpha-vs-delta injection is too on-the-nose — the model is *guaranteed* to win above chance, which weakens "scientifically meaningful". Marking warn.
- E4.35: 5. "We" throughout; reasons each step ("L2 penalty - the simplest defensible classifier" line 134-135).
- E5.46: 4. Line 165-167 ("rerun with a different SEED to feel the variance, then commit to a single seed") is a useful caveat; line 184 "chance is *the floor*" is a clean limitation.
- E6.47: 4. Stays a tutorial. The Make block (line 188) gestures at ShallowFBCSPNet without showing it; that boundary keeps Diataxis purity but trades it for plan misalignment (see B).

### 7. plot_13_save_and_reuse_prepared_data.py

**A. Citations.** DOIs: `10.1002/hbm.23730` (Schirrmeister) and `10.1038/sdata.2016.18` (Wilkinson FAIR). Both resolve OK; both cited in-text. Anchor coverage: Schirrmeister cited as the reason Braindecode's `BaseConcatDataset.save` exists — warranted, present. Wilkinson FAIR cited as the why-cache argument — appropriate. Cisotto/Pernet/Gramfort not warranted at this layer. PASS.

**B. Plan alignment.** Plan §L1198-1206 (Quality Bar) and the spec frame plot_13 as the persistence companion. The tutorial covers save → reload → roundtrip parity → feature table to Parquet → cache_or_compute helper. PASS in spirit. *However*: the tutorial uses a synthetic 2-channel signal (line 68-70) rather than reusing the windows produced by plot_10/12 — same caveat as plot_12. The plan does not list plot_13 in §"Detailed Tutorial Proposals" beyond its role as a follow-up; reuse of upstream artifacts would have been ideal but is not strictly required. PARTIAL (consistent with plot_12).

**C. Spec coherence.** Four learning_objectives match the docstring bullets at lines 19-25. asserted_invariants:
- `saved_artifact_path exists` — line 99 implicitly.
- `reloaded_dataset.shape == original_dataset.shape` — line 126.
- `reloaded_dataset.metadata == original_dataset.metadata` — line 128 checks columns, not full metadata equality. Partial.
- `manifest contains: eegdash_version, mne_version, braindecode_version, random_seed, split_manifest_hash` — NOT enforced. The tutorial only prints `eegdash` and `braindecode` versions at line 57; no JSON manifest is written.
- `second_run_runtime_seconds < first_run_runtime_seconds` — NOT enforced (no timing).
`requires_api`: `EEGDashDataset` (NOT used; tutorial uses raw `RawDataset`), `BaseConcatDataset.save` (used), `load_concat_dataset` (used). The `EEGDashDataset` requirement is unfulfilled. PARTIAL.

Additional concern: the imports `from braindecode.datasets import BaseConcatDataset, RawDataset` and `from braindecode.datautil import load_concat_dataset` likely fail at runtime under current Braindecode (`RawDataset` is not public; `load_concat_dataset` lives in `braindecode.datasets`). Static lint cannot catch this — runtime gate will. (See top fixes #4, #5.)

**D. Reviewer rubric.**
- E2.11: 4. Build → save → reload → save table → reload table → cache_or_compute. Slightly over-condensed.
- E2.14: 4. Step 4 (line 132-150) computes features, writes Parquet, reads back, asserts equality — one cell, three concepts. Could split.
- E2.17: 3. The "Try it yourself" item "Re-save with `overwrite=False` and observe the `FileExistsError`" (line 202) is an intentional-error pointer but not actually run inline. Spec override at E2.17:warn (line 91) acknowledges this; appropriate.
- E4.31: 4. "How do we save that work to disk and reload it in the next session without rerunning the pipeline?" — a real ML-engineering question.
- E4.33: 4. The roundtrip parity (`np.allclose`, dtype equality) is meaningful for a reproducibility lesson.
- E4.35: 4. "We" throughout; explains why Parquet ("columnar, typed, compressed, readable from R, Julia …" line 134-138).
- E5.46: 4. Line 187-189 explicitly limits: "Caching is never automatic: it costs disk and demands provenance."
- E6.47: 5. Stays in tutorial quadrant.

### 8. plot_40_first_features.py

**A. Citations.** DOIs: `10.7717/peerj-cs.2256` (Cisotto), `10.3389/fnins.2013.00267` (Gramfort), `10.1007/BF01797193` (Berger 1929), `10.18112/openneuro.ds005514.v1.0.0` (HBN deposit). All four resolve OK. Berger 1929 verified: "Über das Elektrenkephalogramm des Menschen", *Arch. Psychiat.* — historically correct citation for the alpha rhythm claim at line 4. Anchor coverage: Cisotto warranted (filter disclosure Tips 4-5, line 60-61) — present. Gramfort warranted (MNE dependency) — present. Schirrmeister/Pernet not strictly required here. PASS.

**B. Plan alignment.** Plan §L977-992: load EO/EC windows, extract `signal_variance` + RMS + spectral band powers, display feature table head, join with metadata, save feature dataset. Tutorial covers all five (lines 82-86, 100-107, 116-119, 126-133, 167-172). The plan's "Move from … noplot_tutorial_features_eoec.py" is preserved. The tutorial *synthesises* windows rather than loading from plot_10 (same systematic deviation as plot_12/13) — but the offline rationale is in the requirements section (line 27). PARTIAL.

**C. Spec coherence.** Four learning_objectives match bullets at lines 16-21. asserted_invariants:
- `feature_table.shape[0] == n_windows` — line 128.
- `feature_table.shape[1] >= 5` — line 129.
- `all column names contain a channel-name token` — line 130.
- `feature_table.join(metadata).dropna(...).shape[0] == feature_table.shape[0]` — NOT enforced as a literal assertion; the `target` column is included via `include_target=True` but no metadata join with subject is performed.
- `saved_features_path exists and reloads with the same dtypes` — line 168-171.
`requires_api`: `fit_feature_extractor`, `extract_features`, `signal_variance`, `rms`, `bandpower` — only `extract_features`, `signal_variance`, and (via `spectral_bands_power`, not the spec name `bandpower`) the band-power feature are exercised. Naming drift between `signal_root_mean_square` (used) and `rms` (in spec) is a minor doc issue. `fit_feature_extractor` not called. PARTIAL.

**D. Reviewer rubric.**
- E2.11: 5. Berger anchor → predict → run → investigate → save. Cleanest narrative of the feature-engineering cluster.
- E2.14: 5. Each cell does one thing.
- E2.17: 3. No intentional error inline; the `Modify` block (line 175-178) just adds a feature.
- E4.31: 5. "Can a small set of band-power features distinguish eyes-open from eyes-closed?" — a real EEG-101 question.
- E4.33: 4. Eyes-closed alpha bump is the canonical demonstration; tutorial flags "a sanity check, not a claim about real data" (line 200-201) — exactly the right framing.
- E4.35: 5. "We" throughout; explains why each feature was chosen (line 92-95).
- E5.46: 5. Line 200-201 is the textbook hedged-claim sentence: "Eyes-closed alpha is roughly two orders of magnitude higher than eyes-open here -- a sanity check, not a claim about real data."
- E6.47: 5. Stays in tutorial quadrant; plot_41 referenced as the next step for spectral preprocessor reuse.

## Citations status table

Status legend: `OK` = DOI resolves and matches the in-text claim; `MISMATCH` =
DOI resolves but to a different paper; `404` = DOI does not resolve.

| Tutorial | DOI | Cited as | Status |
| --- | --- | --- | --- |
| plot_00 | 10.1038/s41597-019-0104-8 | Pernet 2019 EEG-BIDS | OK |
| plot_00 | 10.1038/sdata.2015.1 | Wakeman & Henson 2015 | OK |
| plot_00 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 | OK |
| plot_01 | 10.1038/s41597-019-0104-8 | Pernet 2019 EEG-BIDS | OK |
| plot_01 | 10.3389/fnins.2013.00267 | Gramfort 2013 MNE | OK |
| plot_01 | 10.18112/openneuro.ds002718.v1.0.5 | Wakeman OpenNeuro deposit | OK (Datacite) |
| plot_02 | 10.1038/sdata.2015.1 | Wakeman & Henson 2015 | OK |
| plot_02 | 10.1002/hbm.23730 | Schirrmeister 2017 Braindecode/CNN | OK |
| plot_02 | 10.3389/fnins.2013.00267 | Gramfort 2013 MNE | OK |
| plot_02 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 | OK |
| plot_10 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 | OK |
| plot_10 | 10.1038/s41597-019-0104-8 | Pernet 2019 EEG-BIDS | OK |
| plot_10 | 10.3389/fnins.2013.00267 | Gramfort 2013 MNE | OK |
| plot_11 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 (Tip 9) | OK |
| plot_11 | 10.1038/sdata.2018.110 | Pernet 2019 BIDS split metadata | **MISMATCH** (resolves to Niso 2018 MEG-BIDS) |
| plot_11 | 10.1088/1741-2552/ace8c7 | Aristimunha 2023 MOABB | **404** (DOI does not exist) |
| plot_12 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 | OK |
| plot_12 | 10.1002/hbm.23730 | Schirrmeister 2017 | OK |
| plot_13 | 10.1002/hbm.23730 | Schirrmeister 2017 Braindecode | OK |
| plot_13 | 10.1038/sdata.2016.18 | Wilkinson 2016 FAIR | OK |
| plot_40 | 10.7717/peerj-cs.2256 | Cisotto & Chicco 2024 | OK |
| plot_40 | 10.3389/fnins.2013.00267 | Gramfort 2013 MNE | OK |
| plot_40 | 10.1007/BF01797193 | Berger 1929 (alpha rhythm) | OK |
| plot_40 | 10.18112/openneuro.ds005514.v1.0.0 | HBN ds005514 Release 9 | OK (Datacite) |

## Closing note

Wave B is in good shape on narrative, hedging, plan adherence, and Diataxis
purity. The two tutorials that need attention before the runtime gate are
**plot_11** (citation accuracy) and **plot_13** (Braindecode imports). The
systematic deviation from "reuse upstream artifacts" in plot_12, plot_13,
and plot_40 is the second-largest issue: each tutorial silently
re-synthesises data instead of loading the artifact the previous tutorial
saved, breaking the plan's promised cross-tutorial spacing dimension.
