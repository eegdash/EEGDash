# Tutorial review: real EEGDash data and public processing APIs

Reviewed 2026-09-10 against `7550e96d983069c364738f0745a30d8ef0e9bec7`.
Scope: all 41 public gallery scripts, their former plotting helpers, and the documentation execution profile.

[Issue #410](https://github.com/EEGDash/EEGDash/issues/410) identified malformed introductory text and a dataset citation attached to generated cross-subject results. The original gallery also contained generated signals, participant targets, feature tables and fallback scores elsewhere. The replacements use recorded EEG/EMG, actual image stimuli or observed metadata throughout.

## Final implementation

- EEGDash owns dataset discovery, acquisition, cached/offline loading and participant/event metadata. Feature examples use its `FeatureExtractor`, `extract_features`, spectral preprocessing and published feature functions instead of local FFT or variance implementations.
- EEGPrep handles the applicable EEG preprocessing through Braindecode's public adapters. The resting-state example runs the full artifact-cleaning pipeline; ERP, HBN, preprocessing, HPC and checkpoint examples use the relevant individual operations. Already-processed derivatives, metadata-only examples, EMG and bipolar sleep recordings do not receive an unrelated EEG cleaning pipeline.
- No free-standing custom function definitions remain in the 41 public scripts. Training uses native Braindecode/skorch or explicit framework operations; retrieval uses sklearn top-k scoring and NeuralSet frozen image features. The one MOABB adapter class remains because its external dataset interface requires methods. Obsolete plotting helpers are removed.
- Scripts retain rendered step-by-step explanations, observed targets, explicit cohort/recording selection, training-only model transformations, meaningful checks and measured output interpretation. Model failures never substitute invented results.
- The scripts use the current Braindecode `on_last_window` API and are validated with Braindecode 1.8.1, which satisfies the repository's declared minimum.

EEGPrep conversions can round annotation times or clear measurement dates. Where the selected operations preserve the recording timeline, examples retain original annotations and dates, verify origin/rate/duration, then restore the observed event times before windowing. This restoration is not valid after removing time segments. Tutorial 30 disables whole-window removal during cleaning, preserves source BAD spans across HBN reannotation, and enables MNE epoch rejection during window construction.

Full EEGPrep artifact calibration in tutorial 30 uses each complete recording independently, including the unlabelled held-out recording. Its prose explicitly describes an offline per-recording calibration protocol. The classifier still fits only on training participants; the example does not claim a frozen cleaner or causal online deployment.

## Instructional detail compared with Braindecode

Execution alone did not establish teaching depth. The first replacements were often too terse, so every page received a second, content-focused review. The direct comparators included [basic trialwise decoding](https://braindecode.org/stable/auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.html), [train/test/tune](https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html), [load/save](https://braindecode.org/stable/auto_examples/datasets_io/plot_load_save_datasets.html), [explicit PyTorch training](https://braindecode.org/stable/auto_examples/model_building/plot_train_in_pure_pytorch_and_pytorch_lightning.html) and [EEGPrep integration](https://braindecode.org/1.3/auto_examples/model_building/plot_bcic_iv_2a_eegprep_cleaning.html).

Each page was checked for:

1. A concrete task, prerequisites, bounded real input and the output to inspect.
2. Rendered explanations before conceptual steps, including why important settings were chosen.
3. Array axes, physical units, event/participant identity and representation changes at API boundaries.
4. Fitting versus testing boundaries, or the exact verification performed by a non-predictive example.
5. Interpretation of printed quantities and plot axes, weak results, limitations and a concrete next analysis.

The explanations are task-specific. A search page teaches selection and metadata rather than adding an unrelated training loop. An averaged ERP is distinguished from single-trial prediction. The challenge seed workflows state their narrower tasks and do not claim complete official competition evaluation. The review does not infer pedagogical equivalence from word counts.

## Per-page review

Every row below was reviewed and executed on real data. An unchanged API path means it already used the appropriate public interface, not that the page was skipped.

| Page | Final API choice and instructional focus |
|---|---|
| 00 first search | EEGDash catalogue queries; string identifiers, query limits, recordings versus participants and selection before acquisition. |
| 01 first recording | EEGDashDataset plus MNE inspection; lazy acquisition, voltage/time axes, source event meaning and PSD interpretation. |
| 02 DataLoader | EEGDashDataset → Braindecode windows → PyTorch batches; axes, crop indices, metadata order and separate training/test loaders. |
| 10 preprocess/window | EEGPrep DC-offset removal and common-average reference; source processing, limited posterior-cap reference, verified annotation preservation and reusable windows. |
| 11 leakage-safe split | Actual window metadata and sklearn group splitting; trial fraction versus group fraction and what the overlap checks establish. |
| 12 baseline | EEGDash Welch feature extraction; 12 narrow stimulus bands × 8 channels, physical power, training-only scaling and held-out confusion interpretation. |
| 13 save/reuse | Public Braindecode serialization of real windows; source cache versus prepared representation, manifest, precision and roundtrip checks. |
| 20 visual P300 | EEGDash acquisition and mean-amplitude features, EEGPrep centering/reference and MNE epochs; observed event mapping, first-returned ERP attribution and participant-held-out decoding. |
| 21 auditory oddball | EEGDash acquisition and interval mean, EEGPrep centering/reference; response-qualified observed labels, epoch counts, mean ERP difference and descriptive limits. |
| 30 eyes open/closed | Full EEGPrep cleaning before channel selection, verified instruction timing, retained BAD spans and EEGDash spectral power; offline calibration and subject-held-out interpretation. |
| 40 first features | Native EEGDash feature tree, DataFrame and schema; Welch settings, channel/band columns, physical values versus plot scaling and metadata exclusion from predictors. |
| 41 feature trees | Native tree reuse and actual equivalence/timing; matched columns, worker overhead and measured slowdowns. |
| 42 sklearn handoff | Consumes tutorial 40's real table/schema; predictor allowlist, training-only scaling, coefficient limits and participant split. |
| 50 within subject | EEGDash 96-feature Welch representation; real trial stratification, per-participant calibration and balanced recall. |
| 51 cross subject | Same public features, actual LOSO folds and measured plots; complete participant isolation, pooled confusion versus participant variability. |
| 52 cross session | EEGDash Welch mu/beta features from genuine sessions; half-open band bounds, physical power and directional generalization. |
| 53 learning curves | Native features and nested real training subjects; fixed validation group, reset fitting state and composition confounding. |
| 54 compare pipelines | Native features and paired real folds; shrinkage LDA versus logistic regression, paired differences and the limits of three-participant inference. |
| 55 MOABB interop | Necessary public BaseDataset adapter carries EEGDash recordings into MOABB; EEGDash variance, epoch endpoints and the limits of the interoperability demonstration. |
| 70 challenge basics | EEGChallengeDataset metadata; derivative provenance, eligible/requested/matched counts, repeated runs and acquisition cost. |
| 71 cross-task transfer | EEGClassifier/EEGRegressor replace a custom training function; observed reaction time, auxiliary cue labels, head replacement and disjoint participants. |
| 72 subject-invariant regression | EEGDash spectral functions on recorded challenge EEG; phenotype joins, participant rows, physical band power and subject-independent evaluation versus proven invariance. |
| 73 pretrained model | Real pinned CBraMod checkpoint and EEGPrep resampling; source event/date preservation, patch/head dimensions, frozen state and validation-only selection. |
| 74 NeuroAI interop | EEGDash voltages enter actual NeuralSet Segmenter/EegExtractor/DataLoader; native sampling, event origins and independent all-window numerical verification. |
| Applied age | Original HBN + EEGPrep drift/resampling + EEGDash feature tables; observed ages, participant aggregation, leave-one-out MAE and training-mean baseline. |
| Applied sex | Same public preparation/features; observed categories, one row per participant and the limits of a six-person cohort. |
| Applied p-factor features | Same native feature workflow with observed phenotype scores; aligned participant metadata, released-score units and baseline comparison. |
| Applied p-factor deep | EEGPrep preparation and native PyTorch model operations; grouped folds, training-only normalization, loss/optimizer sequence and participant aggregation. |
| Applied clinical summary | EEGDash metadata-only analysis; participant deduplication, missing values and acquisition duration versus clean data. |
| Applied P300 transfer | EEGDash recordings, EEGPrep centering/reference and native tensor operations; no custom MMD function, explicit label access and non-comparable training objectives. |
| EEG2025 challenge 1 | EEGDash event helpers and signal variance; measured reaction-time targets, prestimulus predictors and participant-held-out error. |
| EEG2025 challenge 2 | Observed externalizing scores, matching the final 2025 target; participant-level features, held-out predictions and training-mean/NRMSE comparison. |
| EEG2026 track 1 | Actual EEG, sidecar IDs and 60 JPGs; pinned DINOv2-giant targets via NeuralSet, with disjoint training/validation/test identities and held-out candidate galleries. |
| EEG2026 track 2 | Small real motor-imagery sessions, EEGPrep resampling and EEGDash variance; event-time preservation and binary seed-task scope. |
| EEG2026 track 3 | Actual first-N2 annotations and EEGDash band powers; five-second time-remaining targets capped at 600 seconds, participant-held-out predictions and official bMAE bin conventions. |
| EEG2026 track 4 | NM000281 EMG2Pose: 16 measured EMG channels and 20 observed joint-angle trajectories; published splits, BAD_IK rejection, five-second dense outputs and angular MAE. |
| Download how-to | Public EEGDash acquisition and local reopen; complete sidecars, byte counts and limited sample-equality coverage. |
| Offline how-to | EEGDash local discovery; online staging, preserved reference lifetime and the distinction between loading data and other network operations. |
| HPC-cache how-to | EEGDash loading plus stdlib staging; persistent versus scratch ownership, complete copies and real sample verification. |
| Parallel-feature how-to | EEGDash extract_features and signal_variance; actual serial/parallel equality, timing boundaries and memory/worker tradeoffs. |
| HPC/Slurm example | EEGPrep components, observed HBN event helpers and native LayerNorm replace a normalization function; timeline checks, fixed training budget and participant-held-out outputs. |

The spectral refactor is an intentional representation change, not a numerical-equivalence claim: general SSVEP models now use 96 channel-specific Welch band powers instead of 33 channel-averaged FFT magnitudes. Band reduction uses lower-inclusive/upper-exclusive limits; density sums are multiplied by bin spacing where physical power is needed. New scores are measured from the revised code.

## Data and CI

The regular documentation job executes 22 introductory/core/feature/evaluation/how-to and Track 2/4 scripts using three small EEGDash sources:

| Source | Explicit subset | Signal download |
|---|---|---|
| nm000118 / Nakanishi2015 | Subjects 1, 2, 3; session 0; run 0 | About 21.1 MB |
| nm000135 / BNCI2014-004 | Subject 1; sessions 0train and 1train; run 0 | About 10.6 MB |
| nm000281 / EMG2Pose | Three exact right-wrist recordings, one per published split | About 23.6 MB |

The PR cache is restored rather than only looked up. No acquisition or model failure is replaced by generated data. Full-gallery execution additionally downloads the task-specific HBN, P300, challenge, sleep, image and EMG sources disclosed on those pages. Cropping reduces computation rather than source-file download size. The regular CI profile renders these larger pages but does not execute them; a full-gallery run regenerates their outputs.

EEGPrep's EEG/MNE conversion dependency and NeuralSet's frozen-image dependencies are explicitly declared in the documentation extra. The existing EEGPrep 0.2.x bound is retained for the Braindecode integration.

## Reproduction and validation

After installing the project with its documentation dependencies, run the small example with:

```sh
PYTHONPATH=. MPLBACKEND=Agg EEGDASH_CACHE_DIR=.eegdash_cache \
  python examples/tutorials/50_evaluation/plot_51_cross_subject_evaluation.py
```

Run the gallery contracts and actual age-target integration check with:

```sh
PYTHONPATH=. MPLBACKEND=Agg EEGDASH_CACHE_DIR=.eegdash_cache \
  python -m pytest -q -o addopts='' \
  tests/unit_tests/test_eeg2026_gallery.py tests/integration/test_tutorial_age.py
```

Before the challenge-track follow-up, all 41 scripts passed individual real-data execution after the public-API refactor. Three regression tests, Ruff, Python/Sphinx-Gallery parsing and independent compilation of every gallery code cell passed. The final focused Sphinx-Gallery build executed all 41 of 41 public scripts successfully against real cached recordings and completed without Sphinx warnings. This includes the full EEGPrep pipeline, actual checkpoint/NeuralSet paths, image stimuli and EMG decoding. Slurm submission and the complete repository API/documentation build are separate from this focused gallery validation.

Local environment: Braindecode 1.8.1, EEGPrep 0.2.23, eeglabio 0.1.3, MNE 1.11.0, NumPy 1.26.4, torch 2.2.2 and MOABB 1.2.0. NeuralSet 0.3.1 interoperability executes successfully here, but this NumPy/torch pair is below its declared requirements; these runs do not establish compatibility with every dependency combination installed by the documentation extra.

## Challenge definitions checked against the live websites

The follow-up crawled both [2025](https://eeg2025.github.io/) and
[2026](https://neural-interfaces26.github.io/tracks.html), then their linked
starter kits before editing. The final 2025 target is externalizing; both 2025
examples now also report the starter kit's RMSE / target standard deviation.
The 2026 pose task replaces the former typing task. Track 1 uses the published
DINOv2-giant extraction settings, and Track 3 follows the executable first-N2,
window-stop and bin-edge conventions. The 2026 site schedules exclusive releases
for September 21, 2026; small public-data evaluations remain distinct from hidden
competition scoring. The source links and differences are explained on each page.

Real pose acquisition exposed an omitted `recording` entity in EEGDash's
MNE-BIDS path. Preserving it in the shared loader allows the source channel
sidecar to distinguish EMG predictors from joint-angle targets; a metadata-only
regression check covers current and older catalogue records.

Follow-up validation: all six 2025/2026 pages executed in a focused Sphinx-Gallery
build on real data with warnings treated as errors (zero warnings). Track 1's
default giant and optional small encoders both executed. The dataset unit suite
passed 626 tests with one skip; the four gallery/recording-entity checks and
pre-commit hooks passed. Track 4's three signal files total 23,578,821 bytes;
its cached gallery execution took 1.24 seconds. Cold acquisition also encountered
a temporary NEMAR manifest 503; the existing downloader successfully retried and
fell back to the original annex files, without substituting data.
