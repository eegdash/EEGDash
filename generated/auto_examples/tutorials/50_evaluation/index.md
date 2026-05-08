# Evaluation and Benchmarking

Five lessons that treat decoding evaluation as a core skill, drawing on
MOABB (Chevallier, Aristimunha et al. 2024) and the evaluation guidance
in Cisotto and Chicco (2024). Difficulty 2-3; assumes the core workflow
track and the leakage-safe split lesson in particular.

Evaluation is where most EEG decoding claims fall apart: the model
trained on a single subject does not generalise to a held-out one, the
single-split accuracy hides session drift, and a paired comparison
between two pipelines is replaced by a bar chart with no statistics.
This category builds, in order, from a single within-subject split
toward a benchmark-grade paired comparison: which evaluation regime is
honest for your claim, and how do you report it. Sourced from
`docs/tutorial_restructure_plan.md` Category F (lines 425-442).

What you will learn:

- When within-subject diagnostics are appropriate, and when they are
  marketing.
- How to run a cross-subject evaluation – the gold standard for any
  generalisation claim – with `GroupKFold` and EEGDash’s split
  helpers.
- How to detect calibration drift across sessions of the same subject.
- How to plot a learning curve as a function of training subjects,
  trials, or windows, and read it for sample-efficiency claims.
- How to compare two pipelines on the same split and report the
  paired-Wilcoxon p-value the right way.

Run the lessons in order:

1. `plot_50_within_subject_evaluation.py` – single-subject
   diagnostics.
2. `plot_51_cross_subject_evaluation.py` – the gold standard for
   generalisation.
3. `plot_52_cross_session_evaluation.py` – calibration drift across
   sessions.
4. `plot_53_learning_curves.py` – performance vs training data
   size.
5. `plot_54_compare_two_pipelines.py` – paired comparison with the
   Wilcoxon test.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Cross-subject generalisation (plot_11) is the headline benchmark for EEG papers, but a calibration-style P300 speller, a clinical seizure detector tuned to one patient, or a lab paradigm where inter-subject variance dominates the contrast all care about a single brain at a time. In those cases subject overlap between train and test is intentional. This tutorial builds a 5-fold within-subject split, proves it is trial-disjoint, fits one LogisticRegression per subject, and puts the per-subject scores next to a leave-one-subject-out cross-subject reference on the same data.">  <div class="sphx-glr-thumbnail-title">When is within-subject decoding the right scientific question?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Cross-subject generalisation is the gold standard for any decoding claim. Train on N-1 subjects, test on the held-out one, repeat for every subject: that is leave-one-subject-out cross-validation (LOSO), the protocol behind the MOABB benchmark aristimunha2023transferstructure and the de-facto evaluation in clinical-EEG decoding. Brookshire et al. 2024 surveyed 81 deep-learning EEG papers and found data leakage in roughly half; on properly subject-held-out splits, the same architectures dropped on average from 0.83 accuracy to 0.62. Cisotto &amp; Chicco 2024 (Tip 9) name leakage the single most common reporting mistake. ds002718 wakeman2015, reachable through NEMAR delorme2022nemar, is the running example throughout the gallery.">  <div class="sphx-glr-thumbnail-title">How well does an EEG decoder generalise to a never-seen subject?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="A model that scores 90% on session 1 of sub-03 can slump to 70% on session 2 of the same subject. The cap and electrodes were placed a millimetre off, the gel impedance dropped, the participant slept poorly, or the recording fell at the wrong time of day. The decoder inherits all of those nuisance factors and the score drops the next morning. This is calibration drift: a covariate shift in the feature distribution that the within-session score never had to handle (Jayaram &amp; Barachant 2018, doi:10.1088/1741-2552/aabea9).">  <div class="sphx-glr-thumbnail-title">How much does a within-session decoder drift across sessions of the same subject?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Recording another month of EEG is expensive. Before committing more budget to data collection, ask whether the model is data-starved or whether the bottleneck sits elsewhere: features, architecture, label quality. The textbook answer is the learning curve (Hoffmann et al. 2014, doi:10.1109/TBME.2014.2300855): hold the validation pool fixed, grow the training pool over orders of magnitude, and read the slope. This tutorial sweeps training sizes from 50 to ~1000 windows on a synthetic cohort that mirrors a NEMAR delorme2022nemar OpenNeuro dataset with 24 subjects, scores balanced accuracy with sklearn.model_selection.learning_curve, and renders two panels: the curve itself and the train-minus-val gap that names the bias-variance regime. Cisotto &amp; Chicco 2024 (doi:10.7717/peerj-cs.2256, Tip 9) flag chance-aware reporting as the most-violated rule in clinical EEG; the chance line is on the figure. Schirrmeister et al. 2017 (doi:10.1002/hbm.23730, Braindecode) and the MOABB benchmark (Chevallier, Aristimunha et al. 2024, doi:10.48550/arXiv.2404.15319) sweep the same protocol on real EEG pipelines. The deliverable is one number: at what training-set size does the model first reach 90% of its plateau accuracy?">  <div class="sphx-glr-thumbnail-title">How does decoding accuracy scale with training-set size, and where does the curve plateau?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="A new decoding pipeline beats your linear baseline by three accuracy points on the held-out subject in a hackathon notebook. The gap looks big until somebody asks the obvious follow-up: would the same gap show up if you swap the test subject for any of the others sitting in NEMAR (Delorme et al. 2022, doi:10.1093/database/baac096)? When the same N subjects are scored by both pipelines you can answer that directly with a paired statistical test on the per-subject deltas. Demsar 2006 (Statistical comparisons of classifiers, JMLR) is the canonical reference for the recipe; Cisotto &amp; Chicco 2024 (doi:10.7717/peerj-cs.2256, Tip 9) flag the unpaired comparison as the single most common over-claim in clinical EEG. So: does the win survive a paired test, and how big is the effect once we strip the between-subject variance?">  <div class="sphx-glr-thumbnail-title">Is Pipeline A really better than Pipeline B, or did it luck out on one subject?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="EEGDash and MOABB sit on opposite ends of the BCI evaluation pipeline. EEGDash is a metadata index over BIDS-curated EEG pernet2019eegbids served from NEMAR delorme2022nemar; MOABB is the de-facto benchmark suite that pairs paradigm definitions (~moabb.paradigms.MotorImagery, P300) with evaluation procedures (~moabb.evaluations.CrossSessionEvaluation, CrossSubjectEvaluation) and a reproducibility study covering 30+ datasets (Aristimunha et al. 2023, Chevallier et al. 2024). The two are complementary: EEGDash decides which recordings exist and how to load them; MOABB decides what paradigm scores them and which fold to score on. The bridge braindecode.datasets.BaseConcatDataset.get_metadata returns (y, metadata) for any MOABB stratified splitter.">  <div class="sphx-glr-thumbnail-title">How do I benchmark an EEGDash dataset with MOABB?</div>
</div>
<!-- thumbnail-parent-div-close --></div>
