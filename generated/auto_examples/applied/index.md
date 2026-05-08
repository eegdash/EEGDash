# Applied Research Projects

Project-style examples that target a concrete scientific question –
age regression, p-factor prediction, sex classification, P300 transfer,
clinical-catalog summary – with realistic data sizes, runtimes, and
limitations. Difficulty 2-3; assumes the *Start Here* trio and the core
workflow track.

These are not first-week tutorials. They are scaffolds for your own
analyses: each script frames a research question, picks an appropriate
evaluation regime, runs a defensible baseline, and surfaces the
limitations honestly so you know what would have to change before any
result here could be cited. Compared with tutorials, these projects
emphasise labels, splits, baselines, and reporting rather than the
individual EEGDash API calls. Sourced from
`docs/tutorial_restructure_plan.md` Category G (lines 1052-1100,
“Applied Examples To Keep But Reframe”).

What you will learn:

- How to frame an EEG-from-population study (age, sex, p-factor) as a
  single regression or classification problem with an honest baseline.
- How to choose between a feature pipeline and a deep model based on
  data size and the question being asked.
- How to apply transfer learning across paradigms (P300 transfer
  across subjects and sessions) without leaking labels.
- How to summarise a clinical catalogue (subjects, sessions,
  conditions, hours of recording) for inclusion in a paper.
- How to write up the limitations section that an EEG paper actually
  needs (Cisotto and Chicco 2024; Pernet et al. 2019 for BIDS).

Treat each script as a starting point for your own work, not a
prescriptive recipe.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Can a feature-based regression head predict a child&#x27;s age from a few seconds of resting-state EEG, on subjects the model has never seen? This applied case study takes the Healthy Brain Network release ds005505 (Alexander et al. 2017, doi:10.1038/sdata.2017.181), surfaced through NEMAR (Delorme et al. 2022, doi:10.1093/nargab/lqac023), wires up an eegdash.EEGDashDataset query, builds a strict subject-aware split (Cisotto and Chicco 2024, doi:10.7717/peerj-cs.2256), fits a sklearn.linear_model.Ridge head on band-power features, and reports Pearson r, Spearman rho, R^2, and MAE against a median-baseline predictor. EEG-based brain-age regression has a long line of prior work (Zoubi et al. 2018, doi:10.3389/fnhum.2018.00461). The headline question here is not whether we beat the published literature; the honest one is, does the model beat the train-set median predictor on never-seen subjects?">  <div class="sphx-glr-thumbnail-title">Age regression from EEG (applied case study)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="A starter project: pull metadata for the OpenNeuro ds004504 clinical EEG release miltiadous2023 through EEGDashDataset, without downloading any signal bytes, and answer four questions a project plan needs answered before any modelling. The pool is served via NEMAR delorme2022nemar; the polished clinical workflow recipe follows Cisotto and Chicco 2024. The deliverable is one pandas.DataFrame with per-condition counts and one three-panel figure rendered from the live catalog numbers. Cohort imbalance, age confounds, recording-length mismatch, and channel-count drift are the four dataset-level pitfalls that silently break clinical EEG decoders before training even starts, so why not answer them first?">  <div class="sphx-glr-thumbnail-title">How do I survey a clinical EEG dataset before training a model?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Two laboratories run a visual oddball task on different participants, different head-caps, different software stacks. Both pipelines produce EEG epochs locked to a rare target and a frequent standard; both target the centro-parietal P3 component (Polich 2007, doi:10.1016/j.clinph.2007.04.019). Yet a P3 decoder trained on cohort A and evaluated on cohort B systematically loses several accuracy points relative to a target-trained ceiling (Cisotto &amp; Chicco 2024, Tip 8, doi:10.7717/peerj-cs.2256). This case study wires adversarial-style maximum mean discrepancy (AS-MMD; Long et al. 2015, https://arxiv.org/abs/1502.02791) between source and target, trains a small encoder, and asks the same applied question that cross-task pretraining (Banville et al. 2021, doi:10.1109/TNSRE.2020.3040290; Defossez et al. 2023, doi:10.1038/s42256-023-00714-5) and the EEG2025 cross-task transfer benchmark (Aristimunha et al. 2025, doi:10.48550/arXiv.2506.19141) ask through EEGChallengeDataset on the NEMAR archive (Delorme et al. 2022, doi:10.1093/database/baac096): by how much does AS-MMD close the naive-to-oracle gap, and does the alignment preserve the underlying P3 component?">  <div class="sphx-glr-thumbnail-title">Cross-cohort P3 transfer with AS-MMD: train on one oddball, deploy on another</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is the deep-learning case study for the p-factor regression project. The companion script project_pfactor_features.py covers the feature-based regime; this one trains braindecode.models.EEGConformer end-to-end on raw resting-state windows from the Healthy Brain Network release ds005505 (Alexander et al. 2017, doi:10.1038/sdata.2017.181), surfaced through NEMAR (Delorme et al. 2022, doi:10.1093/nargab/lqac023). The p-factor is a transdiagnostic score from the Child Behavior Checklist (Caspi et al. 2014, doi:10.1177/2167702613497473) and the modelling contract is the clinical-cautious one Cisotto and Chicco 2024 (doi:10.7717/peerj-cs.2256) ask for: cross-subject split, baseline alongside score, no diagnostic claim. Three regimes shape the framing, mirroring cousin tutorial plot_73: train from scratch, fine-tune a pretrained Braindecode encoder (Schirrmeister et al. 2017, doi:10.1002/hbm.23730), and read back where the network looks. The deliverable is a 3-panel figure plus printed metrics. So can a small EEGConformer beat the train-mean predictor on held-out subjects?">  <div class="sphx-glr-thumbnail-title">Predict p-factor from EEG with a Braindecode model (deep-learning regime)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Companion to project_pfactor_deep.py: same target, different model class. The deep variant asks whether an EEGConformer can learn the p_factor from raw windows, this one asks which interpretable EEG features carry the signal. The p-factor (Caspi et al. 2014, doi:10.1177/2167702613497473) is a transdiagnostic mental-health summary score derived from parent-and-child psychiatric questionnaires; the EEG side comes from the Healthy Brain Network release distributed on OpenNeuro and on the EEG2025 Challenge mirror as EEG2025r5 (Alexander et al. 2017, doi:10.1038/sdata.2017.181), surfaced through NEMAR (Delorme et al. 2022, doi:10.1093/nargab/lqac023). Splits stay strictly subject-disjoint per Cisotto and Chicco 2024 Tip 9 (doi:10.7717/peerj-cs.2256).">  <div class="sphx-glr-thumbnail-title">Predicting p-factor from EEG with hand-crafted features (project starter)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="A canonical &quot;is this signal even predictive?&quot; benchmark task on resting-state EEG from OpenNeuro ds005505 (Healthy Brain Network; Alexander et al. 2017), reachable through NEMAR delorme2022nemar. Log band-power features feed sklearn.pipeline.Pipeline pedregosa2011sklearn with StandardScaler and LogisticRegression. A 3-fold cross-subject loop via GroupKFold keeps every subject in exactly one fold; held-out predictions feed three sklearn display helpers, DecisionBoundaryDisplay, RocCurveDisplay, and ConfusionMatrixDisplay. Do the features separate the classes, how stable is held-out AUC across subjects, and which class does the model confuse?">  <div class="sphx-glr-thumbnail-title">Is resting-state EEG even predictive of the BIDS sex label? (Project Starter)</div>
</div>
<!-- thumbnail-parent-div-close --></div>
