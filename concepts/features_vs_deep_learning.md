<a id="concepts-features-vs-deep-learning"></a>

# Features vs. deep learning

A recurring question in EEG decoding is whether to engineer features —
band power, common spatial patterns (CSP), Riemannian covariance — or
hand the raw signal to a convolutional or transformer-style network and
let it learn its own representation. The answer depends on three things:
how much data you have, how stationary the signal is, and how much
inductive bias you can afford to bake in.

The honest summary is that neither family dominates universally. On
small, single-task, single-cohort decoding problems, well-tuned feature
pipelines are often competitive with — and frequently better than —
deep nets trained from scratch. On large, heterogeneous corpora and on
tasks where the relevant features are not known a priori, end-to-end
deep models clearly win. Schirrmeister et al. (2017) <sup>[1](#id3)</sup> remains the
clearest demonstration that ConvNets *can* match expert feature
pipelines on motor-imagery decoding, but their result holds at scale and
with careful regularisation.

## When handcrafted features tend to win

Pick features when at least two of the following are true:

- **You have less than ~50 subjects.** A logistic regression or SVM on
  10 well-chosen features fits with minimal regularisation. A
  convolutional network has hundreds of thousands of parameters and
  needs orders of magnitude more data to avoid memorizing subject
  identity (see [Leakage and evaluation](leakage_and_evaluation.md)).
- **The relevant rhythm is known.** If you are decoding alpha-band
  modulation, log-power in 8–13 Hz is a near-optimal feature; a deep
  net will rediscover it in the best case and miss it in the worst.
- **You need interpretability.** Feature pipelines come with named,
  reportable inputs (“alpha at Pz”, “central-mu lateralisation”). Deep
  features come with saliency maps that are notoriously hard to read.
- **You need cross-dataset transfer.** Riemannian and CSP-based
  pipelines have well-understood invariances (channel permutation,
  reference change). A vanilla ConvNet trained on one montage can fail
  on another for trivial reasons.
- **You are CPU-bound.** Feature pipelines fit on a laptop in seconds.

The [Extract band-power features](../generated/auto_examples/tutorials/40_features/plot_40_first_features.md)
tutorial shows the simplest version of this argument: build a band-power
table, fit a logistic regression, and read off the result.

## When deep learning tends to win

Pick a deep model when at least two of the following are true:

- **You have hundreds of subjects.** Foundation-model-style deep
  decoders need large, diverse training sets to learn subject-invariant
  filters. EEGDash’s HBN-derived corpora make this regime accessible
  for the first time.
- **The relevant feature is not known.** Tasks like cognitive workload,
  emotional state, or fatigue rarely have a single canonical band; an
  end-to-end model can carve out a useful representation that no human
  has named.
- **You can afford to combine multiple datasets.** Deep models gain
  more from data diversity than from data volume; one large dataset is
  worth less than three medium ones with different montages.
- **You can apply augmentation.** Mixup, channel dropout, time masking,
  and frequency masking close most of the gap between deep nets and
  hand-tuned features on small data; without augmentation, the deep
  model is usually under-regularised.
- **You will fine-tune downstream.** A pre-trained deep model is a
  reusable asset; a hand-tuned feature pipeline is bespoke per task.

The [Train a leakage-safe baseline](../generated/auto_examples/tutorials/10_core_workflow/plot_12_train_a_baseline.md)
tutorial trains a small braindecode ConvNet on the same data the feature
tutorials use, so that you can compare the two pipelines head to head.

## The “feature first” rule

A reliable practical workflow is to make the feature pipeline mandatory.
Before you commit to a deep architecture for a new dataset:

1. Build a one-page feature pipeline (band power, ratios, simple
   covariance summary). The
   [Extract band-power features](../generated/auto_examples/tutorials/40_features/plot_40_first_features.md)
   recipe is enough.
2. Fit a logistic regression or shallow tree on top.
3. Use exactly the same split (preferably subject-aware; see
   [Leakage and evaluation](leakage_and_evaluation.md)).
4. Record the score and the variance across folds.

Now you have a baseline. Anything that costs ten times more compute
should outperform it on more than just the headline number — it should
beat it under cross-subject evaluation, with smaller variance, and on
held-out cohorts. If it does not, the feature pipeline *is* your
deliverable. You will save weeks of GPU time, and your paper will be
honest.

The
[Compose EEG markers from Welch PSD](../generated/auto_examples/tutorials/40_features/plot_41_feature_trees.md)
and
[EEGDash features to scikit-learn](../generated/auto_examples/tutorials/40_features/plot_42_features_to_sklearn.md)
tutorials extend the feature baseline to richer models (gradient
boosting, full scikit-learn pipelines) without leaving the
feature-engineering regime.

## What the literature says

Three observations recur across the EEG-deep-learning reviews:

- **Architecture matters less than regularisation and split discipline.**
  Roy et al. (2019) survey 156 deep EEG papers and find no consistent
  architecture winner; what changes results is whether the split
  respected subject identity (it often did not).
- **At small N, ConvNets and feature pipelines are within noise.**
  Schirrmeister et al. (2017) <sup>[1](#id3)</sup> explicitly tune their ConvNet to
  match FBCSP on motor imagery; both reach high accuracy and the gap
  is dataset-dependent.
- **Subject-invariant claims require evidence, not only architecture
  language.** A model labelled “subject-invariant” must be evaluated
  cross-subject on a held-out cohort, which loops back to
  [Leakage and evaluation](leakage_and_evaluation.md).

The takeaway is not “always use features” or “always use deep nets”. It
is that the choice is an experiment in itself, and the only way to make
it honestly is to run both pipelines under the same evaluation.

## Related tutorials

- [Extract band-power features](../generated/auto_examples/tutorials/40_features/plot_40_first_features.md)
  — minimal band-power feature pipeline.
- [Compose EEG markers from Welch PSD](../generated/auto_examples/tutorials/40_features/plot_41_feature_trees.md)
  — tree-based models on the same features.
- [EEGDash features to scikit-learn](../generated/auto_examples/tutorials/40_features/plot_42_features_to_sklearn.md)
  — full scikit-learn integration.
- [Train a leakage-safe baseline](../generated/auto_examples/tutorials/10_core_workflow/plot_12_train_a_baseline.md)
  — a deep braindecode baseline on the same data, suitable for a head-
  to-head comparison.

## Further reading

* <a id='id3'>**[1]**</a> Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420. [https://doi.org/10.1002/hbm.23730](https://doi.org/10.1002/hbm.23730)
- Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., Falk, T. H., &
  Faubert, J. (2019). Deep learning-based electroencephalography
  analysis: a systematic review. *Journal of Neural Engineering*, 16(5),
  051001. [https://doi.org/10.1088/1741-2552/ab260c](https://doi.org/10.1088/1741-2552/ab260c)
- Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung,
  C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural
  network for EEG-based brain-computer interfaces. *Journal of Neural
  Engineering*, 15(5), 056013.
  [https://doi.org/10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)
- Cisotto, G., & Chicco, D. (2024). Ten quick tips for clinical
  electroencephalographic (EEG) data acquisition and signal processing.
  *PeerJ Computer Science*, 10, e2256.
  [https://doi.org/10.7717/peerj-cs.2256](https://doi.org/10.7717/peerj-cs.2256)
