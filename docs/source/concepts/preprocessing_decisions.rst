.. _concepts-preprocessing-decisions:

Preprocessing decisions
=======================

Preprocessing is where most EEG decoding results are silently won or lost.
The defaults you accept — high-pass cutoff, line-noise notch, montage,
reference, channel set, artifact policy — encode strong assumptions about
the signal you intend to study. They also interact: a 0.5 Hz high-pass
plus an average reference plus ICA-based blink removal is a different
pipeline from a 1 Hz high-pass plus a mastoid reference plus no ICA, and
the two can disagree about a class boundary.

This page does not prescribe a recipe. EEGDash is preprocessing-agnostic
and forwards your data to MNE-Python (Gramfort et al. 2013) [1]_ and
braindecode for transformation. What this page does is name the choices
that matter, describe what flips when you change them, and point at the
tutorial where each one is exercised. The framing follows Tips 4–7 of
Cisotto & Chicco (2024) [2]_, which give the same advice in a clinical
setting.

A short audit before you start
------------------------------

Before you run a single ``filter()`` call, write down the answers to four
questions. They constrain the rest of the pipeline.

1. **What frequency band carries the signal you care about?** ERPs live
   below 30 Hz; sensorimotor rhythms peak at 8–30 Hz; high-gamma
   coupling lives above 60 Hz. There is no universal default.
2. **What is the dominant noise?** 50/60 Hz mains, slow drift,
   sweat-driven 0.1 Hz baseline, EMG bursts, blink saccades. Each one
   has its own remedy.
3. **What is the unit of analysis?** Single trial, averaged ERP, time-
   frequency map, spectral feature. Different analyses tolerate different
   amounts of damage from filtering.
4. **What is the comparison set?** If you are reporting numbers next to
   another paper, your filter, reference, and montage should be either
   identical or explicitly justified as a deliberate change.

Filtering: high-pass and low-pass
---------------------------------

Filtering is necessary and lossy. High-pass filters remove slow drift but
also distort late, slow ERP components if the cutoff is too high. Low-pass
filters remove muscle and noise but also smear sharp transients.

A few rules of thumb:

- For ERP work, a high-pass between 0.1 Hz and 0.5 Hz with a slow,
  zero-phase FIR is the safest default. Pushing to 1 Hz or above
  introduces measurable distortion in late slow components.
- For oscillation work, a high-pass at 1 Hz is often acceptable and
  sometimes desirable; the exact value should be reported.
- A notch filter at the line frequency (50 Hz in the EU, 60 Hz in
  the US) is almost always called for. Verify that the notch is *not*
  inside your analysis band.
- A low-pass at 40–45 Hz is a sensible default for ERP and slow
  oscillation analyses; a low-pass at 100 Hz is appropriate for
  high-gamma work and is what HBN-style data are typically delivered at.

Tutorial
:doc:`/generated/auto_examples/tutorials/10_core_workflow/plot_10_preprocess_and_window`
walks through this on a real recording, including the choice of filter
order and zero-phase vs. causal filtering.

Resampling
----------

If your downstream model only consumes data up to, say, 40 Hz, there is
no benefit to feeding it 1024 Hz samples. Downsampling to 256 Hz or
even 128 Hz reduces memory and compute by a large factor. Always
low-pass *before* downsampling to avoid aliasing; MNE's ``raw.resample``
does this for you.

The flip side: never upsample EEG to mimic a higher rate. You are not
adding information.

Re-referencing
--------------

EEG is fundamentally relative. The reference electrode is part of the
signal, and changing it can flip the polarity of certain components and
re-distribute power across channels. Common choices:

- **Common average reference (CAR).** Re-references each channel to
  the average of all channels. Sensible when you have a dense montage
  and care about spatial topographies. Sensitive to outlier channels;
  always run after bad-channel detection.
- **Mastoid or linked-mastoid reference.** Common in ERP work because
  a stable, near-zero reference at the mastoid emphasises peri-vertex
  signals. Disqualifies CAR-based pipeline comparisons.
- **REST reference.** A cortical-source-based virtual reference that
  attempts to approximate an infinity reference. Useful when comparing
  across labs that used different physical references.

The choice is part of your pipeline, not a preprocessing afterthought.
Pernet et al. (2019) [3]_ require the reference to be declared in the
BIDS sidecar; respect that.

Montage and channel selection
-----------------------------

A montage tells MNE the 3D position of every electrode. Without one,
topographic plots, source localisation, and certain spatial filters
(CSP, Laplacian) cannot run. EEGDash datasets ship a montage when the
underlying BIDS sidecar provides one; otherwise you must set it
manually with ``raw.set_montage("standard_1020")`` or a vendor-specific
file.

Channel selection is the other side of the same coin. Dropping a flaky
channel is fine; dropping half the array because "it makes the model
faster" is not — your reference, montage, and any spatial filter will
silently change. If you must reduce channels, do it once, document it,
and apply it identically to train and test.

Artifact policy
---------------

Eye blinks, saccades, jaw EMG, and movement spikes are present in every
real EEG recording. Three families of mitigations exist, in roughly
decreasing severity:

1. **ICA decomposition.** Decompose into independent components and
   remove the eye-blink and EMG components. Most expensive, most
   transparent.
2. **Threshold-based rejection.** Drop windows whose amplitude exceeds
   a threshold. Cheap; loses data.
3. **Robust models.** Train models that tolerate residual artifacts
   (e.g., temporal augmentation, dropout). Cheapest, weakest signal.

A well-known pattern is to combine (1) for blinks with (2) for residual
muscle bursts. This is the default in many MNE-based pipelines and is
what the EEGDash tutorial gallery demonstrates. The choice matters
because aggressive rejection can remove the signal you wanted to study
— for instance, motor-imagery training data with high muscle artifact.

Default parameters are not neutral
----------------------------------

Every default in this list is a *position*, not the absence of one. A
half-sentence in the methods that says "standard preprocessing was
applied" is unverifiable; a sentence that says "0.5–40 Hz FIR filter,
common average reference, ICA-based blink removal, 2-second windows
with 50% overlap" is reproducible.

When you write your tutorial, paper, or report, list these decisions
explicitly. When you read someone else's, look for them. If a result is
sensitive to those defaults — and many EEG decoding results are — the
list is the most important paragraph in the paper.

Related tutorials
-----------------

- :doc:`/generated/auto_examples/tutorials/10_core_workflow/plot_10_preprocess_and_window`
  is the runnable companion to this page. It applies a defensible
  default pipeline to an EEGDash recording and explains each step.
- :doc:`/generated/auto_examples/tutorials/30_resting_state/plot_30_eyes_open_closed`
  shows how preprocessing interacts with a class-boundary (eyes open
  vs. closed) where alpha-band content is the main feature.
- :doc:`/generated/auto_examples/tutorials/20_event_related/plot_20_visual_p300_oddball`
  uses ERP-style preprocessing (low high-pass, ICA blink removal).

Further reading
---------------

.. [1] Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier,
   D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., &
   Hämäläinen, M. S. (2013). MEG and EEG data analysis with MNE-Python.
   *Frontiers in Neuroscience*, 7, 267.
   https://doi.org/10.3389/fnins.2013.00267
.. [2] Cisotto, G., & Chicco, D. (2024). Ten quick tips for clinical
   electroencephalographic (EEG) data acquisition and signal processing.
   *PeerJ Computer Science*, 10, e2256.
   https://doi.org/10.7717/peerj-cs.2256
.. [3] Pernet, C. R., et al. (2019). EEG-BIDS, an extension to the brain
   imaging data structure for electroencephalography. *Scientific Data*,
   6(1), 103. https://doi.org/10.1038/s41597-019-0104-8

- Widmann, A., & Schröger, E. (2012). Filter effects and filter
  artifacts in the analysis of electroencephalographic signals.
  *Frontiers in Psychology*, 3, 233.
  https://doi.org/10.3389/fpsyg.2012.00233
- Tanner, D., Morgan-Short, K., & Luck, S. J. (2015). How inappropriate
  high-pass filters can produce artifactual effects and incorrect
  conclusions in ERP studies of language and cognition.
  *Psychophysiology*, 52(8), 997–1009.
  https://doi.org/10.1111/psyp.12437
