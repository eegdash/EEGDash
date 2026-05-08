# Event-Related Decoding

Two lessons covering tasks where labels come from events and BIDS
annotations rather than continuous state. Difficulty 2; assumes the
core workflow track.

Event-locked decoding is where BIDS metadata earns its keep: you select
trials by `trial_type`, align windows to a stimulus or response onset,
and decode contrasts (target vs standard, congruent vs incongruent).
The visual P300 oddball lesson is the canonical first event-related
decoding task; the auditory variant is staged as a contrast that holds
the paradigm fixed and changes only the modality. Sourced from
`docs/tutorial_restructure_plan.md` Category C (lines 410-425), with
BIDS event handling per Pernet et al. (2019).

What you will learn:

- How to filter recordings to include only event-locked trials of a
  given type using BIDS `events.tsv` annotations.
- How to construct event-aligned windows around stimulus onsets, with
  matched baselines.
- How to train a P3 target-versus-standard classifier and report
  performance with the appropriate chance level.
- How to compare two event-related paradigms (visual vs auditory
  oddball) holding the analysis pipeline constant.

Run the lessons in order:

1. `plot_20_visual_p300_oddball.py` – visual P3 target vs standard.
2. `plot_21_auditory_oddball.py` – auditory oddball as a controlled
   contrast.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="A child watches letters flash one by one on a screen. Most letters are standards; one is the block&#x27;s target. The brain answers the rare target with a positive deflection over centro-parietal cortex peaking 300-450 ms after stimulus onset, the classic visual P300 (Polich 2007; Picton 1992). This tutorial loads one BIDS recording from OpenNeuro ds005863 (the visualoddball task, reachable through NEMAR; Delorme et al. 2022), turns the BrainVision event codes into a target-vs-standard label, epochs around stimulus onset with baseline correction (Cisotto &amp; Chicco 2024, Tip 7), and produces three artefacts side by side: an ERP at Pz, a scalp topography of the difference wave at the peak, and a 3-fold leave-one-subject-out accuracy on a logistic-regression decoder. Where does the textbook P300 actually live in the data?">  <div class="sphx-glr-thumbnail-title">How does the brain answer a rare visual target?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="The visual oddball of /auto_examples/tutorials/20_event_related/plot_20_visual_p300_oddball delivered a parietal positive bump near 350 ms. Swap the eyes for ears and the same paradigm structure (rare deviant inside a stream of standards) yields a different brain answer: an early mismatch negativity (MMN, ~150-250 ms) followed by a frontal-central P3a/P3b family (~250-400 ms). The latency is shorter, the topography is shifted, and the subcomponent vocabulary changes (Polich 2007, doi:10.1016/j.clinph.2007.04.019; Naatanen et al. 2007, doi:10.1016/j.clinph.2007.04.026; Squires et al. 1975, doi:10.1016/0013-4694(75)90263-1).">  <div class="sphx-glr-thumbnail-title">How does the auditory P300 differ from the visual P300 of plot_20?</div>
</div>
<!-- thumbnail-parent-div-close --></div>
