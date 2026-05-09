# Event-Related Decoding

Two lessons covering tasks where labels come from events and BIDS
annotations rather than continuous state. Difficulty 2; assumes the
core workflow track.

Event-locked decoding is where BIDS metadata earns its keep: you select
trials by `trial_type`, align windows to a stimulus or response onset,
and decode contrasts (target vs standard, congruent vs incongruent).
The visual P300 oddball lesson is the canonical first event-related
decoding task; the auditory variant is staged as a contrast that holds
the paradigm fixed and changes only the modality, with BIDS event
handling per Pernet et al. (2019).

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
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Visual P300 oddball decoding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2 | Runtime: 3m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Auditory P300 oddball decoding</div>
</div>
<!-- thumbnail-parent-div-close --></div>
