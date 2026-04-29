# Notes for the toturial files

- In the tutorial files, places with notes are marked with [?]_#, where # is the nore number.

## Tutorial 1: Basic pipeline

File at: examples/new_tutorials/tutorial_1_basic_pipeline.ipynb

**R1 corrections applied: 2026-04-29**

### General Thoughts

- ~~The `RestingState` trials contain eyes open/close markers compatible with MNE, with regular cycle - 20sec open and 40sec close. Should we aknowledge it? if so - maybe exclude alpha bands?~~ → **Done:** added eyes-open/closed note to data loading section.
- Should we increase the dataset's size? curretly it's ~280 subjects. → **Open:** team decision.
- ICA pipline - consult Tom and Oren → **Open:** pending.

- ~~Features - we need more feature from the spectral and connectivity families.~~ → **Done:** added `spectral_entropy` and `spectral_slope` to extractor.
- Features - we need a more complicated features tree. optimally we'll use another sub-type of spetral features and the `pick_channels` option. → **Open:** needs team input on exact intent.

- Should we do hyper-parameter search? if so, we need to add validation sets. → **Open:** pending.

### Notes

1. ~~the headline say part of the tutorial is too learn "Why subject-level train/test splitting is essential to avoid data leakage". This could be unrelated to the topic since our main goal is the teach ml and dl engeineers how to use brain-data. NWEVERMIND! this is actualy important here.~~ → **Resolved:** `[?]_1` marker removed; text kept as-is.
2. ~~in the 'data loading' panel, we use the line `ds = BaseConcatDataset(ds1.datasets + ds2.datasets)`, need to make sure it is correct.~~ → **Resolved:** verified correct (standard braindecode pattern); `[?]_2` marker removed.
3. ~~The windows-overlap is currently 50%. We need to make sure it's the right approach here.~~ → **Resolved:** 50% overlap is appropriate for feature-based ML; added explanation to windowing markdown.

## Tutorial 2: custom features

**R1 corrections applied: 2026-04-29**

### General Thoughts

- ~~The opening text needs fixing.~~ → **Done:** rewrote intro to list explicit learning goals and fix "seven features" → "six features + one preprocessor".
- We should have tutorials from different types of variation. → **Open:** currently only univariate features; consider adding a bivariate example in a future pass.
- ~~The recreated functions's logic/script should be different than the original.~~ → **Done:** `my_zero_crossings` changed to product-sign approach; `my_line_length` rewritten as explicit two-step form, both now demonstrably distinct from the bank.
- ~~The tutorial should emphasize the specific details (work on last axis, decorators order...)~~ → **Done:** section 1 now has an axis=-1 shape diagram and a mandatory-order callout; section 2 intro explicitly states "reduce `axis=-1`" rule.

### Notes
4. ~~Decorator System - should verify with Aviv and correct the text.~~ → **Partially done:** rewrote section 1 as a table with clearer explanations; decorator API verified against WORKPLAN/PACKAGE_OVERVIEW. Final sign-off with Aviv still recommended.
5. rephrase the sentence. → **Blocked:** `[?]_5` marker is absent from the notebook — the target sentence was never marked. Please add the `[?]_5` tag to the sentence you want rephrased and re-flag this note.

## Tutorial 3:

**R1 corrections applied: 2026-04-29**

### Code
- show how to use metadata (`bin_avalanches`): receive & change → **Open:** `bin_avalanches` already shows the pattern in code; may need a dedicated callout cell or better comment. Team to decide.
- explain why we split the standardization into two functions and the use of `StandardizedSignalType` → **Blocked:** `StandardizedSignalType` not present in current code; design source unclear.
- explain why we didn't need `preprocessor_output_type` for other preprocessors (generity) → **Blocked:** all preprocessors currently use the decorator; which ones should omit it is unclear.
- Replace `_fit_truncated_power_law` with a simpler function/logic. → **Open:** function currently named `_fit_power_law`; no simpler replacement identified without team input.
- `starts` and `ends` are switched from list to nparray multiple times - should decide on one representation. → **Open:** team decision needed.
- ~~for `tau_exponent`: review the "lab" t-max method. probably remove this.~~ → **Done:** removed the `"lab"` branch; `tau_exponent` now uses only `durs.max()` as `t_max`.
- in `gamma_exponent`: consider changing to base e. and change to linear fit instead of poly. → **Open:** marked as "consider"; deferred.

### Tutorial notes
1. change to intro for part 0. (Gal will rewrite). → **Open:** Gal will rewrite; no change made.
2. change to "original" code, or remove it completely. → **Open:** "change to original" vs "remove" — decision needed from team.
3. in part 2 - we dont need to "imagine" a ds - we need some real data (meg?) → **Open:** needs real dataset; TBD.
4. ~~remove the "caching issue" - the package is not suitable for this.~~ → **Done:** removed "No caching" row from The Scaling Problem table; updated "solves all four" → "solves all three".
5. note: the functions/features names changed. → **Open:** team to confirm which names changed and what the current correct names are.
6. ~~shape fix: i want a real example from the avalanche code. also, we need the emphasize the use of -1,-2 indexing for ch, time.~~ → **Done:** section 4 now shows actual 2-D lab code snippet and 3-D adaptation with `axis=-1`/`axis=-2`; rule-of-thumb explanation added.
7. ~~for part 5: remove all references to the "old api".~~ → **Done:** OLD API code block removed from section 5.
8. ~~for part 5: does position really matter? emphasize that we can USE metadata without `metadata_preprocessor`, but not CHANGE it.~~ → **Done:** decorator table now explicitly states "any function can read `_metadata` without this decorator — only *changing* it requires it."
9. ~~feature names in the dict will be neglected for `""` and non-string keys. change example to ints.~~ → **Done:** `""` keys changed to `0` in extractor and diagram; explanation added.
10. ~~talk about the use of `partial`.~~ → **Done:** added a `# Why partial()?` explanation block in the extractor cell.
11. change featureExtractor order of preprocessors and dict. → **Blocked:** intent unclear (positional vs. nesting order vs. keyword argument order). Team to clarify.
12. ~~for part 7: change the explanation - we cannot use 2 NODES as predecessors for one feature.~~ → **Done:** DCC section now explains "each node has exactly one predecessor; a feature node cannot receive another feature node's output."
13. ~~to use DDC - change to DataFrame instead of looping.~~ → **Done:** DCC computation replaced with `pd.concat` + vectorised column operation.