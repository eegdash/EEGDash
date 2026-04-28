# Notes for the toturial files

- In the tutorial files, places with notes are marked with [?]_#, where # is the nore number.

## Tutorial 1: Basic pipeline

File at: examples/new_tutorials/tutorial_1_basic_pipeline.ipynb

### General Thoughts

- The `RestingState` trials contain eyes open/close markers compatible with MNE, with regular cycle - 20sec open and 40sec close. Should we aknowledge it? if so - maybe exclude alpha bands?
- Should we increase the dataset's size? curretly it's ~280 subjects.
- ICA pipline - consult Tom and Oren

- Features - we need more feature from the spectral and connectivity families.
- Features - we need a more complicated features tree. optimally we'll use another sub-type of spetral features and the `pick_channels` option.

- Should we do hyper-parameter search? if so, we need to add validation sets.

### Notes

1. the headline say part of the tutorial is too learn "Why subject-level train/test splitting is essential to avoid data leakage". This could be unrelated to the topic since our main goal is the teach ml and dl engeineers how to use brain-data. NWEVERMIND! this is actualy important here.
2. in the 'data loading' panel, we use the line `ds = BaseConcatDataset(ds1.datasets + ds2.datasets)`, need to make sure it is correct.
3. The windows-overlap is currently 50%. We need to make sure it's the right approach here.

## Tutorial 2: custom features

### General Thoughts

- The opening text is needs fixing.
- We should have tutorials from different types of variation.
- The recreated funtions's logic/script should be different than the original.
- The tutorial should emphasize the specific details (work on last axis, decorators order...)

### Notes
4. Decorator System - should verify with Aviv and correct the text.
5. rephrase the sentence.

## Tutorial 3:

### General Thoughts
- show how to use metedata (`bin_avalanches`): recive & change
- explain why we splitted the standartization itno two funcions and the use of `StandardizedSignalType`