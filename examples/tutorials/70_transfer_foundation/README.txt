Transfer, Foundation Models, and EEG2025
========================================

Five lessons using recorded EEG, observed targets, and explicit evaluation
boundaries. Challenge examples use small R5 mini participant subsets; their
scores describe these instructional splits rather than the full leaderboard.

1. ``plot_70_challenge_dataset_basics.py`` -- inspect challenge metadata and
   distinguish participant counts from recording counts.
2. ``plot_71_cross_task_transfer.py`` -- pretrain on observed resting eye-state
   cues and adapt to reaction-time regression, excluding the test participant
   from both training stages.
3. ``plot_72_subject_invariant_regression.py`` -- evaluate observed p-factor
   with exactly one held-out prediction per participant.
4. ``plot_73_finetune_pretrained_model.py`` -- adapt the published CBraMod
   checkpoint; six subject-grouped folds score each of eighteen participants
   once and compare scratch, linear probe and fine-tuning.
5. ``plot_74_neuroai_interop.py`` -- extract actual voltage windows through
   NeuralSet Segmenter and EegExtractor and batch them with PyTorch DataLoader.
