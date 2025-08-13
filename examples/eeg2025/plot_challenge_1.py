""".. _tutorial-challenge-1:

Challenge 1: Transfer Learning task
=====================================

# # Tutorial for Contrast-Change Detection (CCD) Task - EEG 2025 Competition
# 
# This tutorial demonstrates how to load EEG data for the contrast-change detection (CCD) task from the EEG 2025 competition, extract epochs, and calculate response times and correctness information. We'll use `EEGDash` and `braindecode`.
# This tutorial does NOT address the use of `SuS` task for challenge 1.
# 
# ## Key Features:
# - Load data for subject `NDARAG340ERT` from dataset `ds005507` (This data is available in the R3 minsets as well)
# - Extract stimulus events (`left_target`, `right_target`) and calculate response times and correctness from button presses and feedback
# - Epoch the data based on contrast trial start events
"""
# %%
import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from eegdash import EEGDashDataset
from braindecode.preprocessing import create_windows_from_events
import warnings
from IPython.display import display

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# %% [markdown]
# ## 1. Loading the Data
# 
# We'll load the data for subject `NDARAG340ERT` from the `ds005507` dataset. `EEGDashDataset` will handle the download and preprocessing automatically.
# 

# %%
# Load the dataset
cache_dir = "~/.eegdash_cache"  # keep the cache in the home directory
dataset_name = "ds005507"
dataset = EEGDashDataset({
    "dataset": dataset_name, 
    "subject": "NDARAG340ERT",
    "task": "contrastChangeDetection", "run": 1,
})

# Get the raw EEG data
raw = dataset.datasets[0].raw
print(f"Dataset loaded successfully!")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} seconds")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Channel names: {raw.ch_names[:10]}...")  # Show first 10 channels


# %% [markdown]
# ## 2. Reading BIDS Events File with Additional Columns
# 
# The power of BIDS-formatted datasets is that they include rich metadata in standardized formats. The events.tsv file contains additional columns like `feedback` that aren't available through MNE's annotation system. Let's read the BIDS events file directly using pandas to access ALL the columns: 
# 

# %%
# The key insight: We can read the BIDS events.tsv file directly using pandas!
# This gives us access to ALL columns including the crucial 'feedback' column

# Get the events file path from the EEGDashDataset
bids_args = dataset.datasets[0].get_raw_bids_args()
events_file = os.path.join(cache_dir, dataset_name, f"sub-{bids_args['subject']}/eeg/sub-{bids_args['subject']}_task-{bids_args['task']}_run-{bids_args['run']}_events.tsv")

# Read the events.tsv file using pandas
events_df = pd.read_csv(events_file, sep='\t')

print("BIDS Events File Structure:")
print(f"Shape: {events_df.shape}")
print(f"Columns: {list(events_df.columns)}")
print(f"\nFirst 10 rows:")
display(events_df.head(10))

print(f"\nFeedback column unique values:")
print(events_df['feedback'].value_counts())


# %% [markdown]
# ## 3. Calculate Response Times and Correctness from BIDS Events
# 
# Now we'll calculate response times and correctness by matching stimulus events with their corresponding button presses and feedback. This approach uses the temporal sequence of events in the BIDS file.
# 

# %%
def calculate_behavioral_metrics_from_bids(events_df):
    """
        Calculate response times and correctness from BIDS events DataFrame.
        
        This function matches stimulus events with subsequent button presses and feedback.
    """
    # Get stimulus events
    stimuli = events_df[events_df['value'].isin(['left_target', 'right_target'])].copy()
    
    # Get button press events
    responses = events_df[events_df['value'].isin(['left_buttonPress', 'right_buttonPress'])]

    # Get contrast trial start events
    contrast_trials = events_df[events_df['value'] == 'contrastTrial_start']
    
    # Initialize columns
    stimuli['response_time'] = np.nan
    stimuli['correct'] = None
    stimuli['response_type'] = None
    stimuli['contrast_trial_start'] = None
    
    for idx, stimulus in stimuli.iterrows():
        # Find the next button press after this stimulus, but make sure it is before next 'contrastTrial_start'
        next_contrast_start = contrast_trials[contrast_trials['onset'] > stimulus['onset']].iloc[0]['onset']
        future_responses = responses[
            (responses['onset'] > stimulus['onset']) & 
            (responses['onset'] < next_contrast_start)
        ]
        stimuli.loc[idx, 'contrast_trial_start'] = contrast_trials[contrast_trials['onset'] < stimulus['onset']].iloc[-1]['onset']
        if len(future_responses) > 0:
            # Get the first (closest) response
            next_response = future_responses.iloc[0]
            # Calculate response time
            response_time = next_response['onset'] - stimulus['onset']
            stimuli.loc[idx, 'response_time'] = response_time
            stimuli.loc[idx, 'response_type'] = next_response['value']
            # We can use the feedback column directly!
            # Find feedback that corresponds to the button press
            if len(next_response['feedback']) > 0:
                feedback = next_response['feedback']
                # Map feedback to correctness
                if feedback == 'smiley_face':
                    stimuli.loc[idx, 'correct'] = True
                elif feedback == 'sad_face':
                    stimuli.loc[idx, 'correct'] = False
        # Note: 'non_target' feedback might indicate a different type of trial
    return stimuli


# Calculate behavioral metrics
stimulus_metadata = calculate_behavioral_metrics_from_bids(events_df)
print(f"Behavioral Analysis Results:")
print(f"Total stimulus events: {len(stimulus_metadata)}")
print(f"Events with responses: {stimulus_metadata['response_time'].notna().sum()}")
print(f"Correct responses: {stimulus_metadata['correct'].sum()}")
print(f"Incorrect responses: {stimulus_metadata['response_time'].notna().sum()-stimulus_metadata['correct'].sum()}")
print(f"Response time statistics:")
print(stimulus_metadata['response_time'].describe())
print(f"First few trials with calculated metrics:")
display(stimulus_metadata[['onset', 'value', 'response_time', 'correct', 'response_type', 'contrast_trial_start']].head(8))


# %% [markdown]
# ## 4. Creating Epochs with Braindecode and BIDS Metadata
# 
# Now we'll create epochs using `braindecode`'s `create_windows_from_events`. According to the EEG 2025 challenge requirements, epochs should start from **contrast trial starts** and be **2 seconds long**. This epoching approach ensures we capture:
# 
# - The entire trial from contrast trial start (t=0)
# - The stimulus presentation (usually ~2.8 seconds after trial start)
# - The response window (usually within 2 seconds of stimulus)
# - Full behavioral context for each trial
# 
# We'll use our enhanced metadata that includes the behavioral information extracted from the BIDS events file.
# 

# %%
# Create epochs from contrast trial starts with 2-second duration as per EEG 2025 challenge
# IMPORTANT: Only epoch trials that have valid behavioral data (stimulus + response)

# First, get all contrast trial start events from the BIDS events
all_contrast_trials = events_df[events_df['value'] == 'contrastTrial_start'].copy()
print(f"Found {len(all_contrast_trials)} total contrast trial start events")

# Filter to only include contrast trials that have valid behavioral data
# Get the contrast trial start times that correspond to trials with valid stimulus/response data
valid_contrast_times = stimulus_metadata['contrast_trial_start'].dropna().unique()
print(f"Found {len(valid_contrast_times)} contrast trials with valid behavioral data")

# Filter contrast trial events to only those with valid behavioral data
valid_contrast_trials = all_contrast_trials[
    all_contrast_trials['onset'].isin(valid_contrast_times)
].copy()

print(f"Epoching {len(valid_contrast_trials)} contrast trials (only those with behavioral data)")
print(f"Excluded {len(all_contrast_trials) - len(valid_contrast_trials)} trials without behavioral data")

# Convert valid contrast trial start onset times to samples for MNE
valid_contrast_trials['sample_mne'] = (valid_contrast_trials['onset'] * raw.info['sfreq']).astype(int)

# Create new events array with valid contrast trial starts only
# Format: [sample, previous_sample, event_id]
new_events = np.column_stack([
    valid_contrast_trials['sample_mne'].values,
    np.zeros(len(valid_contrast_trials), dtype=int),
    np.full(len(valid_contrast_trials), 99, dtype=int)  # Use event_id 99 for contrast_trial_start
])

# Create new annotations from these events to replace the original annotations
# This is the key step - we need to replace the annotations in the raw object
annot_from_events = mne.annotations_from_events(
    events=new_events,
    event_desc={99: "contrast_trial_start"},
    sfreq=raw.info['sfreq'],
    orig_time=raw.info['meas_date']
)

# Replace the annotations in the raw object
print(f"Original annotations: {len(raw.annotations)} events")
raw.set_annotations(annot_from_events)
print(f"New annotations: {len(raw.annotations)} contrast trial start events (valid trials only)")

# Verify the new annotations
events_check, event_id_check = mne.events_from_annotations(raw)
print(f"Events from new annotations: {len(events_check)} events")
print(f"Event ID mapping: {event_id_check}")

# Now use braindecode's create_windows_from_events to create 2-second epochs
# Calculate the window size in samples (2 seconds * sampling rate)
window_size_samples = int(2.0 * raw.info['sfreq'])  # 2 seconds in samples
print(f"Window size: {window_size_samples} samples ({window_size_samples / raw.info['sfreq']:.1f} seconds)")

# Create 2-second epochs from valid contrast trial starts only
windows_dataset = create_windows_from_events(
    dataset,  # The EEGDashDataset
    trial_start_offset_samples=0,  # Start from the contrast trial start (no offset)
    trial_stop_offset_samples=window_size_samples,  # End 2 seconds later
    preload=True
)

print(f"Created {len(windows_dataset)} epochs with behavioral data")
print(f"All epochs should now have valid stimulus and response information")


# %% [markdown]
# ## Conclusion
# - The epoched data is now ready under `windows_dataset`.
# - The response time is under `stimulus_metadata['response_time']`. (required for challenge 1 regression task)
# - The correctness is under `stimulus_metadata['correct']`. (required for challenge 1 classification task)
# - The stimulus type (left or right) is under `stimulus_metadata['value']`. (might be useful)


