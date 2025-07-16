config = {
    "required_fields": ["data_name"],
    # Default set of user-facing primary record attributes expected in the database. Records
    # where any of these are missing will be loaded with the respective attribute set to None.
    # Additional fields may be returned if they are present in the database, notably bidsdependencies.
    "attributes": {
        "data_name": "str",
        "dataset": "str",
        "bidspath": "str",
        "subject": "str",
        "task": "str",
        "session": "str",
        "run": "str",
        "sampling_frequency": "float",
        "modality": "str",
        "nchans": "int",
        "ntimes": "int",  # note: this is really the number of seconds in the data, rounded down
    },
    # queryable descriptive fields for a given recording
    "description_fields": ["subject", "session", "run", "task", "age", "gender", "sex"],
    # list of filenames that may be present in the BIDS dataset directory that are used
    # to load and interpret a given BIDS recording.
    "bids_dependencies_files": [
        "dataset_description.json",
        "participants.tsv",
        "events.tsv",
        "events.json",
        "eeg.json",
        "electrodes.tsv",
        "channels.tsv",
        "coordsystem.json",
    ],
    "accepted_query_fields": ["data_name", "dataset"],
}
