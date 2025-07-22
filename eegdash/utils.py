def __init__mongo_client():
    from mne.utils import get_config, set_config

    if get_config("EEGDASH_DB_URI") is None:
        # Set the default MongoDB URI for EEGDash
        # This is a placeholder and should be replaced with your actual MongoDB URI

        set_config(
            "EEGDASH_DB_URI",
            "mongodb+srv://eegdash-user:mdzoMjQcHWTVnKDq@cluster0.vz35p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        )
