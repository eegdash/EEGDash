def __init__mongo_client():
    from mne.utils import set_config

    set_config(
        "EEGDASH_DB_URI",
        "mongodb+srv://eegdash-user:mdzoMjQcHWTVnKDq@cluster0.vz35p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    )
