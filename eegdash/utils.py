from mne.utils import get_config, set_config, use_log_level


def _init_mongo_client():
    with use_log_level("ERROR"):
        if get_config("EEGDASH_DB_URI") is None:
            set_config(
                "EEGDASH_DB_URI",
                "mongodb+srv://eegdash-user:mdzoMjQcHWTVnKDq@cluster0.vz35p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
                set_env=True,
            )
