from eegdash.records import adapt_record_v1_to_v2


def test_adapt_openneuro_v1_to_v2_preserves_alphanumeric_run():
    rec_v1 = {
        "dataset": "ds000001",
        "data_name": "ds000001_sub-01_task-test_run-5F_eeg.vhdr",
        "bidspath": "ds000001/sub-01/eeg/sub-01_task-test_run-5F_eeg.vhdr",
        "bidsdependencies": [
            "ds000001/participants.tsv",
            "ds000001/sub-01/eeg/sub-01_task-test_run-5F_events.tsv",
        ],
        "subject": "01",
        "task": "test",
        "session": None,
        "run": "5F",
        "modality": "eeg",
    }

    out1 = adapt_record_v1_to_v2(rec_v1)
    out2 = adapt_record_v1_to_v2(rec_v1)

    assert out1["schema_version"] == 2
    assert out1["variant"] == "openneuro_raw"
    assert out1["record_id"] == out2["record_id"]

    assert out1["bids_relpath"] == "sub-01/eeg/sub-01_task-test_run-5F_eeg.vhdr"

    assert out1["entities"]["run"] == "5F"
    assert out1["entities_mne"]["run"] is None

    assert out1["storage"]["base"] == "s3://openneuro.org/ds000001"
    assert out1["storage"]["raw_key"] == out1["bids_relpath"]
    assert out1["storage"]["dep_keys"] == [
        "participants.tsv",
        "sub-01/eeg/sub-01_task-test_run-5F_events.tsv",
    ]

    assert out1["cache"]["dataset_subdir"] == "ds000001"
    assert out1["cache"]["raw_relpath"] == out1["bids_relpath"]
    assert out1["cache"]["dep_relpaths"] == out1["storage"]["dep_keys"]


def test_adapt_challenge_v1_to_v2_converts_set_to_bdf_and_strips_dataset_prefix():
    rec_v1 = {
        "dataset": "ds005509",
        "data_name": "ds005509_sub-NDARAH793FBF_task-DespicableMe_eeg.set",
        "bidspath": "ds005509/sub-NDARAH793FBF/eeg/sub-NDARAH793FBF_task-DespicableMe_eeg.set",
        "bidsdependencies": [
            "ds005509/dataset_description.json",
            "ds005509/sub-NDARAH793FBF/eeg/sub-NDARAH793FBF_task-DespicableMe_events.tsv",
            # Defensive: if a legacy record includes a .set dep, adapt it to .bdf
            "ds005509/sub-NDARAH793FBF/eeg/sub-NDARAH793FBF_task-DespicableMe_eeg.set",
        ],
        "subject": "NDARAH793FBF",
        "task": "DespicableMe",
        "session": None,
        "run": None,
        "modality": "eeg",
    }

    s3_bucket = "s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf"

    out1 = adapt_record_v1_to_v2(
        rec_v1,
        s3_bucket=s3_bucket,
        variant="challenge_l100_bdf_mini",
    )
    out2 = adapt_record_v1_to_v2(
        rec_v1,
        s3_bucket=s3_bucket,
        variant="challenge_l100_bdf_mini",
    )

    assert out1["schema_version"] == 2
    assert out1["variant"] == "challenge_l100_bdf_mini"
    assert out1["record_id"] == out2["record_id"]

    assert out1["bids_relpath"].endswith("_eeg.bdf")
    assert out1["bids_relpath"].startswith("sub-NDARAH793FBF/eeg/")

    assert out1["storage"]["base"] == s3_bucket
    assert out1["storage"]["raw_key"] == out1["bids_relpath"]

    assert out1["storage"]["dep_keys"] == [
        "dataset_description.json",
        "sub-NDARAH793FBF/eeg/sub-NDARAH793FBF_task-DespicableMe_events.tsv",
        "sub-NDARAH793FBF/eeg/sub-NDARAH793FBF_task-DespicableMe_eeg.bdf",
    ]

    assert out1["cache"]["dataset_subdir"] == "ds005509-bdf-mini"


def test_infer_variant_from_s3_bucket_for_legacy_records():
    rec_v1 = {
        "dataset": "ds005509",
        "bidspath": "ds005509/sub-NDARAH793FBF/eeg/sub-NDARAH793FBF_task-DespicableMe_eeg.set",
        "bidsdependencies": [],
        "subject": "NDARAH793FBF",
        "task": "DespicableMe",
        "run": None,
        "modality": "eeg",
    }

    out = adapt_record_v1_to_v2(
        rec_v1,
        s3_bucket="s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf",
        variant=None,
    )
    assert out["variant"] == "challenge_l100_bdf_mini"

