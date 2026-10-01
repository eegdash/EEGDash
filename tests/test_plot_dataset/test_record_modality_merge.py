"""The packaged CSV must never blank a recording modality the API provides."""

import pandas as pd
from table_tag_utils import prefer_api_record_modality


def _merged(api_rows, csv_rows):
    api = pd.DataFrame(api_rows)
    csv = pd.DataFrame(csv_rows)
    return api.merge(csv, on="dataset", how="left", suffixes=("", "_csv"))


def test_api_value_kept_when_dataset_missing_from_csv():
    df = _merged(
        {"dataset": ["on000117"], "record_modality": ["meg"]},
        {"dataset": ["ds000117"], "record_modality": ["meg"]},
    )
    out = prefer_api_record_modality(df)
    assert out["record_modality"].tolist() == ["meg"]
    assert out["recording_modality"].tolist() == ["meg"]
    assert "record_modality_csv" not in out.columns


def test_api_wins_over_csv():
    df = _merged(
        {"dataset": ["nm1"], "record_modality": ["eeg, fnirs"]},
        {"dataset": ["nm1"], "record_modality": ["eeg"]},
    )
    assert prefer_api_record_modality(df)["record_modality"].tolist() == ["eeg, fnirs"]


def test_csv_fills_empty_api_cell():
    df = _merged(
        {"dataset": ["nm1", "nm2"], "record_modality": ["", None]},
        {"dataset": ["nm1", "nm2"], "record_modality": ["ieeg", "emg"]},
    )
    assert prefer_api_record_modality(df)["record_modality"].tolist() == ["ieeg", "emg"]


def test_no_csv_column_is_a_noop():
    df = pd.DataFrame({"dataset": ["x"], "record_modality": ["eeg"]})
    assert prefer_api_record_modality(df).equals(df)
