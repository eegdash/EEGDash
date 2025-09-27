import os
from pathlib import Path

import pandas as pd
import pytest

from eegdash.bids_eeg_metadata import (
    enrich_from_participants,
    merge_participants_fields,
    participants_extras_from_tsv,
    participants_row_for_subject,
)


class DummyRaw:
    def __init__(self):
        self.info = {}


def write_participants(tmpdir: Path, rows: list[dict[str, str]]) -> Path:
    """Helper to write a minimal participants.tsv in tmpdir and return bids_root."""
    bids_root = Path(tmpdir)
    bids_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(bids_root / "participants.tsv", sep="\t", index=False)
    return bids_root


def test_participants_row_lookup_accepts_sub_prefix(tmp_path: Path):
    bids_root = write_participants(
        tmp_path,
        [
            {"participant_id": "sub-01", "sex": "F"},
            {"participant_id": "sub-02", "sex": "M"},
        ],
    )
    # lookup with raw label
    row = participants_row_for_subject(bids_root, "01")
    assert row is not None
    assert row["participant_id"] == "sub-01"

    # lookup with full id
    row = participants_row_for_subject(bids_root, "sub-02")
    assert row is not None
    assert row["participant_id"] == "sub-02"


def test_participants_extras_filters_na_like(tmp_path: Path):
    bids_root = write_participants(
        tmp_path,
        [
            {
                "participant_id": "sub-01",
                "sex": "F",
                "age": "",
                "diagnosis": "n/a",
                "note": " Unknown ",
                "custom": "42",
            }
        ],
    )
    extras = participants_extras_from_tsv(bids_root, subject="01")
    # id columns are removed; NA-like values dropped; others preserved
    assert "participant_id" not in extras
    assert extras == {"sex": "F", "custom": "42"}


def test_merge_preserves_requested_names_and_no_overwrite():
    description = {"sex": "M"}  # already present should not be overwritten
    participants_row = {"Sex": "F", "p-factor": "3.14", "site": "A"}
    merged = merge_participants_fields(
        description, participants_row, description_fields=["P Factor", "sex", "Site"]
    )
    # keeps existing description["sex"]
    assert merged["sex"] == "M"
    # fills requested field name using normalized match
    assert merged["P Factor"] == "3.14"
    assert merged["Site"] == "A"
    # adds remaining participants under normalized key (since not requested)
    assert "p_factor" not in merged  # captured by requested field


def test_enrich_from_participants_attaches_to_raw_and_description(tmp_path: Path):
    bids_root = write_participants(
        tmp_path, [{"participant_id": "sub-01", "group": "control", "age": "20"}]
    )

    # Dummy BIDSPath-like object with .subject attribute
    class BP:
        subject = "01"

    raw = DummyRaw()
    description = {}
    extras = enrich_from_participants(bids_root, BP, raw, description)

    assert extras == {"group": "control", "age": "20"}
    # raw.info enrichment
    assert "subject_info" in raw.info
    assert raw.info["subject_info"]["participants_extras"]["group"] == "control"
    # description enrichment (no overwrites, new keys only)
    assert description["group"] == "control"


def test_enrich_handles_description_as_series(tmp_path: Path):
    bids_root = write_participants(
        tmp_path, [{"participant_id": "sub-01", "handedness": "R"}]
    )

    class BP:
        subject = "01"

    raw = DummyRaw()
    # start with a Series-like description
    description = pd.Series({"dataset": "dummy"})
    enrich_from_participants(bids_root, BP, raw, description)

    assert description.loc["handedness"] == "R"


BASE = Path("participants_files/downloaded_files")


def _find_participants_files(base: Path) -> list[Path]:
    paths: list[Path] = []
    if not base.exists():
        return paths
    for root, _, files in os.walk(base):
        if "participants.tsv" in files:
            paths.append(Path(root) / "participants.tsv")
    return paths


ALL_PARTICIPANTS = _find_participants_files(BASE)


@pytest.mark.skipif(not BASE.exists(), reason="downloaded_files directory not present")
@pytest.mark.parametrize(
    "participants_tsv_path",
    ALL_PARTICIPANTS,
    ids=lambda p: str(p).replace("\n", "\\n"),
)
def test_real_participants_extras_smoke(participants_tsv_path: Path):
    # Read the file and pick a subject id from any supported id column
    df = pd.read_csv(
        participants_tsv_path, sep="\t", dtype="string", keep_default_na=False
    )
    id_cols = [
        c for c in ("participant_id", "participant", "subject") if c in df.columns
    ]
    # Choose a subject from the first available id column; fallback to a dummy
    if not df.empty and id_cols:
        subject_val = str(df.iloc[0][id_cols[0]])
    else:
        subject_val = "sub-01"
    bids_root = participants_tsv_path.parent

    extras = participants_extras_from_tsv(bids_root, subject=subject_val)

    # Always a dict; may be empty if there are no extra columns
    assert isinstance(extras, dict)

    # Must not contain identifier columns and not include NA-like values
    for bad_key in ("participant_id", "participant", "subject"):
        assert bad_key not in extras
    for v in extras.values():
        assert isinstance(v, str)
        assert v.strip() != ""
        assert v.lower() not in {"n/a", "na", "nan", "unknown", "none"}
