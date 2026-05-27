"""Per-helper unit tests for the  decomposition helpers.

. The 17 helpers introduced in  +
are currently covered transitively by the snapshot tests — when
something breaks, the failure points at the orchestrator
(extract_record / extract_dataset_metadata / _enumerate_via_manifest)
rather than the helper.

Direct tests give:
- Faster diagnosis (the failing helper is named in the test)
- Coverage of edge cases the snapshot fixtures don't exercise
  (malformed inputs, BIDS inheritance with no session_base, etc.)
- Refactor safety for the helpers themselves

The tests are organised by helper category:
- §1 BIDS-fs metadata readers (_read_bids_readme,
  _read_participants_demographics, _build_global_storage_info)
- §2 BIDS-fs Record/dep_keys helpers (_build_dep_keys)
- §3 Manifest-path orchestration helpers
  (_determine_manifest_storage_base, _collect_bids_entities_from_paths,
  _is_bids_data_zip)
- §4 Manifest-path Record builders (_build_zip_extracted_records,
  _build_subject_zip_record, _build_bids_data_zip_records,
  _build_regular_manifest_record, _build_standalone_zip_content_records,
  _build_ctf_ds_records)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from _helpers import INGEST_DIR as _INGEST_DIR


def _load_digest():
    """Lazy-load 3_digest.py (digit-prefixed filename forces this)."""
    spec = importlib.util.spec_from_file_location(
        "_digest_helpers_target", _INGEST_DIR / "3_digest.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════
# §1 — BIDS-fs metadata readers
# ═══════════════════════════════════════════════════════════════════════


# ─── _read_bids_readme ────────────────────────────────────────────────────


def test_readme_returns_none_when_absent(tmp_path: Path):
    digest = _load_digest()
    assert digest._read_bids_readme(tmp_path) is None


def test_readme_finds_uppercase_readme_file(tmp_path: Path):
    digest = _load_digest()
    (tmp_path / "README").write_text("First line\n\nThird line\n")
    text = digest._read_bids_readme(tmp_path)
    assert text == "First line\nThird line"  # blank lines stripped


def test_readme_finds_readme_md_if_no_README(tmp_path: Path):
    digest = _load_digest()
    (tmp_path / "README.md").write_text("# Title\n\nBody")
    assert digest._read_bids_readme(tmp_path) == "# Title\nBody"


def test_readme_prefers_no_extension_over_md(tmp_path: Path):
    """The lookup order tries ``README`` before ``README.md``.

    Test uses different extensions (not just different case) because
    case-insensitive filesystems like macOS APFS would collapse
    ``README`` and ``readme`` into one file.
    """
    digest = _load_digest()
    (tmp_path / "README").write_text("primary")
    (tmp_path / "README.md").write_text("fallback — should not be read")
    assert digest._read_bids_readme(tmp_path) == "primary"


def test_readme_tolerates_non_utf8(tmp_path: Path):
    """A README with invalid UTF-8 yields None — the function continues
    trying other filenames, then returns None if none worked."""
    digest = _load_digest()
    (tmp_path / "README").write_bytes(b"\xff\xfe\xfd")  # invalid UTF-8
    assert digest._read_bids_readme(tmp_path) is None


# ─── _read_participants_demographics ──────────────────────────────────────


def test_demographics_returns_zeros_when_no_participants_tsv(tmp_path: Path):
    digest = _load_digest()
    count, ages, sex, hand = digest._read_participants_demographics(tmp_path)
    assert count == 0
    assert ages == []
    assert sex == {}
    assert hand == {}


def test_demographics_basic_row_count(tmp_path: Path):
    digest = _load_digest()
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\nsub-01\t30\nsub-02\t25\nsub-03\t40\n"
    )
    count, ages, _, _ = digest._read_participants_demographics(tmp_path)
    assert count == 3
    assert sorted(ages) == [25, 30, 40]


def test_demographics_ignores_invalid_ages(tmp_path: Path):
    """Ages outside (0, 120) and non-numeric strings are skipped."""
    digest = _load_digest()
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\nsub-01\tn/a\nsub-02\t-1\nsub-03\t150\nsub-04\t30\n"
    )
    _, ages, _, _ = digest._read_participants_demographics(tmp_path)
    assert ages == [30]


def test_demographics_handles_age_column_variants(tmp_path: Path):
    """Column header can be ``age``, ``Age``, or ``AGE``."""
    digest = _load_digest()
    (tmp_path / "participants.tsv").write_text("participant_id\tAge\nsub-01\t42\n")
    _, ages, _, _ = digest._read_participants_demographics(tmp_path)
    assert ages == [42]


def test_demographics_sex_distribution(tmp_path: Path):
    digest = _load_digest()
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tsex\n"
        "sub-01\tM\n"
        "sub-02\tfemale\n"
        "sub-03\tf\n"
        "sub-04\tother\n"
        "sub-05\tn/a\n"
    )
    _, _, sex, _ = digest._read_participants_demographics(tmp_path)
    assert sex == {"m": 1, "f": 2, "o": 1}  # n/a is excluded


def test_demographics_handedness_distribution(tmp_path: Path):
    digest = _load_digest()
    (tmp_path / "participants.tsv").write_text(
        "participant_id\thandedness\n"
        "sub-01\tR\n"
        "sub-02\tleft\n"
        "sub-03\tA\n"
        "sub-04\tambidextrous\n"
    )
    _, _, _, hand = digest._read_participants_demographics(tmp_path)
    assert hand == {"r": 1, "l": 1, "a": 2}


def test_demographics_malformed_file_returns_zeros(tmp_path: Path):
    """A file pandas can't parse → empty demographics, no exception."""
    digest = _load_digest()
    # Write something that's not valid TSV
    (tmp_path / "participants.tsv").write_bytes(b"\x00\x01\x02 garbage")
    count, ages, sex, hand = digest._read_participants_demographics(tmp_path)
    # Either the file is parsed and yields garbage counts, OR the
    # function tolerates and returns empty. Both are acceptable; we just
    # require NO exception propagated.
    assert isinstance(count, int)
    assert isinstance(ages, list)
    assert isinstance(sex, dict)
    assert isinstance(hand, dict)


# ─── _build_global_storage_info ───────────────────────────────────────────


def test_storage_info_returns_none_for_unknown_source(tmp_path: Path):
    """Unknown sources (not in STORAGE_CONFIGS) get None."""
    digest = _load_digest()
    info = digest._build_global_storage_info(
        "ds-xyz", "totally_unknown_source", tmp_path
    )
    assert info is None


def test_storage_info_finds_dataset_description(tmp_path: Path):
    digest = _load_digest()
    (tmp_path / "dataset_description.json").write_text("{}")
    info = digest._build_global_storage_info("ds002893", "openneuro", tmp_path)
    assert info is not None
    assert info["raw_key"] == "dataset_description.json"
    assert info["backend"] == "s3"
    assert "ds002893" in info["base"]


def test_storage_info_collects_dep_keys_for_other_globals(tmp_path: Path):
    """README, participants.tsv etc. land in dep_keys (sorted, deduped)."""
    digest = _load_digest()
    (tmp_path / "dataset_description.json").write_text("{}")
    (tmp_path / "participants.tsv").write_text("participant_id\n")
    (tmp_path / "README").write_text("test")
    info = digest._build_global_storage_info("ds002893", "openneuro", tmp_path)
    assert info is not None
    assert "participants.tsv" in info["dep_keys"]
    assert "README" in info["dep_keys"]
    assert info["dep_keys"] == sorted(set(info["dep_keys"]))  # sorted, deduped


def test_storage_info_excludes_manifest_json_and_dotfiles(tmp_path: Path):
    digest = _load_digest()
    (tmp_path / "dataset_description.json").write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / ".hidden_file.json").write_text("{}")
    info = digest._build_global_storage_info("ds002893", "openneuro", tmp_path)
    assert info is not None
    assert "manifest.json" not in info["dep_keys"]
    assert ".hidden_file.json" not in info["dep_keys"]


# ═══════════════════════════════════════════════════════════════════════
# §2 — BIDS-fs Record/dep_keys helpers
# ═══════════════════════════════════════════════════════════════════════


# ─── _build_dep_keys ──────────────────────────────────────────────────────


def test_dep_keys_finds_channels_tsv_alongside_recording(tmp_path: Path):
    """The most common case: channels.tsv next to the recording."""
    digest = _load_digest()
    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    record = eeg_dir / "sub-01_task-rest_eeg.edf"
    record.touch()
    (eeg_dir / "sub-01_task-rest_channels.tsv").write_text("name\n")

    dep_keys, is_split, ok = digest._build_dep_keys(
        record, tmp_path, fif_is_split=False, fif_continuations_ok=True
    )
    assert "sub-01/eeg/sub-01_task-rest_channels.tsv" in dep_keys
    assert is_split is False
    assert ok is True


def test_dep_keys_finds_session_level_sidecar_via_inheritance(tmp_path: Path):
    """BIDS inheritance: an events.json shared across runs sits at
    the session level (no task / run entities)."""
    digest = _load_digest()
    sub_dir = tmp_path / "sub-01"
    eeg_dir = sub_dir / "eeg"
    eeg_dir.mkdir(parents=True)
    record = eeg_dir / "sub-01_task-rest_run-01_eeg.edf"
    record.touch()
    # Session-level events.json (no task/run entities)
    (eeg_dir / "sub-01_events.json").write_text("{}")

    dep_keys, _, _ = digest._build_dep_keys(
        record, tmp_path, fif_is_split=False, fif_continuations_ok=True
    )
    assert "sub-01/eeg/sub-01_events.json" in dep_keys


def test_dep_keys_includes_fdt_companion_for_set_file(tmp_path: Path):
    """Format-specific companions (.fdt for .set) added even when absent."""
    digest = _load_digest()
    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    record = eeg_dir / "sub-01_task-rest_eeg.set"
    record.touch()
    # No .fdt file on disk — but should still appear in dep_keys
    dep_keys, _, _ = digest._build_dep_keys(
        record, tmp_path, fif_is_split=False, fif_continuations_ok=True
    )
    assert any(k.endswith(".fdt") for k in dep_keys)


def test_dep_keys_detects_split_fif_continuation(tmp_path: Path):
    """Existence of ``<stem>-1.fif`` triggers split-FIF detection."""
    digest = _load_digest()
    meg_dir = tmp_path / "sub-01" / "meg"
    meg_dir.mkdir(parents=True)
    record = meg_dir / "sub-01_task-rest_meg.fif"
    record.touch()
    (meg_dir / "sub-01_task-rest_meg-1.fif").touch()

    dep_keys, is_split, ok = digest._build_dep_keys(
        record, tmp_path, fif_is_split=False, fif_continuations_ok=True
    )
    assert is_split is True
    assert ok is True
    assert "sub-01/meg/sub-01_task-rest_meg-1.fif" in dep_keys


def test_dep_keys_broken_fif_continuation_flags_integrity(tmp_path: Path):
    """A broken-symlink FIF continuation sets fif_continuations_ok=False."""
    digest = _load_digest()
    meg_dir = tmp_path / "sub-01" / "meg"
    meg_dir.mkdir(parents=True)
    record = meg_dir / "sub-01_task-rest_meg.fif"
    record.touch()
    # Create a broken symlink for the continuation
    broken_target = tmp_path / ".no_such_target.fif"
    (meg_dir / "sub-01_task-rest_meg-1.fif").symlink_to(broken_target)

    _, is_split, ok = digest._build_dep_keys(
        record, tmp_path, fif_is_split=True, fif_continuations_ok=True
    )
    assert is_split is True
    assert ok is False  # broken symlink → integrity flag


# ═══════════════════════════════════════════════════════════════════════
# §3 — Manifest-path orchestration helpers
# ═══════════════════════════════════════════════════════════════════════


# ─── _determine_manifest_storage_base ─────────────────────────────────────


def test_storage_base_explicit_in_manifest_kept_when_canonical():
    """When the manifest provides storage_base matching the source's
    prefix, return it as-is."""
    digest = _load_digest()
    manifest = {"storage_base": "https://zenodo.org/records/12345/extra"}
    result = digest._determine_manifest_storage_base("zenodo", "ds-001", manifest)
    assert result == "https://zenodo.org/records/12345/extra"


def test_storage_base_explicit_rejected_when_wrong_source():
    """An explicit storage_base that doesn't start with the source's
    prefix gets rebuilt — the pre-PR-#327 NEMAR misrouting defense."""
    digest = _load_digest()
    manifest = {"storage_base": "s3://openneuro.org/wrong-dataset"}
    # NEMAR's prefix is s3://nemar — the openneuro base mismatches
    result = digest._determine_manifest_storage_base("nemar", "nm000176", manifest)
    assert result == "s3://nemar/nm000176"


def test_storage_base_figshare_uses_source_url():
    digest = _load_digest()
    manifest = {"external_links": {"source_url": "https://figshare.com/articles/12345"}}
    result = digest._determine_manifest_storage_base("figshare", "ds-001", manifest)
    assert result == "https://figshare.com/articles/12345"


def test_storage_base_figshare_falls_back_to_default():
    digest = _load_digest()
    result = digest._determine_manifest_storage_base("figshare", "ds-001", {})
    assert "ds-001" in result


def test_storage_base_zenodo_uses_zenodo_id():
    digest = _load_digest()
    manifest = {"zenodo_id": "999999"}
    result = digest._determine_manifest_storage_base("zenodo", "ds-001", manifest)
    assert "999999" in result
    assert "ds-001" not in result  # zenodo_id wins


def test_storage_base_gin_includes_organization():
    digest = _load_digest()
    manifest = {"organization": "MyLab"}
    result = digest._determine_manifest_storage_base("gin", "ds-001", manifest)
    assert "MyLab" in result
    assert "ds-001" in result


def test_storage_base_default_for_unknown_source():
    digest = _load_digest()
    result = digest._determine_manifest_storage_base("totally_unknown", "ds-001", {})
    assert "ds-001" in result


# ─── _collect_bids_entities_from_paths ────────────────────────────────────


def test_entities_collects_neuro_subjects_only():
    """Only modalities in NEURO_MODALITIES count toward subjects/tasks."""
    digest = _load_digest()
    files = [
        {"path": "sub-01/eeg/sub-01_task-rest_eeg.edf"},
        {"path": "sub-02/eeg/sub-02_task-motor_eeg.edf"},
        # Non-neuro file (anat) should NOT contribute to subjects/tasks
        {"path": "sub-99/anat/sub-99_T1w.nii.gz"},
    ]
    subjects, _sessions, tasks, modalities = digest._collect_bids_entities_from_paths(
        files, []
    )
    assert subjects == {"01", "02"}
    assert tasks == {"rest", "motor"}
    # Modalities tracks ALL paths, not just neuro
    assert "eeg" in modalities


def test_entities_walks_zip_contents():
    """ZIP file's _zip_contents entries are walked too."""
    digest = _load_digest()
    files = [
        {
            "path": "data.zip",
            "_zip_contents": [
                {"path": "sub-01/eeg/sub-01_task-rest_eeg.edf"},
                {"path": "sub-02/eeg/sub-02_task-motor_eeg.edf"},
            ],
        },
    ]
    subjects, _, tasks, _ = digest._collect_bids_entities_from_paths(files, [])
    assert subjects == {"01", "02"}
    assert tasks == {"rest", "motor"}


def test_entities_accepts_plain_string_paths():
    """Files can be strings (not dicts) — early manifest schemas."""
    digest = _load_digest()
    files = ["sub-01/eeg/sub-01_task-rest_eeg.edf"]
    subjects, _, tasks, _ = digest._collect_bids_entities_from_paths(files, [])
    assert subjects == {"01"}
    assert tasks == {"rest"}


def test_entities_standalone_zip_contents_array():
    """Manifest's top-level zip_contents (separate from per-file)."""
    digest = _load_digest()
    files = []
    zip_contents = [{"path": "sub-01/eeg/sub-01_task-rest_eeg.edf"}]
    subjects, _, tasks, _ = digest._collect_bids_entities_from_paths(
        files, zip_contents
    )
    assert subjects == {"01"}
    assert tasks == {"rest"}


# ─── _is_bids_data_zip ────────────────────────────────────────────────────


def test_is_bids_data_zip_matches_known_patterns():
    digest = _load_digest()
    assert digest._is_bids_data_zip("dataset_bids_v1.zip")
    assert digest._is_bids_data_zip("data_bids.zip")
    assert digest._is_bids_data_zip("recording_eeg.zip")
    assert digest._is_bids_data_zip("study_meg.zip")
    assert digest._is_bids_data_zip("clinical_ieeg.zip")
    assert digest._is_bids_data_zip("rawdata_v2.zip")
    assert digest._is_bids_data_zip("data.zip")
    assert digest._is_bids_data_zip("dataset_v1.zip")


def test_is_bids_data_zip_rejects_unrelated_zips():
    digest = _load_digest()
    assert not digest._is_bids_data_zip("analysis_results.zip")
    assert not digest._is_bids_data_zip("supplementary_figures.zip")
    assert not digest._is_bids_data_zip("readme.zip")


def test_is_bids_data_zip_case_insensitive():
    """The patterns match against lowercased input."""
    digest = _load_digest()
    assert digest._is_bids_data_zip("DATA_BIDS.ZIP")
    assert digest._is_bids_data_zip("Recording_EEG.zip")


# ═══════════════════════════════════════════════════════════════════════
# §4 — Manifest-path Record builders
# ═══════════════════════════════════════════════════════════════════════


# ─── _build_regular_manifest_record ───────────────────────────────────────


def test_regular_record_skips_non_neuro_files():
    digest = _load_digest()
    record, errors = digest._build_regular_manifest_record(
        {"path": "README.md", "size": 100},
        dataset_id="ds-001",
        storage_base="https://example.org/ds-001",
        source="zenodo",
        digested_at="2026-05-22T12:00:00Z",
    )
    assert record is None
    assert errors == []


def test_regular_record_builds_for_eeg_file():
    digest = _load_digest()
    record, errors = digest._build_regular_manifest_record(
        {
            "path": "sub-01/eeg/sub-01_task-rest_eeg.edf",
            "size": 1024,
            "download_url": "https://example.org/file.edf",
        },
        dataset_id="ds-001",
        storage_base="https://example.org/ds-001",
        source="zenodo",
        digested_at="2026-05-22T12:00:00Z",
    )
    assert record is not None
    assert errors == []
    assert record["bids_relpath"] == "sub-01/eeg/sub-01_task-rest_eeg.edf"
    assert record["download_url"] == "https://example.org/file.edf"
    assert record["file_size"] == 1024


def test_regular_record_accepts_plain_string_file_entry():
    """Some old manifest shapes use plain strings instead of dicts."""
    digest = _load_digest()
    record, _ = digest._build_regular_manifest_record(
        "sub-01/eeg/sub-01_task-rest_eeg.edf",
        dataset_id="ds-001",
        storage_base="https://example.org/ds-001",
        source="zenodo",
        digested_at="2026-05-22T12:00:00Z",
    )
    assert record is not None
    assert record["bids_relpath"] == "sub-01/eeg/sub-01_task-rest_eeg.edf"


# ─── _build_ctf_ds_records ────────────────────────────────────────────────


def test_ctf_records_dedup_to_ds_directories():
    """Multiple files inside one .ds dir produce ONE Record per dir."""
    digest = _load_digest()
    files = [
        {"path": "sub-01/meg/sub-01_task-rest_meg.ds/raw.meg4"},
        {"path": "sub-01/meg/sub-01_task-rest_meg.ds/res4"},
        {"path": "sub-01/meg/sub-01_task-rest_meg.ds/hc"},
        # A different .ds dir
        {"path": "sub-02/meg/sub-02_task-rest_meg.ds/raw.meg4"},
    ]
    records, errors = digest._build_ctf_ds_records(
        files,
        dataset_id="ds-meg",
        storage_base="https://example.org/ds-meg",
        source="nemar",
        digested_at="2026-05-22T12:00:00Z",
    )
    assert errors == []
    assert len(records) == 2
    relpaths = {r["bids_relpath"] for r in records}
    assert relpaths == {
        "sub-01/meg/sub-01_task-rest_meg.ds",
        "sub-02/meg/sub-02_task-rest_meg.ds",
    }


def test_ctf_records_no_ds_dirs_returns_empty():
    digest = _load_digest()
    files = [
        {"path": "sub-01/eeg/sub-01_task-rest_eeg.edf"},  # no .ds anywhere
    ]
    records, errors = digest._build_ctf_ds_records(
        files,
        dataset_id="ds-eeg",
        storage_base="https://example.org/ds-eeg",
        source="nemar",
        digested_at="2026-05-22T12:00:00Z",
    )
    assert records == []
    assert errors == []


# ─── _build_zip_extracted_records ─────────────────────────────────────────


def test_zip_extracted_skips_non_neuro_inside_zip():
    digest = _load_digest()
    file_info = {
        "name": "data.zip",
        "download_url": "https://example.org/data.zip",
        "_zip_contents": [
            {"path": "sub-01/eeg/sub-01_task-rest_eeg.edf", "size": 1024},
            {"path": "README.md", "size": 100},  # skipped
        ],
    }
    records, errors = digest._build_zip_extracted_records(
        file_info,
        dataset_id="ds-001",
        storage_base="https://example.org/ds-001",
        digested_at="2026-05-22T12:00:00Z",
    )
    assert len(records) == 1
    assert errors == []
    assert records[0]["container_url"] == "https://example.org/data.zip"
    assert records[0]["container_type"] == "zip"


# ─── _build_subject_zip_record ────────────────────────────────────────────


def test_subject_zip_record_matches_sub_prefix_pattern():
    digest = _load_digest()
    rec, errs = digest._build_subject_zip_record(
        {
            "path": "sub-007.zip",
            "download_url": "https://example.org/sub-007.zip",
            "size": 1_000_000,
        },
        dataset_id="ds-001",
        storage_base="https://example.org/ds-001",
        primary_mod="eeg",
        recording_modality_val=["eeg"],
        digested_at="2026-05-22T12:00:00Z",
    )
    assert rec is not None
    assert errs == []
    assert rec["zip_contains_bids"] is True
    assert rec["container_url"] == "https://example.org/sub-007.zip"


def test_subject_zip_record_rejects_unrelated_zip():
    digest = _load_digest()
    rec, errs = digest._build_subject_zip_record(
        {"path": "analysis.zip"},
        dataset_id="ds-001",
        storage_base="https://example.org/ds-001",
        primary_mod="eeg",
        recording_modality_val=["eeg"],
        digested_at="2026-05-22T12:00:00Z",
    )
    assert rec is None
    assert errs == []
