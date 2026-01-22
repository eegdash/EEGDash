"""Tests for VHDR metadata parser."""

import json
import sys
from pathlib import Path

import pytest

# Add scripts/ingestions to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "ingestions"))

from _vhdr_parser import find_datasets_needing_redigestion, parse_vhdr_metadata


class TestParseVhdrMetadata:
    """Tests for parse_vhdr_metadata function."""

    def test_basic_vhdr_parsing(self, tmp_path: Path):
        """Test parsing a standard VHDR file."""
        vhdr_content = """\
Brain Vision Data Exchange Header File Version 1.0

[Common Infos]
Codepage=UTF-8
DataFile=test.eeg
MarkerFile=test.vmrk
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels=31
SamplingInterval=2000

[Channel Infos]
Ch1=Fp1,,0.0488281,µV
Ch2=Fp2,,0.0488281,µV
Ch3=F3,,0.0488281,µV
Ch4=F4,,0.0488281,µV
"""
        vhdr_path = tmp_path / "test.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        assert result["nchans"] == 31
        # 1,000,000 / 2000 = 500 Hz
        assert result["sampling_frequency"] == 500.0
        assert result["ch_names"] == ["Fp1", "Fp2", "F3", "F4"]

    def test_nonexistent_file(self, tmp_path: Path):
        """Test handling of nonexistent file."""
        vhdr_path = tmp_path / "nonexistent.vhdr"
        result = parse_vhdr_metadata(vhdr_path)
        assert result is None

    def test_broken_symlink(self, tmp_path: Path):
        """Test handling of broken symlinks (git-annex scenario)."""
        # Create a symlink pointing to a nonexistent target
        vhdr_path = tmp_path / "broken.vhdr"
        target_path = tmp_path / "nonexistent_target.vhdr"

        # Create symlink (skip on Windows if symlinks aren't supported)
        try:
            vhdr_path.symlink_to(target_path)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        result = parse_vhdr_metadata(vhdr_path)
        assert result is None

    def test_utf8_encoding(self, tmp_path: Path):
        """Test parsing VHDR file with UTF-8 encoding."""
        vhdr_content = """\
[Common Infos]
NumberOfChannels=3
SamplingInterval=1000

[Channel Infos]
Ch1=Fp1,,0.1,µV
Ch2=Cz,,0.1,µV
Ch3=O1,,0.1,µV
"""
        vhdr_path = tmp_path / "utf8.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        assert result["nchans"] == 3
        assert result["sampling_frequency"] == 1000.0
        assert result["ch_names"] == ["Fp1", "Cz", "O1"]

    def test_latin1_encoding_fallback(self, tmp_path: Path):
        """Test parsing VHDR file with latin-1 encoding."""
        vhdr_content = """\
[Common Infos]
NumberOfChannels=2
SamplingInterval=500

[Channel Infos]
Ch1=Fp1,,0.1,µV
Ch2=Fp2,,0.1,µV
"""
        vhdr_path = tmp_path / "latin1.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="latin-1")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        assert result["nchans"] == 2
        # 1,000,000 / 500 = 2000 Hz
        assert result["sampling_frequency"] == 2000.0

    def test_escaped_commas_in_channel_names(self, tmp_path: Path):
        r"""Test parsing channel names with escaped commas (\1 -> ,)."""
        vhdr_content = """\
[Common Infos]
NumberOfChannels=3
SamplingInterval=2000

[Channel Infos]
Ch1=EMG\\1Left,,0.1,µV
Ch2=EMG\\1Right,,0.1,µV
Ch3=EOG,,0.1,µV
"""
        vhdr_path = tmp_path / "escaped.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        assert result["ch_names"] == ["EMG,Left", "EMG,Right", "EOG"]

    def test_missing_channel_names_uses_generic(self, tmp_path: Path):
        """Test that empty channel names get generic names."""
        vhdr_content = """\
[Common Infos]
NumberOfChannels=3
SamplingInterval=1000

[Channel Infos]
Ch1=,,0.1,µV
Ch2=Fp2,,0.1,µV
Ch3=,,0.1,µV
"""
        vhdr_path = tmp_path / "generic.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        assert result["ch_names"] == ["Ch1", "Fp2", "Ch3"]

    def test_malformed_file_no_sections(self, tmp_path: Path):
        """Test handling of malformed file without valid sections."""
        vhdr_content = "This is not a valid VHDR file"
        vhdr_path = tmp_path / "malformed.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)
        assert result is None

    def test_incomplete_file_only_nchans(self, tmp_path: Path):
        """Test parsing file with only NumberOfChannels."""
        vhdr_content = """\
[Common Infos]
NumberOfChannels=16
"""
        vhdr_path = tmp_path / "incomplete.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        assert result["nchans"] == 16
        assert "sampling_frequency" not in result
        assert "ch_names" not in result

    def test_sampling_interval_zero(self, tmp_path: Path):
        """Test that SamplingInterval=0 is skipped."""
        vhdr_content = """\
[Common Infos]
NumberOfChannels=8
SamplingInterval=0
"""
        vhdr_path = tmp_path / "zero_interval.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        assert result["nchans"] == 8
        # sampling_frequency should not be present since interval is 0
        assert "sampling_frequency" not in result

    def test_channel_section_derives_nchans(self, tmp_path: Path):
        """Test that nchans is derived from channel section if missing."""
        vhdr_content = """\
[Common Infos]
SamplingInterval=1000

[Channel Infos]
Ch1=Fp1,,0.1,µV
Ch2=Fp2,,0.1,µV
Ch3=Cz,,0.1,µV
Ch4=Pz,,0.1,µV
"""
        vhdr_path = tmp_path / "derive_nchans.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        # nchans derived from channel names
        assert result["nchans"] == 4
        assert len(result["ch_names"]) == 4

    def test_channel_numbers_out_of_order(self, tmp_path: Path):
        """Test that channels are sorted by number."""
        vhdr_content = """\
[Common Infos]
NumberOfChannels=3
SamplingInterval=1000

[Channel Infos]
Ch3=O1,,0.1,µV
Ch1=Fp1,,0.1,µV
Ch2=Cz,,0.1,µV
"""
        vhdr_path = tmp_path / "unordered.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        # Channels should be sorted by number
        assert result["ch_names"] == ["Fp1", "Cz", "O1"]

    def test_case_insensitive_section_parsing(self, tmp_path: Path):
        """Test that section and key parsing is case-insensitive."""
        vhdr_content = """\
[COMMON INFOS]
NUMBEROFCHANNELS=5
samplinginterval=4000

[channel infos]
ch1=Fp1,,0.1,µV
CH2=Fp2,,0.1,µV
"""
        vhdr_path = tmp_path / "case.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        assert result["nchans"] == 5
        # 1,000,000 / 4000 = 250 Hz
        assert result["sampling_frequency"] == 250.0
        assert result["ch_names"] == ["Fp1", "Fp2"]

    def test_channel_with_reference(self, tmp_path: Path):
        """Test parsing channels with reference electrode specified."""
        vhdr_content = """\
[Common Infos]
NumberOfChannels=2
SamplingInterval=2000

[Channel Infos]
Ch1=Fp1,REF,0.1,µV
Ch2=Fp2,REF,0.1,µV
"""
        vhdr_path = tmp_path / "with_ref.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        assert result["ch_names"] == ["Fp1", "Fp2"]

    def test_floating_point_sampling_interval(self, tmp_path: Path):
        """Test parsing floating point sampling interval."""
        vhdr_content = """\
[Common Infos]
NumberOfChannels=10
SamplingInterval=3906.25
"""
        vhdr_path = tmp_path / "float_interval.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        result = parse_vhdr_metadata(vhdr_path)

        assert result is not None
        # 1,000,000 / 3906.25 = 256 Hz
        assert abs(result["sampling_frequency"] - 256.0) < 0.01

    def test_string_path_input(self, tmp_path: Path):
        """Test that string paths work as well as Path objects."""
        vhdr_content = """\
[Common Infos]
NumberOfChannels=8
SamplingInterval=1000
"""
        vhdr_path = tmp_path / "string_path.vhdr"
        vhdr_path.write_text(vhdr_content, encoding="utf-8")

        # Pass as string instead of Path
        result = parse_vhdr_metadata(str(vhdr_path))

        assert result is not None
        assert result["nchans"] == 8


class TestFindDatasetsNeedingRedigestion:
    """Tests for find_datasets_needing_redigestion function."""

    def test_finds_datasets_with_missing_metadata(self, tmp_path: Path):
        """Test finding datasets with VHDR files missing metadata."""
        # Create a dataset directory with missing metadata
        dataset_dir = tmp_path / "ds000001"
        dataset_dir.mkdir()

        records_data = {
            "records": [
                {
                    "bids_relpath": "sub-01/eeg/sub-01_eeg.vhdr",
                    "sampling_frequency": None,
                    "nchans": None,
                }
            ]
        }
        records_file = dataset_dir / "ds000001_records.json"
        with open(records_file, "w") as f:
            json.dump(records_data, f)

        result = find_datasets_needing_redigestion(tmp_path)

        assert result == ["ds000001"]

    def test_skips_datasets_with_complete_metadata(self, tmp_path: Path):
        """Test that datasets with complete metadata are not flagged."""
        dataset_dir = tmp_path / "ds000002"
        dataset_dir.mkdir()

        records_data = {
            "records": [
                {
                    "bids_relpath": "sub-01/eeg/sub-01_eeg.vhdr",
                    "sampling_frequency": 500.0,
                    "nchans": 31,
                }
            ]
        }
        records_file = dataset_dir / "ds000002_records.json"
        with open(records_file, "w") as f:
            json.dump(records_data, f)

        result = find_datasets_needing_redigestion(tmp_path)

        assert result == []

    def test_skips_non_vhdr_files(self, tmp_path: Path):
        """Test that non-VHDR files with missing metadata are skipped."""
        dataset_dir = tmp_path / "ds000003"
        dataset_dir.mkdir()

        records_data = {
            "records": [
                {
                    "bids_relpath": "sub-01/eeg/sub-01_eeg.set",
                    "sampling_frequency": None,
                    "nchans": None,
                }
            ]
        }
        records_file = dataset_dir / "ds000003_records.json"
        with open(records_file, "w") as f:
            json.dump(records_data, f)

        result = find_datasets_needing_redigestion(tmp_path)

        assert result == []

    def test_handles_missing_records_file(self, tmp_path: Path):
        """Test that directories without records files are skipped."""
        dataset_dir = tmp_path / "ds000004"
        dataset_dir.mkdir()

        result = find_datasets_needing_redigestion(tmp_path)

        assert result == []

    def test_handles_malformed_json(self, tmp_path: Path):
        """Test that malformed JSON files are skipped."""
        dataset_dir = tmp_path / "ds000005"
        dataset_dir.mkdir()

        records_file = dataset_dir / "ds000005_records.json"
        records_file.write_text("{ invalid json }")

        result = find_datasets_needing_redigestion(tmp_path)

        assert result == []

    def test_returns_unique_sorted_list(self, tmp_path: Path):
        """Test that result is unique and sorted."""
        # Create multiple datasets, some needing redigestion
        for ds_id in ["ds000003", "ds000001", "ds000002"]:
            dataset_dir = tmp_path / ds_id
            dataset_dir.mkdir()

            needs_redigestion = ds_id in ["ds000003", "ds000001"]
            records_data = {
                "records": [
                    {
                        "bids_relpath": "sub-01/eeg/sub-01_eeg.vhdr",
                        "sampling_frequency": None if needs_redigestion else 500.0,
                        "nchans": None if needs_redigestion else 31,
                    }
                ]
            }
            records_file = dataset_dir / f"{ds_id}_records.json"
            with open(records_file, "w") as f:
                json.dump(records_data, f)

        result = find_datasets_needing_redigestion(tmp_path)

        # Should be sorted
        assert result == ["ds000001", "ds000003"]
