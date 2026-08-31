"""Guard the NM000134 EEG-to-image start-kit tutorial contract."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "examples/eeg2026/tutorial_eeg_to_image_start_kit.py"


def test_eeg_to_image_start_kit_uses_the_real_nm000134_test_record():
    """Keep the lesson executable, real-data-only, and viewer-native."""
    assert TUTORIAL.is_file()
    source = TUTORIAL.read_text(encoding="utf-8")

    assert 'DATASET = "nm000134"' in source
    assert 'SUBJECT = "09"' in source
    assert 'SESSION = "01"' in source
    assert 'RUN = "02"' in source
    assert 'TASK = "images"' in source
    assert "sub-09/ses-01/eeg/sub-09_ses-01_task-images_run-02_eeg.edf" in source
    assert "EXPECTED_STIM_TEST_EVENTS = 936" in source
    assert "EXPECTED_UNIQUE_IMAGE_IDS = 200" in source
    assert "EXPECTED_EEG_CHANNELS = 32" in source
    assert "SAMPLING_FREQUENCY = 256.0" in source
    assert "assert len(dataset.records) == 1" in source
    assert "len(dataset)" not in source
    assert "stim_test" in source
    assert "event_intervals_s" in source
    assert "raw.get_data" in source
    assert "dataset.plot(index=0, height=520)" in source
    assert "NeuralBench owns" in source
    assert "not an official train/test split" in source
    assert "EEGDASH_RUN_2026_TUTORIALS" not in source
    assert "os.environ" not in source
    assert "np.random" not in source
    assert "def main()" not in source
    assert ".. figure::" not in source
    assert source.count("plt.show()") >= 2
    assert "# Step 1." in source
    assert "# Step 2." in source
    assert "# Step 3." in source
    assert "# Step 4." in source
    assert "# Step 5." in source
