from unittest.mock import patch

import numpy as np


def test_hbn_reannotation_no_events():
    """Test that warning is logged when no events found."""
    # Create mock raw with no matching annotations
    import mne

    from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation

    info = mne.create_info(["EEG"], 256, ch_types=["eeg"])
    raw = mne.io.RawArray(np.random.randn(1, 2560), info)
    # Add some other annotations that don't match
    raw.set_annotations(mne.Annotations([0], [1], ["other_event"]))

    # Instantiate the preprocessor
    preprocessor = hbn_ec_ec_reannotation()

    with patch("eegdash.hbn.preprocessing.logger") as mock_logger:
        result = preprocessor.transform(raw)
        # Should return original raw and log warning
        assert result is raw
        mock_logger.warning.assert_called_once()


def test_hbn_preprocessor_no_annotations():
    """Test warning when no eye events found (lines 87-91)."""
    import mne
    import numpy as np

    from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation

    # Create raw with no relevant annotations
    info = mne.create_info(ch_names=["EEG"], sfreq=256, ch_types=["eeg"])
    data = np.random.randn(1, 5000)
    raw = mne.io.RawArray(data, info)
    raw.set_annotations(mne.Annotations(onset=[0], duration=[1], description=["other"]))

    preprocessor = hbn_ec_ec_reannotation()
    # Should return raw unchanged with warning
    result = preprocessor.transform(raw)
    assert result is raw


def test_hbn_preprocessor_with_valid_annotations():
    """Test with valid eye annotations."""
    import mne
    import numpy as np

    from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation

    info = mne.create_info(ch_names=["EEG"], sfreq=256, ch_types=["eeg"])
    data = np.random.randn(1, 256 * 60)  # 60 seconds
    raw = mne.io.RawArray(data, info)
    raw.set_annotations(
        mne.Annotations(
            onset=[1, 25],
            duration=[1, 1],
            description=["instructed_toCloseEyes", "instructed_toOpenEyes"],
        )
    )

    preprocessor = hbn_ec_ec_reannotation()
    result = preprocessor.transform(raw)
    # Should have new annotations
    assert len(result.annotations) > 0
