import pytest
from unittest.mock import MagicMock, patch
from eegdash.dataset.dataset import EEGDashDataset
from eegdash.features.extractors import FeatureExtractor, TrainableFeature


def test_dataset_init_exception_gap(tmp_path):
    # Cover lines 297-298 in dataset.py: Exception handling in loop

    # Create valid dataset dir for checks
    cache = tmp_path
    (cache / "ds001").mkdir()

    with patch(
        "eegdash.dataset.dataset.EEGDashDataset._find_local_bids_records"
    ) as mock_find:
        # Mock records
        mock_find.return_value = [
            {"path": "s1", "bidspath": "foo/bar", "dataset": "ds001"},
            {"path": "s2", "bidspath": "foo/baz", "dataset": "ds001"},
        ]

        # Try patching the class in the module
        with patch("eegdash.dataset.dataset.EEGDashRaw"):
            # Mock get_entities_from_record (plural) called in dataset.py
            with patch(
                "eegdash.dataset.dataset.get_entities_from_record",
                return_value={"sub": "01"},
            ):
                # Mock participants_row_for_subject is NOT called directly, usage is:
                # part_row = bids_ds.subject_participant_tsv(local_file)
                # So we mock EEGBIDSDataset or its method?
                # In the code: bids_ds = EEGBIDSDataset(...)
                # We should patch EEGBIDSDataset class in dataset.py
                with patch("eegdash.dataset.dataset.EEGBIDSDataset") as mock_bids_cls:
                    mock_bids = mock_bids_cls.return_value
                    mock_bids.subject_participant_tsv.return_value = {}

                    # Mock merge_participants_fields to raise Exception
                    with patch(
                        "eegdash.dataset.dataset.merge_participants_fields",
                        side_effect=Exception("Boom"),
                    ):
                        ds = EEGDashDataset(
                            cache_dir=cache,
                            dataset="ds001",
                            check_files=False,
                            download=False,
                        )
                        # Should swallow exception and continue
                        assert len(ds.datasets) == 2


def test_dataset_init_kwargs_gap(tmp_path):
    # Cover lines 317, 319 in dataset.py: database and auth_token init
    with patch(
        "eegdash.dataset.dataset.EEGDashDataset._find_datasets",
        return_value=[MagicMock()],
    ):
        with patch("eegdash.api.EEGDash") as mock_api:
            # Must provide cache_dir
            ds = EEGDashDataset(
                cache_dir=tmp_path,
                dataset="ds001",
                query={"subject": "sub-01"},
                database="mydb",
                auth_token="mytoken",
            )

            # Check if EEGDash was init with correct kwargs
            call_args = mock_api.call_args[1]
            assert call_args["database"] == "mydb"
            assert call_args["auth_token"] == "mytoken"


def test_extractor_preprocess_tuple_gap():
    # Cover line 251 in extractors.py: if not isinstance(z, tuple): z = (z,)

    # Mock FeatureExtractor with a preprocess that returns a single item
    class SingleRetExtractor(FeatureExtractor):
        def preprocess(self, x):
            return x  # Single item, not tuple

    child = MagicMock(spec=TrainableFeature)
    fe = SingleRetExtractor({"c": child})

    # partial_fit calls preprocess
    fe.partial_fit("data")

    # Child should receive it wrapped
    assert child.partial_fit.call_args[0][0] == "data"
