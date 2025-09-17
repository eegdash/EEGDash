from pathlib import Path

from ..api import EEGDashDataset
from ..bids_eeg_metadata import build_query_from_kwargs
from ..const import RELEASE_TO_OPENNEURO_DATASET_MAP, SUBJECT_MINI_RELEASE_MAP
from ..logging import logger
from .registry import register_openneuro_datasets


class EEGChallengeDataset(EEGDashDataset):
    """EEG 2025 Challenge dataset helper.

    This class provides a convenient wrapper around :class:`EEGDashDataset`
    configured for the EEG 2025 Challenge releases. It maps a given
    ``release`` to its corresponding OpenNeuro dataset and optionally restricts
    to the official "mini" subject subset.

    Parameters
    ----------
    release : str
        Release name. One of ["R1", ..., "R11"].
    mini : bool, default True
        If True, restrict subjects to the challenge mini subset.
    query : dict | None
        Additional MongoDB-style filters to AND with the release selection.
        Must not contain the key ``dataset``.
    s3_bucket : str | None, default "s3://nmdatasets/NeurIPS25"
        Base S3 bucket used to locate the challenge data.
    **kwargs
        Passed through to :class:`EEGDashDataset`.

    """

    def __init__(
        self,
        release: str,
        cache_dir: str,
        mini: bool = True,
        query: dict | None = None,
        s3_bucket: str | None = "s3://nmdatasets/NeurIPS25",
        **kwargs,
    ):
        self.release = release
        self.mini = mini

        if release not in RELEASE_TO_OPENNEURO_DATASET_MAP:
            raise ValueError(
                f"Unknown release: {release}, expected one of {list(RELEASE_TO_OPENNEURO_DATASET_MAP.keys())}"
            )

        dataset_parameters = []
        if isinstance(release, str):
            dataset_parameters.append(RELEASE_TO_OPENNEURO_DATASET_MAP[release])
        else:
            raise ValueError(
                f"Unknown release type: {type(release)}, the expected type is str."
            )

        if query and "dataset" in query:
            raise ValueError(
                "Query using the parameters `dataset` with the class EEGChallengeDataset is not possible."
                "Please use the release argument instead, or the object EEGDashDataset instead."
            )

        if self.mini:
            # When using the mini release, restrict subjects to the predefined subset.
            # If the user specifies subject(s), ensure they all belong to the mini subset;
            # otherwise, default to the full mini subject list for this release.

            allowed_subjects = set(SUBJECT_MINI_RELEASE_MAP[release])

            # Normalize potential 'subjects' -> 'subject' for convenience
            if "subjects" in kwargs and "subject" not in kwargs:
                kwargs["subject"] = kwargs.pop("subjects")

            # Collect user-requested subjects from kwargs/query. We canonicalize
            # kwargs via build_query_from_kwargs to leverage existing validation,
            # and support Mongo-style {"$in": [...]} shapes from a raw query.
            requested_subjects: list[str] = []

            # From kwargs
            if "subject" in kwargs and kwargs["subject"] is not None:
                # Use the shared query builder to normalize scalars/lists
                built = build_query_from_kwargs(subject=kwargs["subject"])
                s_val = built.get("subject")
                if isinstance(s_val, dict) and "$in" in s_val:
                    requested_subjects.extend(list(s_val["$in"]))
                elif s_val is not None:
                    requested_subjects.append(s_val)  # type: ignore[arg-type]

            # From query (top-level only)
            if query and isinstance(query, dict) and "subject" in query:
                qval = query["subject"]
                if isinstance(qval, dict) and "$in" in qval:
                    requested_subjects.extend(list(qval["$in"]))
                elif isinstance(qval, (list, tuple, set)):
                    requested_subjects.extend(list(qval))
                elif qval is not None:
                    requested_subjects.append(qval)

            # Validate if any subjects were explicitly requested
            if requested_subjects:
                invalid = sorted(
                    {s for s in requested_subjects if s not in allowed_subjects}
                )
                if invalid:
                    raise ValueError(
                        "Some requested subject(s) are not part of the mini release for "
                        f"{release}: {invalid}. Allowed subjects: {sorted(allowed_subjects)}"
                    )
                # Do not override user selection; keep their (validated) subjects as-is.
            else:
                # No subject specified by the user: default to the full mini subset
                kwargs["subject"] = sorted(allowed_subjects)

            s3_bucket = f"{s3_bucket}/{release}_mini_L100_bdf"
        else:
            s3_bucket = f"{s3_bucket}/{release}_L100_bdf"

        logger.info(
            "\n\n"
            "[EEGChallengeDataset] EEG 2025 Competition Data Notice:\n"
            "-------------------------------------------------------\n"
            "This object loads the HBN dataset that has been preprocessed for the EEG Challenge:\n"
            "  - Downsampled from 500Hz to 100Hz\n"
            "  - Bandpass filtered (0.5–50 Hz)\n"
            "\n"
            "For full preprocessing applied for competition details, see:\n"
            "  https://github.com/eeg2025/downsample-datasets\n"
            "\n"
            "The HBN dataset have some preprocessing applied by the HBN team:\n"
            "  - Re-reference (Cz Channel)\n"
            "\n"
            "IMPORTANT: The data accessed via `EEGChallengeDataset` is NOT identical to what you get from `EEGDashDataset` directly.\n"
            "If you are participating in the competition, always use `EEGChallengeDataset` to ensure consistency with the challenge data.\n"
            "\n",
            UserWarning,
            module="eegdash",
        )

        super().__init__(
            dataset=RELEASE_TO_OPENNEURO_DATASET_MAP[release],
            query=query,
            cache_dir=cache_dir,
            s3_bucket=s3_bucket,
            _suppress_comp_warning=True,
            **kwargs,
        )


registered_classes = register_openneuro_datasets(
    summary_file=Path(__file__).with_name("dataset_summary.csv"),
    base_class=EEGDashDataset,
    namespace=globals(),
)


__all__ = ["EEGChallengeDataset"] + list(registered_classes.keys())
