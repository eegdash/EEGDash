from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..api import EEGDashDataset
from ..bids_eeg_metadata import build_query_from_kwargs
from ..const import RELEASE_TO_OPENNEURO_DATASET_MAP, SUBJECT_MINI_RELEASE_MAP
from ..logging import logger
from .registry import register_openneuro_datasets


class EEGChallengeDataset(EEGDashDataset):
    """A dataset helper for the EEG 2025 Challenge.

    This class simplifies access to the EEG 2025 Challenge datasets. It is a
    specialized version of :class:`~eegdash.api.EEGDashDataset` that is
    pre-configured for the challenge's data releases. It automatically maps a
    release name (e.g., "R1") to the corresponding OpenNeuro dataset and handles
    the selection of subject subsets (e.g., "mini" release).

    Parameters
    ----------
    release : str
        The name of the challenge release to load. Must be one of the keys in
        :const:`~eegdash.const.RELEASE_TO_OPENNEURO_DATASET_MAP`
        (e.g., "R1", "R2", ..., "R11").
    cache_dir : str
        The local directory where the dataset will be downloaded and cached.
    mini : bool, default True
        If True, the dataset is restricted to the official "mini" subset of
        subjects for the specified release. If False, all subjects for the
        release are included.
    query : dict, optional
        An additional MongoDB-style query to apply as a filter. This query is
        combined with the release and subject filters using a logical AND.
        The query must not contain the ``dataset`` key, as this is determined
        by the ``release`` parameter.
    s3_bucket : str, optional
        The base S3 bucket URI where the challenge data is stored. Defaults to
        the official challenge bucket.
    **kwargs
        Additional keyword arguments that are passed directly to the
        :class:`~eegdash.api.EEGDashDataset` constructor.

    Raises
    ------
    ValueError
        If the specified ``release`` is unknown, or if the ``query`` argument
        contains a ``dataset`` key. Also raised if ``mini`` is True and a
        requested subject is not part of the official mini-release subset.

    See Also
    --------
    EEGDashDataset : The base class for creating datasets from queries.

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

        message_text = Text.from_markup(
            "This object loads the HBN dataset that has been preprocessed for the EEG Challenge:\n"
            "  * Downsampled from 500Hz to 100Hz\n"
            "  * Bandpass filtered (0.5-50 Hz)\n\n"
            "For full preprocessing applied for competition details, see:\n"
            "  [link=https://github.com/eeg2025/downsample-datasets]https://github.com/eeg2025/downsample-datasets[/link]\n\n"
            "The HBN dataset have some preprocessing applied by the HBN team:\n"
            "  * Re-reference (Cz Channel)\n\n"
            "[bold red]IMPORTANT[/bold red]: The data accessed via `EEGChallengeDataset` is [u]NOT[/u] identical to what you get from [link=https://github.com/sccn/EEGDash/blob/develop/eegdash/api.py]EEGDashDataset[/link] directly.\n"
            "If you are participating in the competition, always use `EEGChallengeDataset` to ensure consistency with the challenge data."
        )

        warning_panel = Panel(
            message_text,
            title="[yellow]EEG 2025 Competition Data Notice[/yellow]",
            subtitle="[cyan]Source: EEGChallengeDataset[/cyan]",
            border_style="yellow",
        )

        # Render the panel directly to the console so it displays in IPython/terminals
        try:
            Console().print(warning_panel)
        except Exception:
            warning_message = str(message_text)
            logger.warning(warning_message)

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
