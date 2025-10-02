from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from eegdash.logging import logger

__all__ = [
    "BubbleSizeMetric",
    "DatasetSummaryRecord",
    "LegendConfig",
    "LegendLabelItem",
    "LegendSizeBin",
    "SummaryBubbleArtifacts",
    "SummaryBubbleConfig",
    "load_dataset_summary",
    "modality_color_map",
    "record_to_bubble_kwargs",
    "summary_to_bubble_kwargs",
]

#: Path to the packaged dataset summary table.
_DEFAULT_DATASET_SUMMARY_PATH = (
    Path(__file__).resolve().parent.parent / "dataset" / "dataset_summary.csv"
)

#: Canonical colour palette replicated from the documentation utilities so that
#: plotting adapters do not need to import from the Sphinx-only package.
#: The mapping keys use canonical modality labels.
_MODALITY_COLOR_MAP = {
    "Visual": "#2563eb",
    "Auditory": "#0ea5e9",
    "Tactile": "#10b981",
    "Somatosensory": "#10b981",
    "Multisensory": "#ec4899",
    "Motor": "#f59e0b",
    "Resting State": "#6366f1",
    "Rest": "#6366f1",
    "Sleep": "#7c3aed",
    "Other": "#14b8a6",
    "Unknown": "#94a3b8",
}

#: Canonicalisation helpers replicated from ``docs/plot_dataset/colours.py``.
_CANONICAL_MAP = {
    "type_subject": {
        "healthy controls": "Healthy",
        "healthy": "Healthy",
        "control": "Healthy",
        "clinical": "Clinical",
        "patient": "Clinical",
    },
    "modality": {
        "visual": "Visual",
        "auditory": "Auditory",
        "tactile": "Tactile",
        "somatosensory": "Somatosensory",
        "multisensory": "Multisensory",
        "motor": "Motor",
        "rest": "Resting State",
        "resting state": "Resting State",
        "resting-state": "Resting State",
        "sleep": "Sleep",
        "other": "Other",
    },
    "experiment_type": {
        "perception": "Perception",
        "decision making": "Decision-making",
        "decision-making": "Decision-making",
        "rest": "Rest",
        "resting state": "Rest",
        "resting-state": "Rest",
        "sleep": "Sleep",
        "cognitive": "Cognitive",
        "clinical": "Clinical",
        "other": "Other",
    },
}

#: Separators that may appear in the modality column. They are normalised to
#: commas before selecting the primary modality token.
_MODALITY_SEPARATORS = ("/", "|", ";")

#: Palette applied to Type Subject badges when structured legends are rendered.
_TYPE_SUBJECT_COLOR_MAP = {
    "Healthy": "#15803d",
    "Clinical": "#b91c1c",
    "Other": "#0ea5e9",
    "Unknown": "#94a3b8",
    "Unspecified": "#94a3b8",
}

#: Fallback size override used when the selected metric lacks usable values.
_MIN_SIZE_OVERRIDE = 1.0

#: Number of dataset names to include in summarised warning messages.
_WARNING_NAME_LIMIT = 6


@dataclass(slots=True)
class DatasetSummaryRecord:
    """Structured representation of a dataset summary row.

    Attributes
    ----------
    dataset:
        Dataset identifier.
    modality:
        Canonical modality label used for grouped bubble colours.
    n_subjects:
        Number of unique subjects listed for the dataset (minimum 0).
    n_records:
        Total number of EEG recordings aggregated across the dataset (minimum 0).
    duration_hours_total:
        Total recording duration (hours), if available.
    n_tasks:
        Optional count of experimental tasks captured within the dataset.
    type_subject:
        Canonicalised subject type label (e.g., ``"Healthy"``, ``"Clinical"``).
    experiment_type:
        Canonicalised experiment type label.
    n_sessions:
        Inferred session count for plotting purposes. Defaults to ``1`` until
        richer metadata is integrated.
    n_trials:
        Inferred trials per session, derived from ``n_records``.
    trial_len:
        Approximate trial duration in seconds, falling back to ``1.0`` when the
        CSV does not provide sufficient information.

    """

    dataset: str
    modality: str
    n_subjects: int
    n_records: int
    duration_hours_total: float | None
    n_tasks: int | None
    type_subject: str | None
    experiment_type: str | None
    n_sessions: int
    n_trials: int
    trial_len: float


class BubbleSizeMetric(str, Enum):
    """Enumerates bubble scaling metrics supported by the adapter."""

    N_RECORDS = "n_records"

    @property
    def heading(self) -> str:
        return "Record volume"

    @property
    def description(self) -> str:
        return "number of aggregated EEG recordings"


@dataclass(slots=True)
class LegendSizeBin:
    """Legend entry describing a quantitative bubble size bucket."""

    label: str
    value: float
    gid: str


@dataclass(slots=True)
class LegendLabelItem:
    """Legend entry representing a qualitative label (modality or badge)."""

    label: str
    gid: str
    color: str | None = None


@dataclass(slots=True)
class LegendConfig:
    """Structured legend specification understood by ``dataset_bubble_plot``."""

    size_bins: tuple[LegendSizeBin, ...] = ()
    modalities: tuple[LegendLabelItem, ...] = ()
    type_subjects: tuple[LegendLabelItem, ...] = ()
    heading: str | None = None


@dataclass(slots=True)
class SummaryBubbleConfig:
    """Configuration driving conversion from summary records to bubble payloads.

    Parameters
    ----------
    size_metric:
        Bubble size metric applied to each record. Defaults to ``n_records``.
    sort_by_modality:
        When ``True`` (default) datasets are ordered by modality before size.
    sort_descending_size:
        Controls whether larger metrics appear first within each modality.
    limit:
        Optional maximum number of datasets to include in the plotting payload.
    include_type_subject_badges:
        Flag enabling Type Subject badge rendering in structured legends.
    palette_override:
        Mapping of modality labels to overriding colour codes.
    min_size_override:
        Lower bound applied whenever the selected size metric is missing.
    trial_len_default:
        Safety fallback (seconds) when ``record.trial_len`` is non-positive.
    legend_enabled:
        Toggles structured legend generation; defaults to ``True``.
    legend_heading:
        Optional text injected above the structured legend group.
    scale:
        Default scale factor forwarded to :func:`dataset_bubble_plot`.
    shape:
        Bubble shape forwarded to :func:`dataset_bubble_plot`.
    meta_gap:
        Gap between dataset clusters forwarded to :class:`_ClusterDatasetPlotter`.
    layout_seed:
        Optional seed propagating deterministic jitter into the bubble layout.
    collapse_iterations:
        Iteration budget used by :class:`_BubbleChart` when clustering datasets.
    dataset_url_resolver:
        Optional callable resolving dataset URLs for interactive outputs. The
        callable receives each :class:`DatasetSummaryRecord` and should return an
        absolute or relative URL string. When omitted, URL metadata is not
        attached to generated artifacts.

    """

    size_metric: BubbleSizeMetric = BubbleSizeMetric.N_RECORDS
    sort_by_modality: bool = True
    sort_descending_size: bool = True
    limit: int | None = 12
    include_type_subject_badges: bool = False
    palette_override: Mapping[str, str] | None = None
    min_size_override: float = _MIN_SIZE_OVERRIDE
    trial_len_default: float = 1.0
    legend_enabled: bool = True
    legend_heading: str | None = None
    scale: float = 0.4
    shape: str = "hexagon"
    meta_gap: float = 10.0
    layout_seed: int | None = None
    collapse_iterations: int = 120
    dataset_url_resolver: Callable[["DatasetSummaryRecord"], str | None] | None = None


@dataclass(slots=True)
class SummaryBubbleArtifacts:
    """Container bundling plotting payloads with derived metadata."""

    bubble_kwargs: list[dict[str, float | int | str]] = field(default_factory=list)
    color_map: dict[str, str] = field(default_factory=dict)
    legend_config: LegendConfig | None = None
    size_metric_values: tuple[float, ...] = ()
    records: tuple[DatasetSummaryRecord, ...] = ()
    config: SummaryBubbleConfig | None = None


def load_dataset_summary(path: str | Path | None = None) -> list[DatasetSummaryRecord]:
    """Load and normalise dataset summary records ready for plotting.

    The loader performs lightweight cleaning so downstream plotting utilities
    can assume the following invariants:

    * Superfluous whitespace is removed from string columns.
    * Rows whose ``dataset`` column is exactly ``"test"`` (case insensitive)
      are dropped—these rows appear in intermediate QA exports.
    * Numeric columns required for plotting are converted to integers or
      floats, with missing values coerced to ``0`` (integers) or ``None``.
    * Modality values are normalised to canonical labels using the replicated
      map from the documentation package.

    Notes
    -----
    The plotting workflow currently assumes a single session with one-second
    trials when richer metadata is unavailable. These defaults are documented
    here to aid future phases that may replace the heuristics.

    Parameters
    ----------
    path:
        Optional filesystem path to ``dataset_summary.csv``. When omitted, the
        packaged resource bundled with EEG-DaSh is used.

    Returns
    -------
    list[DatasetSummaryRecord]
        Cleaned records ready for grouped bubble plotting.

    Raises
    ------
    FileNotFoundError
        If *path* is provided and does not exist.
    ValueError
        When mandatory columns are missing from the CSV file.

    """
    csv_path = Path(path) if path else _DEFAULT_DATASET_SUMMARY_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset summary CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {
        "dataset",
        "n_records",
        "n_subjects",
        "duration_hours_total",
        "Type Subject",
        "modality of exp",
        "type of exp",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"dataset_summary is missing required columns: {sorted(missing)}"
        )

    # Rename key columns to snake_case for easier attribute access.
    df = df.rename(
        columns={
            "Type Subject": "type_subject",
            "modality of exp": "modality",
            "type of exp": "experiment_type",
        }
    )

    # Trim whitespace in string columns to keep categories consistent.
    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype(str).str.strip()

    # Filter out QA rows explicitly marked as "test".
    df = df[df["dataset"].str.lower() != "test"]

    # Drop empty dataset identifiers.
    df = df[df["dataset"].notna() & (df["dataset"].astype(str).str.strip() != "")]

    # Cast the numeric columns required for plotting.
    df["n_records"] = (
        pd.to_numeric(df["n_records"], errors="coerce").fillna(0).astype(int)
    )
    df["n_subjects"] = (
        pd.to_numeric(df["n_subjects"], errors="coerce").fillna(0).astype(int)
    )
    df["duration_hours_total"] = pd.to_numeric(
        df["duration_hours_total"], errors="coerce"
    )
    if "n_tasks" in df.columns:
        df["n_tasks"] = pd.to_numeric(df["n_tasks"], errors="coerce")
    else:
        df["n_tasks"] = float("nan")

    # Canonicalise categorical labels.
    df["type_subject"] = df["type_subject"].map(
        lambda value: _canonicalise(value, "type_subject")
    )
    df["modality"] = df["modality"].map(_normalise_modality)
    df["experiment_type"] = df["experiment_type"].map(
        lambda value: _canonicalise(value, "experiment_type")
    )

    records: list[DatasetSummaryRecord] = []
    missing_modality: set[str] = set()
    missing_duration: set[str] = set()

    for row in df.itertuples(index=False):
        dataset_name = str(getattr(row, "dataset"))
        n_records = max(int(getattr(row, "n_records", 0)), 0)
        n_subjects = max(int(getattr(row, "n_subjects", 0)), 0)
        duration_hours_total = _as_optional_float(
            getattr(row, "duration_hours_total", None)
        )
        n_tasks = _as_optional_int(getattr(row, "n_tasks", None))
        if n_tasks is not None and n_tasks < 0:
            n_tasks = 0

        n_sessions = _infer_session_count(n_records)
        n_trials = _infer_trials_per_session(n_records, n_sessions)
        trial_len = _infer_trial_length_seconds(duration_hours_total, n_records)

        modality = str(getattr(row, "modality"))
        if modality == "Unknown":
            missing_modality.add(dataset_name)
        if duration_hours_total is None or duration_hours_total <= 0:
            missing_duration.add(dataset_name)

        records.append(
            DatasetSummaryRecord(
                dataset=dataset_name,
                modality=modality,
                n_subjects=n_subjects,
                n_records=n_records,
                duration_hours_total=duration_hours_total,
                n_tasks=n_tasks,
                type_subject=_as_optional_str(getattr(row, "type_subject", None)),
                experiment_type=_as_optional_str(getattr(row, "experiment_type", None)),
                n_sessions=n_sessions,
                n_trials=n_trials,
                trial_len=trial_len,
            )
        )

    if missing_modality:
        logger.warning(
            "Modality information missing for %d dataset(s): %s",
            len(missing_modality),
            _summarise_names(missing_modality),
        )
    if missing_duration:
        logger.warning(
            "Recording duration unavailable for %d dataset(s); default trial length of 1.0s will be used.",
            len(missing_duration),
        )

    return records


def record_to_bubble_kwargs(
    record: DatasetSummaryRecord, *, trial_len_default: float = 1.0
) -> dict[str, float | int | str]:
    """Convert a :class:`DatasetSummaryRecord` into ``dataset_bubble_plot`` kwargs.

    Parameters
    ----------
    record:
        Dataset summary structure created by :func:`load_dataset_summary`.
    trial_len_default:
        Duration to use when ``record.trial_len`` is missing or not greater
        than zero. The default (``1.0`` second) keeps bubble scaling stable
        until richer metadata is introduced.

    Returns
    -------
    dict[str, float | int | str]
        Keyword arguments ready to be expanded into
        :func:`dataset_bubble_plot`. The dictionary uses the ``paradigm``
        key to remain compatible with the existing plotting API even though
        the values originate from EEG-DaSh modalities.

    """
    trial_len = record.trial_len if record.trial_len > 0 else trial_len_default
    return {
        "dataset_name": record.dataset,
        "paradigm": record.modality,
        "n_subjects": max(record.n_subjects, 1),
        "n_sessions": max(record.n_sessions, 1),
        "n_trials": max(record.n_trials, 1),
        "trial_len": trial_len,
    }


def modality_color_map(
    records: Sequence[DatasetSummaryRecord],
    palette_override: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a colour map covering the modalities present in *records*.

    Parameters
    ----------
    records:
        Iterable of dataset summary records whose modalities should be
        supported by the returned colour map.
    palette_override:
        Optional mapping of modality labels to colour overrides. Entries are
        merged after detecting the canonical palette.

    Returns
    -------
    dict[str, str]
        Mapping suitable for the ``color_map`` argument of
        :func:`dataset_bubble_plot`.

    """
    observed_modalities = {record.modality for record in records}
    colour_map: dict[str, str] = {}
    for modality in observed_modalities:
        colour_map[modality] = _MODALITY_COLOR_MAP.get(
            modality, _MODALITY_COLOR_MAP["Unknown"]
        )
    if palette_override:
        colour_map.update(palette_override)
    # Preserve deterministic ordering for reproducible plots.
    return {key: colour_map[key] for key in sorted(colour_map)}


def summary_to_bubble_kwargs(
    records: Sequence[DatasetSummaryRecord],
    *,
    config: SummaryBubbleConfig | None = None,
) -> SummaryBubbleArtifacts:
    """Convert dataset summary records into plotting payloads.

    The converter enriches each payload with ``size_override`` reflecting the
    requested :class:`BubbleSizeMetric`. When the selected metric is missing or
    non-positive the value is clamped to :attr:`SummaryBubbleConfig.min_size_override`
    and a warning is emitted via :mod:`eegdash.logging`.

    Parameters
    ----------
    records:
        Dataset summary records ready for plotting.
    config:
        Optional :class:`SummaryBubbleConfig` controlling ordering, legend
        generation and palette overrides. A default configuration is used
        when omitted.

    Returns
    -------
    SummaryBubbleArtifacts
        Bundle containing ready-to-plot kwargs, the derived colour map,
        structured legend configuration, and the metric values applied.

    """
    cfg = config or SummaryBubbleConfig()
    if cfg.size_metric is not BubbleSizeMetric.N_RECORDS:
        logger.warning(
            "Unsupported bubble size metric %s requested; defaulting to n_records.",
            cfg.size_metric.value,
        )
        cfg = replace(cfg, size_metric=BubbleSizeMetric.N_RECORDS)
    if not records:
        return SummaryBubbleArtifacts(config=cfg)

    entries: list[tuple[DatasetSummaryRecord, float]] = []
    missing_modality: set[str] = set()
    missing_badge: set[str] = set()

    for record in records:
        if record.modality == "Unknown":
            missing_modality.add(record.dataset)
        size_value = _resolve_size_metric_value(
            record, cfg.size_metric, cfg.min_size_override
        )
        entries.append((record, size_value))
        if cfg.include_type_subject_badges and not record.type_subject:
            missing_badge.add(record.dataset)

    if cfg.sort_by_modality:
        entries.sort(
            key=lambda item: (
                (item[0].modality or "Unknown").lower(),
                -item[1] if cfg.sort_descending_size else item[1],
                item[0].dataset.lower(),
            )
        )
    else:
        entries.sort(
            key=lambda item: (
                -item[1] if cfg.sort_descending_size else item[1],
                item[0].dataset.lower(),
            )
        )

    if cfg.limit is not None:
        entries = entries[: cfg.limit]

    selected_records = tuple(record for record, _ in entries)
    color_map = modality_color_map(selected_records, cfg.palette_override)

    bubble_kwargs: list[dict[str, float | int | str]] = []
    for index, (record, size_value) in enumerate(entries):
        payload = record_to_bubble_kwargs(
            record, trial_len_default=cfg.trial_len_default
        )
        payload["size_override"] = size_value
        if cfg.dataset_url_resolver is not None:
            try:
                dataset_url = cfg.dataset_url_resolver(record)
            except Exception as error:  # pragma: no cover - defensive logging
                logger.warning(
                    "Dataset URL resolver failed for %s: %s",
                    record.dataset,
                    error,
                    exc_info=True,
                )
                dataset_url = None
            if dataset_url:
                payload["dataset_url"] = dataset_url
        if cfg.layout_seed is not None:
            payload["layout_seed"] = int(cfg.layout_seed) + index
        bubble_kwargs.append(payload)

    legend_config = (
        _build_structured_legend(entries, color_map, cfg)
        if cfg.legend_enabled
        else None
    )

    if missing_modality:
        logger.warning(
            "Modality palette fallback applied to %d dataset(s): %s",
            len(missing_modality),
            _summarise_names(missing_modality),
        )
    if missing_badge:
        logger.warning(
            "Type Subject metadata missing for %d dataset(s); badge legends will show 'Unspecified'.",
            len(missing_badge),
        )

    return SummaryBubbleArtifacts(
        bubble_kwargs=bubble_kwargs,
        color_map=color_map,
        legend_config=legend_config,
        size_metric_values=tuple(size for _, size in entries),
        records=selected_records,
        config=cfg,
    )


def _canonicalise(value: object, field: str) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    lookup = _CANONICAL_MAP.get(field, {})
    result = lookup.get(text.lower())
    return result or text


def _normalise_modality(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"

    text = str(value).strip()
    if not text:
        return "Unknown"

    for sep in _MODALITY_SEPARATORS:
        text = text.replace(sep, ",")

    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        return "Unknown"

    primary = tokens[0]
    canonical = _CANONICAL_MAP["modality"].get(primary.lower())
    if canonical:
        return canonical

    if primary in _MODALITY_COLOR_MAP:
        return primary

    title_variant = primary.title()
    if title_variant in _MODALITY_COLOR_MAP:
        return title_variant

    return "Other"


def _summarise_names(
    names: Sequence[str] | set[str], *, limit: int = _WARNING_NAME_LIMIT
) -> str:
    sorted_names = sorted(set(names))
    if len(sorted_names) <= limit:
        return ", ".join(sorted_names)
    head = ", ".join(sorted_names[: limit - 1])
    return f"{head}, …"


def _resolve_size_metric_value(
    record: DatasetSummaryRecord, metric: BubbleSizeMetric, min_value: float
) -> float:
    if metric is not BubbleSizeMetric.N_RECORDS:
        logger.warning(
            "Dataset %s requested unsupported bubble size metric %s; defaulting to record counts.",
            record.dataset,
            metric.value,
        )
    raw_value = float(record.n_records)
    if raw_value <= 0:
        logger.warning(
            "Dataset %s lacks usable record counts; falling back to minimum bubble radius %.3f.",
            record.dataset,
            min_value,
        )
        return float(min_value)
    return max(raw_value, float(min_value))


def _build_structured_legend(
    entries: Sequence[tuple[DatasetSummaryRecord, float]],
    color_map: Mapping[str, str],
    cfg: SummaryBubbleConfig,
) -> LegendConfig | None:
    values = [value for _, value in entries if value > 0]
    if not values:
        return None

    min_value = min(values)
    median_value = statistics.median(values)
    max_value = max(values)

    unique_bins: list[float] = []
    for candidate in (min_value, median_value, max_value):
        if not unique_bins or not math.isclose(
            candidate, unique_bins[-1], rel_tol=1e-6, abs_tol=1e-9
        ):
            unique_bins.append(candidate)

    size_bins = tuple(
        LegendSizeBin(
            label=_format_metric_label(cfg.size_metric, candidate),
            value=candidate,
            gid=f"legend/size/{cfg.size_metric.value}/{index}",
        )
        for index, candidate in enumerate(unique_bins)
    )

    modalities = tuple(
        LegendLabelItem(
            label=modality,
            gid=f"legend/modality/{_slugify_label(modality)}",
            color=color_map.get(modality, _MODALITY_COLOR_MAP["Unknown"]),
        )
        for modality in sorted({record.modality for record, _ in entries})
    )

    type_subjects: tuple[LegendLabelItem, ...] = ()
    if cfg.include_type_subject_badges:
        seen: dict[str, LegendLabelItem] = {}
        for record, _ in entries:
            label = _normalise_type_subject_label(record.type_subject)
            if label not in seen:
                seen[label] = LegendLabelItem(
                    label=label,
                    gid=f"legend/type_subject/{_slugify_label(label)}",
                    color=_TYPE_SUBJECT_COLOR_MAP.get(
                        label, _TYPE_SUBJECT_COLOR_MAP["Unknown"]
                    ),
                )
        type_subjects = tuple(seen[label] for label in sorted(seen))

    heading = cfg.legend_heading or cfg.size_metric.heading

    return LegendConfig(
        size_bins=size_bins,
        modalities=modalities,
        type_subjects=type_subjects,
        heading=heading,
    )


def _format_metric_label(metric: BubbleSizeMetric, value: float) -> str:
    rounded = int(round(value))
    return f"{rounded:,} record{'s' if rounded != 1 else ''}"


def _slugify_label(label: str) -> str:
    return (
        "".join(char.lower() if char.isalnum() else "-" for char in label).strip("-")
        or "value"
    )


def _normalise_type_subject_label(label: str | None) -> str:
    return label or "Unspecified"


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_session_count(n_records: int) -> int:
    """Infer session counts for datasets without explicit metadata.

    Current metadata lacks a direct ``n_sessions`` field, so we assume a single
    session per dataset for the initial integration phase. The helper is kept
    separate to centralise the inference logic when richer metadata becomes
    available.
    """
    _ = n_records  # Placeholder for future heuristics.
    return 1


def _infer_trials_per_session(n_records: int, n_sessions: int) -> int:
    """Estimate the number of trials per session.

    The estimation keeps the product ``n_sessions * n_trials`` proportional to
    the total record count while ensuring both operands remain strictly
    positive so that logarithmic bubble scaling remains defined.
    """
    if n_sessions <= 0:
        n_sessions = 1
    if n_records <= 0:
        return 1
    return max(int(math.ceil(n_records / n_sessions)), 1)


def _infer_trial_length_seconds(
    duration_hours_total: float | None, n_records: int
) -> float:
    """Estimate average trial duration when duration metadata is available."""
    if duration_hours_total is None or duration_hours_total <= 0 or n_records <= 0:
        return 1.0
    seconds_total = duration_hours_total * 3600.0
    approx = seconds_total / n_records
    return max(float(approx), 1e-3)
