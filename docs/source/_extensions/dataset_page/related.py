"""Related-datasets kNN index — Gower distance over mixed metadata.

Gower distance is the canonical similarity measure for records whose
features mix categorical and numerical types — it normalises numerical
differences by per-feature range and treats categorical mismatch as a
binary 0/1 contribution, then averages across features (weighted).
References: Gower 1971 ("A General Coefficient of Similarity"); the
``gower`` / ``gower-metric`` Python packages implement the same recipe.

We inline the algorithm so the docs build picks up no new runtime dep —
numpy is already pulled in by sphinx-gallery, and the resulting matrix
is tiny (~780² f32 ≈ 2.4 MB) even at full catalog size.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from sphinx.util import logging

from .data_loaders import _clean_value

LOGGER = logging.getLogger(__name__)


# (name, kind, weight) — weights are domain judgment: modality and
# paradigm dominate, source/license barely move the needle.
_RELATED_FEATURE_SPECS: tuple[tuple[str, str, float], ...] = (
    ("record_modality", "cat", 4.0),  # eeg / meg / emg / ieeg — primary axis
    ("modality of exp", "cat", 3.0),  # visual / motor / auditory / …
    ("type of exp", "cat", 2.5),  # perception / clinical / resting-state
    ("Type Subject", "cat", 2.0),  # healthy / clinical / development
    ("source", "cat", 0.5),  # openneuro / nemar — provenance
    ("n_subjects", "num", 1.5),
    ("n_records", "num", 1.0),
    ("n_channels", "num", 1.5),  # extracted from nchans_set top val
    ("sampling_rate", "num", 1.0),  # extracted from sampling_freqs top val
    ("duration_hours_total", "num", 0.8),
)


def _extract_top_numeric_val(value: object) -> float:
    """Return the most-common numeric ``val`` from a ``[{val, count}]`` array.

    Used to collapse channel / sampling-rate frequency lists to a single
    representative number for distance calculations. Returns ``nan`` when
    no usable value is present.
    """
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]", "[ ]"}:
        return float("nan")
    if text.startswith("["):
        try:
            items = json.loads(text)
        except (ValueError, json.JSONDecodeError):
            return float("nan")
        best_val: float | None = None
        best_count = -1
        for item in items:
            if not isinstance(item, dict):
                continue
            v = item.get("val")
            c = item.get("count", 0) or 0
            try:
                v_num = float(v)
            except (TypeError, ValueError):
                continue
            if c > best_count:
                best_val = v_num
                best_count = c
        return best_val if best_val is not None else float("nan")
    # Bare numeric column (n_subjects, n_records, …).
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _build_related_index_gower(
    dataset_names: Sequence[str],
    dataset_rows: Mapping[str, Mapping[str, object]],
    k: int = 6,
) -> dict[str, list[str]]:
    """Compute per-dataset top-K related neighbours via Gower distance.

    Returns ``{name: [neighbour_name, …]}``. An entry is empty when the
    dataset has no usable features. Falls back to ``{}`` (callers then
    use the legacy modality grouping) if numpy is unavailable.
    """
    try:
        import numpy as np
    except ImportError:
        LOGGER.warning(
            "[dataset-docs] numpy not available; related-datasets falls back "
            "to modality grouping"
        )
        return {}

    names = list(dataset_names)
    n = len(names)
    if n < 2:
        return {name: [] for name in names}

    specs = _RELATED_FEATURE_SPECS

    # Build typed columns once, indexed positionally so the distance loop
    # is plain numpy.
    cat_cols: dict[str, np.ndarray] = {}
    cat_missing: dict[str, np.ndarray] = {}
    num_cols: dict[str, np.ndarray] = {}
    num_missing: dict[str, np.ndarray] = {}
    num_ranges: dict[str, float] = {}

    for col, kind, _ in specs:
        if kind == "cat":
            arr = np.empty(n, dtype=object)
            miss = np.zeros(n, dtype=bool)
            for i, name in enumerate(names):
                row = dataset_rows.get(name) or {}
                raw = _clean_value(row.get(col))
                if not raw:
                    miss[i] = True
                    arr[i] = ""
                else:
                    arr[i] = raw.lower()
            cat_cols[col] = arr
            cat_missing[col] = miss
        else:  # numerical
            arr = np.full(n, np.nan, dtype=np.float64)
            for i, name in enumerate(names):
                row = dataset_rows.get(name) or {}
                if col == "n_channels":
                    val = _extract_top_numeric_val(row.get("nchans_set"))
                elif col == "sampling_rate":
                    val = _extract_top_numeric_val(row.get("sampling_freqs"))
                else:
                    val = _extract_top_numeric_val(row.get(col))
                arr[i] = val
            miss = np.isnan(arr)
            num_cols[col] = arr
            num_missing[col] = miss
            present = arr[~miss]
            if present.size:
                rng = float(present.max() - present.min())
                num_ranges[col] = rng if rng > 0 else 0.0
            else:
                num_ranges[col] = 0.0

    # Accumulate weighted pairwise distance + weight tracker.
    D = np.zeros((n, n), dtype=np.float32)
    W = np.zeros((n, n), dtype=np.float32)

    for col, kind, weight in specs:
        if kind == "cat":
            arr = cat_cols[col]
            miss = cat_missing[col]
            pair_d = (arr[:, None] != arr[None, :]).astype(np.float32)
            present = (~miss[:, None]) & (~miss[None, :])
        else:
            arr = num_cols[col]
            miss = num_missing[col]
            rng = num_ranges[col]
            if rng <= 0:
                # All present rows share one value → contributes nothing,
                # but the pair still "covers" this feature for normalisation.
                pair_d = np.zeros((n, n), dtype=np.float32)
            else:
                diff = np.abs(arr[:, None] - arr[None, :]) / rng
                pair_d = diff.astype(np.float32)
            present = (~miss[:, None]) & (~miss[None, :])
            # nan minus nan stayed nan in the diff above — clamp to 0 so
            # masking does the right thing.
            pair_d = np.where(present, pair_d, 0)
        D += weight * pair_d
        W += weight * present.astype(np.float32)

    # Normalise by per-pair weight coverage; pairs with zero shared
    # features get a distance of 1 (treated as "unrelated").
    no_overlap = W <= 0
    W = np.where(no_overlap, 1, W)
    D = D / W
    D = np.where(no_overlap, 1.0, D)
    np.fill_diagonal(D, 0.0)

    # Top-K nearest per row, excluding self and entries with zero distance
    # only when there is more than one such entry (perfect-match dupes are
    # still useful to surface).
    related: dict[str, list[str]] = {}
    for i, name in enumerate(names):
        order = np.argsort(D[i], kind="stable")
        out: list[str] = []
        for j in order:
            if j == i:
                continue
            if D[i, j] >= 1.0:  # no shared features
                continue
            out.append(names[j])
            if len(out) >= k:
                break
        related[name] = out
    return related


def _related_meta_for(
    name: str,
    related_names: Sequence[str],
    rows: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build the ``related_meta`` list consumed by editorial footnotes.

    Each entry has ``{name, modality, n_subjects, same_authors}``. The
    same-author flag flips when ANY clean-cased author name overlaps
    between the source row and the neighbour row.
    """
    self_row = rows.get(name) or {}
    self_authors = {
        a.strip().lower() for a in _normalize_author_field(self_row.get("authors")) if a
    }

    out: list[dict[str, object]] = []
    for rel in related_names:
        row = rows.get(rel) or {}
        modality = (
            _clean_value(row.get("record_modality"))
            or _clean_value(row.get("modality of exp"))
            or ""
        )
        n_sub = _clean_value(row.get("n_subjects"))
        their_authors = {
            a.strip().lower() for a in _normalize_author_field(row.get("authors")) if a
        }
        same_authors = bool(
            self_authors and their_authors and (self_authors & their_authors)
        )
        out.append(
            {
                "name": rel,
                "modality": modality,
                "n_subjects": n_sub,
                "same_authors": same_authors,
            }
        )
    return out


def _normalize_author_field(value: object) -> list[str]:
    """Light-weight author normaliser for the same-author flag.

    Accepts either a list, a JSON-encoded list, or a free-form string;
    returns a plain ``list[str]``. Mirrors ``_normalize_list`` without
    importing ``data_loaders`` at module load (which would force a
    circular import path through ``sections``).
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, json.JSONDecodeError):
            pass
    return [text]
