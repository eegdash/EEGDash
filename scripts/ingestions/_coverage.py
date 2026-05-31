"""Per-field technical-metadata coverage aggregation.

Answers "what is missing, and where did each value come from" across a corpus of
digested records — the measurement the cheap-resolver work is graded against.
Reads only the already-emitted ``*_records.json`` (each record carries the
inferable fields plus ``_metadata_provenance``); does NOT touch signal data.

Coverage is reported overall and broken down by provenance source, by file
format (extension), and by modality, so the formats/modalities where the cheap
path still fails are visible at a glance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# The technical-metadata fields a record can carry that the resolver fills.
INFERABLE_FIELDS: tuple[str, ...] = (
    "sampling_frequency",
    "nchans",
    "ntimes",
    "ch_names",
    "duration_seconds",
)


def _blank_field_stat() -> dict[str, Any]:
    return {"resolved": 0, "missing": 0, "coverage": 0.0, "by_source": defaultdict(int)}


def _is_resolved(value: Any) -> bool:
    """A field is resolved when it is present and non-empty."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, str)) and len(value) == 0:
        return False
    return True


def _record_modalities(record: dict[str, Any]) -> list[str]:
    mods = record.get("recording_modality")
    if isinstance(mods, list) and mods:
        return [str(m) for m in mods]
    datatype = record.get("datatype")
    return [str(datatype)] if datatype else ["unknown"]


def _update_field_stat(stat: dict[str, Any], value: Any, source: Any) -> None:
    if _is_resolved(value):
        stat["resolved"] += 1
    else:
        stat["missing"] += 1
    # ``source`` is the provenance entry; missing/absent -> empty-string bucket.
    stat["by_source"][source or ""] += 1


def aggregate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-field coverage over *records*.

    Returns a report dict with ``total_records``, ``fields`` (overall per-field
    stats), ``by_format`` (ext -> field -> stats), and ``by_modality``.
    """
    fields: dict[str, dict[str, Any]] = {
        f: _blank_field_stat() for f in INFERABLE_FIELDS
    }
    by_format: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {f: _blank_field_stat() for f in INFERABLE_FIELDS}
    )
    by_modality: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {f: _blank_field_stat() for f in INFERABLE_FIELDS}
    )

    total = 0
    for record in records:
        total += 1
        ext = (record.get("extension") or "").lower()
        prov = record.get("_metadata_provenance") or {}
        modalities = _record_modalities(record)
        for field in INFERABLE_FIELDS:
            value = record.get(field)
            source = prov.get(field)
            _update_field_stat(fields[field], value, source)
            _update_field_stat(by_format[ext][field], value, source)
            # by_modality is intentionally NOT a partition: a co-recorded
            # EEG+MEG file legitimately contributes to BOTH modality buckets, so
            # per-modality counts can sum to more than total_records. The headline
            # number is ``fields`` (each record counted once).
            for mod in modalities:
                _update_field_stat(by_modality[mod][field], value, source)

    _finalize(fields)
    for ext_stats in by_format.values():
        _finalize(ext_stats)
    for mod_stats in by_modality.values():
        _finalize(mod_stats)

    return {
        "total_records": total,
        "fields": fields,
        "by_format": _undefault(by_format),
        "by_modality": _undefault(by_modality),
    }


def _finalize(field_stats: dict[str, dict[str, Any]]) -> None:
    """Compute coverage ratios and convert defaultdicts to plain dicts."""
    for stat in field_stats.values():
        seen = stat["resolved"] + stat["missing"]
        stat["coverage"] = round(stat["resolved"] / seen, 4) if seen else 0.0
        stat["by_source"] = dict(stat["by_source"])


def _undefault(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items()}


def iter_records_from_output(output_dir: Path | str) -> Iterator[dict[str, Any]]:
    """Yield every record from all ``*_records.json`` under *output_dir*."""
    output_dir = Path(output_dir)
    for records_path in sorted(output_dir.glob("*/*_records.json")):
        try:
            payload = json.loads(records_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for record in payload.get("records", []):
            if isinstance(record, dict):
                yield record


def aggregate_output_dir(output_dir: Path | str) -> dict[str, Any]:
    """Aggregate coverage over a whole digestion-output directory."""
    return aggregate_records(iter_records_from_output(output_dir))


def format_coverage_summary(report: dict[str, Any]) -> str:
    """Render a compact human-readable coverage summary."""
    lines: list[str] = []
    total = report["total_records"]
    lines.append(f"Records analyzed: {total}")
    lines.append("")
    lines.append("Per-field coverage (resolved / total):")
    for field in INFERABLE_FIELDS:
        stat = report["fields"][field]
        seen = stat["resolved"] + stat["missing"]
        lines.append(
            f"  {field:<20} {stat['resolved']:>7}/{seen:<7} "
            f"{stat['coverage'] * 100:6.2f}%"
        )
    lines.append("")
    lines.append("ntimes by source:")
    for source, count in sorted(
        report["fields"]["ntimes"]["by_source"].items(),
        key=lambda kv: -kv[1],
    ):
        label = source or "<missing>"
        lines.append(f"  {label:<22} {count:>7}")
    lines.append("")
    lines.append("Worst formats for ntimes (lowest coverage first):")
    fmts = sorted(
        report["by_format"].items(),
        key=lambda kv: kv[1]["ntimes"]["coverage"],
    )
    for ext, stats in fmts:
        nt = stats["ntimes"]
        seen = nt["resolved"] + nt["missing"]
        if seen == 0:
            continue
        lines.append(
            f"  {ext or '<none>':<10} ntimes {nt['coverage'] * 100:6.2f}% "
            f"({nt['resolved']}/{seen})"
        )
    return "\n".join(lines)


__all__ = [
    "INFERABLE_FIELDS",
    "aggregate_output_dir",
    "aggregate_records",
    "format_coverage_summary",
    "iter_records_from_output",
]
