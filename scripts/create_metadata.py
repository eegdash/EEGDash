import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

# ---------- helpers ----------


def _canonical_key(v: Any) -> str:
    """Stable, comparable key for dedupe (keeps original value intact elsewhere)."""
    try:
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(v)


def _parse_subject_from_name(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    m = re.search(r"sub-([A-Za-z0-9]+)", name)
    return m.group(1) if m else None


def _safe_duration_seconds(rec: Dict[str, Any]) -> float | None:
    rd = rec.get("rawdatainfo") or {}
    if (
        isinstance(rd.get("ntimes"), (int, float))
        and isinstance(rd.get("sampling_frequency"), (int, float))
        and rd.get("sampling_frequency")
    ):
        return float(rd["ntimes"]) / float(rd["sampling_frequency"])

    ej = rec.get("eeg_json") or {}
    if isinstance(ej.get("RecordingDuration"), (int, float)):
        return float(ej["RecordingDuration"])

    sf = (
        rec.get("sampling_frequency")
        or rd.get("sampling_frequency")
        or ej.get("SamplingFrequency")
    )
    nt = rec.get("ntimes")
    if isinstance(nt, (int, float)):
        if isinstance(sf, (int, float)) and sf:
            return float(nt) / float(sf) if nt > 24 * 3600 else float(nt)
        return float(nt)
    return None


def _to_py_scalar(x):
    """Make numpy scalars JSON-serializable if they sneak in."""
    try:
        import numpy as np  # type: ignore

        if isinstance(x, (np.generic,)):
            return x.item()
    except Exception:
        pass
    return x


# ---------- main aggregation ----------


def normalize_to_dataset(
    records: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Collapse file-level records into one JSON blob per dataset.

    Output per dataset:
      {
        'dataset': str,
        'n_records': int,
        'subject_id': [unique subjects],
        'task': [unique tasks],
        'nchans': [unique channel counts],
        'duration': {'seconds_total': float, 'hours_total': float},
        'extra_info': { eeg_json_key: [unique original values] },
        'sampling_frequency': [unique Hz],
        'channel_types': [unique channel types],
      }
    """
    agg = defaultdict(
        lambda: {
            "dataset": None,
            "n_records": 0,
            "subject_id": set(),
            "task": set(),
            "nchans": set(),
            "sampling_frequency": set(),
            "channel_types": set(),
            "duration_seconds_total": 0.0,
            # store {canon_key -> original_value} so we dedupe but keep originals
            "extra_info": defaultdict(dict),
        }
    )

    for rec in records:
        ds = rec.get("dataset")
        if not ds:
            continue
        a = agg[ds]
        a["dataset"] = ds
        a["n_records"] += 1

        # subjects
        subj = (
            rec.get("subject")
            or (rec.get("rawdatainfo") or {}).get("subject_id")
            or _parse_subject_from_name(
                rec.get("data_name") or rec.get("bidspath") or ""
            )
        )
        if subj:
            a["subject_id"].add(subj)

        # tasks
        task = rec.get("task") or (rec.get("rawdatainfo") or {}).get("task")
        if task:
            a["task"].add(task)

        # nchans
        nchan = (
            rec.get("nchans")
            or (rec.get("rawdatainfo") or {}).get("nchans")
            or (rec.get("eeg_json") or {}).get("EEGChannelCount")
        )
        if isinstance(nchan, (int, float)):
            a["nchans"].add(int(nchan))

        # sampling frequency
        sf = (
            rec.get("sampling_frequency")
            or (rec.get("rawdatainfo") or {}).get("sampling_frequency")
            or (rec.get("eeg_json") or {}).get("SamplingFrequency")
        )
        if isinstance(sf, (int, float)):
            a["sampling_frequency"].add(float(sf))

        # channel types
        cts = (
            rec.get("channel_types")
            or (rec.get("rawdatainfo") or {}).get("channel_types")
            or []
        )
        for ct in cts:
            a["channel_types"].add(ct)

        # duration
        dur = _safe_duration_seconds(rec)
        if isinstance(dur, (int, float)):
            a["duration_seconds_total"] += float(dur)

        # EEG JSON extra info (deduplicated, keep original values)
        eeg = rec.get("eeg_json") or {}
        for k, v in eeg.items():
            if v is None:
                continue
            canon = _canonical_key(v)
            a["extra_info"][k][canon] = v

    # finalize: convert sets to sorted lists, package duration
    out: Dict[str, Dict[str, Any]] = {}
    for ds, a in agg.items():
        extra_info = {
            k: sorted((vals.values()), key=lambda x: _canonical_key(x))
            for k, vals in a["extra_info"].items()
        }
        ds_blob = {
            "dataset": a["dataset"],
            "n_records": int(a["n_records"]),
            "subject_id": sorted(a["subject_id"]),
            "task": sorted(a["task"]),
            "nchans": sorted(int(x) for x in a["nchans"]),
            "duration": {
                "seconds_total": round(float(a["duration_seconds_total"]), 3),
                "hours_total": round(float(a["duration_seconds_total"]) / 3600.0, 3),
            },
            "extra_info": extra_info,
            "sampling_frequency": sorted(float(x) for x in a["sampling_frequency"]),
            "channel_types": sorted(a["channel_types"]),
        }
        # sanitize possible numpy scalars
        ds_blob = json.loads(json.dumps(ds_blob, default=_to_py_scalar))
        out[ds] = ds_blob
    return out


def dataset_summary_table(
    dataset_json: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for ds, blob in dataset_json.items():
        rows.append(
            {
                "dataset": ds,
                "n_records": blob["n_records"],
                "n_subjects": len(blob["subject_id"]),
                "n_tasks": len(blob["task"]),
                "nchans_set": ",".join(map(str, blob["nchans"])),
                "sampling_freqs": ",".join(
                    sorted(
                        {
                            str(int(f)) if float(f).is_integer() else str(f)
                            for f in blob.get("sampling_frequency", [])
                        }
                    )
                ),
                "duration_hours_total": blob["duration"]["hours_total"],
            }
        )
    return rows


# ---------- saving ----------


def save_consolidation(
    dataset_json: Dict[str, Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    out_dir: str | Path = "consolidated_output",
    *,
    split_per_dataset_json: bool = True,
    all_in_one_json: bool = True,
    write_summary_csv: bool = True,
) -> Dict[str, str]:
    """Save:
      - one JSON per dataset (optional),
      - a single combined JSON (optional),
      - a CSV summary table (optional).

    Returns dict of created file paths.
    """
    out = {}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # combined JSON
    if all_in_one_json:
        combined = out_path / "datasets_consolidated.json"
        with combined.open("w", encoding="utf-8") as f:
            json.dump(dataset_json, f, ensure_ascii=False, indent=2)
        out["combined_json"] = str(combined)

    # per-dataset JSON
    if split_per_dataset_json:
        per_dir = out_path / "datasets"
        per_dir.mkdir(exist_ok=True)
        for ds, blob in dataset_json.items():
            p = per_dir / f"{ds}.json"
            with p.open("w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
        out["per_dataset_dir"] = str(per_dir)

    # CSV summary
    if write_summary_csv and summary_rows:
        csv_path = out_path / "dataset_summary.csv"
        fieldnames = list(summary_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        out["summary_csv"] = str(csv_path)

    return out


# ---------- example usage ----------
from eegdash import EEGDash

records = EEGDash().find(query={})


records_to_table = normalize_to_dataset(records)
summary_rows = dataset_summary_table(records_to_table)

files = save_consolidation(
    records_to_table,
    summary_rows,
    out_dir="consolidated",
    split_per_dataset_json=True,
    all_in_one_json=True,
    write_summary_csv=True,
)

print("Saved:", files)
# Access your consolidated JSON for ds005509 if needed:
dataset_info = records_to_table.get("ds005509", {})
