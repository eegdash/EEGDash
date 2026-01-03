import csv
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from s3fs import S3FileSystem


# Initialize anonymous S3 client for the OpenNeuro public bucket
def get_s3_dataset_info(dataset: str, fs: S3FileSystem | None = None) -> dict:
    """Fetch dataset information from S3 (OpenNeuro public bucket).

    Returns dict with keys:
      - total_size: total bytes under the dataset prefix
    """
    # Reuse a provided filesystem when running in parallel to avoid extra overhead
    local_fs = fs or S3FileSystem(anon=True, client_kwargs={"region_name": "us-east-2"})
    prefix = "s3://openneuro.org"
    path = f"{prefix}/{dataset}"

    try:
        total_size = int(local_fs.du(path, total=True) or 0)
    except Exception:
        total_size = 0
    return {"total_size": total_size}


def human_readable_size(num_bytes: int) -> str:
    """Format bytes using the closest unit among MB, GB, TB (fallback to KB/B).

    Chooses the largest unit such that the value is >= 1. Uses base 1024.
    """
    if num_bytes is None:
        return "0 B"
    size = float(num_bytes)
    units = [
        (1024**4, "TB"),
        (1024**3, "GB"),
        (1024**2, "MB"),
        (1024**1, "KB"),
        (1, "B"),
    ]
    for factor, unit in units:
        if size >= factor:
            value = size / factor
            # Use no decimals for B/KB; two decimals otherwise
            if unit in ("B", "KB"):
                return f"{int(round(value))} {unit}"
            return f"{value:.2f} {unit}"
    return "0 B"


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
        'session': [unique sessions],
        'run': [unique runs],
        'extension': [unique file extensions],
        'datatype': [unique datatypes],
        'suffix': [unique suffixes],
        'recording_modality': [unique recording modalities],
        'storage_backend': str (e.g., 's3'),
        'storage_base': str (e.g., 's3://openneuro.org/ds001785'),
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
            "session": set(),
            "run": set(),
            "extension": set(),
            "datatype": set(),
            "suffix": set(),
            "recording_modality": set(),
            "storage_backend": None,
            "storage_base": None,
            "nchans": set(),
            "sampling_frequency": set(),
            "channel_types": set(),
            "duration_seconds_total": 0.0,
            "dep_keys_count": 0,
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

        # sessions (new)
        session = rec.get("session")
        if session:
            a["session"].add(session)

        # runs (new)
        run = rec.get("run")
        if run:
            a["run"].add(run)

        # extension (new)
        ext = rec.get("extension")
        if ext:
            a["extension"].add(ext)

        # datatype (new)
        datatype = rec.get("datatype")
        if datatype:
            a["datatype"].add(datatype)

        # suffix (new)
        suffix = rec.get("suffix")
        if suffix:
            a["suffix"].add(suffix)

        # recording_modality (new)
        modality = rec.get("recording_modality")
        if modality:
            a["recording_modality"].add(modality)

        # storage info (new)
        storage = rec.get("storage") or {}
        if storage.get("backend") and not a["storage_backend"]:
            a["storage_backend"] = storage["backend"]
        if storage.get("base") and not a["storage_base"]:
            a["storage_base"] = storage["base"]

        # count companion files (dep_keys)
        dep_keys = storage.get("dep_keys") or []
        a["dep_keys_count"] += len(dep_keys)

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
            "n_subjects": len(a["subject_id"]),
            "task": sorted(a["task"]),
            "n_tasks": len(a["task"]),
            "session": sorted(a["session"]),
            "n_sessions": len(a["session"]),
            "run": sorted(a["run"]),
            "n_runs": len(a["run"]),
            "extension": sorted(a["extension"]),
            "datatype": sorted(a["datatype"]),
            "suffix": sorted(a["suffix"]),
            "recording_modality": sorted(a["recording_modality"]),
            "storage_backend": a["storage_backend"],
            "storage_base": a["storage_base"],
            "dep_keys_count": a["dep_keys_count"],
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


def enrich_with_s3_size(
    dataset_json: Dict[str, Dict[str, Any]],
    *,
    max_workers: int = 16,
) -> Dict[str, Dict[str, Any]]:
    """Augment each dataset blob with S3 size info and human-readable size.

    Parallelizes S3 queries across datasets to speed up the process.

    Adds keys per dataset:
      - size_bytes
      - size
      - s3_item_count
    """
    if not dataset_json:
        return dataset_json

    fs = S3FileSystem(anon=True, client_kwargs={"region_name": "us-east-2"})

    def worker(ds: str) -> tuple[str, dict]:
        return ds, get_s3_dataset_info(ds, fs=fs)

    datasets = list(dataset_json.keys())
    # Bound workers to avoid overwhelming the network
    workers = max(1, min(max_workers, len(datasets)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker, ds): ds for ds in datasets}
        for fut in as_completed(futures):
            ds, info = fut.result()
            blob = dataset_json.get(ds, {})
            size_bytes = int(info.get("total_size") or 0)
            blob["size_bytes"] = size_bytes
            blob["size"] = human_readable_size(size_bytes)
    return dataset_json


def dataset_summary_table(
    dataset_json: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for ds, blob in dataset_json.items():
        rows.append(
            {
                "dataset": ds,
                "n_records": blob["n_records"],
                "n_subjects": blob.get("n_subjects", len(blob.get("subject_id", []))),
                "n_tasks": blob.get("n_tasks", len(blob.get("task", []))),
                "n_sessions": blob.get("n_sessions", len(blob.get("session", []))),
                "n_runs": blob.get("n_runs", len(blob.get("run", []))),
                "tasks": ",".join(blob.get("task", [])),
                "extensions": ",".join(blob.get("extension", [])),
                "datatypes": ",".join(blob.get("datatype", [])),
                "suffixes": ",".join(blob.get("suffix", [])),
                "recording_modalities": ",".join(blob.get("recording_modality", [])),
                "storage_backend": blob.get("storage_backend", ""),
                "storage_base": blob.get("storage_base", ""),
                "dep_keys_count": blob.get("dep_keys_count", 0),
                "nchans_set": ",".join(map(str, blob.get("nchans", []))),
                "sampling_freqs": ",".join(
                    sorted(
                        {
                            str(int(f)) if float(f).is_integer() else str(f)
                            for f in blob.get("sampling_frequency", [])
                        }
                    )
                ),
                "channel_types": ",".join(blob.get("channel_types", [])),
                "duration_hours_total": blob.get("duration", {}).get("hours_total", 0),
                "size": blob.get("size", "0 B"),
                "size_bytes": blob.get("size_bytes", 0),
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
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset metadata summary from EEGDash API"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="eegdash_staging",
        help="Database name (default: eegdash_staging)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="https://data.eegdash.org",
        help="API URL (default: https://data.eegdash.org)",
    )
    parser.add_argument(
        "--mongodb-uri",
        type=str,
        default=None,
        help="MongoDB URI for direct connection (e.g., mongodb://user:pass@host:27017). "
        "If provided, bypasses HTTP API for faster access.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="consolidated",
        help="Output directory (default: consolidated)",
    )
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Skip S3 size enrichment (faster)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=60,
        help="Max workers for S3 queries (default: 60)",
    )
    args = parser.parse_args()

    # Fetch records - either via MongoDB directly or HTTP API
    if args.mongodb_uri:
        from pymongo import MongoClient

        print(f"Connecting directly to MongoDB database={args.database}...")
        client = MongoClient(args.mongodb_uri)
        db = client[args.database]
        records = list(db.records.find({}))
        print(f"Found {len(records)} records")
        client.close()
    else:
        from eegdash import EEGDash

        print(f"Fetching records from {args.api_url} database={args.database}...")
        client = EEGDash(api_url=args.api_url, database=args.database)
        records = list(client.find({}))
        print(f"Found {len(records)} records")

    print("Normalizing to dataset level...")
    records_to_table = normalize_to_dataset(records)
    print(f"Found {len(records_to_table)} unique datasets")

    if not args.skip_s3:
        print(f"Enriching with S3 sizes (workers={args.workers})...")
        records_to_table = enrich_with_s3_size(
            records_to_table, max_workers=args.workers
        )
    else:
        print("Skipping S3 size enrichment")

    summary_rows = dataset_summary_table(records_to_table)

    files = save_consolidation(
        records_to_table,
        summary_rows,
        out_dir=args.output,
        split_per_dataset_json=True,
        all_in_one_json=True,
        write_summary_csv=True,
    )

    print("Saved:", files)


if __name__ == "__main__":
    main()
