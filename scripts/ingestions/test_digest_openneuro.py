"""Test OpenNeuro digestion and compare to ground truth.

Usage:
    python scripts/ingestions/test_digest_openneuro.py ds004477 --dataset-dir data/cloned_all/ds004477 --ground-truth eegdash_ds004477_extended.json

If dataset_id is omitted, defaults to ds004477. If --all is provided, the script will
process all directories under data/cloned_all that begin with 'ds'.

This script uses the eegdash library with EEGBIDSDataset (allow_symlinks=True) to
extract BIDS metadata from cloned git-annex datasets. It falls back to manual
parsing only when the library fails (e.g., datasets with no EEG recordings).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from eegdash.bids_eeg_metadata import load_eeg_attrs_from_bids_file
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    _HAVE_EEGDASH = True
except Exception:
    # Fallback to local, lightweight BIDS parsing if core dependencies are not installed
    _HAVE_EEGDASH = False
    load_eeg_attrs_from_bids_file = None
    EEGBIDSDataset = None


def process_dataset(dataset_id: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """Process a BIDS dataset using eegdash library.

    Uses EEGBIDSDataset with allow_symlinks=True for git-annex support.
    Falls back to manual parsing when the library fails.
    """
    # Try eegdash library first, fall back to manual parsing if it fails
    if _HAVE_EEGDASH and EEGBIDSDataset is not None:
        try:
            bids_dataset = EEGBIDSDataset(
                data_dir=str(dataset_dir), dataset=dataset_id, allow_symlinks=True
            )
            files = bids_dataset.get_files()
            records = []
            for bids_file in files:
                try:
                    # Use library's metadata extraction directly
                    record = load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                    # Normalize the record for comparison
                    try:
                        record = normalize_record(record, dataset_dir)
                    except Exception:
                        pass
                    records.append(record)
                except Exception as exc:
                    print(f"Error processing {bids_file}: {exc}")
            if records:
                return records
            # If no records from eegdash, fall through to manual parsing
        except Exception as exc:
            print(
                f"EEGBIDSDataset failed for {dataset_id}, falling back to manual parsing: {exc}"
            )

    # Fallback: manual parsing
    return _process_dataset_manual(dataset_id, dataset_dir)


def get_sidecar_dependencies(p: Path, dataset_dir: Path, dataset_id: str) -> List[str]:
    """Return a list of dataset_id-relative sidecar paths for primary data file `p`.

    Simplified version for manual fallback - finds direct sidecars in same directory.
    The primary data file itself is NOT included.
    """
    deps = []
    parent_dir = p.parent
    stem = p.stem

    # Direct sidecar patterns for the same stem
    sidecar_suffixes = [
        ".json",
        "_channels.tsv",
        "_events.tsv",
        "_events.json",
        "_electrodes.tsv",
        "_coordsystem.json",
    ]

    # Companion files for specific formats
    if p.suffix.lower() == ".vhdr":
        sidecar_suffixes.extend([".eeg", ".vmrk"])
    if p.suffix.lower() == ".set":
        sidecar_suffixes.append(".fdt")

    # Find sidecars in same directory
    if parent_dir.exists():
        for suffix in sidecar_suffixes:
            if suffix.startswith("_"):
                # Pattern like stem_channels.tsv
                sidecar = parent_dir / f"{stem}{suffix}"
            else:
                # Pattern like stem.json
                sidecar = parent_dir / f"{stem}{suffix}"

            if sidecar.exists() or sidecar.is_symlink():
                try:
                    deps.append(
                        f"{dataset_id}/{sidecar.relative_to(dataset_dir).as_posix()}"
                    )
                except Exception:
                    deps.append(f"{dataset_id}/{sidecar.name}")

    # Extract task from filename for task-level root files
    task = None
    for token in stem.split("_"):
        if token.startswith("task-"):
            task = token.split("task-", 1)[1]
            break

    # Include task-level files at dataset root
    if task:
        for pattern in [
            f"task-{task}_events.json",
            f"task-{task}_eeg.json",
            f"task-{task}_channels.tsv",
            f"task-{task}_events.tsv",
        ]:
            task_file = dataset_dir / pattern
            if task_file.exists() or task_file.is_symlink():
                deps.append(f"{dataset_id}/{pattern}")

    return sorted(set(deps))


def _process_dataset_manual(dataset_id: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """Fallback manual parsing when eegdash library fails or is unavailable."""
    import csv
    import json as _json

    records = []
    # Find EEG files inside dataset_dir recursively
    eeg_extensions = [".bdf", ".edf", ".vhdr", ".set", ".fif", ".cnt", ".eeg"]

    # BIDS folders to exclude (not raw data)
    excluded_dirs = {"sourcedata", "derivatives", "code", "stimuli"}

    # collect files grouped by stem (canonical BIDS basename without ext)
    file_groups = {}
    for p in dataset_dir.rglob("*"):
        # Accept regular files and symlinks (git-annex will create symlinks to large data files)
        if not (p.is_file() or p.is_symlink()):
            continue

        # Skip files in excluded directories
        rel_parts = p.relative_to(dataset_dir).parts
        if any(part in excluded_dirs for part in rel_parts):
            continue

        ext = p.suffix.lower()
        if ext not in eeg_extensions:
            continue
        stem = p.name.rsplit(".", 1)[0]
        file_groups.setdefault(stem, []).append(p)

    # canonical extension priority for one record per stem
    ext_priority = [".bdf", ".edf", ".vhdr", ".set", ".eeg", ".fif", ".cnt"]
    chosen_files = []
    for stem, files in file_groups.items():
        # pick best candidate per ext priority
        chosen = None
        for ext in ext_priority:
            for f in files:
                if f.suffix.lower() == ext:
                    chosen = f
                    break
            if chosen:
                break
        if chosen:
            chosen_files.append(chosen)

    for p in chosen_files:
        file_name = p.name
        # find sidecar JSON
        json_path = p.with_suffix(p.suffix + ".json") if p.suffix else None
        if not json_path or (not json_path.exists() and not json_path.is_symlink()):
            # try with .json extension only (e.g., .vhdr -> .json)
            side_json = p.with_suffix(".json")
            if side_json.exists() or side_json.is_symlink():
                json_path = side_json
        eeg_json = None
        try:
            if json_path and json_path.exists():
                with json_path.open("r", encoding="utf-8") as fh:
                    eeg_json = _json.load(fh)
        except Exception:
            eeg_json = None

        # Build record
        # openneuro path: dataset_id + relative path from dataset_dir
        try:
            rel = p.relative_to(dataset_dir)
        except ValueError:
            rel = Path(p.name)
        openneuro_path = f"{dataset_id}/{rel.as_posix()}"

        # parse BIDS entities from path segments
        # expect structure: sub-XXX[/ses-XXX]/eeg/...
        parts = rel.parts
        subject = None
        session = None
        task = None
        run = None
        modality = "eeg"
        # find sub-xxx
        for seg in parts:
            if seg.startswith("sub-") and subject is None:
                subject = seg.split("sub-")[1]
            if seg.startswith("ses-") and session is None:
                session = seg.split("ses-")[1]
        # parse filename entities
        stem = p.stem
        # BIDS filename patterns contain keys like sub-01_task-PES_run-1
        for token in stem.split("_"):
            if token.startswith("task-"):
                task = token.split("task-")[1]
            if token.startswith("run-"):
                run = token.split("run-")[1]

        # extract fields
        sampling_frequency = None
        nchans = None
        ntimes = None
        if eeg_json:
            sampling_frequency = (
                eeg_json.get("SamplingFrequency")
                or eeg_json.get("samplingFrequency")
                or eeg_json.get("sfreq")
            )
            nchans = (
                eeg_json.get("EEGChannelCount")
                or eeg_json.get("n_channels")
                or eeg_json.get("nchans")
            )
            ntimes = (
                eeg_json.get("RecordingDuration")
                or eeg_json.get("Duration")
                or eeg_json.get("recordingDuration")
            )

        # find participants row
        participants_row = None
        tsv = dataset_dir / "participants.tsv"
        if tsv.exists() and subject is not None:
            try:
                with tsv.open("r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh, delimiter="\t")
                    for r in reader:
                        # participants.tsv might use participant_id or participant
                        found = False
                        # Normalize keys (strip BOM) and compare
                        normalized = {
                            k.strip().lstrip("\ufeff"): v for k, v in r.items()
                        }
                        for key in ("participant_id", "participant", "subject"):
                            if key in normalized and normalized[key] in (
                                f"{subject}",
                                f"sub-{subject}",
                            ):
                                participants_row = normalized
                                found = True
                                break
                        if found:
                            break
            except Exception:
                participants_row = None

        # Build bidsdependencies
        bidsdependencies = []
        # dataset-level files: deterministic order
        ds_files = ["dataset_description.json", "participants.tsv"]
        for fname in ds_files:
            fpath = dataset_dir / fname
            if fpath.exists() or fpath.is_symlink():
                bidsdependencies.append(f"{dataset_id}/{fname}")

        # file-level deps: deterministic set of same-stem sidecars
        # Keep only relevant sidecars directly related to the chosen data file `p`.
        bidsdependencies.extend(get_sidecar_dependencies(p, dataset_dir, dataset_id))

        record = {
            "data_name": f"{dataset_id}_{file_name}",
            "dataset": dataset_id,
            "bidspath": openneuro_path,
            "subject": subject,
            "task": task,
            "session": session,
            "run": run,
            "modality": modality,
            "sampling_frequency": sampling_frequency,
            "nchans": nchans,
            "ntimes": ntimes,
            "participant_tsv": participants_row,
            "eeg_json": eeg_json,
            "bidsdependencies": bidsdependencies,
        }
        # Normalize record
        record = normalize_record(record, dataset_dir)
        records.append(record)
    return records


def _try_int(v):
    try:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        s = str(v).strip()
        if s == "" or s.lower() in ("n/a", "na", "nan"):
            return None
        if s.isdigit():
            return int(s)
        # remove padded zeros
        if s.lstrip("0").isdigit():
            return int(s)
        try:
            f = float(s)
            return int(f) if f.is_integer() else f
        except Exception:
            return None
    except Exception:
        return None


def normalize_record(rec: Dict[str, Any], dataset_dir: Path) -> Dict[str, Any]:
    """Normalize fields of a generated metadata record.

    - normalize run to int
    - normalize subject to no 'sub-' prefix
    - normalize session to no 'ses-' prefix and to None when empty
    - normalize modality to lower-case
    - normalize participant_tsv fields (age->int, n/a => None)
    - normalize bidsdependencies to sorted unique list
    - normalize sampling_frequency/nchans to int/float when possible
    """
    r = dict(rec)
    # run
    r_run = r.get("run")
    if isinstance(r_run, str):
        r_run = r_run.strip()
    r_run_val = _try_int(r_run)
    r["run"] = r_run_val

    # subject
    subject = r.get("subject")
    if isinstance(subject, str) and subject.startswith("sub-"):
        r["subject"] = subject.split("sub-")[-1]

    # session
    session = r.get("session")
    if isinstance(session, str) and session.startswith("ses-"):
        session = session.split("ses-")[-1]
    if session == "" or session is None:
        r["session"] = None
    else:
        r["session"] = session

    # modality
    mod = r.get("modality")
    if isinstance(mod, str):
        r["modality"] = mod.strip().lower()

    # participant_tsv normalization
    pt = r.get("participant_tsv")
    if isinstance(pt, dict):
        norm_pt = {}
        for k, v in pt.items():
            k_norm = k.strip().lstrip("\ufeff")
            # numeric parsing for age
            if k_norm.lower() in ("age",):
                age_val = _try_int(v)
                norm_pt["age"] = age_val
            elif k_norm.lower() in ("gender", "sex"):
                norm_pt["sex"] = (
                    None
                    if (
                        v is None
                        or (
                            isinstance(v, str)
                            and v.strip().lower() in ("n/a", "nan", "none")
                        )
                    )
                    else v
                )
            elif k_norm.lower() in ("handedness", "hand"):
                norm_pt["hand"] = (
                    None
                    if (
                        v is None
                        or (
                            isinstance(v, str)
                            and v.strip().lower() in ("n/a", "nan", "none")
                        )
                    )
                    else v
                )
            else:
                norm_pt[k_norm] = (
                    None
                    if (
                        v is None
                        or (
                            isinstance(v, str)
                            and v.strip().lower() in ("n/a", "nan", "none")
                        )
                    )
                    else v
                )
        r["participant_tsv"] = norm_pt

    # sampling_frequency and nchans
    sf = r.get("sampling_frequency")
    if sf is not None:
        try:
            r["sampling_frequency"] = float(sf)
        except Exception:
            r["sampling_frequency"] = sf

    nc = r.get("nchans")
    if nc is not None:
        nc_i = _try_int(nc)
        r["nchans"] = nc_i

    # bidsdependencies -> unique & sort
    bd = r.get("bidsdependencies") or []
    if isinstance(bd, list):
        unique = sorted({str(x) for x in bd})
        r["bidsdependencies"] = unique

    # Fill channel metadata if missing using channels.tsv (look at parent folder)
    if (
        not r.get("channel_names")
        or not r.get("channel_types")
        or r.get("nchans") is None
    ):
        try:
            # parent path is included in bidspath: dataset_id/rel
            # we compute the actual path on disk by joining dataset dir with the relative part
            # derive bids path relative component (strip dataset_id/)
            bidspath = r.get("bidspath")
            if isinstance(bidspath, str) and "/" in bidspath:
                # e.g. ds004477/sub-001/eeg/sub-001_task-PES_eeg.bdf
                components = bidspath.split("/")
                rel_components = components[1:-1]
                parent_dir = (
                    Path(dataset_dir).joinpath(*rel_components)
                    if rel_components
                    else Path(dataset_dir)
                )
                channels_file = parent_dir / (Path(bidspath).stem + "_channels.tsv")
                # fallback: common channels filename in parent
                if not channels_file.exists():
                    # check at parent dir for any channels.tsv
                    parent_dir = (
                        Path(dataset_dir).joinpath(*rel_components)
                        if rel_components
                        else Path(dataset_dir)
                    )
                    if parent_dir.exists():
                        # find channel tsv files matching _channels.tsv in parent
                        for cand in parent_dir.iterdir():
                            if cand.name.endswith("_channels.tsv"):
                                channels_file = cand
                                break
                if channels_file.exists():
                    import csv as _csv

                    names = []
                    types = []
                    with channels_file.open("r", encoding="utf-8") as fh:
                        reader = _csv.DictReader(fh, delimiter="\t")
                        for row in reader:
                            # typical fields: name, type
                            name = (
                                row.get("name")
                                or row.get("channel_name")
                                or row.get("label")
                            )
                            ctype = (
                                row.get("type")
                                or row.get("channel_type")
                                or row.get("coil")
                            )
                            if name:
                                names.append(name)
                            if ctype:
                                types.append(ctype)
                    if names:
                        r["channel_names"] = names
                    if types:
                        r["channel_types"] = types
                    if r.get("nchans") in (None, ""):
                        r["nchans"] = len(names) if names else r.get("nchans")
        except Exception:
            # ignore channel failures
            pass

    # normalize bidspath to be dataset + relative path
    bidspath = r.get("bidspath")
    if isinstance(bidspath, str):
        # remove duplicate leading slashes
        r["bidspath"] = bidspath.lstrip("/")

    return r


def compare_records(
    gt_records: List[Dict[str, Any]], gen_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare ground truth to generated records. Return a structured diff summary."""
    # Index generated records by data_name
    gen_index = {r.get("data_name"): r for r in gen_records}
    gt_index = {r.get("data_name"): r for r in gt_records}

    missing_in_gen = [name for name in gt_index if name not in gen_index]
    extra_in_gen = [name for name in gen_index if name not in gt_index]

    mismatches = []
    # Compare fieldwise for matched names
    for name, gt in gt_index.items():
        gen = gen_index.get(name)
        if not gen:
            continue
        # Compare keys present in gt record; ignore ntimes if missing
        for key, gt_val in gt.items():
            if key == "ntimes":
                # allowed to be missing
                continue
            if key == "_id":
                # Skip DB-specific _id fields
                continue
            gen_val = gen.get(key)
            if key == "bidsdependencies":
                # Compare as sets (order can vary)
                gt_set = set(gt_val or [])
                gen_set = set(gen_val or [])

                # The library doesn't include the primary data file in bidsdependencies,
                # but some GT records do. Tolerate this difference by ignoring the primary
                # data file when comparing.
                bidspath = gt.get("bidspath", "")
                primary_data_file = (
                    bidspath  # e.g., "ds004477/sub-001/eeg/sub-001_task-PES_eeg.bdf"
                )

                # Remove primary data file from both sets for comparison
                gt_set.discard(primary_data_file)
                gen_set.discard(primary_data_file)

                if gt_set != gen_set:
                    mismatches.append(
                        {
                            "data_name": name,
                            "field": key,
                            "ground_truth": gt_val,
                            "generated": gen_val,
                        }
                    )
                continue
            if key == "session":
                # Treat empty string and None as equivalent for session
                if gt_val in (None, "") and gen_val in (None, ""):
                    continue
                if str(gt_val) == str(gen_val):
                    continue
                mismatches.append(
                    {
                        "data_name": name,
                        "field": key,
                        "ground_truth": gt_val,
                        "generated": gen_val,
                    }
                )
                continue
            if key == "run":
                # Compare as integers (handle string vs int)
                try:
                    gt_int = int(gt_val) if gt_val is not None else None
                    gen_int = int(gen_val) if gen_val is not None else None
                    if gt_int == gen_int:
                        continue
                except (ValueError, TypeError):
                    # Fallback to string comparison
                    if str(gt_val) == str(gen_val):
                        continue
                mismatches.append(
                    {
                        "data_name": name,
                        "field": key,
                        "ground_truth": gt_val,
                        "generated": gen_val,
                    }
                )
                continue
            if key == "modality":
                # Compare case-insensitive
                if (
                    isinstance(gt_val, str)
                    and isinstance(gen_val, str)
                    and gt_val.lower() == gen_val.lower()
                ):
                    continue
                if gt_val == gen_val:
                    continue
                mismatches.append(
                    {
                        "data_name": name,
                        "field": key,
                        "ground_truth": gt_val,
                        "generated": gen_val,
                    }
                )
                continue
            if key == "eeg_json":
                if isinstance(gt_val, dict) and isinstance(gen_val, dict):
                    # Ignore RecordingDuration differences
                    gtv = dict(gt_val)
                    gnv = dict(gen_val)
                    gtv.pop("RecordingDuration", None)
                    gnv.pop("RecordingDuration", None)
                    if gtv != gnv:
                        mismatches.append(
                            {
                                "data_name": name,
                                "field": key,
                                "ground_truth": gt_val,
                                "generated": gen_val,
                            }
                        )
                else:
                    if gt_val != gen_val:
                        mismatches.append(
                            {
                                "data_name": name,
                                "field": key,
                                "ground_truth": gt_val,
                                "generated": gen_val,
                            }
                        )
                continue
            if key == "participant_tsv":
                # Compare only a subset of participant fields if present
                if isinstance(gt_val, dict) and isinstance(gen_val, dict):
                    # Ground truth may use different keys: Gender/Age/Handedness
                    gt_to_norm = {}
                    for k, v in gt_val.items():
                        k_norm = k.strip().lower()
                        if k_norm in ("gender", "sex"):
                            gt_to_norm["sex"] = v
                        if k_norm in ("age",):
                            gt_to_norm["age"] = v
                        if k_norm in ("handedness", "hand"):
                            gt_to_norm["hand"] = v
                    for pkey in ("age", "sex", "hand"):
                        gtv = gt_val.get(pkey)
                        gnv = gen_val.get(pkey)
                        gtv = gt_to_norm.get(pkey, gtv)
                        # normalize numeric ages
                        try:
                            gnvn = (
                                int(gnv)
                                if gnv is not None
                                and str(gnv).strip() != ""
                                and str(gnv).lower() != "n/a"
                                else None
                            )
                        except Exception:
                            gnvn = gnv
                        try:
                            # Handle MongoDB {"$numberDouble": "NaN"} format
                            if isinstance(gtv, dict) and "$numberDouble" in gtv:
                                v = gtv["$numberDouble"]
                                try:
                                    gtvn = int(float(v))
                                except Exception:
                                    gtvn = None
                            else:
                                gtvn = (
                                    int(gtv)
                                    if gtv is not None and str(gtv).strip() != ""
                                    else None
                                )
                        except Exception:
                            gtvn = gtv
                        if gtvn != gnvn:
                            mismatches.append(
                                {
                                    "data_name": name,
                                    "field": f"participant_tsv.{pkey}",
                                    "ground_truth": gtv,
                                    "generated": gnv,
                                }
                            )
                else:
                    if gt_val != gen_val:
                        mismatches.append(
                            {
                                "data_name": name,
                                "field": key,
                                "ground_truth": gt_val,
                                "generated": gen_val,
                            }
                        )
                continue
            # Default compare
            if gen_val != gt_val:
                mismatches.append(
                    {
                        "data_name": name,
                        "field": key,
                        "ground_truth": gt_val,
                        "generated": gen_val,
                    }
                )

    return {
        "missing_in_generated": missing_in_gen,
        "extra_in_generated": extra_in_gen,
        "mismatches": mismatches,
        "generated_count": len(gen_records),
        "ground_truth_count": len(gt_records),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Test OpenNeuro digestion for one dataset"
    )
    parser.add_argument("dataset_id", nargs="?", default="ds004477")
    parser.add_argument("--dataset-dir", type=Path, required=False)
    parser.add_argument(
        "--ground-truth", type=Path, default=Path("eegdash_ds004477_extended.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("digestion_output"))
    parser.add_argument("--all", action="store_true", help="Process all ds* datasets")

    args = parser.parse_args(argv)

    cloned_dir = Path("data/cloned_all")
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        dataset_dir = cloned_dir / args.dataset_id

    # Load ground truth
    if not args.ground_truth.exists():
        print(f"Ground truth file not found: {args.ground_truth}")
        return 2

    with args.ground_truth.open("r", encoding="utf-8") as fh:
        gt_data = json.load(fh)

    # Filter ground truth for dataset
    gt_records = [r for r in gt_data if r.get("dataset") == args.dataset_id]
    if not gt_records:
        print(f"No ground truth records found for {args.dataset_id}")

    out_base = args.output_dir
    out_base.mkdir(parents=True, exist_ok=True)

    if args.all:
        # Process all ds* directories under cloned_dir
        ds_dirs = [
            p for p in cloned_dir.iterdir() if p.is_dir() and p.name.startswith("ds")
        ]
        all_summary = []
        for ds_dir in ds_dirs:
            ds_id = ds_dir.name
            print(f"Processing {ds_id}...")
            gen_records = process_dataset(ds_id, ds_dir)
            # Save generated manifest
            ds_out = out_base / ds_id
            ds_out.mkdir(parents=True, exist_ok=True)
            gen_path = ds_out / f"{ds_id}_generated.json"
            with gen_path.open("w", encoding="utf-8") as fh:
                json.dump(gen_records, fh, indent=2, default=str)
            summary = {
                "dataset_id": ds_id,
                "generated_count": len(gen_records),
            }
            all_summary.append(summary)
        # Save overall summary
        with (out_base / "all_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(all_summary, fh, indent=2)
        print(f"Processed {len(ds_dirs)} datasets. Results under {out_base}")
        return 0

    print(f"Processing dataset: {args.dataset_id} @ {dataset_dir}")
    gen_records = process_dataset(args.dataset_id, dataset_dir)

    ds_out = out_base / args.dataset_id
    ds_out.mkdir(parents=True, exist_ok=True)
    gen_path = ds_out / f"{args.dataset_id}_generated.json"

    with gen_path.open("w", encoding="utf-8") as fh:
        json.dump(gen_records, fh, indent=2, default=str)

    summary = compare_records(gt_records, gen_records)

    # Save comparison summary
    with (ds_out / f"{args.dataset_id}_compare_summary.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("Comparison summary:")
    print(f"  Ground truth count: {summary['ground_truth_count']}")
    print(f"  Generated count: {summary['generated_count']}")
    print(f"  Missing in generated: {len(summary['missing_in_generated'])}")
    print(f"  Extra in generated: {len(summary['extra_in_generated'])}")
    print(f"  Field mismatches: {len(summary['mismatches'])}")

    if summary["missing_in_generated"]:
        print("Missing data_names:")
        for m in summary["missing_in_generated"]:
            print(f"  - {m}")

    if summary["mismatches"]:
        print("Mismatches (some examples):")
        for mm in summary["mismatches"][:10]:
            print(mm)

    print(f"Generated manifest saved to: {gen_path}")
    print(
        f"Comparison summary saved to: {ds_out / f'{args.dataset_id}_compare_summary.json'}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
