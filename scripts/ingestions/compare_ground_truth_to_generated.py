"""Compare ground truth JSONs in `ground_true` with generated metadata in `digestion_output`.

Produces per-dataset compare summary JSON and an overall report printed to console.

Usage:
  PYTHONPATH=$PWD python3 scripts/ingestions/compare_ground_truth_to_generated.py

"""

import json
from pathlib import Path
from typing import Any, Dict, List

GROUND_TRUE_DIR = Path("ground_true")
DIGESTION_DIR = Path("digestion_output")
OUT_DIR = DIGESTION_DIR


def load_ground_truth_files() -> List[Path]:
    files = sorted(GROUND_TRUE_DIR.glob("*.json"))
    return files


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compare_records(
    gt_records: List[Dict[str, Any]], gen_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Lightweight comparison logic: reuses rules from test_digest_openneuro

    - Skip _id
    - Ignore ntimes being missing
    - Compare bidsdependencies as sets (ignoring primary data file)
    - Compare session/run as string/int equivalent
    - Compare modality case-insensitively
    - Compare participant_tsv by age/sex/hand (normalize age)
    """
    gen_index = {r.get("data_name"): r for r in gen_records}
    gt_index = {r.get("data_name"): r for r in gt_records}

    missing_in_gen = [name for name in gt_index if name not in gen_index]
    extra_in_gen = [name for name in gen_index if name not in gt_index]

    mismatches = []
    for name, gt in gt_index.items():
        gen = gen_index.get(name)
        if not gen:
            continue
        for key, gt_val in gt.items():
            if key == "ntimes":
                continue
            if key == "_id":
                continue
            gen_val = gen.get(key)

            if key == "bidsdependencies":
                # Compare as sets, ignoring primary data file difference
                gt_set = set(gt_val or [])
                gen_set = set(gen_val or [])

                # The library doesn't include the primary data file in bidsdependencies,
                # but some GT records do. Tolerate this difference.
                bidspath = gt.get("bidspath", "")
                gt_set.discard(bidspath)
                gen_set.discard(bidspath)

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
                # Treat empty string and None as equivalent
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
                # Compare case-insensitively
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

            if key == "participant_tsv":
                if isinstance(gt_val, dict) and isinstance(gen_val, dict):
                    # Map GT keys to normalized keys (gender->sex, handedness->hand)
                    gt_normalized = {}
                    for k, v in gt_val.items():
                        k_lower = k.lower().strip()
                        if k_lower in ("gender", "sex"):
                            gt_normalized["sex"] = v
                        elif k_lower in ("handedness", "hand"):
                            gt_normalized["hand"] = v
                        elif k_lower == "age":
                            gt_normalized["age"] = v

                    for pkey in ("age", "sex", "hand"):
                        gtv = gt_normalized.get(pkey)
                        gnv = gen_val.get(pkey)
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
                            # Some ground truth uses nested structures like {"$numberDouble": "NaN"}; we try to coerce
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
                                    else gtv
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


def main():
    gt_files = load_ground_truth_files()
    overall = {
        "datasets_compared": 0,
        "datasets_with_mismatches": 0,
        "datasets_with_missing": 0,
        "dataset_summaries": [],
    }

    for gt_path in gt_files:
        try:
            gt_json = load_json(gt_path)
        except Exception as exc:
            print(f"Failed to load ground truth {gt_path}: {exc}")
            continue
        # ground truth file name pattern e.g. eegdash_ds004477_records.json
        name = gt_path.name
        if not name.startswith("eegdash_") or not name.endswith("_records.json"):
            continue
        dataset_id = name.split("eegdash_")[1].split("_records.json")[0]

        gen_path = DIGESTION_DIR / dataset_id / f"{dataset_id}_generated.json"
        if not gen_path.exists():
            overall["datasets_with_missing"] += 1
            msg = f"Generated output not found for {dataset_id} ({gen_path})"
            print(msg)
            overall["dataset_summaries"].append(
                {"dataset_id": dataset_id, "error": "generated_missing"}
            )
            continue

        gt_records = gt_json
        gen_records = load_json(gen_path)
        summary = compare_records(gt_records, gen_records)
        # compute high-level stats
        mismatches = summary.get("mismatches", [])
        missing = summary.get("missing_in_generated", [])
        if mismatches or missing:
            overall["datasets_with_mismatches"] += 1
        overall["datasets_compared"] += 1
        out_dir = OUT_DIR / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{dataset_id}_gt_compare.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(summary, fh, indent=2, default=str)
        overall_summary = {
            "dataset_id": dataset_id,
            "ground_truth_count": summary.get("ground_truth_count"),
            "generated_count": summary.get("generated_count"),
            "mismatch_count": len(summary.get("mismatches", [])),
            "missing_count": len(summary.get("missing_in_generated", [])),
        }
        overall["dataset_summaries"].append(overall_summary)
        print(
            f"Compared {dataset_id}: computed {overall_summary['generated_count']} vs ground-truth {overall_summary['ground_truth_count']} (mismatches: {overall_summary['mismatch_count']}, missing: {overall_summary['missing_count']})"
        )

    # Save overall summary
    overall_path = OUT_DIR / "ground_truth_overall_compare.json"
    with overall_path.open("w", encoding="utf-8") as fh:
        json.dump(overall, fh, indent=2, default=str)
    print("\nOverall summary:")
    print(json.dumps(overall, indent=2, default=str))
    print(f"Per-dataset compare summaries written to {OUT_DIR}/*_gt_compare.json")


if __name__ == "__main__":
    main()
