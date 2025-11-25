"""Aggregate compare summary files in digestion_output to summarize mismatch types.

Usage:
  python3 scripts/ingestions/aggregate_gt_comparison_stats.py

Outputs a human-friendly report and a JSON summary at `digestion_output/gt_aggregate_stats.json`.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("digestion_output")

compare_files = sorted(BASE.glob("*/**/*_gt_compare.json"))

field_counter_overall = Counter()
per_dataset_stats = []

for cf in compare_files:
    try:
        with cf.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        continue
    dataset = cf.parent.name
    mismatches = data.get("mismatches", [])
    miss_count = len(mismatches)
    missing_in_generated = len(data.get("missing_in_generated", []))
    extra_in_generated = len(data.get("extra_in_generated", []))
    # gather field counter
    fc = Counter()
    examples = defaultdict(list)
    for mm in mismatches:
        field = mm.get("field")
        fc[field] += 1
        if len(examples[field]) < 2:
            examples[field].append(
                {
                    "data_name": mm.get("data_name"),
                    "gt": mm.get("ground_truth"),
                    "gen": mm.get("generated"),
                }
            )
    for k, v in fc.items():
        field_counter_overall[k] += v
    # get top 3 fields
    top_fields = fc.most_common(5)
    per_dataset_stats.append(
        {
            "dataset": dataset,
            "ground_truth_count": data.get("ground_truth_count"),
            "generated_count": data.get("generated_count"),
            "missing_in_generated": missing_in_generated,
            "extra_in_generated": extra_in_generated,
            "mismatches_total": miss_count,
            "top_mismatch_fields": [
                {"field": f, "count": c, "examples": examples.get(f, [])}
                for f, c in top_fields
            ],
        }
    )

# print summary
print("Overall mismatch field counts (top 10):")
for field, count in field_counter_overall.most_common(10):
    print(f"  {field}: {count}")

out_summary = {
    "dataset_count": len(per_dataset_stats),
    "field_counts": field_counter_overall,
    "datasets": per_dataset_stats,
}

# write summary json
with (BASE / "gt_aggregate_stats.json").open("w", encoding="utf-8") as fh:
    # Convert counters to dicts
    out = {
        "dataset_count": out_summary["dataset_count"],
        "field_counts": dict(out_summary["field_counts"]),
        "datasets": out_summary["datasets"],
    }
    json.dump(out, fh, indent=2, default=str)

print("\nWrote: digestion_output/gt_aggregate_stats.json")
