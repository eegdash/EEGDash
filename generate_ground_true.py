from time import sleep
import pandas as pd
from eegdash import EEGDash

from bson import json_util

df = pd.read_csv(
    "/Users/baristim/Projects/EEGDash-2/eegdash/dataset/dataset_summary.csv"
)
eegdash = EEGDash()
for _, row in df.iterrows():
    dataset = row["dataset"]
    docs = eegdash.find(dataset=dataset)  # returns list[dict]

    # # Dump as extended JSON (includes $oid, $date, $numberDouble, etc.)
    with open(
        "/Users/baristim/Projects/EEGDash-2/ground_true/"
        + f"eegdash_{dataset}_records.json",
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write(json_util.dumps(docs, indent=2))

    print(f"Saved {len(docs)} records for dataset {dataset}")
    sleep(1)  # to avoid overwhelming the server
