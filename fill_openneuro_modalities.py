#!/usr/bin/env python3
"""
Fill `record_modality` from OpenNeuro for each dataset row.

Usage:
  python fill_openneuro_modalities.py path/to/dataset_summary.csv \
      --id-col openneuro_id \
      --out path/to/dataset_summary.with_modalities.csv \
      --cache .openneuro_mod_cache.json

Notes:
- OPENNEURO_API_KEY (optional) for private/rate-limited datasets.
- Auto-detects delimiter (.csv => ',', .tsv => '\t') unless --sep provided.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://openneuro.org/crn/graphql"
TIMEOUT = 30
ID_RE = re.compile(r"^ds\d{6}$", re.IGNORECASE)

FOLDER_TO_MODALITY = {
    "eeg": "EEG",
    "ieeg": "iEEG",
    "meg": "MEG",
    "pet": "PET",
    "func": "fMRI",
    "dwi": "DWI",
    "asl": "ASL",
    "anat": "MRI",  # structural
    "fmap": "FieldMap",
}


def detect_sep(path: Path, given: str | None) -> str:
    if given:
        return given
    if path.suffix.lower() == ".tsv":
        return "\t"
    return ","


def load_cache(cache_path: Path) -> dict:
    if cache_path.is_file():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache: dict, cache_path: Path):
    try:
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    except Exception:
        pass


def gql(query: str, variables: dict, session: requests.Session) -> dict:
    headers = {"Content-Type": "application/json"}
    if k := os.getenv("OPENNEURO_API_KEY"):
        headers["Authorization"] = f"Bearer {k}"
    r = session.post(
        API,
        headers=headers,
        json={"query": query, "variables": variables},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"]))
    return data["data"]


def normalize_modalities(mods) -> list[str]:
    if not mods:
        return []
    out = []
    for m in mods:
        s = str(m or "").strip()
        if not s:
            continue
        su = s.upper()
        su = {"FMRI": "fMRI"}.get(su, su)  # pretty-case fMRI
        if su not in out:
            out.append(su)
    return out


def get_latest_snapshot_and_modalities(dsid: str, session: requests.Session):
    q = """
    query($id:ID!){
      dataset(id:$id){
        latestSnapshot{
          tag
          summary{ modalities }
        }
      }
    }
    """
    d = gql(q, {"id": dsid}, session)
    snap = (d.get("dataset") or {}).get("latestSnapshot") or {}
    tag = snap.get("tag")
    mods = normalize_modalities(((snap.get("summary") or {}).get("modalities")) or [])
    return tag, mods


def list_top_files(dsid: str, tag: str, session: requests.Session):
    q = """
    query($id:ID!,$tag:String!){
      snapshot(datasetId:$id, tag:$tag){
        files{ filename directory }
      }
    }
    """
    d = gql(q, {"id": dsid, "tag": tag}, session)
    return ((d.get("snapshot") or {}).get("files")) or []


def infer_from_folders(files: list[dict]) -> list[str]:
    found = set()
    for f in files:
        if f.get("directory"):
            name = (f.get("filename") or "").lower()
            if name in FOLDER_TO_MODALITY:
                found.add(FOLDER_TO_MODALITY[name])
    # prioritize electrophysiology first
    ordered = [m for m in ["EEG", "MEG", "iEEG"] if m in found]
    ordered += sorted(found - set(ordered))
    return ordered


def fetch_modalities_for_id(
    dsid: str, session: requests.Session, retries: int = 2, backoff: float = 1.5
) -> list[str]:
    err = None
    for attempt in range(retries + 1):
        try:
            tag, mods = get_latest_snapshot_and_modalities(dsid, session)
            if mods:
                return mods
            if not tag:
                raise RuntimeError("No latestSnapshot tag")
            files = list_top_files(dsid, tag, session)
            inferred = infer_from_folders(files)
            if inferred:
                return inferred
            raise RuntimeError(
                "No modalities in summary and none inferred from folders"
            )
        except Exception as e:
            err = e
            if attempt < retries:
                time.sleep((backoff**attempt))
                continue
            raise err


def main():
    ap = argparse.ArgumentParser(
        description="Fill record_modality from OpenNeuro GraphQL for each dataset row."
    )
    ap.add_argument("table", type=Path, help="Input CSV/TSV table")
    ap.add_argument(
        "--id-col",
        default="dataset",
        help="Column with OpenNeuro ID (default: dataset)",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="Output path (default: overwrite input)"
    )
    ap.add_argument(
        "--sep", default=None, help="Column separator (auto by extension if omitted)"
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path(".openneuro_mod_cache.json"),
        help="Path to cache JSON",
    )
    ap.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests (optional)",
    )
    args = ap.parse_args()

    sep = detect_sep(args.table, args.sep)
    df = pd.read_csv(args.table, sep=sep, dtype=str, keep_default_na=False)

    if args.id_col not in df.columns:
        raise SystemExit(
            f"Column '{args.id_col}' not found. Available: {list(df.columns)}"
        )

    # Ensure output column exists
    if "record_modality" not in df.columns:
        df["record_modality"] = ""

    cache = load_cache(args.cache)
    session = requests.Session()

    # Iterate rows
    for idx, dsid in enumerate(df[args.id_col]):
        dsid = (dsid or "").strip()
        if not dsid:
            continue
        # normalize to lowercase 'ds######'
        m = ID_RE.match(dsid)
        if not m:
            continue

        key = dsid.lower()
        if key in cache:
            mods = cache[key]
        else:
            try:
                mods = fetch_modalities_for_id(key, session)
                cache[key] = mods
                if args.rate > 0:
                    time.sleep(args.rate)
            except Exception as e:
                # keep an empty entry, but store error note in cache for visibility
                cache[key] = {"error": str(e)}
                mods = []

        # Only write modalities if we actually have a list
        if isinstance(mods, list) and mods:
            df.at[idx, "record_modality"] = ", ".join(mods)

    # Save cache and table
    save_cache(cache, args.cache)
    out_path = args.out if args.out else args.table
    df.to_csv(out_path, sep=sep, index=False)
    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
