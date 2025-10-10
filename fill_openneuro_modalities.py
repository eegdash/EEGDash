#!/usr/bin/env python3
"""
Batch-fill `record_modality` for OpenNeuro datasets.

What it does
------------
1) Sends ONE (or a few, chunked) GraphQL request(s) using field aliases:
     dataset(id){ latestSnapshot{ tag summary{ modalities } } }
   -> Gives you curated modalities when available.
2) For rows that still have no modality but DO have a snapshot `tag`,
   it sends another batched request to list top-level files:
     snapshot(datasetId, tag){ files{ filename directory } }
   -> Infers modality from BIDS root folders (eeg/, meg/, ieeg/, ...).
3) Writes the resulting modalities into the `record_modality` column.

Usage
-----
python fill_record_modality_batched.py path/to/table.csv \
  --id-col dataset \
  --out table.with_modalities.csv \
  --chunk-size 40 \
  --cache .openneuro_mod_cache.json

Notes
-----
- For private or rate-limited access: export OPENNEURO_API_KEY=...
- Auto-detects delimiter by extension (.csv => ',', .tsv => '\\t'), or use --sep.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

API = "https://openneuro.org/crn/graphql"
TIMEOUT = 30
ID_RE = re.compile(r"^ds\d{6}$", re.IGNORECASE)
ALIAS_SAFE = re.compile(r"[^A-Za-z0-9_]")


# ---------- Helpers ----------


def detect_sep(path: Path, given: str | None) -> str:
    if given:
        return given
    return "\t" if path.suffix.lower() == ".tsv" else ","


def normalize_modalities(mods) -> list[str]:
    if not mods:
        return []
    out: list[str] = []
    for m in mods:
        s = str(m or "").strip()
        if not s:
            continue
        su = s.upper()
        su = {"FMRI": "fMRI"}.get(su, su)  # pretty-case fMRI
        if su not in out:
            out.append(su)
    return out


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
    "perf": "PERF",
}


def infer_from_folders(files: list[dict]) -> list[str]:
    found = set()
    for f in files or []:
        if f.get("directory"):
            name = (f.get("filename") or "").lower()
            if name in FOLDER_TO_MODALITY:
                found.add(FOLDER_TO_MODALITY[name])
    ordered = [m for m in ["EEG", "MEG", "iEEG"] if m in found]  # electrophys first
    ordered += sorted(found - set(ordered))
    return ordered


def _alias_for_id(dsid: str) -> str:
    # GraphQL alias must start with a letter; sanitize to be safe.
    return "a_" + ALIAS_SAFE.sub("_", dsid.lower())


def gql(
    query: str, variables: dict | None = None, session: requests.Session | None = None
) -> dict:
    session = session or requests.Session()
    headers = {"Content-Type": "application/json"}
    if k := os.getenv("OPENNEURO_API_KEY"):
        headers["Authorization"] = f"Bearer {k}"
    r = session.post(
        API,
        headers=headers,
        json={"query": query, "variables": variables or {}},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return data["data"]


# ---------- Batched queries ----------


def batch_fetch_snapshot_info(
    ids: list[str], chunk_size: int = 40, session: requests.Session | None = None
) -> dict[str, dict]:
    """
    Returns { id: {"tag": <str|None>, "modalities": [..]} } using as few requests as possible.
    """
    session = session or requests.Session()
    out: dict[str, dict] = {}
    for start in range(0, len(ids), chunk_size):
        batch = ids[start : start + chunk_size]
        fields = []
        for dsid in batch:
            alias = _alias_for_id(dsid)
            fields.append(
                f'''{alias}: dataset(id:"{dsid}") {{
                      latestSnapshot {{
                        tag
                        summary {{ modalities }}
                      }}
                    }}'''
            )
        query = "query Q { " + "\n".join(fields) + " }"
        data = gql(query, session=session)
        for dsid in batch:
            alias = _alias_for_id(dsid)
            node = data.get(alias) or {}
            snap = node.get("latestSnapshot") or {}
            tag = snap.get("tag")
            mods = normalize_modalities(
                ((snap.get("summary") or {}).get("modalities")) or []
            )
            out[dsid] = {"tag": tag, "modalities": mods}
    return out


def batch_list_files_for_snapshots(
    id_to_tag: dict[str, str],
    chunk_size: int = 35,
    session: requests.Session | None = None,
) -> dict[str, list[dict]]:
    """
    Given mapping {id: tag}, returns {id: files[]} for top-level snapshot files.
    """
    session = session or requests.Session()
    out: dict[str, list[dict]] = {}
    items = list(id_to_tag.items())
    for start in range(0, len(items), chunk_size):
        part = items[start : start + chunk_size]
        fields = []
        for dsid, tag in part:
            alias = _alias_for_id(f"{dsid}_{tag}")
            fields.append(
                f'''{alias}: snapshot(datasetId:"{dsid}", tag:"{tag}") {{
                      files {{ filename directory }}
                    }}'''
            )
        query = "query F { " + "\n".join(fields) + " }"
        data = gql(query, session=session)
        for dsid, tag in part:
            alias = _alias_for_id(f"{dsid}_{tag}")
            node = data.get(alias) or {}
            files = node.get("files") or []
            out[dsid] = files
    return out


# ---------- Cache ----------


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


# ---------- Main pipeline ----------


def main():
    ap = argparse.ArgumentParser(
        description="Batch fill record_modality from OpenNeuro."
    )
    ap.add_argument("table", type=Path, help="Input CSV/TSV file")
    ap.add_argument(
        "--id-col",
        default="openneuro_id",
        help="Column with OpenNeuro dataset IDs (default: openneuro_id)",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="Output path (default: overwrite input)"
    )
    ap.add_argument(
        "--sep", default=None, help="Separator (auto from extension if omitted)"
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=40,
        help="Max IDs per GraphQL batch (default: 40)",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path(".openneuro_mod_cache.json"),
        help="Cache JSON path",
    )
    args = ap.parse_args()

    sep = detect_sep(args.table, args.sep)
    df = pd.read_csv(args.table, sep=sep, dtype=str, keep_default_na=False)

    if args.id_col not in df.columns:
        raise SystemExit(
            f"Column '{args.id_col}' not found. Available: {list(df.columns)}"
        )

    if "record_modality" not in df.columns:
        df["record_modality"] = ""

    # Collect IDs to resolve
    ids: list[str] = []
    for v in df[args.id_col].astype(str).tolist():
        s = (v or "").strip()
        if s and ID_RE.match(s):
            ids.append(s.lower())

    if not ids:
        out_path = args.out if args.out else args.table
        df.to_csv(out_path, sep=sep, index=False)
        print(f"No valid OpenNeuro IDs found. Wrote: {out_path}")
        return

    # Load cache
    cache = load_cache(args.cache)

    # Split: which need querying (not in cache as a list of modalities)
    ids_to_query = [i for i in ids if not (isinstance(cache.get(i), list) and cache[i])]

    session = requests.Session()

    # 1) Batched snapshot info (tag + curated modalities)
    if ids_to_query:
        snap_info = batch_fetch_snapshot_info(
            ids_to_query, chunk_size=args.chunk_size, session=session
        )
        for dsid, info in snap_info.items():
            if info.get("modalities"):
                cache[dsid] = info["modalities"]  # resolved
            else:
                # Keep tag for possible folder inference
                cache.setdefault(dsid, {})
                if isinstance(cache[dsid], dict):
                    cache[dsid]["tag"] = info.get("tag")

    # 2) Folder inference for those still empty but with a tag
    pending_for_folders = {
        k: (cache[k]["tag"])
        for k in ids
        if (
            isinstance(cache.get(k), dict)
            and cache[k].get("tag")
            and not isinstance(cache.get(k), list)
        )
    }
    if pending_for_folders:
        files_map = batch_list_files_for_snapshots(
            pending_for_folders,
            chunk_size=max(10, args.chunk_size - 5),
            session=session,
        )
        for dsid, files in files_map.items():
            inferred = infer_from_folders(files)
            if inferred:
                cache[dsid] = inferred
            else:
                # mark as unknown to avoid re-querying next run
                cache[dsid] = []

    # Write cache
    save_cache(cache, args.cache)

    # 3) Write column back
    def _mods_for(idv: str) -> str:
        key = (idv or "").strip().lower()
        v = cache.get(key)
        if isinstance(v, list) and v:
            return ", ".join(v)
        return ""  # leave blank if unknown

    df["record_modality"] = df[args.id_col].map(_mods_for)

    out_path = args.out if args.out else args.table
    df.to_csv(out_path, sep=sep, index=False)
    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
