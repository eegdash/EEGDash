"""Fetch NEMAR datasets via data.nemar.org (replaces GitHub org scan).

Catalogue:    GET /?format=json
Per-dataset:  GET /{id}/metadata.json  (neuroschema v0.3 dataset doc)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _http import get_client, request_json  # noqa: E402
from _serialize import save_datasets_deterministically, setup_paths  # noqa: E402

setup_paths()
from eegdash.schemas import Dataset, create_dataset  # noqa: E402

BASE_URL = "https://data.nemar.org"


# ---- data.nemar.org client ---------------------------------------------


class NemarApiError(RuntimeError):
    pass


def _get(url: str, *, timeout: float = 30.0, retries: int = 3):
    payload, resp = request_json(
        "get", url, timeout=timeout, retries=retries, client=get_client()
    )
    if resp is not None and resp.status_code == 404:
        return None
    if resp is None or resp.status_code != 200:
        raise NemarApiError(f"{url} -> {resp.status_code if resp else 'no-response'}")
    return payload


def iter_catalogue(
    *,
    base_url: str = BASE_URL,
    id_prefixes: tuple[str, ...] = ("nm", "on"),
    skip_unpublished: bool = True,
    timeout: float = 30.0,
    retries: int = 3,
) -> Iterator[dict]:
    payload = _get(
        f"{base_url.rstrip('/')}/?format=json", timeout=timeout, retries=retries
    )
    if not isinstance(payload, dict) or "datasets" not in payload:
        raise NemarApiError("catalogue missing 'datasets' key")
    for entry in payload["datasets"] or []:
        dataset_id = str(entry.get("id") or "")
        if not dataset_id or (id_prefixes and not dataset_id.startswith(id_prefixes)):
            continue
        if skip_unpublished and not entry.get("latest"):
            continue
        yield entry


def fetch_metadata(
    dataset_id: str,
    *,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
    retries: int = 3,
) -> dict | None:
    return _get(
        f"{base_url.rstrip('/')}/{dataset_id}/metadata.json",
        timeout=timeout,
        retries=retries,
    )


# ---- neuroschema -> EEGDash Dataset mapping ----------------------------


def _person_name(a) -> str:
    if isinstance(a, str):
        return a.strip()
    if not isinstance(a, dict):
        return ""
    name = (a.get("name") or "").strip()
    if name:
        return name
    return " ".join(
        p for p in (a.get("given_name", ""), a.get("family_name", "")) if p
    ).strip()


def _funding_lines(funding) -> list[str]:
    out = []
    for f in funding or []:
        if isinstance(f, str):
            out.append(f.strip())
            continue
        if not isinstance(f, dict):
            continue
        funder = (f.get("funder_name") or "").strip()
        if not funder:
            continue
        pieces = [funder, f.get("award_number") or "", f.get("award_title") or ""]
        out.append(" - ".join(p for p in pieces if p))
    return out


def _paper_doi(related) -> str | None:
    for r in related or []:
        if not isinstance(r, dict) or (r.get("identifier_type") or "").upper() != "DOI":
            continue
        rel = (r.get("relation_type") or "").strip()
        if rel and rel not in {"IsSupplementTo", "IsDescribedBy", "References"}:
            continue
        doi = (r.get("identifier") or "").strip()
        if doi:
            return doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return None


def _ages(demographics) -> list[int]:
    d = demographics or {}
    return [
        int(x) for x in (d.get("age_min"), d.get("age_max")) if isinstance(x, (int, float))
    ]


def build_dataset_document(metadata: dict, *, catalogue_entry: dict) -> Dataset:
    dataset_id = str(metadata.get("dataset_id") or catalogue_entry.get("id") or "")
    ext = metadata.get("external_links") or {}
    prov = metadata.get("provenance") or {}
    demo = metadata.get("demographics") or {}
    summary = metadata.get("data_summary") or {}
    authors = metadata.get("authors") or []
    modality = metadata.get("recording_modality") or ["EEG"]

    return create_dataset(
        dataset_id=dataset_id,
        name=metadata.get("name") or catalogue_entry.get("title") or dataset_id,
        source="nemar",
        recording_modality=[str(m).lower() for m in modality],
        bids_version=metadata.get("bids_version"),
        license=metadata.get("license"),
        authors=[n for n in (_person_name(a) for a in authors) if n],
        funding=_funding_lines(metadata.get("funding")),
        dataset_doi=ext.get("dataset_doi") or catalogue_entry.get("doi"),
        associated_paper_doi=_paper_doi(metadata.get("related_identifiers")),
        tasks=metadata.get("tasks") or None,
        sessions=metadata.get("sessions") or None,
        total_files=summary.get("total_files"),
        size_bytes=summary.get("size_bytes"),
        subjects_count=demo.get("subjects_count"),
        ages=_ages(demo),
        sex_distribution=demo.get("sex_distribution"),
        species=demo.get("species") or "Human",
        source_url=f"https://data.nemar.org{catalogue_entry.get('browse_url') or '/' + dataset_id + '/'}",
        github_url=ext.get("github_url"),
        senior_author=_person_name(authors[-1]) if authors else None,
        dataset_modified_at=prov.get("publish_date"),
    )


# ---- Stage 1 driver ----------------------------------------------------


def fetch_datasets(
    *,
    base_url: str = BASE_URL,
    limit: int | None = None,
    id_prefixes: tuple[str, ...] = ("nm", "on"),
    skip_unpublished: bool = True,
    timeout: float = 30.0,
    retries: int = 3,
) -> Iterator[Dataset]:
    fetched = 0
    for entry in iter_catalogue(
        base_url=base_url,
        id_prefixes=id_prefixes,
        skip_unpublished=skip_unpublished,
        timeout=timeout,
        retries=retries,
    ):
        if limit and fetched >= limit:
            break
        dataset_id = entry.get("id")
        try:
            metadata = fetch_metadata(
                dataset_id, base_url=base_url, timeout=timeout, retries=retries
            )
        except NemarApiError as exc:
            print(f"  [skip] {dataset_id}: {exc}", file=sys.stderr)
            continue
        if not metadata:
            print(f"  [skip] {dataset_id}: no metadata.json", file=sys.stderr)
            continue
        fetched += 1
        yield build_dataset_document(metadata, catalogue_entry=entry)
        if fetched % 20 == 0:
            print(f"  Processed {fetched} datasets...")


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch NEMAR datasets via data.nemar.org.")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/nemar_datasets.json"),
    )
    p.add_argument(
        "--id-prefixes",
        default="nm,on",
        help="Comma-separated id prefixes (default: nm,on).",
    )
    p.add_argument(
        "--include-unpublished",
        action="store_true",
        help="Include datasets with no 'latest' version.",
    )
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--digested-at", default=None)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    prefixes = tuple(s.strip() for s in args.id_prefixes.split(",") if s.strip())
    print(f"Fetching NEMAR datasets from: {args.base_url}  (prefixes={prefixes})")

    datasets = list(
        fetch_datasets(
            base_url=args.base_url,
            limit=args.limit,
            id_prefixes=prefixes,
            skip_unpublished=not args.include_unpublished,
            timeout=args.timeout,
            retries=args.retries,
        )
    )

    if args.digested_at:
        for ds in datasets:
            if "timestamps" in ds:
                ds["timestamps"]["digested_at"] = args.digested_at

    save_datasets_deterministically(datasets, args.output)
    print(f"\nSaved {len(datasets)} datasets to {args.output}")


if __name__ == "__main__":
    main()
