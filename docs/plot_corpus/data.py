"""Readers, the status vocabulary and the headline numbers of the EEGDash Corpus.

Every function is a pure function of ``rows`` (``atlas.csv``, one row per
dataset of the universe), ``buckets`` (one row per channel set of the
converted v0 Zarr store), ``added`` (one row per channel set of the datasets v1
adds, from EEGDash's records) or ``electrodes`` (one entry per electrode position
used by those channel sets).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

# The one status vocabulary, copied literally from the source of truth (the
# corpus tables' ``STATUSES``); a row with any other status is refused.
CONVERTED, TO_CONVERT, TO_INGEST = (
    "Ingested and Converted (v0)",
    "Ingested, but not Converted (v1)",
    "To Be Ingested (v1)",
)
BLOCKED, GATED, EXCLUDED, PAPER_ONLY = (
    "Blocked (License)",
    "Gated (To Be Ingested)",
    "Excluded",
    "Paper Only",
)
STATUSES = [CONVERTED, TO_CONVERT, TO_INGEST, BLOCKED, GATED, EXCLUDED, PAPER_ONLY]
# CSS class suffix per status: ``corpus-fill-<slug>`` / ``corpus-bg-<slug>`` /
# ``corpus-bd-<slug>`` read the colour variable ``--corpus-st-<slug>``.
SLUGS = {
    CONVERTED: "converted",
    TO_CONVERT: "to-convert",
    TO_INGEST: "to-ingest",
    BLOCKED: "blocked",
    GATED: "gated",
    EXCLUDED: "excluded",
    PAPER_ONLY: "paper-only",
}
# Short labels for legends and KPI boxes.
SHORT = {
    CONVERTED: "v0, converted",
    TO_CONVERT: "v1, to convert",
    TO_INGEST: "to ingest",
    BLOCKED: "blocked (licence)",
    GATED: "gated",
    EXCLUDED: "excluded",
    PAPER_ONLY: "paper only",
}

ATLAS_COLUMNS = [
    "dataset_id", "name", "short_name", "source", "archive", "paradigm", "datatypes", "status",
    "in_reve_paper", "in_release_card", "in_v0", "in_v1", "licence", "redistributable",
    "n_subjects", "n_sessions", "hours", "mean_channels", "native_sfreq", "raw_GB",
    "provenance", "note", "link",
]  # fmt: skip
BUCKET_COLUMNS = ["bucket", "dataset", "n_chans", "n_sessions", "hours", "channels"]
PROV_KEY = {
    "n_subjects": "subjects",
    "n_sessions": "sessions",
    "hours": "hours",
    "mean_channels": "channels",
    "native_sfreq": "sfreq",
    "raw_GB": "size",
}
DIGITS = {
    "n_subjects": 0,
    "n_sessions": 0,
    "hours": 1,
    "mean_channels": 1,
    "native_sfreq": 0,
    "raw_GB": 1,
}
PROVENANCE = {
    "zarr": "measured in the converted Zarr store",
    "eegdash": "EEGDash's catalogue records",
    "card": "REVE's release card",
    "stated": "REVE's paper",
    "download": "size of a downloaded copy",
}
# The conversion recipe (one place; the pages read it through numbers.inc).
RECIPE = {"recipe_sfreq": "200", "recipe_band": "0.5–99.5", "recipe_window": "10"}
# The integrity check of the retraining. Stated from the retraining results,
# not derivable from the shipped tables: keep in step with the benchmark CSVs.
CHECKS = {
    "check_loader_datasets": "two",
    "check_bench_tasks": "eight",
    "check_bench_gap": "0.011 (linear probe) and 0.024 (fine-tuning)",
}
# What comes next, in order; {keys} are filled from the rows at build time.
ROADMAP = [
    "Host the converted Zarr store eegdash-corpus-v0 ({v0_datasets} datasets, {v0_hours} h) through EEGDash",
    "One conversion pass for v1: the {to_convert} datasets ingested but not converted "
    "({to_convert_hours} h measured so far)",
    "Deposit the datasets still to ingest ({ingest_names}) on NEMAR so EEGDash serves them",
    "Multi-dataset download in EEGDash: one call fetches a list of dataset ids",
]
ISSUES = "https://github.com/eegdash/EEGDash/issues"


def read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    """A CSV as dicts; refuse a file whose header misses an expected column."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return list(reader)


def read_rows(path: Path) -> list[dict[str, str]]:
    """``atlas.csv``; every status must be one of ``STATUSES``."""
    rows = read_csv(path, ATLAS_COLUMNS)
    check_statuses(rows)
    return rows


def check_statuses(rows: list[dict[str, str]]) -> None:
    for r in rows:
        if r["status"] not in STATUSES:
            raise ValueError(
                f"{r['dataset_id']}: status {r['status']!r} is not one of {STATUSES}"
            )


def read_buckets(path: Path) -> list[dict[str, str]]:
    return read_csv(path, BUCKET_COLUMNS)


def read_electrodes(path: Path) -> list[dict]:
    """``electrodes.json``: one entry per position, ``names``, ``xyz`` (metres), ``hours``, ``datasets``, ``buckets``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["electrodes"] if isinstance(data, dict) else data


def fnum(v: str) -> float | None:
    """A cell as a positive number, or None (empty, non-numeric or zero cannot go on a log axis)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def label_of(r: dict[str, str], width: int = 26) -> str:
    name = r.get("short_name") or r.get("name") or r.get("dataset_id", "")
    return name if len(name) <= width else name[: width - 1] + "…"


def short_title(r: dict[str, str]) -> str:
    """A dataset's short label without a trailing version (``Siena … v1.0.0`` -> its title)."""
    return re.sub(
        r"(?i)\s+v\d[\w.]*$", "", r["short_name"] or r["name"] or r["dataset_id"]
    ).strip()


def of_status(rows: list[dict[str, str]], *statuses: str) -> list[dict[str, str]]:
    return [r for r in rows if r["status"] in statuses]


def v0_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return of_status(rows, CONVERTED)


def v1_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """v1 = v0 converted, the datasets ingested and waiting for conversion, and those still to ingest: the
    three statuses whose name says (v1).
    """
    return of_status(rows, CONVERTED, TO_CONVERT, TO_INGEST)


def v1_added(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The datasets v1 adds to v0: to convert first, then to ingest."""
    return of_status(rows, TO_CONVERT) + of_status(rows, TO_INGEST)


def hours_of(rows: list[dict[str, str]]) -> float:
    return sum(fnum(r["hours"]) or 0 for r in rows)


def total(rows: list[dict[str, str]], col: str) -> float:
    return sum(fnum(r[col]) or 0 for r in rows)


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


def kpis(rows: list[dict[str, str]]) -> dict:
    """The headline numbers: datasets per status, v0 hours and subjects, v1 hours, datasets pending."""
    by = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    v0, v1 = v0_rows(rows), v1_rows(rows)
    return {
        "datasets": len(rows),
        **by,
        "v0_hours": hours_of(v0),
        "v0_subjects": int(total(v0, "n_subjects")),
        "v0_sessions": int(total(v0, "n_sessions")),
        "v1_hours": hours_of(v1),
        "v1_subjects": int(total(v1, "n_subjects")),
        "v1_datasets": len(v1),
        "pending": by[TO_INGEST] + by[BLOCKED],
        "paper": sum(r["in_reve_paper"] == "yes" for r in rows),
        "card": sum(r["in_release_card"] == "yes" for r in rows),
        "card_only": sum(
            r["in_release_card"] == "yes" and r["in_reve_paper"] != "yes" for r in rows
        ),
    }


def progress(rows: list[dict[str, str]]) -> list[dict]:
    """The steps from what is converted to what is out of reach, with counts, hours and names."""
    names = lambda rs: ", ".join(short_title(r) for r in rs)  # noqa: E731
    v0, v1, ing, blk, gat = (
        of_status(rows, s) for s in (CONVERTED, TO_CONVERT, TO_INGEST, BLOCKED, GATED)
    )
    unmeasured = [r for r in v1 if not fnum(r["hours"])]
    return [
        dict(status=CONVERTED, title="v0 converted", count=len(v0), hours=hours_of(v0), sub=f"{hours_of(v0):,.0f} h"),
        dict(status=TO_CONVERT, title="v1 to convert", count=len(v1), hours=hours_of(v1),
             sub=f"+{hours_of(v1):,.0f} h measured" + (f"; {names(unmeasured)} unmeasured" if unmeasured else "")),
        dict(status=TO_INGEST, title="to ingest", count=len(ing), hours=hours_of(ing), sub=names(ing) or "none"),
        dict(status=BLOCKED, title="blocked (licence)", count=len(blk), hours=hours_of(blk),
             sub="NoDerivatives or no licence"),
        dict(status=GATED, title="gated", count=len(gat), hours=hours_of(gat), sub=names(gat) or "none"),
    ]  # fmt: skip


def roadmap_items(rows: list[dict[str, str]]) -> list[str]:
    k = kpis(rows)
    ing = [short_title(r) for r in of_status(rows, TO_INGEST)]
    return [
        s.format(
            v0_datasets=k[CONVERTED],
            v0_hours=f"{k['v0_hours']:,.0f}",
            to_convert=k[TO_CONVERT],
            to_convert_hours=f"{hours_of(of_status(rows, TO_CONVERT)):,.0f}",
            ingest_names=", ".join(ing) or "none",
        )
        for s in ROADMAP
    ]


def numbers(
    rows: list[dict[str, str]],
    buckets: list[dict[str, str]],
    electrodes: list[dict],
    synced: str,
    added: list[dict[str, str]] | None = None,
) -> dict:
    """The substitutions of ``numbers.inc``: every figure the RST prose quotes."""
    k = kpis(rows)
    v0 = v0_rows(rows)
    new = v1_added(rows)
    added = added or []
    mapped = {b["dataset"] for b in added}
    unmapped = [short_title(r) for r in new if r["dataset_id"] not in mapped]
    return {
        "universe": str(k["datasets"]),
        "paper_named": str(k["paper"]),
        "card_entries": str(k["card"]),
        "card_only": str(k["card_only"]),
        "v0_datasets": str(k[CONVERTED]),
        "v0_hours": f"{k['v0_hours']:,.0f}",
        "v0_subjects": f"{k['v0_subjects']:,}",
        "v0_sessions": f"{k['v0_sessions']:,}",
        "v0_buckets": f"{len(buckets):,}",
        "v0_positions": str(sum(e["hours"] > 0 for e in electrodes)),
        "v1_positions": str(
            sum(e["hours"] + e.get("added_hours", 0) > 0 for e in electrodes)
        ),
        "v1_buckets": f"{len(buckets) + len(added):,}",
        "v1_mapped": str(len(mapped)),
        "v1_unmapped": ", ".join(unmapped) or "none",
        "v0_paradigms": str(len({r["paradigm"] for r in v0 if r["paradigm"]})),
        "v1_datasets": str(k["v1_datasets"]),
        "v1_hours": f"{k['v1_hours']:,.0f}",
        "v1_new": str(len(new)),
        "v1_new_hours": f"{hours_of(new):,.0f}",
        "v1_to_convert": str(k[TO_CONVERT]),
        "v1_unmeasured": str(sum(not fnum(r["hours"]) for r in new)),
        "to_ingest": str(k[TO_INGEST]),
        "blocked": str(k[BLOCKED]),
        "gated": str(k[GATED]),
        "excluded": str(k[EXCLUDED]),
        "pending": str(k["pending"]),
        "synced": synced,
        **RECIPE,
        **CHECKS,
    }
