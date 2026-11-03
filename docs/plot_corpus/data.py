"""Readers, the status vocabulary and the headline numbers of the EEGDash Corpus.

Every function is a pure function of ``rows`` (``atlas.csv``, one row per
dataset of the universe), ``buckets`` (one row per channel set of the
released Zarr store, EEGDash Corpus v0), ``electrodes`` (one entry per
electrode position used by those channel sets), ``recordings`` (one row per dataset: recording
lengths, recordings and hours per subject, tokens) or ``demographics`` (one row
per dataset with ages or sex known).

The universe is every row of ``atlas.csv``: the union of the datasets REVE's
paper names and the entries its public release card lists. Every count, status
chart and substitution is over all of them; an earlier copy's ``tracked``
column is read and ignored. Four statuses: converted and released (v0),
blocked by the licence, gated by an access agreement, excluded.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

# The one status vocabulary, copied literally from the source of truth (the
# corpus tables' ``STATUSES``); a row with any other status is refused.
CONVERTED = (
    "Ingested and Converted (v0)"  # in the released Zarr store, EEGDash Corpus v0
)
BLOCKED = "Blocked (ND Licence)"  # a No Derivatives licence, or none: a converted copy may not be redistributed
GATED = "Gated (Distribution Forbidden)"  # an access agreement forbids redistribution
EXCLUDED = (
    "Excluded"  # out of the corpus for another reason (the reason column says why)
)
# Earlier spellings still accepted in a data file, mapped to the current one (until the next sync).
RENAMED = {
    "Gated (To Be Ingested)": GATED,
    "Blocked (License)": BLOCKED,
}
STATUSES = [CONVERTED, BLOCKED, GATED, EXCLUDED]
OUT = (BLOCKED, GATED)  # cannot be redistributed
# CSS class suffix per status: ``corpus-fill-<slug>`` / ``corpus-bg-<slug>`` /
# ``corpus-bd-<slug>`` read the colour variable ``--corpus-st-<slug>``.
SLUGS = {
    CONVERTED: "converted",
    BLOCKED: "blocked",
    GATED: "gated",
    EXCLUDED: "excluded",
}
# Short labels for legends and KPI boxes.
SHORT = {
    CONVERTED: "converted (v0)",
    BLOCKED: "blocked (ND licence)",
    GATED: "gated (distribution forbidden)",
    EXCLUDED: "excluded",
}

ATLAS_COLUMNS = [
    "dataset_id", "name", "short_name", "source", "archive", "paradigm", "datatypes", "status",
    "in_reve_paper", "in_release_card", "in_v0", "licence", "redistributable",
    "n_subjects", "n_sessions", "hours", "mean_channels", "native_sfreq", "raw_GB",
    "provenance", "reason", "link",
]  # fmt: skip
BUCKET_COLUMNS = ["bucket", "dataset", "n_chans", "n_sessions", "hours", "channels"]
IGNORED = "tracked"  # a column of earlier copies (the rule of 2026-10-13, reversed): read and dropped
CSV_COLUMNS = ATLAS_COLUMNS  # the columns of the CSV downloads
RECORDING_COLUMNS = [
    "dataset_id", "n_recordings", "n_subjects", "minutes_p10", "minutes_p25", "minutes_p50", "minutes_p75",
    "minutes_p90", "minutes_max", "rec_per_subject_p50", "rec_per_subject_max", "hours_per_subject_p50",
    "hours_per_subject_max", "tokens",
]  # fmt: skip
# The corpus table: one row per dataset of v0, the list ``eegdash.corpus.EEGDashCorpus`` takes (its ``dataset_id``
# column) and the pages offer as a download; published as the source's table without its internal ``note`` column.
TABLE = "eegdash-corpus-v0.csv"
TABLE_COLUMNS = [
    "dataset_id", "eegdash_id", "short_name", "name", "paradigm", "archive", "datatypes", "status", "n_subjects",
    "n_sessions", "hours", "mean_channels", "native_sfreq", "raw_GB", "licence",
]  # fmt: skip
DEMOGRAPHIC_COLUMNS = [
    "dataset_id", "year", "n_ages", "age_min", "age_p25", "age_p50", "age_p75", "age_max", "n_male", "n_female",
    "tasks",
]  # fmt: skip
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
BAND = (0.5, 99.5)  # Hz
BAND_CAP = (
    0.45  # the converter caps the band's upper edge at this fraction of the native rate
)
RECIPE = {
    "recipe_sfreq": "200",
    "recipe_band": f"{BAND[0]:g}–{BAND[1]:g}",
    "recipe_window": "10",
}
RECIPE["recipe_points"] = (
    f"{int(RECIPE['recipe_sfreq']) * int(RECIPE['recipe_window']):,}"  # points per sample
)
# The integrity check of the conversion. Stated from the retraining results,
# not derivable from the shipped tables: keep in step with the benchmark CSVs.
CHECKS = {
    "check_loader_datasets": "two",
    "check_bench_tasks": "eight",
    "check_bench_gap": "0.011 (linear probe) and 0.024 (fine-tuning)",
}
# The released store's facts the shipped tables do not carry (its sample count, size and file count, from the
# store and its checksum list), the repository's release tag and the SHA-256 of the checksum list at that tag: keep in
# step with the release.
STORE = {"v0_samples": "5,045,111", "v0_size": "3.60 TB", "v0_files": "44,801", "v0_tag": "v1.0",
         "v0_sums_sha256": "0cba6a7b2106d8f9853c04d07f647256c32aa0c51ae4f8bc0953eaa882e32e54"}  # fmt: skip
ISSUES = "https://github.com/eegdash/EEGDash/issues"
WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    """A CSV as dicts; refuse a file whose header misses an expected column."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return list(reader)


def read_rows(path: Path) -> list[dict[str, str]]:
    """``atlas.csv``; every status must be one of ``STATUSES`` (an earlier spelling is mapped, ``RENAMED``).

    A ``tracked`` column (copies synced before the source dropped it) is ignored: every row counts.
    """
    rows = read_csv(path, ATLAS_COLUMNS)
    for r in rows:
        r["status"] = RENAMED.get(r["status"], r["status"])
        r.pop(IGNORED, None)
    check_statuses(rows)
    return rows


def read_optional(path: Path, columns: list[str]) -> list[dict[str, str]]:
    """A derived table the pages can do without (empty when the data copy does not carry it yet)."""
    return read_csv(path, columns) if Path(path).is_file() else []


def read_table(path: Path) -> list[dict[str, str]]:
    """The corpus table (``TABLE``); refuse an empty or repeated ``dataset_id``, which ``EEGDashCorpus`` would refuse."""
    table = read_csv(path, TABLE_COLUMNS)
    ids = [r["dataset_id"].strip() for r in table]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: every row needs its own dataset_id")
    return table


def by_id(table: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {r["dataset_id"]: r for r in table}


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


def read_unplaced(path: Path) -> list[dict]:
    """``electrodes.json``'s channel names the position bank lacks (placed from their dataset's own coordinates):
    ``name``, ``hours``, ``datasets``; empty for a copy synced before the list.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("unplaced", []) if isinstance(data, dict) else []


def fnum(v: str) -> float | None:
    """A cell as a positive number, or None (empty, non-numeric or zero cannot go on a log axis)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def name_of(r: dict[str, str], width: int = 26) -> str:
    """A dataset's short name, cut to ``width`` characters (for captions and tooltips, which name the id apart)."""
    name = r.get("short_name") or r.get("name") or r.get("dataset_id", "")
    return name if len(name) <= width else name[: width - 1] + "…"


def label_of(r: dict[str, str], width: int = 22) -> str:
    """A chart's axis label, ``name (id)``: the name cut to ``width``, the id never; a row whose name is its id (or a
    label with no id, a paradigm) shows it once.
    """
    i = r.get("dataset_id", "")
    name = r.get("short_name") or r.get("name") or ""
    if not i or not name or name == i:
        return name_of(r, width)
    return f"{name_of(r, width)} ({i})"


def short_title(r: dict[str, str]) -> str:
    """A dataset's short label without a trailing version (``TDBRAIN Dataset V3.1`` -> ``TDBRAIN Dataset``)."""
    return re.sub(
        r"(?i)\s+v\d[\w.]*$", "", r["short_name"] or r["name"] or r["dataset_id"]
    ).strip()


def of_status(rows: list[dict[str, str]], *statuses: str) -> list[dict[str, str]]:
    return [r for r in rows if r["status"] in statuses]


def v0_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return of_status(rows, CONVERTED)


def hours_of(rows: list[dict[str, str]]) -> float:
    return sum(fnum(r["hours"]) or 0 for r in rows)


def total(rows: list[dict[str, str]], col: str) -> float:
    return sum(fnum(r[col]) or 0 for r in rows)


def join_and(words: list[str]) -> str:
    """``a, b and c`` (``none`` for an empty list), for prose substitutions."""
    if not words:
        return "none"
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def is_no_derivatives(r: dict[str, str]) -> bool:
    """A licence with No Derivatives terms (``CC-BY-NC-ND-4.0``, ``CC-BY-ND-4.0``)."""
    return bool(re.search(r"(^|-)ND(-|$)", r["licence"]))


def reason(r: dict[str, str]) -> str:
    """Why a dataset is not in the corpus (empty for a converted one): the published ``reason`` column, derived at
    sync time (``sync_data.derive_reason``); the internal notes are never published.
    """
    return r["reason"]


def excluded_prose(rows: list[dict[str, str]]) -> str:
    """The excluded datasets and why (``A: held out …; B: an fMRI accession …``)."""
    return (
        "; ".join(f"{short_title(r)}: {reason(r)}" for r in of_status(rows, EXCLUDED))
        or "none"
    )


def licence_prose(rows: list[dict[str, str]]) -> str:
    """The licences of the rows, most datasets first, a licence carried by one dataset naming it."""
    by: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by.setdefault(r["licence"] or "none stated", []).append(r)
    parts = []
    for lic in sorted(by, key=lambda k: (-len(by[k]), k)):
        n = len(by[lic])
        what = (
            plural(n, "dataset")
            if not parts
            else f"{n}: {short_title(by[lic][0])}"
            if n == 1
            else str(n)
        )
        parts.append(f"{lic} ({what})")
    return join_and(parts)


def capped_band(v0: list[dict[str, str]]) -> dict[str, str]:
    """The datasets whose native rate is too low for the band's upper edge (``BAND_CAP`` × the rate below it): how many,
    at which rates, and the band they get (from the lowest rate).
    """
    rates = [
        x for x in (fnum(r["native_sfreq"]) for r in v0) if x and x * BAND_CAP < BAND[1]
    ]
    low = sorted(set(rates))
    return {
        "v0_capped": WORDS.get(len(rates), str(len(rates))),
        "v0_capped_rates": join_and([f"{x:g}" for x in low]),
        "v0_capped_band": f"{BAND[0]:g}–{BAND_CAP * low[0]:g}"
        if low
        else RECIPE["recipe_band"],
    }


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


def kpis(rows: list[dict[str, str]]) -> dict:
    """The headline numbers over every row: datasets per status, v0's datasets, hours and subjects."""
    by = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    v0 = v0_rows(rows)
    return {
        "datasets": len(rows),
        **by,
        "v0_datasets": len(v0),
        "v0_hours": hours_of(v0),
        "v0_subjects": int(total(v0, "n_subjects")),
        "v0_sessions": int(total(v0, "n_sessions")),
        "out": by[BLOCKED] + by[GATED],
        "blocked_nd": sum(is_no_derivatives(r) for r in of_status(rows, BLOCKED)),
        "paper": sum(r["in_reve_paper"] == "yes" for r in rows),
        "card": sum(r["in_release_card"] == "yes" for r in rows),
        "card_only": sum(
            r["in_release_card"] == "yes" and r["in_reve_paper"] != "yes" for r in rows
        ),
    }


def status_summary(rows: list[dict[str, str]]) -> str:
    """Where the datasets tracked stand, in one sentence: converted and released, cannot be redistributed (blocked by
    the licence, gated), excluded.
    """
    k = kpis(rows)
    return (f"{k[CONVERTED]} of the {k['datasets']} datasets tracked are converted and released in v0; {k['out']} "
            f"cannot be redistributed ({k[BLOCKED]} blocked by the licence, {k[GATED]} gated by an access agreement) "
            f"and {k[EXCLUDED]} {'is' if k[EXCLUDED] == 1 else 'are'} excluded.")  # fmt: skip


# ------------------------------------------------------------------------------------------------- montages
TENTEN = re.compile(
    r"(?i)^(fp|af|f|ft|fc|t|c|tp|cp|p|po|o|i)(z|\d{1,2})$"
)  # a 10-20 / 10-10 electrode name
TENTEN_ORDER = [
    "fp",
    "af",
    "f",
    "ft",
    "fc",
    "t",
    "c",
    "tp",
    "cp",
    "p",
    "po",
    "o",
    "i",
]  # front to back
MONTAGE_MIN = (
    10  # a row of the montage heatmap is a name used by at least this many datasets
)


def electrode_sets(
    buckets: list[dict[str, str]], electrodes: list[dict]
) -> dict[str, set[str]]:
    """Per dataset its distinct electrodes over its channel sets, each under its position's first spelling."""
    first = {n: el["names"][0] for el in electrodes for n in el["names"]}
    have: dict[str, set[str]] = {}
    for b in buckets:
        have.setdefault(b["dataset"], set()).update(
            first.get(c, c) for c in b["channels"].split("|")
        )
    return have


def montage_names(
    have: dict[str, set[str]], electrodes: list[dict], min_datasets: int = MONTAGE_MIN
) -> tuple[list[str], dict[str, int]]:
    """The 10-20 / 10-10 names at least ``min_datasets`` datasets use, front to back then left to right, and the
    number of datasets using each name.
    """
    x_of = {el["names"][0]: el["xyz"][0] for el in electrodes}
    uses: dict[str, int] = {}
    for names in have.values():
        for n in names:
            if TENTEN.match(n):
                uses[n] = uses.get(n, 0) + 1
    shared = [n for n in uses if uses[n] >= min_datasets]
    shared.sort(
        key=lambda n: (
            TENTEN_ORDER.index(TENTEN.match(n).group(1).lower()),
            x_of.get(n, 0),
            n,
        )
    )
    return shared, uses


def numbers(
    rows: list[dict[str, str]],
    buckets: list[dict[str, str]],
    electrodes: list[dict],
    synced: str,
    recordings: list[dict[str, str]] | None = None,
) -> dict:
    """The substitutions of ``numbers.inc``: every figure the RST prose quotes, from the rows (the universe: the
    paper's datasets and the card's entries), the v0 store's buckets and electrodes and its recording table.
    """
    k = kpis(rows)
    v0 = v0_rows(rows)
    in_v0 = {r["dataset_id"] for r in v0}
    tokens = sum(
        fnum(q["tokens"]) or 0 for q in recordings or [] if q["dataset_id"] in in_v0
    )
    names, _ = montage_names(electrode_sets(buckets, electrodes), electrodes)
    return {
        "universe": str(k["datasets"]),
        "paper_named": str(k["paper"]),
        "card_entries": str(k["card"]),
        "card_only": str(k["card_only"]),
        "v0_datasets": str(k["v0_datasets"]),
        "v0_hours": f"{k['v0_hours']:,.0f}",
        "v0_subjects": f"{k['v0_subjects']:,}",
        "v0_sessions": f"{k['v0_sessions']:,}",
        "v0_buckets": f"{len(buckets):,}",
        "v0_positions": str(sum(e["hours"] > 0 for e in electrodes)),
        "v0_paradigms": str(len({r["paradigm"] for r in v0 if r["paradigm"]})),
        "v0_tokens": f"{tokens / 1e9:,.2f} B" if tokens else "not counted",
        "v0_licences": licence_prose(v0),
        "not_redistributable": str(k["out"]),
        "blocked": str(k[BLOCKED]),
        "blocked_nd": str(k["blocked_nd"]),
        "blocked_none": str(k[BLOCKED] - k["blocked_nd"]),
        "gated": str(k[GATED]),
        "gated_names": join_and([short_title(r) for r in of_status(rows, GATED)]),
        "excluded": str(k[EXCLUDED]),
        "excluded_prose": excluded_prose(rows),
        "montage_rows": str(len(names)),
        "montage_min": str(MONTAGE_MIN),
        "synced": synced,
        **capped_band(v0),
        **RECIPE,
        **CHECKS,
        **STORE,
    }
