"""Readers, the status vocabulary and the headline numbers of the EEGDash Corpus.

Every function is a pure function of ``rows`` (``atlas.csv``, one row per
dataset of the universe), ``buckets`` (one row per channel set of the
converted v0 Zarr store), ``added`` (one row per channel set of the datasets v1
adds, from EEGDash's records), ``electrodes`` (one entry per electrode position
used by those channel sets), ``recordings`` (one row per dataset: recording
lengths, recordings and hours per subject, tokens) or ``demographics`` (one row
per dataset with ages or sex known).

The universe is every row of ``atlas.csv``: the union of the datasets REVE's
paper names and the entries its public release card lists. Every count, status
chart, roadmap and substitution is over all of them; an earlier copy's
``tracked`` column is read and ignored.
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
    "Gated (Distribution Forbidden)",
    "Excluded",
    "Paper Only",
)
# Earlier spellings still accepted in a data file, mapped to the current one (until the next sync).
RENAMED = {"Gated (To Be Ingested)": GATED}
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
    GATED: "gated (distribution forbidden)",
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
IGNORED = "tracked"  # a column of earlier copies (the rule of 2026-10-13, reversed): read and dropped
CSV_COLUMNS = ATLAS_COLUMNS  # the columns of the CSV downloads
RECORDING_COLUMNS = [
    "dataset_id", "n_recordings", "n_subjects", "minutes_p10", "minutes_p25", "minutes_p50", "minutes_p75",
    "minutes_p90", "minutes_max", "rec_per_subject_p50", "rec_per_subject_max", "hours_per_subject_p50",
    "hours_per_subject_max", "tokens",
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


def join_and(words: list[str]) -> str:
    """``a, b and c`` (``none`` for an empty list), for prose substitutions."""
    if not words:
        return "none"
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


# Why an excluded row is out, read from its note: (pattern, words), first match wins; otherwise the note's first
# clause.
EXCLUDED_WHY = [
    (r"test task", "the benchmark's test task"),
    (r"fMRI", "an fMRI accession listed by mistake"),
    (r"names no dataset", "names no dataset"),
]


def excluded_why(r: dict[str, str]) -> str:
    """Why one excluded row is out, in a few words (its note)."""
    for pat, words in EXCLUDED_WHY:
        if re.search(pat, r["note"]):
            return words
    reason = re.sub(r"^excluded from the corpus:\s*", "", r["note"])
    return re.split(r"[;,]", reason)[0].strip() or "not a dataset for the corpus"


def card_only_prose(rows: list[dict[str, str]]) -> str:
    """What the release-card entries the paper does not name are, and the excluded datasets of the paper, in words
    from the rows: the converted entries, the excluded ones with their reason, the total excluded.
    """
    card_only = [
        r for r in rows if r["in_release_card"] == "yes" and r["in_reve_paper"] != "yes"
    ]
    conv = [r for r in card_only if r["status"] == CONVERTED]
    excl = [r for r in card_only if r["status"] == EXCLUDED]
    rest = [r for r in card_only if r not in conv and r not in excl]
    paper_excl = [
        r for r in rows if r["status"] == EXCLUDED and r["in_reve_paper"] == "yes"
    ]
    word = lambda n: WORDS.get(n, str(n))  # noqa: E731
    parts = []
    if conv:
        ids = join_and([r["dataset_id"] for r in conv])
        parts.append(f"{ids} {'is a real dataset that was' if len(conv) == 1 else 'are real datasets that were'} "
                     f"converted with the rest" + (" (in v0)" if all(r["in_v0"] == "yes" for r in conv) else ""))  # fmt: skip
    if excl:
        lead = f"the other {word(len(excl))}" if conv and not rest else word(len(excl))
        why = ", ".join(f"{r['dataset_id']} {excluded_why(r)}" for r in excl)
        parts.append(f"{lead} {'is' if len(excl) == 1 else 'are'} excluded ({why})")
    if rest:
        parts.append(
            join_and([f"{r['dataset_id']} is {SHORT[r['status']]}" for r in rest])
        )
    text = join_and(parts) if parts else "none"
    if paper_excl:
        names = join_and(
            [f"{short_title(r)} from the paper ({excluded_why(r)})" for r in paper_excl]
        )
        text += f", as {'is' if len(paper_excl) == 1 else 'are'} {names}"
    n_excl = sum(r["status"] == EXCLUDED for r in rows)
    return text + (f": {word(n_excl)} excluded in all" if excl and paper_excl else "")


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


def kpis(rows: list[dict[str, str]]) -> dict:
    """The headline numbers over every row: datasets per status, the versions' datasets, hours and subjects."""
    by = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    v0, v1 = v0_rows(rows), v1_rows(rows)
    return {
        "datasets": len(rows),
        **by,
        "v0_datasets": len(v0),
        "v0_hours": hours_of(v0),
        "v0_subjects": int(total(v0, "n_subjects")),
        "v0_sessions": int(total(v0, "n_sessions")),
        "v1_hours": hours_of(v1),
        "v1_subjects": int(total(v1, "n_subjects")),
        "v1_datasets": len(v1),
        "v1_to_convert": len(of_status(v1, TO_CONVERT)),
        "v1_to_ingest": len(of_status(v1, TO_INGEST)),
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
        dict(status=CONVERTED, title="converted (v0)", count=len(v0), hours=hours_of(v0),
             sub=f"{hours_of(v0):,.0f} h"),
        dict(status=TO_CONVERT, title="v1 to convert", count=len(v1), hours=hours_of(v1),
             sub=f"+{hours_of(v1):,.0f} h measured" + (f"; {names(unmeasured)} unmeasured" if unmeasured else "")),
        dict(status=TO_INGEST, title="to ingest", count=len(ing), hours=hours_of(ing), sub=names(ing) or "none"),
        dict(status=BLOCKED, title="blocked (licence)", count=len(blk), hours=hours_of(blk),
             sub="NoDerivatives or no licence"),
        dict(status=GATED, title="gated (distribution forbidden)", count=len(gat), hours=hours_of(gat),
             sub=f"{names(gat)}; never redistributable" if gat else "none"),
    ]  # fmt: skip


def roadmap_items(rows: list[dict[str, str]]) -> list[str]:
    k = kpis(rows)
    ing = [short_title(r) for r in of_status(rows, TO_INGEST)]
    return [
        s.format(
            v0_datasets=k["v0_datasets"],
            v0_hours=f"{k['v0_hours']:,.0f}",
            to_convert=k[TO_CONVERT],
            to_convert_hours=f"{hours_of(of_status(rows, TO_CONVERT)):,.0f}",
            ingest_names=", ".join(ing) or "none",
        )
        for s in ROADMAP
        if ing
        or "{ingest_names}"
        not in s  # nothing left to deposit: the step is done, not listed
    ]


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
    added: list[dict[str, str]] | None = None,
) -> dict:
    """The substitutions of ``numbers.inc``: every figure the RST prose quotes. The release card's entries are
    over every row of the universe (the paper's datasets and the card's entries).
    """
    card_only = [
        r for r in rows if r["in_release_card"] == "yes" and r["in_reve_paper"] != "yes"
    ]
    k = kpis(rows)
    v0 = v0_rows(rows)
    new = v1_added(rows)
    added = added or []
    mapped = {b["dataset"] for b in added}
    unmapped = [short_title(r) for r in new if r["dataset_id"] not in mapped]
    names, _ = montage_names(electrode_sets(buckets, electrodes), electrodes)
    return {
        "universe": str(k["datasets"]),
        "paper_named": str(k["paper"]),
        "card_entries": str(k["card"]),
        "card_only": str(k["card_only"]),
        "card_only_names": join_and([r["dataset_id"] for r in card_only]),
        "card_only_prose": card_only_prose(rows),
        "v0_datasets": str(k["v0_datasets"]),
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
        "v1_to_convert": str(k["v1_to_convert"]),
        # the added datasets in words, without a "0 still to ingest" once every one is on NEMAR
        "v1_added_prose": f"{k['v1_to_convert']} ingested by EEGDash and waiting for conversion"
        + (f", {k['v1_to_ingest']} still to be ingested" if k["v1_to_ingest"] else ""),
        "v1_added_short": f"{k['v1_to_convert']} to convert"
        + (f", {k['v1_to_ingest']} to ingest" if k["v1_to_ingest"] else ""),
        "v1_unmeasured": str(sum(not fnum(r["hours"]) for r in new)),
        "to_ingest": str(k[TO_INGEST]),
        "blocked": str(k[BLOCKED]),
        "gated": str(k[GATED]),
        "excluded": str(k[EXCLUDED]),
        "montage_rows": str(len(names)),
        "montage_min": str(MONTAGE_MIN),
        "synced": synced,
        **RECIPE,
        **CHECKS,
    }
