"""Dataset-level BIDS metadata extraction Seam.

Extracted from ``3_digest.py``. Reads README + participants.tsv demographics,
builds the dataset-level storage doc, harvests dataset_description.json extras, and
assembles the Dataset document via ``create_dataset``. Pure leaf logic — depends
only on ``_bids_path``, ``_file_utils``, ``_constants`` and ``eegdash`` — never on
``3_digest``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from _bids_path import normalize_modality
from _constants import NEURO_MODALITIES
from _file_utils import get_annex_file_size
from eegdash.dataset._source_inference import DEFAULT_STORAGE_CONFIG, STORAGE_CONFIGS
from eegdash.schemas import AGE_UNITS_PER_YEAR, Storage, create_dataset
from source_adapter import SourceAdapter

logger = logging.getLogger(__name__)

__all__ = ["extract_dataset_metadata"]


def _read_bids_readme(bids_root: Path) -> str | None:
    """Return cleaned README text, or None if absent / unreadable."""
    for readme_name in ("README", "README.md", "README.txt", "readme", "readme.md"):
        readme_path = bids_root / readme_name
        if not readme_path.exists():
            continue
        try:
            raw_readme = readme_path.read_text(encoding="utf-8")
            return "\n".join(
                [line.rstrip() for line in raw_readme.splitlines() if line.strip()]
            )
        except (OSError, UnicodeDecodeError):
            continue
    return None


# participants.tsv age columns, in order of preference.
_AGE_COLUMNS: tuple[str, ...] = ("age", "Age", "AGE")

# Spellings of a participants.json age ``Units`` value, mapped to the keys of
# AGE_UNITS_PER_YEAR. BIDS defines ``age`` in years; another unit is applied
# only when participants.json declares it for the age column, never guessed
# from the values or the column name.
_AGE_UNIT_SPELLINGS: dict[str, str] = {
    "y": "years",
    "yr": "years",
    "yrs": "years",
    "year": "years",
    "years": "years",
    "mo": "months",
    "mos": "months",
    "mon": "months",
    "mth": "months",
    "mths": "months",
    "month": "months",
    "months": "months",
    "w": "weeks",
    "wk": "weeks",
    "wks": "weeks",
    "week": "weeks",
    "weeks": "weeks",
    "d": "days",
    "day": "days",
    "days": "days",
    "h": "hours",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
    "hours": "hours",
}

# Words a ``Units`` value may carry besides the unit itself ("Year(s)",
# "decimal years", "age in days"). Anything else makes the value free text,
# which is read as years (the BIDS default), never parsed for a unit.
_AGE_UNIT_FILLER_WORDS: frozenset[str] = frozenset({"s", "in", "age", "decimal"})


def _canonical_age_unit(raw: object) -> str | None:
    """Map a participants.json ``Units`` value to a key of AGE_UNITS_PER_YEAR.

    The value must name exactly one unit, optionally with filler words
    (``_AGE_UNIT_FILLER_WORDS``); otherwise None.
    """
    if not isinstance(raw, str):
        return None
    tokens = re.findall(r"[a-z]+", raw.lower())
    words = [w for w in tokens if w not in _AGE_UNIT_FILLER_WORDS]
    if len(words) != 1:
        return None
    return _AGE_UNIT_SPELLINGS.get(words[0])


def _read_participants_age_units(bids_root: Path) -> str | None:
    """Unit participants.json declares for the participants.tsv age column.

    Returns a key of AGE_UNITS_PER_YEAR other than ``"years"``, or None when
    the ages are in years: no participants.json, no age entry, no ``Units``,
    a years unit, or a ``Units`` value naming no known unit (read as years,
    the BIDS default, as before).
    """
    participants_path = bids_root / "participants.tsv"
    sidecar_path = bids_root / "participants.json"
    if not participants_path.exists() or not sidecar_path.exists():
        return None
    try:
        columns = pd.read_csv(participants_path, sep="\t", nrows=0).columns
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ):
        return None
    age_col = next((col for col in _AGE_COLUMNS if col in columns), None)
    if age_col is None or not isinstance(sidecar, dict):
        return None
    entry = sidecar.get(age_col)
    if entry is None:
        entry = next(
            (
                value
                for key, value in sidecar.items()
                if isinstance(key, str) and key.lower() == age_col.lower()
            ),
            None,
        )
    if not isinstance(entry, dict):
        return None
    raw = entry.get("Units", entry.get("units"))
    if raw is None or raw == "":
        return None
    unit = _canonical_age_unit(raw)
    if unit is None:
        logger.debug(
            "%s: participants.json age Units %r not recognised; ages read as years",
            bids_root.name,
            raw,
        )
        return None
    return None if unit == "years" else unit


def _age_in_years(value: object, per_year: float) -> float | None:
    """Return one participants.tsv age in years, or None when unusable.

    Keeps ages strictly between 0 and 120 years. Whole years stay ``int``, so
    cohorts recorded in whole years digest exactly as before; fractions are
    kept (infant cohorts used to lose every age below one year).
    """
    try:
        years = round(float(value) / per_year, 4)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(years) or not 0 < years < 120:
        return None
    return int(years) if years.is_integer() else years


def _read_participants_demographics(
    bids_root: Path,
) -> tuple[int, list[float], dict[str, int], dict[str, int]]:
    """Read participants.tsv and return (subjects_count, ages, sex_distribution, handedness_distribution).

    Ages are in years, converted from the unit participants.json declares for
    the age column (see ``_read_participants_age_units``).
    """
    subjects_count = 0
    ages: list[float] = []
    sex_distribution: dict[str, int] = {}
    handedness_distribution: dict[str, int] = {}

    participants_path = bids_root / "participants.tsv"
    if not participants_path.exists():
        return subjects_count, ages, sex_distribution, handedness_distribution

    try:
        df = pd.read_csv(
            participants_path, sep="\t", dtype="string", keep_default_na=False
        )
        subjects_count = len(df)

        age_col = next((col for col in _AGE_COLUMNS if col in df.columns), None)
        if age_col:
            age_units = _read_participants_age_units(bids_root)
            per_year = AGE_UNITS_PER_YEAR[age_units] if age_units else 1.0
            for val in df[age_col]:
                age = _age_in_years(val, per_year)
                if age is not None:
                    ages.append(age)

        sex_col = next(
            (
                col
                for col in ("sex", "Sex", "SEX", "gender", "Gender")
                if col in df.columns
            ),
            None,
        )
        if sex_col:
            for val in df[sex_col]:
                val_lower = str(val).lower().strip()
                if val_lower in ("m", "male"):
                    sex_distribution["m"] = sex_distribution.get("m", 0) + 1
                elif val_lower in ("f", "female"):
                    sex_distribution["f"] = sex_distribution.get("f", 0) + 1
                elif val_lower and val_lower not in (
                    "n/a",
                    "na",
                    "nan",
                    "unknown",
                    "",
                ):
                    sex_distribution["o"] = sex_distribution.get("o", 0) + 1

        hand_col = next(
            (
                col
                for col in ("handedness", "Handedness", "hand", "Hand")
                if col in df.columns
            ),
            None,
        )
        if hand_col:
            for val in df[hand_col]:
                val_lower = str(val).lower().strip()
                if val_lower in ("r", "right"):
                    handedness_distribution["r"] = (
                        handedness_distribution.get("r", 0) + 1
                    )
                elif val_lower in ("l", "left"):
                    handedness_distribution["l"] = (
                        handedness_distribution.get("l", 0) + 1
                    )
                elif val_lower in ("a", "ambidextrous"):
                    handedness_distribution["a"] = (
                        handedness_distribution.get("a", 0) + 1
                    )
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        KeyError,
    ):
        pass

    return subjects_count, ages, sex_distribution, handedness_distribution


# A DOI value: "10.1234/abc", "doi:10.1234/abc" or "https://doi.org/10.1234/abc".
_DOI_PREFIX_RE = re.compile(
    r"^(?:doi:\s*|https?://(?:dx\.|www\.)?doi\.org/)", re.IGNORECASE
)
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")
# Separator before a second DOI in a list ("10.1/a, doi:10.2/b").
_DOI_LIST_SEP_RE = re.compile(
    r"[\s,;]+(?=(?:doi:\s*|https?://(?:dx\.|www\.)?doi\.org/)?10\.\d{4,9}/)",
    re.IGNORECASE,
)


def _bare_doi(value: object) -> str | None:
    """Return the DOI ``value`` holds, without prefix, or None.

    The value must be a DOI, optionally prefixed with ``doi:`` or a doi.org
    URL (whose query string or fragment is dropped); of a list of DOIs the
    first is kept. Free text, other URLs (a publisher page that embeds a DOI)
    and anything else give None, so a page never links to a made-up DOI.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    prefix = _DOI_PREFIX_RE.match(text)
    if prefix:
        text = text[prefix.end() :]
        if prefix.group(0).lower().startswith("http"):
            text = re.split(r"[?#]", text, maxsplit=1)[0]
    doi = _DOI_LIST_SEP_RE.split(text, maxsplit=1)[0].rstrip(".,; ")
    for close, open_ in ((")", "("), ("]", "[")):
        while doi.endswith(close) and doi.count(close) > doi.count(open_):
            doi = doi[:-1]
    return doi if _DOI_RE.fullmatch(doi) else None


def _doi_identity(doi: str) -> str:
    """Compare key of a DOI: lower case, without a version suffix (``.v1.0.0``)."""
    return re.sub(r"\.v\d+(?:\.\d+)*$", "", doi.lower())


# The README "Paper DOI" badge, the paper line a deposit declares:
# [![Paper DOI](https://img.shields.io/...)](https://doi.org/<doi>). Same
# pattern as the dataset page's _README_PAPER_DOI_RE
# (docs/source/_extensions/dataset_page/_constants.py).
_README_PAPER_BADGE_RE = re.compile(
    r"\[!\[Paper[^\]]*\]\([^)]+\)\]\((https?://[^)\s]+)\)", re.IGNORECASE
)
# A deposit's paper-match record: which paper, if any, is confirmed to
# describe the data ({"association": {"verdict": "confirmed" | ...},
# "document": {"doi": ...}}). Most NEMAR fNIRS deposits carry one.
_PAPER_MATCH_RELPATH = Path("code") / "paper_match" / "match.json"
# Sources whose catalogue paper is no confirmation that the paper describes the
# dataset. NEMAR's comes from DataCite related identifiers (IsDescribedBy,
# IsSupplementTo), and some of those name a method paper or the wrong study;
# for these sources only a README badge or a confirmed record gives a paper.
# OpenNeuro's associatedPaperDOI and EEGManyLabs' paper are declared by the
# dataset's authors and are kept as a last resort.
_UNCONFIRMED_CATALOGUE_PAPER_SOURCES: frozenset[str] = frozenset({"nemar"})


def _readme_paper_doi(readme: str | None) -> str | None:
    """DOI of the README's "Paper DOI" badge, or None."""
    if not readme:
        return None
    match = _README_PAPER_BADGE_RE.search(readme)
    return _bare_doi(match.group(1)) if match else None


def _paper_match_doi(bids_root: Path) -> tuple[bool, str | None]:
    """Read the deposit's paper-match record.

    Returns ``(False, None)`` without a record, ``(True, doi)`` when it
    confirms a paper, and ``(True, None)`` when it confirms none (another
    verdict such as "probable" or "wrong-paper", no DOI, or an unreadable
    file).
    """
    path = bids_root / _PAPER_MATCH_RELPATH
    if not path.is_file():
        return False, None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return True, None
    if not isinstance(record, dict):
        return True, None
    association = record.get("association")
    document = record.get("document")
    if not isinstance(association, dict) or not isinstance(document, dict):
        return True, None
    if association.get("verdict") != "confirmed":
        return True, None
    return True, _confirmed_paper_doi(record, association, document)


# The paper an evidence quote was read from, as its ``source`` names it:
# "https://doi.org/<doi> [...]" or "[...; paper doi:<doi>; ...]".
_EVIDENCE_PAPER_DOI_RE = re.compile(
    r"(?:paper doi:\s*|doi\.org/)(10\.\d{4,9}/[^\s;\[\]]+)", re.IGNORECASE
)


def _evidence_paper_dois(association: dict) -> list[str]:
    """DOIs of the texts the evidence quotes were read from, skipping data records."""
    evidence = association.get("evidence")
    dois: list[str] = []
    for item in evidence if isinstance(evidence, list) else []:
        source = item.get("source") if isinstance(item, dict) else None
        if isinstance(source, str) and "repository record" not in source.lower():
            dois.extend(_EVIDENCE_PAPER_DOI_RE.findall(source))
    return dois


def _confirmed_paper_doi(record: dict, association: dict, document: dict) -> str | None:
    """DOI of the paper a confirmed paper-match record names, or None.

    ``_paperenrich.confirmed_paper.doi`` when the record carries it. Else
    ``document.doi`` when the document is a paper; when the document is the
    data record the match read (``kind`` "repository-record", or a
    ``source_kind`` naming a repository record), a paper an evidence quote was
    read from instead. The data record is never the paper, in any version: no
    DOI under its registrant prefix (e.g. every Zenodo version of a Zenodo
    record) is taken.
    """
    is_data_record = document.get("kind") == "repository-record" or (
        "repository record" in str(document.get("source_kind") or "").lower()
    )
    record_doi = _bare_doi(document.get("doi")) if is_data_record else None
    enrich = record.get("_paperenrich")
    confirmed = enrich.get("confirmed_paper") if isinstance(enrich, dict) else None
    candidates = [confirmed.get("doi")] if isinstance(confirmed, dict) else []
    if is_data_record:
        candidates.extend(_evidence_paper_dois(association))
    else:
        candidates.append(document.get("doi"))
    record_prefix = record_doi.split("/", 1)[0] if record_doi else None
    for value in candidates:
        doi = _bare_doi(value)
        if doi is not None and doi.split("/", 1)[0] != record_prefix:
            return doi
    return None


def _paper_doi_for_dataset(
    bids_root: Path,
    readme: str | None,
    metadata: dict,
    dataset_doi: object,
    source: str,
) -> str | None:
    """Return the DOI of the paper that describes the dataset, or None.

    In order: the README "Paper DOI" badge; else the deposit's paper-match
    record, which decides when present (only a confirmed paper counts); else,
    except for ``_UNCONFIRMED_CATALOGUE_PAPER_SOURCES`` (NEMAR), the paper the
    source catalogue gives, which stage 1 put on the manifest as
    ``associated_paper_doi`` (OpenNeuro: ``associatedPaperDOI``). A NEMAR
    dataset with neither a badge nor a confirmed record has no paper. The
    dataset's own DOI, in any version, is never its paper. Methods or toolbox
    papers a dataset only references give no paper link.
    """
    own = _bare_doi(dataset_doi)
    own_identity = _doi_identity(own) if own else None
    badge = _readme_paper_doi(readme)
    if badge is not None and _doi_identity(badge) != own_identity:
        return badge
    recorded, doi = _paper_match_doi(bids_root)
    if not recorded and source not in _UNCONFIRMED_CATALOGUE_PAPER_SOURCES:
        doi = _bare_doi(metadata.get("associated_paper_doi"))
    if doi is None or _doi_identity(doi) == own_identity:
        return None
    return doi


_BIDS_GLOBAL_FILES: tuple[str, ...] = (
    "participants.tsv",
    "participants.json",
    "samples.tsv",
    "samples.json",
    "README",
    "README.md",
    "README.txt",
    "CHANGES",
    "CHANGES.md",
    "LICENSE",
    "authors.tsv",
    "dataset_description.json",
)


def _build_global_storage_info(
    dataset_id: str, source: str, bids_root: Path
) -> Storage | None:
    """Build the Dataset-level storage doc (backend, base URL, root-level dep_keys)."""
    if source not in ("openneuro", "nemar", "gin") and source not in STORAGE_CONFIGS:
        return None

    cfg = STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)
    storage_base = f"{cfg['base']}/{dataset_id}"
    storage_backend = cfg["backend"]

    dep_keys: list[str] = []
    raw_key = "dataset_description.json"  # default "main" file
    found_files: set[str] = set()

    # 1. Check explicit list (case-sensitive then case-insensitive).
    for fname in _BIDS_GLOBAL_FILES:
        fpath = bids_root / fname
        if fpath.exists():
            found_files.add(fname)
            if fname == "dataset_description.json":
                raw_key = fname
            else:
                dep_keys.append(fname)
            continue
        # Case-insensitive fallback
        try:
            found = next(
                x.name for x in bids_root.iterdir() if x.name.lower() == fname.lower()
            )
            found_files.add(found)
            if found.lower() == "dataset_description.json":
                raw_key = found
            else:
                dep_keys.append(found)
        except StopIteration:
            pass

    # 2. Scan for other root-level BIDS files (sidecars, etc.).
    ignored_files = {"manifest.json", ".ds_store"}
    for item in bids_root.iterdir():
        if not item.is_file():
            continue
        item_name = item.name
        if (
            item_name in found_files
            or item_name.lower() in ignored_files
            or item_name.startswith(".")
        ):
            continue
        if item_name.lower().endswith(
            (".json", ".tsv", ".txt", ".md", ".yaml", ".yml")
        ):
            dep_keys.append(item_name)
            found_files.add(item_name)

    return {
        "backend": storage_backend,  # type: ignore[typeddict-item]
        "base": storage_base,
        "raw_key": raw_key,
        "dep_keys": sorted(set(dep_keys)),
    }


def extract_dataset_metadata(
    bids_dataset,
    dataset_id: str,
    source: str,
    digested_at: str,
    metadata: dict | None = None,
    source_adapter: SourceAdapter | None = None,
) -> dict[str, Any]:
    """Extract Dataset-level metadata from a BIDS dataset."""
    metadata = metadata or {}
    bids_root = Path(bids_dataset.bidsdir)
    # Read dataset_description.json
    description = {}
    desc_path = bids_root / "dataset_description.json"
    if desc_path.exists():
        try:
            with open(desc_path) as f:
                description = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    readme = _read_bids_readme(bids_root)

    name = description.get("Name", dataset_id)
    bids_version = description.get("BIDSVersion")
    license_info = description.get("License")
    authors = description.get("Authors", [])
    funding = description.get("Funding", [])
    dataset_doi = description.get("DatasetDOI")

    files = bids_dataset.get_files()
    modalities = set()
    tasks = set()
    sessions = set()
    subjects = set()

    for bids_file in files:
        mod = bids_dataset.get_bids_file_attribute("modality", bids_file)
        mod_canon = normalize_modality(mod)
        if mod_canon:
            modalities.add(mod_canon)

        if mod_canon in NEURO_MODALITIES:
            task = bids_dataset.get_bids_file_attribute("task", bids_file)
            if task:
                tasks.add(task)
            session = bids_dataset.get_bids_file_attribute("session", bids_file)
            if session:
                sessions.add(session)
            subject = bids_dataset.get_bids_file_attribute("subject", bids_file)
            if subject:
                subjects.add(subject)

    recording_modalities = sorted(
        list({m for m in modalities if m in NEURO_MODALITIES})
    )
    if not recording_modalities:
        recording_modalities = ["eeg"]

    (
        subjects_count,
        ages,
        sex_distribution,
        handedness_distribution,
    ) = _read_participants_demographics(bids_root)
    participants_path = bids_root / "participants.tsv"

    if participants_path.exists() and subjects_count > 0:
        folder_subjects = {
            d.name
            for d in bids_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        }
        if folder_subjects and subjects_count != len(folder_subjects):
            logging.warning(
                "%s: participants.tsv has %d rows but found %d sub-* folders "
                "(possible naming convention mismatch)",
                dataset_id,
                subjects_count,
                len(folder_subjects),
            )

    # Prefer subject count from validated neuro files over participants.tsv row count.
    if subjects:
        subjects_count = len(subjects)
    elif subjects_count == 0:
        subjects_count = len(
            [d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        )

    data_processed = (bids_root / "derivatives").exists()

    if source_adapter is not None:
        source_url = source_adapter.dataset_url()
    else:
        source_url = None
    if source_url is None and source == "gin":
        source_url = f"https://gin.g-node.org/EEGManyLabs/{dataset_id}"

    dataset_created_at = metadata.get("dataset_created_at")
    dataset_modified_at = metadata.get("dataset_modified_at")
    senior_author = metadata.get("senior_author")
    contact_info = metadata.get("contact_info")

    is_clinical = metadata.get("is_clinical")
    clinical_purpose = metadata.get("clinical_purpose")

    if not dataset_modified_at:
        # Try to get from manifest timestamps dict if present
        ts = metadata.get("timestamps", {})
        if isinstance(ts, dict):
            dataset_modified_at = ts.get("dataset_modified_at")
            dataset_created_at = ts.get("dataset_created_at") or dataset_created_at

    # Prefer API-supplied size_bytes; don't trust manifest total_size (wrong for git-annex).
    size_bytes = metadata.get("size_bytes")
    if size_bytes is None and bids_root.exists():
        size_bytes = sum(
            get_annex_file_size(f)
            for f in bids_root.rglob("*")
            if f.is_file() or f.is_symlink()
        )

    storage_info = _build_global_storage_info(dataset_id, source, bids_root)

    dataset = create_dataset(
        dataset_id=dataset_id,
        name=name,
        source=source,
        readme=readme,
        recording_modality=recording_modalities,
        datatypes=sorted(modalities) if modalities else recording_modalities,
        bids_version=bids_version,
        license=license_info,
        authors=authors if isinstance(authors, list) else [authors] if authors else [],
        funding=funding if isinstance(funding, list) else [funding] if funding else [],
        dataset_doi=dataset_doi,
        associated_paper_doi=_paper_doi_for_dataset(
            bids_root, readme, metadata, dataset_doi, source
        ),
        tasks=sorted(tasks),
        sessions=sorted(sessions),
        total_files=len(files),
        size_bytes=size_bytes,
        data_processed=data_processed,
        subjects_count=subjects_count,
        ages=ages,
        age_mean=sum(ages) / len(ages) if ages else None,
        participants_age_units=_read_participants_age_units(bids_root),
        sex_distribution=sex_distribution if sex_distribution else None,
        handedness_distribution=handedness_distribution
        if handedness_distribution
        else None,
        source_url=source_url,
        digested_at=digested_at,
        dataset_created_at=dataset_created_at,
        dataset_modified_at=dataset_modified_at,
        senior_author=senior_author,
        contact_info=contact_info,
        storage=storage_info,
        # Clinical classification
        is_clinical=is_clinical,
        clinical_purpose=clinical_purpose,
    )

    description_extras = _extract_dataset_description_extras(description)
    dataset.update(description_extras)

    return dict(dataset)


_BIDS_DESCRIPTION_DATASET_FIELDS: dict[str, str] = {
    "Acknowledgements": "acknowledgements",
    "HowToAcknowledge": "how_to_acknowledge",
    "EthicsApprovals": "ethics_approvals",
    "ReferencesAndLinks": "references_and_links",
    "GeneratedBy": "generated_by",
    "SourceDatasets": "source_datasets",
}


def _extract_dataset_description_extras(
    description: dict[str, Any],
) -> dict[str, Any]:
    """Pull extra dataset_description.json fields not already captured by extract_dataset_metadata."""
    out: dict[str, Any] = {}
    for desc_key, ds_key in _BIDS_DESCRIPTION_DATASET_FIELDS.items():
        val = description.get(desc_key)
        if val is None or val == "" or val == []:
            continue
        out[ds_key] = val
    return out
