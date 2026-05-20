"""Single source of truth for the dataset exclusion list.

Both :mod:`eegdash.dataset.registry` (which uses the set to keep
problematic dataset IDs out of the public class registry) and
:mod:`eegdash.dataset.snapshot` (which uses it to filter rows in the
docs build) need this set. Keeping it here — in a tiny leaf module
neither of those imports recursively — avoids:

1. The maintenance trap of two divergent copies (which is exactly what
   happened during the B1 snapshot refactor: snapshot.py grew its own
   21-entry subset while registry.py kept the canonical 37-entry list,
   and the snapshot's copy quietly became the only effective filter
   for the docs build).
2. The circular-import dance the previous co-located definitions had
   to do (``registry`` and ``snapshot`` import from each other lazily
   inside functions).

The ingestion scripts (``scripts/ingestions/3_digest.py``,
``scripts/ingestions/5_inject.py``) maintain their own copies for now
— they predate this module and aren't part of the docs/runtime import
graph. Aligning them is tracked separately.
"""

from __future__ import annotations

# Hardcoded list of datasets to skip when registering / filtering.
#
# This is the canonical pre-refactor production filter (the older,
# more curated list that the registry shipped with). If you need to
# add/remove an entry, this is the only place to do it.
EXCLUDED_DATASETS: frozenset[str] = frozenset(
    {
        "ABUDUKADI",
        "ABUDUKADI_2",
        "ABUDUKADI_3",
        "ABUDUKADI_4",
        "AILIJIANG",
        "AILIJIANG_3",
        "AILIJIANG_4",
        "AILIJIANG_5",
        "AILIJIANG_7",
        "AILIJIANG_8",
        "BAIHETI",
        "BAIHETI_2",
        "BAIHETI_3",
        "BIAN_3",
        "BIN_27",
        "BLIX",
        "BOJIN",
        "BOUSSAGOL",
        "AISHENG",
        "ACHOLA",
        "ANASHKIN",
        "ANJUM",
        "BARBIERI",
        "BIN_8",
        "BIN_9",
        "BING_4",
        "BING_8",
        "BOWEN_4",
        "AZIZAH",
        "BAO",
        "BAO-YOU",
        "BAO_2",
        "BENABBOU",
        "BING",
        "BOXIN",
        "test",
        "ds003380",
    }
)


__all__ = ["EXCLUDED_DATASETS"]
