#!/usr/bin/env python3
"""Shrink the two giant auto-generated dataset API pages.

`sphinx-apidoc` writes ``.. automodule:: eegdash.dataset[.dataset] :members:``,
which dumps the ~753 dynamically-registered per-dataset classes (they set
``__module__="eegdash.dataset.dataset"`` and land in ``__all__``), producing two
~7MB near-orphan pages. The canonical per-dataset docs are the Dataset Brief
pages (``eegdash.dataset.DS*.html``), already linked from the ``source_*.rst``
toctrees. Restrict these two ``automodule`` blocks to the hand-written classes.

Runs after ``sphinx-apidoc`` in the Makefile ``apidoc`` target.
"""

from pathlib import Path

API_DIR = Path(__file__).parent / "source" / "api" / "dataset"

# Exact ``:members:`` allow-lists, keyed by generated file name.
PATCHES = {
    "eegdash.dataset.dataset.rst": ":members: EEGDashDataset, EEGChallengeDataset",
    "eegdash.dataset.rst": ":members: EEGDashDataset, EEGChallengeDataset, EEGDashRaw",
}


def main() -> None:
    for name, members_line in PATCHES.items():
        path = API_DIR / name
        if not path.exists():
            print(f"prune_apidoc: {name} not found (skipped)")
            continue
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        patched = []
        changed = False
        for line in lines:
            if line.strip() == ":members:":
                indent = line[: len(line) - len(line.lstrip())]
                patched.append(f"{indent}{members_line}\n")
                changed = True
            else:
                patched.append(line)
        path.write_text("".join(patched), encoding="utf-8")
        print(f"prune_apidoc: {'patched' if changed else 'no :members: line in'} {name}")


if __name__ == "__main__":
    main()
