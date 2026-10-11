"""Build the EEGDash Corpus fragments from the synced data (no network).

Usage::

    python prepare_corpus.py                       # from docs/: plot_corpus/data -> source/_static/corpus_generated
    python prepare_corpus.py --data plot_corpus/data --out source/_static/corpus_generated
    python prepare_corpus.py --if-stale            # skip when the outputs are newer than the data and the renderer

Writes one ``<key>.html`` per chart or table, ``numbers.inc`` (the RST
substitutions), ``manifest.json`` (keys, captions, sizes) and the CSV
downloads. The "as of" date is ``MANIFEST.json``'s ``synced``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plot_corpus.fragments import build, is_stale  # noqa: E402

DATA = HERE / "plot_corpus" / "data"
OUT = HERE / "source" / "_static" / "corpus_generated"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA, help="synced data folder")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder for the fragments")
    ap.add_argument("--if-stale", action="store_true", help="only build when the outputs are out of date")
    a = ap.parse_args(argv)
    if a.if_stale and not is_stale(a.data, a.out):
        print(f"{a.out}: up to date")
        return 0
    m = build(a.data, a.out)
    total = sum(f["bytes"] for f in m["fragments"].values())
    print(f"{a.out}: {len(m['fragments'])} fragments, {total:,} bytes, data as of {m['synced']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
