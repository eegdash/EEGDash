"""Post-process the `sphinx-markdown-builder` output for agent consumption.

Run after `sphinx-build -b markdown` has populated `docs/_build/markdown/`.
Does two things:

1. Mirror every ``*.md`` file into the already-built HTML tree at the same
   relative path, so that `/install/install.md` resolves alongside
   `/install/install.html`. Satisfies buildwithfern.com's
   ``markdown-url-support`` check.

2. Concatenate the same markdown corpus into a single ``llms-full.txt`` at
   the HTML tree root. Some scanners (and LLM ingestion pipelines) look
   for this file alongside the curated ``llms.txt`` index.

The script is intentionally tolerant: if the markdown builder failed on a
page (common for pages with heavy sphinx-design / gallery content), we
skip the missing file and continue. No hard failures.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _section_header(rel_path: Path) -> str:
    """Header separating one page's markdown from the next in llms-full.txt."""
    return (
        f"\n\n<!-- ====================================================== -->\n"
        f"<!-- SOURCE: {rel_path.as_posix()} -->\n"
        f"<!-- ====================================================== -->\n\n"
    )


def copy_siblings(markdown_root: Path, html_root: Path) -> int:
    """Copy every .md under ``markdown_root`` to the matching path in html_root."""
    copied = 0
    for md_path in markdown_root.rglob("*.md"):
        rel = md_path.relative_to(markdown_root)
        target = html_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(md_path, target)
        copied += 1
    return copied


def _llms_full_sort_key(rel_posix: str) -> tuple[int, str]:
    """Deterministic page order for llms-full.txt.

    Narrative docs come first, then API reference, then the long
    sphinx-gallery output, so LLMs read the high-value content before
    they hit context limits.
    """
    if rel_posix == "index.md":
        return (0, rel_posix)
    if rel_posix.startswith("install/"):
        return (1, rel_posix)
    if rel_posix.startswith("user_guide"):
        return (2, rel_posix)
    if rel_posix == "dataset_summary.md":
        return (3, rel_posix)
    if rel_posix.startswith("api/"):
        return (5, rel_posix)
    if rel_posix.startswith("generated/"):
        return (6, rel_posix)
    return (4, rel_posix)


def build_llms_full(markdown_root: Path, output_path: Path) -> int:
    """Concatenate all markdown files into a single llms-full.txt."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_files = sorted(
        markdown_root.rglob("*.md"),
        key=lambda p: _llms_full_sort_key(p.relative_to(markdown_root).as_posix()),
    )

    with output_path.open("w", encoding="utf-8") as out:
        out.write("# EEGDash — full markdown corpus\n\n")
        out.write(
            "Concatenation of every Sphinx-rendered markdown page on "
            "eegdash.org, produced at build time for LLM ingestion.\n"
        )
        out.write(
            "See the curated index at <https://eegdash.org/llms.txt> for "
            "navigation; this file is the raw corpus.\n"
        )
        for md_path in md_files:
            rel = md_path.relative_to(markdown_root)
            out.write(_section_header(rel))
            out.write(md_path.read_text(encoding="utf-8"))

    return len(md_files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Sphinx markdown-builder output root (usually _build/markdown).",
    )
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
        help="HTML output root to mirror .md siblings into (usually _build/html).",
    )
    parser.add_argument(
        "--llms-full",
        type=Path,
        default=None,
        help="Optional explicit path for llms-full.txt. Defaults to <target>/llms-full.txt.",
    )
    args = parser.parse_args()

    source: Path = args.source
    target: Path = args.target

    if not source.is_dir():
        print(f"[build_markdown_assets] source {source} missing; nothing to do.")
        return 0
    if not target.is_dir():
        print(f"[build_markdown_assets] target {target} missing; nothing to do.")
        return 0

    llms_full = args.llms_full or (target / "llms-full.txt")

    copied = copy_siblings(source, target)
    total = build_llms_full(source, llms_full)

    print(
        f"[build_markdown_assets] copied {copied} .md siblings and wrote "
        f"{total}-page llms-full.txt ({llms_full.stat().st_size:,} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
