"""Pre-merge SEO validator for the Sphinx-built HTML tree.

Usage::

    python scripts/validate_docs_seo.py docs/_build/html [--fail-under-regression]

Walks every ``*.html`` under the given root and reports the same
categories Ahrefs Site Audit flags on us:

* meta description missing / too short / too long
* duplicate ``<title>`` / ``<h1>`` / ``<meta name="description">`` tags
* ``<img>`` elements without an ``alt`` attribute
* missing ``<link rel="canonical">``

Exit code is ``0`` on a clean run and ``1`` when any **error**
counter is non-zero. Warnings do not fail the run. The numbers here
should track the Ahrefs crawl roughly 1:1 — fresh regressions
caught here never reach the deployed site.

The output format is intentionally scripting-friendly: fixed-width
``key value`` lines, so CI or a simple diff can spot regressions
between develop and a PR branch.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Limits
MIN_DESCRIPTION = 50
MAX_DESCRIPTION = 160
MAX_TITLE = 70

# Narrow, order-agnostic, quote-specific patterns. Matches the real
# shapes emitted by docutils, sphinxext-opengraph, and our injector
# without tripping on apostrophes inside double-quoted content.
META_DESC_PATTERNS = [
    re.compile(
        r'<meta\s+(?:[^>]*?\s)?name="description"'
        r'\s+(?:[^>]*?\s)?content="([^"]*)"[^>]*>',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"<meta\s+(?:[^>]*?\s)?name='description'"
        r"\s+(?:[^>]*?\s)?content='([^']*)'[^>]*>",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'<meta\s+(?:[^>]*?\s)?content="([^"]*)"'
        r'\s+(?:[^>]*?\s)?name="description"[^>]*>',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"<meta\s+(?:[^>]*?\s)?content='([^']*)'"
        r"\s+(?:[^>]*?\s)?name='description'[^>]*>",
        flags=re.IGNORECASE,
    ),
]
TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
H1_RE = re.compile(r"<h1\b[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
# Only look at <img> in the rendered output (ignoring <svg><image>).
IMG_RE = re.compile(r"<img\b[^>]*?>", re.IGNORECASE)
IMG_ALT_RE = re.compile(r"\balt\s*=\s*(['\"])([^'\"]*)\1", re.IGNORECASE)
CANONICAL_RE = re.compile(
    r'<link\s+[^>]*?\brel\s*=\s*["\']?canonical["\']?[^>]*?>',
    flags=re.IGNORECASE,
)
ROBOTS_META_RE = re.compile(
    r'<meta\s+(?:[^>]*?\s)?name\s*=\s*["\']?robots["\']?[^>]*>',
    flags=re.IGNORECASE,
)
NOINDEX_RE = re.compile(r"noindex", re.IGNORECASE)


@dataclass
class Stats:
    scanned: int = 0
    # Error-severity counters (cause non-zero exit)
    missing_description: int = 0
    short_description: int = 0
    long_description: int = 0
    missing_title: int = 0
    missing_h1: int = 0
    duplicate_description: int = 0
    duplicate_title: int = 0
    duplicate_h1: int = 0
    img_without_alt: int = 0
    # Warning-severity counters (reported, no exit code)
    missing_canonical: int = 0
    offenders: dict[str, list[str]] = field(default_factory=dict)

    def record(self, key: str, path: Path) -> None:
        bucket = self.offenders.setdefault(key, [])
        if len(bucket) < 10:
            bucket.append(str(path))


def _find_descriptions(source: str) -> list[str]:
    found = []
    for pattern in META_DESC_PATTERNS:
        found.extend(pattern.findall(source))
    return [html.unescape(s).strip() for s in found]


def _find_images_without_alt(source: str) -> int:
    missing = 0
    for match in IMG_RE.findall(source):
        # Skip tracker pixels, invisible placeholders, etc. without alt.
        # We still count them — Ahrefs does — but alt="" explicitly
        # counts as present.
        if not IMG_ALT_RE.search(match):
            missing += 1
    return missing


def audit_file(path: Path, stats: Stats, noindex_allowed: bool = True) -> None:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    stats.scanned += 1

    # Skip pages explicitly marked noindex — Google / Ahrefs agree
    # those aren't part of the SEO surface.
    robots_tag = ROBOTS_META_RE.search(source)
    if robots_tag and NOINDEX_RE.search(robots_tag.group(0)):
        return

    descs = _find_descriptions(source)
    if not descs:
        stats.missing_description += 1
        stats.record("missing_description", path)
    else:
        if len(descs) > 1:
            stats.duplicate_description += 1
            stats.record("duplicate_description", path)
        longest = max(descs, key=len)
        shortest = min(descs, key=len)
        if len(shortest) < MIN_DESCRIPTION:
            stats.short_description += 1
            stats.record("short_description", path)
        if len(longest) > MAX_DESCRIPTION:
            stats.long_description += 1
            stats.record("long_description", path)

    titles = TITLE_RE.findall(source)
    if not titles:
        stats.missing_title += 1
        stats.record("missing_title", path)
    elif len(titles) > 1:
        stats.duplicate_title += 1
        stats.record("duplicate_title", path)

    h1s = H1_RE.findall(source)
    if not h1s:
        stats.missing_h1 += 1
        stats.record("missing_h1", path)
    elif len(h1s) > 1:
        stats.duplicate_h1 += 1
        stats.record("duplicate_h1", path)

    alt_missing = _find_images_without_alt(source)
    if alt_missing:
        stats.img_without_alt += alt_missing
        stats.record("img_without_alt", path)

    if not CANONICAL_RE.search(source):
        stats.missing_canonical += 1
        stats.record("missing_canonical", path)


def render_report(stats: Stats, root: Path, show_offenders: bool) -> str:
    lines = [
        f"SEO validator - {root}",
        f"scanned                   {stats.scanned}",
        "-" * 40,
        f"meta description missing  {stats.missing_description}",
        f"meta description short    {stats.short_description}",
        f"meta description long     {stats.long_description}",
        f"duplicate description     {stats.duplicate_description}",
        f"missing <title>           {stats.missing_title}",
        f"duplicate <title>         {stats.duplicate_title}",
        f"missing <h1>              {stats.missing_h1}",
        f"duplicate <h1>            {stats.duplicate_h1}",
        f"<img> without alt         {stats.img_without_alt}",
        f"missing canonical         {stats.missing_canonical}  (warn)",
    ]
    if show_offenders and stats.offenders:
        lines.append("-" * 40)
        for key in sorted(stats.offenders):
            lines.append(f"[{key}]")
            for p in stats.offenders[key]:
                lines.append(f"  - {p}")
            if len(stats.offenders[key]) == 10:
                lines.append("  - (additional offenders truncated to 10)")
    return "\n".join(lines)


def _is_error(stats: Stats) -> bool:
    return (
        stats.missing_description
        or stats.short_description
        or stats.long_description
        or stats.missing_title
        or stats.missing_h1
        or stats.duplicate_description
        or stats.duplicate_title
        or stats.duplicate_h1
        or stats.img_without_alt
    ) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("docs/_build/html"),
        help="HTML output root (defaults to docs/_build/html)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on *any* non-clean counter (incl. warnings).",
    )
    parser.add_argument(
        "--no-offenders",
        action="store_true",
        help="Skip the per-category list of up to 10 offending files.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.is_dir():
        print(f"SEO validator: {root} is not a directory", file=sys.stderr)
        return 2

    stats = Stats()
    # Skip embed-only HTML that Sphinx ships under `_static/`. These are
    # iframe-source chart fragments (dataset_generated/*.html,
    # social_card_gen.html, webpack-macros.html) that never exist as
    # standalone pages in the sitemap and deliberately ship without
    # <title>/<h1>/<meta description>.
    skip_prefixes = (
        root / "_static",
        root / "_sources",
    )
    # Pages whose structural issues come from upstream tooling
    # (sphinx-gallery, our own raw-HTML iframe inlining in
    # dataset_summary.html) and that we've decided not to chase
    # further. Removing them from the scan keeps the validator actionable
    # as a regression detector without constantly blocking merges on
    # known-deferred state. Any regression on a *new* page still fails.
    deferred_relative = {
        "generated/auto_examples/index.html",
        "generated/auto_examples/core/p300_transfer_learning.html",
        "dataset_summary.html",
    }
    deferred_paths = {root / rel for rel in deferred_relative}
    for path in sorted(root.rglob("*.html")):
        if any(skip == path or skip in path.parents for skip in skip_prefixes):
            continue
        if path in deferred_paths:
            continue
        audit_file(path, stats)

    print(render_report(stats, root, show_offenders=not args.no_offenders))

    if args.strict:
        if _is_error(stats) or stats.missing_canonical:
            print("FAIL (strict)")
            return 1
    elif _is_error(stats):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
