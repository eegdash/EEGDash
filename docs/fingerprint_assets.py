"""Post-build asset fingerprinter for the Sphinx HTML output.

Usage::

    python fingerprint_assets.py --html-dir _build/html

Walks ``<html-dir>/_static`` for the CSS/JS files we author — not the
theme's own files — computes a short sha256 prefix, renames every
matched file to ``stem.<hash>.ext``, and rewrites every reference in
``*.html`` and ``*.css`` under ``<html-dir>`` (plus ``sitemap.xml``
where applicable).

Why this is useful even though Sphinx already appends ``?v=<hash>``
query strings:

* Query-string hashes are ignored by some HTTP caches and by Fastly's
  default URL normalization. Filename hashes survive every hop.
* When / if we move eegdash.org behind a reverse proxy that emits
  ``Cache-Control: immutable``, only filename-hashed assets are safe to
  mark immutable.
* It's idempotent — a second pass with no file changes does nothing.

Why it *doesn't* fingerprint theme files:

* pydata-sphinx-theme already hashes its own assets (``?digest=<hash>``)
  and an asset shared across many pages under a stable filename remains
  easy to debug when something breaks.
* Theme assets ship from every Sphinx install at the same path, so any
  custom rename increases the blast radius of a theme upgrade.

See ``docs/SEO_AUDIT_LOG.md`` for the broader PSI / Ahrefs context.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

# Files we own under _static/. Theme files (pydata-sphinx-theme,
# sphinx-design, sphinx-gallery, pygments, copybutton, graphviz) are
# intentionally excluded — they already carry Sphinx's own `?digest`
# hash and renaming them increases theme-upgrade risk. Keep this list
# small and explicit so additions are deliberate.
OWNED_GLOBS = (
    "custom.css",
    "css/*.css",
    "js/*.js",
    "lib/*.js",
    "lib/*.css",
)

HASH_LENGTH = 8
# Match `.stem.<hash>.<ext>` so a second pass doesn't double-hash an
# already-fingerprinted file. Hash segment must be exactly our length
# of hex characters.
ALREADY_FINGERPRINTED = re.compile(r"\.[0-9a-f]{%d}\." % HASH_LENGTH)


def _short_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:HASH_LENGTH]


def _fingerprint_name(path: Path) -> Path:
    """Return ``stem.HASH.ext``. Preserves compound suffixes like
    ``min.js``: ``fuse.min.js`` -> ``fuse.min.HASH.js``.
    """
    if ALREADY_FINGERPRINTED.search(path.name):
        return path
    digest = _short_hash(path)
    # Split on the LAST dot so `.min.js` keeps its `.min` suffix.
    stem, ext = path.stem, path.suffix
    return path.with_name(f"{stem}.{digest}{ext}")


def _collect_owned_assets(static_root: Path) -> list[Path]:
    matched: list[Path] = []
    for pattern in OWNED_GLOBS:
        matched.extend(static_root.glob(pattern))
    # Dedup + filter out already-fingerprinted + missing files.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in matched:
        if not p.is_file() or p in seen:
            continue
        if ALREADY_FINGERPRINTED.search(p.name):
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _rewrite_references(html_root: Path, rename_map: dict[str, str]) -> tuple[int, int]:
    """Replace old asset paths with hashed ones in every html/css/xml
    file under ``html_root``.

    ``rename_map`` keys are full paths relative to ``html_root`` (e.g.
    ``_static/css/custom.css``), values are the new path with the hash
    in the filename (e.g. ``_static/css/custom.abcd1234.css``). We use
    the full relative path — not just the basename — so two files with
    the same filename in different directories don't alias to the same
    hash. Returns ``(files_scanned, files_modified)``.
    """
    if not rename_map:
        return (0, 0)

    # Longest-path first so `_static/css/custom.css` matches before
    # `_static/custom.css` when both appear on the same line.
    sorted_keys = sorted(rename_map.keys(), key=len, reverse=True)
    alternation = "|".join(re.escape(k) for k in sorted_keys)
    # Anchors:
    #   * lookbehind accepts `/` so we match paths with any number of
    #     ``../`` prefixes — subpages reference assets as
    #     ``href="../../_static/custom.css"`` while the homepage uses
    #     ``href="_static/custom.css"``.
    #   * lookbehind also accepts typical attribute / URL-fn start
    #     delimiters (quote, equals, paren, whitespace, start-of-line).
    #   * lookahead ensures we stop at the end of the token (query
    #     string, fragment, closing quote/paren, whitespace, EOL).
    pattern = re.compile(
        rf'(?<=[\s"\'(=/])({alternation})(?=[\s"\')?#]|$)',
        flags=re.MULTILINE,
    )

    scanned = 0
    modified = 0
    for suffix in (".html", ".css", ".xml"):
        for path in html_root.rglob(f"*{suffix}"):
            scanned += 1
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            updated = pattern.sub(lambda m: rename_map[m.group(1)], text)
            if updated != text:
                path.write_text(updated, encoding="utf-8")
                modified += 1
    return scanned, modified


def fingerprint(html_root: Path) -> dict[str, str]:
    static_root = html_root / "_static"
    if not static_root.is_dir():
        print(f"fingerprint_assets: {static_root} missing; skipping")
        return {}

    owned = _collect_owned_assets(static_root)
    if not owned:
        print("fingerprint_assets: nothing to rename")
        return {}

    # Keys/values are paths relative to ``html_root`` so two same-named
    # files under different directories (e.g. `_static/custom.css` and
    # `_static/css/custom.css`) don't alias to one hash.
    rename_map: dict[str, str] = {}
    for path in owned:
        new_path = _fingerprint_name(path)
        if new_path == path:
            continue
        path.rename(new_path)
        old_rel = path.relative_to(html_root).as_posix()
        new_rel = new_path.relative_to(html_root).as_posix()
        rename_map[old_rel] = new_rel
        print(f"  {old_rel} -> {new_rel}")

    scanned, modified = _rewrite_references(html_root, rename_map)
    print(
        f"fingerprint_assets: renamed {len(rename_map)} files, "
        f"rewrote refs in {modified}/{scanned} text files"
    )
    return rename_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=Path("_build/html"),
        help="Path to the Sphinx HTML output root (defaults to _build/html).",
    )
    args = parser.parse_args()

    fingerprint(args.html_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
