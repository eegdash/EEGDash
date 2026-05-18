"""Markdown -> RST conversion for dataset README content.

Pure text transformation with no Sphinx / docutils dependency, so this
module is importable in isolation (useful for unit tests and tooling
that wants to reuse the same README-conversion behaviour as the
documentation build).

The public entry point is :func:`convert_readme_to_rst`. The module
also exposes :func:`_convert_readme_to_rst` as a back-compat alias.

Helpers are all module-scoped (the original lint rule for the
monolithic ``dataset_page.py`` forbade nested functions, so they were
hoisted out then -- this split preserves that structure).
"""

from __future__ import annotations

import re


def _is_decorative_line(s: str) -> bool:
    """Check if line is purely decorative (em-dashes, dashes, equals, etc.)."""
    s = s.strip()
    if len(s) < 3:
        return False
    return bool(re.match(r"^[—\-=_*#~]+$", s)) and len(set(s)) <= 2


# Module-level helpers for ``_convert_readme_to_rst``. The README-to-RST
# converter is large and these helpers were originally written inline; the
# no-nested-functions check requires them at module scope.


def _sanitize_header_text(title: str) -> str:
    """Strip inline markdown that breaks bold headers in RST."""
    title = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", title)
    title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)
    title = re.sub(r"\[([^\]]+)\]\[([^\]]+)\]", r"\1", title)
    title = re.sub(r"`([^`]+)`", r"\1", title)
    title = re.sub(r"\*\*([^*]+)\*\*", r"\1", title)
    title = re.sub(r"\*([^*]+)\*", r"\1", title)
    title = re.sub(r"__([^_]+)__", r"\1", title)
    title = re.sub(r"_([^_]+)_", r"\1", title)
    title = re.sub(r"<(https?://[^>]+)>", r"\1", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"(\w)_(?=[\s.,;:!?\)\]\}]|$)", r"\1\\_", title)
    title = title.replace("|", "\\|")
    return title


def _convert_code_fence(match: re.Match) -> str:
    """``re.sub`` callback for ```...``` blocks -> RST code-block directives."""
    lang = match.group(1) or "text"
    code = match.group(2)
    indented_code = "\n".join("   " + line for line in code.split("\n"))
    return f"\n.. code-block:: {lang}\n\n{indented_code}\n"


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^\|[-:\s|]+\|$", stripped))


def _is_tree_line(line: str) -> bool:
    return bool(re.match(r"^\s*[\|│├└][\s─\-\|│├└]*", line))


def _stash_inline_code_factory(spans: list[str]):
    """Return a ``re.sub`` callback that appends to ``spans`` and emits a placeholder."""

    def stash(m: re.Match) -> str:
        spans.append(m.group(1))
        return f"\x00INLINE_CODE_{len(spans) - 1}\x00"

    return stash


def _replace_md_image(m: re.Match) -> str:
    alt = m.group(1).strip()
    url = m.group(2).strip()
    text = alt or url
    return f"`{text} <{url}>`__"


def _replace_ref_image_factory(ref_link_defs: dict[str, str]):
    """Return a ``re.sub`` callback that resolves ``![alt][ref]`` against ``ref_link_defs``."""

    def replace(m: re.Match) -> str:
        alt = m.group(1).strip()
        ref = m.group(2).strip()
        url = ref_link_defs.get(ref, "")
        if not url:
            return m.group(0)
        text = alt or url
        return f"`{text} <{url}>`__"

    return replace


def _stash_rst_link_factory(stash: list[str]):
    """Return a ``re.sub`` callback that stashes a literal RST link reference."""

    def replace(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"\x00RST_LINK_{len(stash) - 1}\x00"

    return replace


def _replace_ref_link_factory(ref_link_defs: dict[str, str]):
    """Return a ``re.sub`` callback that resolves ``[text][ref]`` against ``ref_link_defs``."""

    def replace(m: re.Match) -> str:
        text = m.group(1)
        ref = m.group(2)
        url = ref_link_defs.get(ref, "")
        if url:
            return f"`{text} <{url}>`__"
        return m.group(0)

    return replace


def _restore_inline_code_factory(spans: list[str]):
    """Return a ``re.sub`` callback that pops back into the stashed inline-code spans."""

    def replace(m: re.Match) -> str:
        code = spans[int(m.group(1))]
        return f"``{code}``"

    return replace


def convert_readme_to_rst(text: str) -> str:
    """Convert README content to RST (headers become bold, not section headers).

    Handles markdown (#) headers, RST-style underline headers, and decorative
    box-style headers (em-dash lines) to avoid messing up the document structure.

    Also handles various Markdown constructs that conflict with RST syntax:
    - Markdown tables (wrapped in code blocks)
    - Directory trees (wrapped in code blocks)
    - Code fences (converted to RST code-block directives)
    - Reference-style links [text][1] with [1]: url
    - Trailing underscores (escaped to avoid hyperlink target errors)
    - Pipe characters (escaped to avoid substitution reference errors)
    - Orphan asterisks in file patterns (escaped)
    - Markdown checkboxes (converted to simple list items)
    """
    text = text.lstrip("﻿")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Downgrade literal HTML headers so they don't collide with the
    # page's own `<h1>`. Upstream READMEs sometimes contain raw `<h1>`
    # tags, which Sphinx passes through verbatim and Ahrefs then flags
    # as "Multiple H1 tags".
    text = re.sub(
        r"<(/?)h1(\b[^>]*)>",
        r"<\g<1>h3\g<2>>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"<(/?)h2(\b[^>]*)>",
        r"<\g<1>h4\g<2>>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\t+", lambda m: "  " * len(m.group(0)), text, flags=re.MULTILINE)

    # === Phase 1: Extract reference-style link definitions ===
    ref_link_defs: dict[str, str] = {}
    ref_def_pattern = re.compile(r"^\s*\[([^\]]+)\]:\s*(.+?)\s*$", re.MULTILINE)
    for match in ref_def_pattern.finditer(text):
        ref_link_defs[match.group(1)] = match.group(2)
    text = ref_def_pattern.sub("", text)

    # === Phase 2: Convert markdown code fences to RST code blocks ===
    code_fence_pattern = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)
    text = code_fence_pattern.sub(_convert_code_fence, text)

    text = re.sub(r"``\s*`", "``", text)
    text = re.sub(r"`\s*``", "``", text)

    lines = text.split("\n")
    result: list[str] = []
    i = 0
    in_code_block = False

    # === Phase 3: Detect and wrap markdown tables ===
    is_table_line = _is_table_line
    is_table_separator = _is_table_separator
    is_tree_line = _is_tree_line

    table_regions: set[int] = set()
    tree_regions: set[int] = set()

    j = 0
    while j < len(lines):
        if is_table_line(lines[j]) or is_table_separator(lines[j]):
            while j < len(lines) and (
                is_table_line(lines[j])
                or is_table_separator(lines[j])
                or lines[j].strip() == ""
            ):
                if lines[j].strip():
                    table_regions.add(j)
                j += 1
        else:
            j += 1

    j = 0
    while j < len(lines):
        if is_tree_line(lines[j]):
            while j < len(lines) and (is_tree_line(lines[j]) or lines[j].strip() == ""):
                if lines[j].strip():
                    tree_regions.add(j)
                j += 1
        else:
            j += 1

    # === Phase 4: Process lines ===
    table_block: list[str] = []
    tree_block: list[str] = []
    in_table = False
    in_tree = False
    in_blockquote = False
    prev_was_list_item = False

    while i < len(lines):
        line = lines[i]

        if line.strip().endswith("::") or line.strip().startswith(".. code-block::"):
            in_code_block = True
        elif (
            in_code_block
            and line.strip()
            and not line.startswith(" ")
            and not line.startswith("\t")
        ):
            in_code_block = False

        is_blockquote_line = False
        if not in_code_block:
            blockquote_match = re.match(r"^\s*>\s?", line)
            if blockquote_match:
                is_blockquote_line = True
                line = line[blockquote_match.end() :]
            elif in_blockquote:
                if result and result[-1].strip():
                    result.append("")
                in_blockquote = False

        if i in table_regions and not in_tree and not is_blockquote_line:
            if not in_table:
                in_table = True
                table_block = []
                if result and result[-1].strip():
                    result.append("")
                result.append(".. code-block:: text")
                result.append("")
            table_block.append("   " + line)
            result.append("   " + line)
            i += 1
            continue
        elif in_table:
            in_table = False
            result.append("")

        if i in tree_regions and not in_table and not is_blockquote_line:
            if not in_tree:
                in_tree = True
                tree_block = []
                if result and result[-1].strip():
                    result.append("")
                result.append(".. code-block:: text")
                result.append("")
            tree_block.append("   " + line)
            result.append("   " + line)
            i += 1
            continue
        elif in_tree:
            in_tree = False
            result.append("")

        if _is_decorative_line(line):
            if i + 2 < len(lines):
                potential_title = lines[i + 1].strip()
                next_decorative = lines[i + 2]
                if potential_title and _is_decorative_line(next_decorative):
                    result.append("")
                    result.append(f"**{_sanitize_header_text(potential_title)}**")
                    result.append("")
                    i += 3
                    prev_was_list_item = False
                    continue
            i += 1
            continue

        if not is_blockquote_line:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if header_match:
                title = _sanitize_header_text(header_match.group(2).strip())
                result.append("")
                result.append(f"**{title}**")
                result.append("")
                i += 1
                prev_was_list_item = False
                continue

        if not is_blockquote_line and i + 1 < len(lines) and len(line.strip()) < 80:
            next_line = lines[i + 1]
            underline_match = re.match(r"^([=\-~^\"\'`—]+)$", next_line.strip())
            if (
                underline_match
                and len(next_line.strip()) >= 3
                and line.strip()
                and len(set(next_line.strip())) == 1
            ):
                title = _sanitize_header_text(line.strip())
                result.append("")
                result.append(f"**{title}**")
                result.append("")
                i += 2
                prev_was_list_item = False
                continue

        if not in_code_block:
            line = re.sub(r"^(\s*)[•⁃‣]\s*", r"\1- ", line)
            line = re.sub(r"^(\s*)-\s*\[([ xX]?)\]\s*", r"\1- ", line)

            inline_code_spans: list[str] = []

            line = re.sub(
                r"(?<![\\`])`([^`]+)`(?!`)",
                _stash_inline_code_factory(inline_code_spans),
                line,
            )

            line = re.sub(
                r"\[!\[([^\]]*)\]\([^)]+\)\]\(([^)]+)\)",
                lambda m: f"`{m.group(1).strip() or m.group(2).strip()} <{m.group(2).strip()}>`__",
                line,
            )

            line = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace_md_image, line)

            line = re.sub(
                r"!\[([^\]]*)\]\[([^\]]+)\]",
                _replace_ref_image_factory(ref_link_defs),
                line,
            )

            rst_link_stash: list[str] = []

            line = re.sub(
                r"`[^`]+\s<[^>]+>`__",
                _stash_rst_link_factory(rst_link_stash),
                line,
            )

            line = re.sub(r"<(https?://[^>]+)>", r"`\1 <\1>`__", line)

            for idx, stashed in enumerate(rst_link_stash):
                line = line.replace(f"\x00RST_LINK_{idx}\x00", stashed)

            line = re.sub(
                r"\[([^\]]+)\]\[([^\]]+)\]",
                _replace_ref_link_factory(ref_link_defs),
                line,
            )

            line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"`\1 <\2>`__", line)

            line = re.sub(r"([a-zA-Z0-9])_(?=[\s.,;:!?\)\]\}]|$)", r"\1\_", line)

            line = re.sub(r"\|([^\|]+)\|", r"\\|\1\\|", line)
            if (
                line.strip().startswith("|")
                and i not in table_regions
                and i not in tree_regions
            ):
                line = line.replace("|", "\\|")

            line = re.sub(r"\\?\*\*([^*:]+:)\*\*?(?=\s|$)", r"*\1*", line)
            line = re.sub(r"\\?\*(\*\*[^*]+\*\*)(?=[a-zA-Z])", r"\1 ", line)
            line = re.sub(r"(\*\*[^*]+)\*\s*\\\*", r"\1**", line)
            line = line.replace("\\*", "\x00ESCAPED_ASTERISK\x00")
            line = re.sub(r"\*(?=[._/])", r"\\*", line)
            line = re.sub(r"(?<=[/\-])\*", r"\\*", line)

            asterisk_count = line.count("*") - line.count("\\*")
            if asterisk_count % 2 == 1:
                line = re.sub(r"(?<=\s)\*(?=\S)", r"\\*", line)

            line = line.replace("\x00ESCAPED_ASTERISK\x00", "\\*")

            line = re.sub(r"(\*\*[^*]+\*\*)(?=[a-zA-Z])", r"\1 ", line)
            line = re.sub(r"([a-zA-Z])(\*\*[^*]+\*\*)", r"\1 \2", line)
            line = re.sub(r"\*\*\s+([^*]+)\*\*", r"**\1**", line)
            line = re.sub(r"\*\*([^*]+)\s+\*\*", r"**\1**", line)
            line = re.sub(r"\*\s+([^*]+)\*(?!\*)", r"*\1*", line)
            line = re.sub(r"\*([^*]+)\s+\*(?!\*)", r"*\1*", line)
            line = re.sub(r"\*\*\[([^\]]+)\]\*\s*\\?\*", r"**[\1]**", line)
            line = re.sub(r"(\*\*[^*]+)\\(\*\*)$", r"\1\2", line)
            line = re.sub(r"(?<=\s)\\?\*(\*\*[^*]+\*\*)", r"\1", line)
            line = re.sub(r"^\\?\*(\*\*[^*]+\*\*)", r"\1", line)
            line = re.sub(r"\*\s+(\w+)\*(?!\*)", r"*\1*", line)
            line = re.sub(r"(\*[^*]+\*)\s*\\\*$", r"\1", line)
            line = re.sub(r"(\*\*[^*]+)\*\s*\\\*$", r"\1**", line)

            if inline_code_spans:
                line = re.sub(
                    r"\x00INLINE_CODE_(\d+)\x00",
                    _restore_inline_code_factory(inline_code_spans),
                    line,
                )

            line = re.sub(r"``\\?`", "``", line)
            line = re.sub(r"`\\?``", "``", line)
            line = re.sub(r"^```\s*$", "", line)

            backtick_count = line.count("`") - line.count("\\`") - 2 * line.count("``")
            if backtick_count % 2 == 1:
                parts = []
                in_backtick = False
                for j, char in enumerate(line):
                    if char == "`" and (j == 0 or line[j - 1] != "\\"):
                        if in_backtick:
                            in_backtick = False
                            parts.append(char)
                        else:
                            remaining = line[j + 1 :]
                            if "`" in remaining and not remaining.startswith(" "):
                                in_backtick = True
                                parts.append(char)
                            else:
                                parts.append("\\`")
                    else:
                        parts.append(char)
                line = "".join(parts)

        is_list_item = bool(
            re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+[.)]\s+", line)
        )

        if is_blockquote_line and not in_blockquote:
            if result and result[-1].strip():
                result.append("")
            in_blockquote = True

        if (
            prev_was_list_item
            and not is_list_item
            and line.strip()
            and not line.startswith(" ")
        ):
            if result and result[-1].strip():
                result.append("")

        if is_blockquote_line:
            if line.strip():
                line = "   " + line
            else:
                line = "   "

        result.append(line)
        prev_was_list_item = is_list_item
        i += 1

    # Post-processing: insert blank lines before unindent transitions to avoid
    # "Block quote ends without a blank line" warnings in RST.
    final: list[str] = []
    for idx, line in enumerate(result):
        if idx > 0 and final:
            prev = final[-1]
            prev_indent = len(prev) - len(prev.lstrip())
            cur_indent = len(line) - len(line.lstrip()) if line.strip() else -1
            if (
                cur_indent >= 0
                and prev.strip()
                and prev_indent > cur_indent
                and final[-1] != ""
            ):
                final.append("")
        final.append(line)

    text = "\n".join(final)

    # --- Final sanitization pass ---
    text = re.sub(r"^\.\.\s+(\[[^\]]+\])", r"\1", text, flags=re.MULTILINE)
    text = re.sub(
        r"``<(https?://[^>]+)>``_",
        r"`\1 <\1>`__",
        text,
    )

    lines = text.split("\n")
    for li, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("..") or stripped.startswith(":"):
            continue

        cleaned = re.sub(r"``[^`]*``", "", line)
        cleaned = re.sub(r"`[^`]+ <[^>]+>`__", "", cleaned)
        cleaned = re.sub(r"`[^`]+`", "", cleaned)
        if cleaned.count("`") % 2 == 1:
            lines[li] = re.sub(r"(?<!`)`(?!`)", r"\`", line)
            line = lines[li]

        cleaned = re.sub(r"\*\*[^*]+\*\*", "", line)
        cleaned = re.sub(r"(?<!\*)\*(?!\*)[^*]+\*(?!\*)", "", cleaned)
        cleaned = re.sub(r"\\\*", "", cleaned)
        remaining_stars = cleaned.count("*")
        if remaining_stars > 0:
            line = re.sub(r"\\\*([^*]+)\*(?!\*)", r"\\\*\1\\*", line)
            line = re.sub(r"(?<![\\*\s])\*(?=\s|$)", r"\\*", line)
            line = re.sub(r"(?:^|(?<=\s))\*(?![*\s])", r"\\*", line)
            lines[li] = line

    return "\n".join(lines)


# Back-compat alias for any caller (and the section formatter) that
# referenced the leading-underscore name when this code lived in the
# monolithic ``dataset_page.py``.
_convert_readme_to_rst = convert_readme_to_rst
