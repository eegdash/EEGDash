"""Shared helpers for wrapping comma-separated values as HTML tags.

These utilities are used by both the model and dataset summary table
preparation scripts so the generated markup stays consistent.
"""

from __future__ import annotations

import re
from html import escape
from typing import Iterable

_TAG_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalize_values(cell: object) -> Iterable[str]:
    """Yield cleaned string tokens from a table cell.

    Handles ``None``/empty strings/"NaN" gracefully and normalises common
    separators (comma/semicolon/pipe/newline).
    """
    if cell is None:
        return []

    if isinstance(cell, (list, tuple, set)):
        values: list[str] = []
        for item in cell:
            values.extend(_normalize_values(item))
        deduped: list[str] = []
        seen_nested: set[str] = set()
        for value in values:
            key = value.lower()
            if key in seen_nested:
                continue
            seen_nested.add(key)
            deduped.append(value)
        return deduped

    text = str(cell).strip()
    if not text:
        return []

    lowered = text.lower()
    if lowered in {"nan", "none", "null", ""}:
        return []

    # Normalise separators so we can split reliably. Keep forward slash to
    # support values like "Clinical / Research" that should remain intact.
    for sep in (";", "|", "\n"):
        text = text.replace(sep, ",")

    tokens = [t.strip() for t in text.split(",") if t.strip()]
    # Remove duplicates while preserving order.
    seen = set()
    unique_tokens = []
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_tokens.append(token)
    return unique_tokens


def _slugify(value: str) -> str:
    slug = _TAG_SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or "tag"


def wrap_tags(
    cell: object,
    *,
    kind: str,
    joiner: str = " ",
    include_slug_class: bool = False,
    normalizer: callable | None = None,
) -> str:
    """Format the given cell as a collection of HTML ``<span class="tag">``.

    Parameters
    ----------
    cell
        Value to wrap. Can be a comma-separated string, list-like, or ``None``.
    kind
        Logical tag family (e.g. ``"categorization"`` or ``"pathology"``).
        Used to attach CSS hooks like ``tag-kind-pathology`` and data
        attributes for client-side styling.
    joiner
        Delimiter inserted between tags. Defaults to a single space.
    include_slug_class
        When ``True`` also add ``tag-value-<slug>`` classes. Helpful when CSS
        wants per-value hooks without relying on ``data-*`` attributes.

    """
    kind_slug = _slugify(kind)
    tokens = _normalize_values(cell)
    if not tokens:
        return ""

    spans = []
    for token in tokens:
        value = normalizer(token) if normalizer else token
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        token_text = escape(value)
        token_slug = _slugify(value)
        classes = ["tag", f"tag-kind-{kind_slug}"]
        if include_slug_class:
            classes.append(f"tag-value-{token_slug}")
        class_attr = " ".join(classes)
        spans.append(
            (
                f'<span class="{class_attr}" '
                f'data-tag-kind="{kind_slug}" '
                f'data-tag-value="{token_slug}" '
                f'data-tag-label="{token_text}">{token_text}</span>'
            )
        )
    return joiner.join(spans)


__all__ = ["wrap_tags"]
