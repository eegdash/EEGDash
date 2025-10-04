from __future__ import annotations

import math
import re

from docutils import nodes

SKIP_CONTAINER_CLASSES = {
    "sphx-glr-script-out",
    "sphx-glr-single-img",
    "sphx-glr-thumbnail",
    "sphx-glr-horizontal",
}


class TextExtractor(nodes.NodeVisitor):
    def __init__(self, document):
        super().__init__(document)
        self.text = []

    def visit_Text(self, node):
        self.text.append(node.astext())

    def visit_literal_block(self, node):
        # Don't visit the children of literal blocks (i.e., code blocks)
        raise nodes.SkipNode

    def visit_figure(self, node):
        raise nodes.SkipNode

    def visit_image(self, node):
        raise nodes.SkipNode

    def visit_container(self, node):
        classes = set(node.get("classes", ()))
        if classes & SKIP_CONTAINER_CLASSES:
            raise nodes.SkipNode

    def unknown_visit(self, node):
        """Pass for all other nodes."""
        pass


EXAMPLE_PREFIX = "generated/auto_examples/"


def _should_calculate(pagename: str) -> bool:
    if not pagename:
        return False
    if not pagename.startswith(EXAMPLE_PREFIX):
        return False
    if pagename.endswith("/sg_execution_times"):
        return False
    if pagename == "generated/auto_examples/index":
        return False
    return True


def html_page_context(app, pagename, templatename, context, doctree):
    """Add estimated reading time directly under tutorial titles."""
    if not doctree or not _should_calculate(pagename):
        context.pop("reading_time", None)
        return

    visitor = TextExtractor(doctree)
    doctree.walk(visitor)

    full_text = " ".join(visitor.text)
    word_count = len(re.findall(r"\w+", full_text))

    wpm = 200  # Median reading speed
    reading_time = math.ceil(word_count / wpm) if wpm > 0 else 0

    if reading_time <= 0:
        context.pop("reading_time", None)
        return

    context["reading_time"] = reading_time

    body = context.get("body")
    if not isinstance(body, str) or "</h1>" not in body:
        return

    minutes_label = "minute" if reading_time == 1 else "minutes"
    badge_html = (
        '<div class="eegdash-reading-time" role="note">'
        '<span class="eegdash-reading-time__label">Estimated reading time:</span>'
        f'<span class="eegdash-reading-time__value">{reading_time} {minutes_label}</span>'
        "</div>"
    )

    insert_at = body.find("</h1>")
    if insert_at == -1:
        return

    context["body"] = body[: insert_at + 5] + badge_html + body[insert_at + 5 :]


def setup(app):
    """Setup the Sphinx extension."""
    app.connect("html-page-context", html_page_context)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
