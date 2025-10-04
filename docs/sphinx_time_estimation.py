import math
import re
from docutils import nodes


class TextExtractor(nodes.NodeVisitor):
    def __init__(self, document):
        super().__init__(document)
        self.text = []

    def visit_Text(self, node):
        self.text.append(node.astext())

    def visit_literal_block(self, node):
        # Don't visit the children of literal blocks (i.e., code blocks)
        raise nodes.SkipNode

    def unknown_visit(self, node):
        """Pass for all other nodes."""
        pass


def html_page_context(app, pagename, templatename, context, doctree):
    """Add estimated reading time to the template context."""
    if not doctree:
        return

    visitor = TextExtractor(doctree)
    doctree.walk(visitor)

    full_text = " ".join(visitor.text)
    word_count = len(re.findall(r"\w+", full_text))

    wpm = 200  # Median reading speed
    reading_time = math.ceil(word_count / wpm) if wpm > 0 else 0

    context["reading_time"] = reading_time


def setup(app):
    """Setup the Sphinx extension."""
    app.connect("html-page-context", html_page_context)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
