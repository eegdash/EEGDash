import math
import re


def get_reading_time(content, wpm=200):
    """
    Estimate the reading time for a given piece of content.
    """
    # Remove HTML tags and count words
    text = re.sub(r"<[^>]+>", "", content)
    words = re.findall(r"\w+", text)
    num_words = len(words)
    minutes = num_words / wpm
    return math.ceil(minutes)


def html_page_context(app, pagename, templatename, context, doctree):
    """
    Add estimated reading time to the template context.
    """
    if not doctree:
        return

    content = doctree.astext()
    reading_time = get_reading_time(content)
    context["reading_time"] = reading_time


def setup(app):
    """
    Setup the Sphinx extension.
    """
    app.connect("html-page-context", html_page_context)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }