"""Sphinx extension for embedding dataset file explorers in documentation."""

import re
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from jinja2 import Template


class DatasetExplorerDirective(Directive):
    """Embed a dataset file explorer.

    Usage::

        .. dataset-explorer::
            :dataset: ds002718
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "dataset": directives.unchanged,
    }

    # Allow any safe dataset identifier: letters, digits, underscore, hyphen.
    # Catalog uses many shapes (DS002718, BNCI2020, EEG2025R1, LEMON, ...),
    # so reject only path-traversal / template-injection characters.
    _DATASET_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

    def run(self) -> list:
        dataset_id: str = self.options.get("dataset", "ds002718")

        if not self._DATASET_ID_RE.match(dataset_id):
            raise self.error(
                f"Invalid dataset_id: {dataset_id!r}. Must match [A-Za-z0-9_-]{{1,64}}."
            )

        template_path: Path = (
            Path(__file__).parent.parent / "_templates" / "dataset-explorer.html"
        )

        try:
            template_text: str = template_path.read_text(encoding="utf-8")
            template: Template = Template(template_text)
            html: str = template.render(dataset_id=dataset_id)
        except FileNotFoundError as e:
            raise self.error(f"Template not found at {template_path}") from e
        except Exception as e:
            raise self.error(f"Failed to render dataset explorer template: {e}")

        # Register the template as a build dependency so incremental
        # builds re-run this directive whenever the template changes.
        # Without this, edits to dataset-explorer.html are silently
        # ignored on cached pages.
        env = self.state.document.settings.env
        env.note_dependency(str(template_path))

        return [nodes.raw("", html, format="html")]


def setup(app) -> dict:
    app.add_directive("dataset-explorer", DatasetExplorerDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
