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

    def run(self) -> list:
        dataset_id: str = self.options.get("dataset", "ds002718")

        if not re.match(r"^ds\d+$", dataset_id):
            raise self.error(
                f"Invalid dataset_id: {dataset_id}. Expected format: ds<digits> (e.g., ds002718)"
            )

        template_path: Path = (
            Path(__file__).parent.parent / "_templates" / "dataset-explorer.html"
        )

        if not template_path.exists():
            raise self.error(f"Template not found at {template_path}")

        try:
            template_text: str = template_path.read_text(encoding="utf-8")
            template: Template = Template(template_text)
            html: str = template.render(dataset_id=dataset_id)
        except Exception as e:
            raise self.error(f"Failed to render dataset explorer template: {e}")

        return [nodes.raw("", html, format="html")]


def setup(app) -> dict:
    app.add_directive("dataset-explorer", DatasetExplorerDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
