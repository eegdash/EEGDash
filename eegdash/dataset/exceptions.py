# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Custom exceptions for EEGDash.

This module defines exceptions used throughout the EEGDash library to provide
informative error messages for common issues.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..logging import logger


class EEGDashError(Exception):
    """Base exception for all EEGDash errors."""

    pass


class DataIntegrityError(EEGDashError):
    """Raised when a dataset record has known data integrity issues.

    This exception is raised when attempting to load a record that has been
    flagged during ingestion as having missing companion files or other
    integrity problems.

    Attributes
    ----------
    record : dict
        The problematic record metadata.
    issues : list[str]
        List of specific integrity issues found.
    authors : list[str]
        Dataset authors who can be contacted about the issue.
    contact_info : list[str] | None
        Contact information for reporting the issue.
    source_url : str | None
        URL to the dataset source for reporting issues.

    Examples
    --------
    >>> try:
    ...     dataset.raw  # Attempt to load data
    ... except DataIntegrityError as e:
    ...     print(f"Cannot load: {e.issues}")
    ...     print(f"Contact authors: {e.authors}")

    """

    def __init__(
        self,
        message: str,
        record: dict[str, Any] | None = None,
        issues: list[str] | None = None,
        authors: list[str] | None = None,
        contact_info: list[str] | None = None,
        source_url: str | None = None,
    ):
        self.record = record or {}
        self.issues = issues or []
        self.authors = authors or []
        self.contact_info = contact_info
        self.source_url = source_url
        super().__init__(message)

    def _build_rich_output(self) -> Panel:
        """Build a rich Panel with formatted error information."""
        dataset_id = self.record.get("dataset", "unknown")
        bids_relpath = self.record.get("bids_relpath", "unknown")

        # Create the main content
        content = Text()

        # File info
        content.append("File: ", style="bold")
        content.append(f"{bids_relpath}\n", style="cyan")
        content.append("Dataset: ", style="bold")
        content.append(f"{dataset_id}\n\n", style="cyan")

        # Issues table
        if self.issues:
            content.append("Issues Found:\n", style="bold red")
            for issue in self.issues:
                content.append("  \u2717 ", style="red")  # ✗
                content.append(f"{issue}\n", style="yellow")
            content.append("\n")

        # Contact information
        if self.authors:
            content.append("Dataset Authors:\n", style="bold")
            for author in self.authors:
                content.append(f"  \u2022 {author}\n", style="dim")

        if self.contact_info:
            content.append("\n\u2709 Contact Email: ", style="bold green")  # ✉
            contact_str = (
                ", ".join(self.contact_info)
                if isinstance(self.contact_info, list)
                else self.contact_info
            )
            content.append(f"{contact_str}\n", style="green")

        if self.source_url:
            content.append("\n\u21b3 Report Issue: ", style="bold blue")  # ↳
            content.append(f"{self.source_url}\n", style="blue underline")

        # Create panel
        panel = Panel(
            content,
            title="[bold red]Data Integrity Error[/bold red]",
            subtitle="[dim]This is a problem with the source data, not EEGDash[/dim]",
            border_style="red",
            padding=(1, 2),
        )

        return panel

    def print_rich(self, console: Console | None = None) -> None:
        """Print a rich formatted version of the error to the console.

        Parameters
        ----------
        console : Console, optional
            Rich console to print to. If None, creates a new one.

        """
        if console is None:
            console = Console(stderr=True)
        console.print(self._build_rich_output())

    def log_error(self) -> None:
        """Log the error using the EEGDash logger with rich formatting."""
        bids_relpath = self.record.get("bids_relpath", "unknown")

        # Log main error
        logger.error(
            "[bold red]Data Integrity Error[/bold red] - Cannot load [cyan]%s[/cyan]",
            bids_relpath,
        )

        # Log issues
        for issue in self.issues:
            logger.error("  [red]\u2717[/red] %s", issue)

        # Log contact info
        if self.authors:
            logger.info(
                "[dim]Dataset authors: %s[/dim]",
                ", ".join(self.authors),
            )

        if self.contact_info:
            contact_str = (
                ", ".join(self.contact_info)
                if isinstance(self.contact_info, list)
                else self.contact_info
            )
            logger.info(
                "[green]\u2709 Contact email: %s[/green]",
                contact_str,
            )

        if self.source_url:
            logger.info(
                "[blue]\u21b3 Report issue at: %s[/blue]",
                self.source_url,
            )

    def log_warning(self) -> None:
        """Log the integrity issues as warnings (non-blocking)."""
        bids_relpath = self.record.get("bids_relpath", "unknown")

        # Log main warning
        logger.warning(
            "[bold yellow]Data Integrity Warning[/bold yellow] - "
            "[cyan]%s[/cyan] has known issues",
            bids_relpath,
        )

        # Log issues as warnings
        for issue in self.issues:
            logger.warning("  [yellow]\u26a0[/yellow] %s", issue)  # ⚠

        # Log contact info
        if self.authors:
            logger.info(
                "[dim]Dataset authors: %s[/dim]",
                ", ".join(self.authors),
            )

        if self.contact_info:
            contact_str = (
                ", ".join(self.contact_info)
                if isinstance(self.contact_info, list)
                else self.contact_info
            )
            logger.info(
                "[green]\u2709 Contact email: %s[/green]",
                contact_str,
            )

        if self.source_url:
            logger.info(
                "[blue]\u21b3 Report issue at: %s[/blue]",
                self.source_url,
            )

    @classmethod
    def warn_from_record(cls, record: dict[str, Any]) -> None:
        """Log a warning about data integrity issues without raising an exception.

        Use this when you want to warn about issues but still allow loading.

        Parameters
        ----------
        record : dict
            Record containing ``_data_integrity_issues`` and optionally
            ``_dataset_authors``, ``_dataset_contact``, ``_source_url``.

        """
        issues = record.get("_data_integrity_issues", [])
        authors = record.get("_dataset_authors", [])
        contact_info = record.get("_dataset_contact")
        source_url = record.get("_source_url")

        # Create temporary instance just for logging
        warning = cls(
            message="Data integrity warning",
            record=record,
            issues=issues,
            authors=authors,
            contact_info=contact_info,
            source_url=source_url,
        )
        warning.log_warning()

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DataIntegrityError":
        """Create a DataIntegrityError from a record with integrity issues.

        Parameters
        ----------
        record : dict
            Record containing ``_data_integrity_issues`` and optionally
            ``_dataset_authors``, ``_dataset_contact``, ``_source_url``.

        Returns
        -------
        DataIntegrityError
            Exception with all relevant context.

        """
        issues = record.get("_data_integrity_issues", [])
        authors = record.get("_dataset_authors", [])
        contact_info = record.get("_dataset_contact")
        source_url = record.get("_source_url")
        dataset_id = record.get("dataset", "unknown")
        bids_relpath = record.get("bids_relpath", "unknown")

        # Build simple message for str(exception)
        msg_parts = [
            f"Cannot load '{bids_relpath}' from dataset '{dataset_id}':",
        ]
        for issue in issues:
            msg_parts.append(f"  - {issue}")

        if authors:
            msg_parts.append(f"Authors: {', '.join(authors)}")

        if contact_info:
            contact_str = (
                ", ".join(contact_info)
                if isinstance(contact_info, list)
                else contact_info
            )
            msg_parts.append(f"Contact email: {contact_str}")

        if source_url:
            msg_parts.append(f"Report at: {source_url}")

        error = cls(
            message="\n".join(msg_parts),
            record=record,
            issues=issues,
            authors=authors,
            contact_info=contact_info,
            source_url=source_url,
        )

        # Log the error with rich formatting
        error.log_error()

        return error


class UnsupportedDataError(EEGDashError):
    """Raised when data cannot be loaded due to format limitations.

    This exception is raised for datasets that fundamentally cannot be loaded
    because of unsupported format variants (e.g., fluorescence SNIRF), corrupted
    or truncated files, or other issues that no repair can fix.

    Attributes
    ----------
    record : dict
        The problematic record metadata.
    reason : str | None
        Short description of why the data is unsupported.

    Examples
    --------
    >>> try:
    ...     dataset.raw  # Attempt to load data
    ... except UnsupportedDataError as e:
    ...     print(f"Cannot load: {e.reason}")

    """

    def __init__(
        self,
        message: str,
        record: dict[str, Any] | None = None,
        reason: str | None = None,
    ):
        self.record = record or {}
        self.reason = reason
        super().__init__(message)


__all__ = ["EEGDashError", "DataIntegrityError", "UnsupportedDataError"]
