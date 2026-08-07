"""Errors shared by the strict in-tree PM4 frontend."""

from __future__ import annotations


class Pm4InspectionError(RuntimeError):
    """Raised when graph or code-object input cannot be proven safe to lower."""
