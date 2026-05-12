"""Model plugin protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelPlugin(Protocol):
    """Architecture-level layer sequence and metadata.

    Real model plugins will also own weight-name maps, chat templates, RoPE config, and
    architecture-specific layer parameters. The scaffold keeps only the layer sequence needed
    to validate registry + fusion planning.
    """

    name: str
    architectures: tuple[str, ...]

    def layer_sequence(self) -> Sequence[str]:
        """Return primitive layer keys in execution order."""
