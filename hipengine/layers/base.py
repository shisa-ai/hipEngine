"""Layer plugin protocol scaffold."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LayerPlugin(Protocol):
    layer: str
    description: str
