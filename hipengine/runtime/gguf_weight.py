"""Model-neutral structural types for resident GGUF runtime dispatch."""

from __future__ import annotations

from typing import Protocol

from hipengine.loading.materialize import DeviceTensorAllocation


class GGUFWeightSpec(Protocol):
    """Minimum resident-layout metadata required by runtime dispatch."""

    layout: str
    quant_key: str


class GGUFDeviceWeight(Protocol):
    """Structural ABI shared by Qwen, Laguna, and future GGUF weight owners."""

    spec: GGUFWeightSpec
    backend: str

    def allocation(self, name: str | None = None) -> DeviceTensorAllocation: ...


__all__ = ["GGUFDeviceWeight", "GGUFWeightSpec"]
