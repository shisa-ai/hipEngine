"""Fusion planner spike.

Given a model's primitive layer chain, choose the longest registered fused composite at each
position. If no fused or primitive kernel is registered, keep the primitive step in the plan
and let registry resolution produce a clean MissingKernelError.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from hipengine.kernels.registry import KernelKey, can_resolve, resolve


@dataclass(frozen=True)
class KernelPlanStep:
    backend: str
    layer: str
    quant: str
    variant: str = ""
    source_layers: tuple[str, ...] = ()

    @property
    def key(self) -> KernelKey:
        return KernelKey(self.backend, self.layer, self.quant, self.variant)


@dataclass(frozen=True)
class BoundKernel:
    step: KernelPlanStep
    kernel: Callable[..., Any]


class FusionPlanner:
    """Longest-match planner over ``+``-joined layer composites."""

    def __init__(self, *, backend: str, quant: str, max_fusion_width: int = 4):
        if max_fusion_width < 1:
            raise ValueError("max_fusion_width must be >= 1")
        self.backend = backend
        self.quant = quant
        self.max_fusion_width = max_fusion_width

    def plan(self, layers: Sequence[str]) -> tuple[KernelPlanStep, ...]:
        layer_tuple = tuple(layers)
        if not layer_tuple:
            return ()
        if any(not layer for layer in layer_tuple):
            raise ValueError("layer names must be non-empty")

        steps: list[KernelPlanStep] = []
        index = 0
        while index < len(layer_tuple):
            max_width = min(self.max_fusion_width, len(layer_tuple) - index)
            chosen = self._choose_layer(layer_tuple, index, max_width)
            source = layer_tuple[index : index + chosen]
            steps.append(
                KernelPlanStep(
                    backend=self.backend,
                    layer="+".join(source),
                    quant=self.quant,
                    source_layers=tuple(source),
                )
            )
            index += chosen
        return tuple(steps)

    def _choose_layer(self, layers: tuple[str, ...], index: int, max_width: int) -> int:
        for width in range(max_width, 1, -1):
            candidate = "+".join(layers[index : index + width])
            if can_resolve(backend=self.backend, layer=candidate, quant=self.quant):
                return width
        return 1


def resolve_plan(plan: Sequence[KernelPlanStep]) -> tuple[BoundKernel, ...]:
    """Resolve every plan step through the kernel registry."""

    bound: list[BoundKernel] = []
    for step in plan:
        kernel = resolve(
            backend=step.backend,
            layer=step.layer,
            quant=step.quant,
            variant=step.variant,
        )
        bound.append(BoundKernel(step=step, kernel=kernel))
    return tuple(bound)
