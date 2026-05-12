"""Toy model plugin for scaffold/spike tests.

This is not an inference model. It is a one-layer layer-sequence fixture that exercises the
4-axis kernel registry and longest-match fusion planner before any real HIP kernels land.
"""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.models.registry import register_model


@dataclass(frozen=True)
class ToyOneLayerModel:
    name: str = "toy_one_layer"
    architectures: tuple[str, ...] = ("HipEngineToyForCausalLM",)

    def layer_sequence(self) -> tuple[str, ...]:
        return (
            "embed",
            "rmsnorm",
            "rotate",
            "qkv_proj",
            "attention_decode",
            "o_proj",
            "lm_head",
        )


TOY_ONE_LAYER = register_model(ToyOneLayerModel())
