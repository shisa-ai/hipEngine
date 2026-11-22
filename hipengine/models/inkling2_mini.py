"""Checkpoint-derived execution geometry for Inkling 2 Mini."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


from hipengine.models.registry import register_model
from hipengine.quant.inkling2_experts import INKLING2_MIXED_QUANT


@dataclass(frozen=True, slots=True)
class Inkling2MiniModel:
    name: str = "inkling2_mini"
    architectures: tuple[str, ...] = ("Inkling2LoopForConditionalGeneration",)
    default_quant: str = INKLING2_MIXED_QUANT

    def layer_sequence(self) -> tuple[str, ...]:
        return ("inkling2_attention", "inkling2_routed_ffn")


INKLING2_MINI = register_model(Inkling2MiniModel())


@dataclass(frozen=True, slots=True)
class Inkling2GGUFModel(Inkling2MiniModel):
    name: str = "inkling2_gguf"
    architectures: tuple[str, ...] = ("inkling2", "inkling2_loop")
    default_quant: str = "gguf_inkling2"


INKLING2_GGUF = register_model(Inkling2GGUFModel())


@dataclass(frozen=True, slots=True)
class Inkling2Visit:
    """One decoder visit; KV and convolution state share the visit's slot ID."""

    weight_layer: int
    state_slot: int
    is_local: bool
    restart_from_prelude: bool = False
    message_pass: int | None = None


@dataclass(frozen=True, slots=True)
class Inkling2VisitPlan:
    visits: tuple[Inkling2Visit, ...]

    @classmethod
    def from_config(cls, config: Mapping) -> Inkling2VisitPlan:
        text = config["text_config"]
        loop = text["loop_config"]
        layers = int(text["num_hidden_layers"])
        start = int(loop["core_start"])
        end = int(loop["core_end"])
        passes = int(loop["num_passes"])
        if not (0 <= start < end <= layers and passes >= 1):
            raise ValueError(f"invalid Inkling 2 loop geometry: {start=}, {end=}, {layers=}, {passes=}")
        local = frozenset(int(i) for i in text["local_layer_ids"])
        if any(not 0 <= i < layers for i in local):
            raise ValueError("Inkling 2 local_layer_ids contains an out-of-range layer")

        visits = [Inkling2Visit(i, i, i in local) for i in range(end)]
        # The first pass shares the physical layer numbering; extra passes own
        # disjoint cache slots after the physical layer range.
        width = end - start
        for pass_index in range(1, passes):
            slot = layers + (pass_index - 1) * width
            visits.extend(
                Inkling2Visit(i, slot + i - start, i in local, i == start, pass_index - 1)
                for i in range(start, end)
            )
        visits.extend(Inkling2Visit(i, i, i in local) for i in range(end, layers))
        return cls(tuple(visits))
