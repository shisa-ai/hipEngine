"""Shared doubles for the Qwen4Exp expert-down arm-count tests.

Both grouped expert-down chains dispatch once per layer whose resident
``layers.<n>.expert_down`` weight carries the chain's quant, so an arm's expected
call count is a property of the loaded artifact - not of a constant. These
doubles model just enough of a resident runner for the campaign's count helpers:
the slot map, the expert intermediate width, and the backend name.

The shipped UD split is 43 ``gguf_q5_1`` down tensors and five ``gguf_q8_0``
across 48 layers (``docs/QWEN4EXP-OPTIMIZATION-LEDGER.md`` ED-3); the builders
default to it but accept any split, so a test can prove the helpers follow the
artifact rather than the constant that used to be here.
"""

from __future__ import annotations

from types import SimpleNamespace


def resident_weight(quant_key: str) -> SimpleNamespace:
    """The attribute the count helper reads off a resident quantized weight."""

    return SimpleNamespace(spec=SimpleNamespace(quant_key=quant_key))


def expert_down_runner(
    *,
    q5_1: int = 43,
    q8_0: int = 5,
    ffn: int = 640,
    backend: str = "hip_gfx1151",
    down_quants: dict[int, str] | None = None,
    extra_slots: dict[str, object] | None = None,
) -> SimpleNamespace:
    """A runner double carrying ``q5_1 + q8_0`` routed expert-down layers.

    ``down_quants`` overrides the layer -> quant map with an explicit split (the
    shipped one interleaves: layers 2, 4, 30, 46, 47 are ``gguf_q8_0``). Every
    layer also carries a Q4_K gate/up slot, because a layer without its down
    weight is the only case the helper treats as non-dispatchable.
    """

    weights: dict[str, object] = {}
    if down_quants is None:
        down_quants = {}
        layer = 0
        for _ in range(q5_1):
            down_quants[layer] = "gguf_q5_1"
            layer += 1
        for _ in range(q8_0):
            down_quants[layer] = "gguf_q8_0"
            layer += 1
    for layer, quant_key in sorted(down_quants.items()):
        weights[f"layers.{layer}.expert_gate"] = resident_weight("gguf_q4_k")
        weights[f"layers.{layer}.expert_down"] = resident_weight(quant_key)
    if extra_slots:
        weights.update(extra_slots)
    return SimpleNamespace(
        resident=SimpleNamespace(device_weights=weights),
        config=SimpleNamespace(expert_feed_forward_length=ffn),
        backend=backend,
    )


#: The shipped artifact's down split: 43 Q5_1 layers and five Q8_0 layers.
SHIPPED_DOWN_QUANTS = {
    layer: ("gguf_q8_0" if layer in (2, 4, 30, 46, 47) else "gguf_q5_1")
    for layer in range(48)
}
