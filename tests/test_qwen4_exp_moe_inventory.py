"""P1 frozen quant/shape/owner inventory test for Qwen4Exp MoE prefill.

This is the RED gate for the P1 "layer 2 and the Q8 expert-down family" unit.
It reads the pinned Unsloth ``UD-Q4_K_XL`` split GGUF (the binding AR target)
and asserts that the per-layer expert quant/shape/owner map matches the frozen
contract before any layer-2 routing or timing work is trusted. Artifact drift
fails here, before timing.

Frozen MoE map (from docs/QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md P1):

* 43 layers of Q4_K / Q4_K / Q5_1  (expert_gate / expert_up / expert_down)
* layer 2 of Q5_K / Q5_K / Q8_0
* layers 4, 30, 46, 47 of Q4_K / Q4_K / Q8_0

The test is model-gated: it reads only metadata + tensor headers (no GPU, no
weight payloads), and skips when the local pinned shard is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.qwen4_exp_gguf import (
    GDN,
    QSA,
    build_qwen4_exp_gguf_tensor_map,
)

UNSLOTH_ROOT = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
_PART0 = UNSLOTH_ROOT / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"

# Frozen MoE expert quant map: layer_id -> (gate, up, down) ggml type names.
FROZEN_EXPERT_QUANT_MAP: dict[int, tuple[str, str, str]] = {
    2: ("Q5_K", "Q5_K", "Q8_0"),
    4: ("Q4_K", "Q4_K", "Q8_0"),
    30: ("Q4_K", "Q4_K", "Q8_0"),
    46: ("Q4_K", "Q4_K", "Q8_0"),
    47: ("Q4_K", "Q4_K", "Q8_0"),
}

FROZEN_BLOCK_COUNT = 48
FROZEN_EXPERTS = 512
FROZEN_FFN = 640
FROZEN_HIDDEN = 2560
# QSA layers are those whose (layer_id % 4) == 3 in the frozen target.
FROZEN_QSA_LAYERS = frozenset(range(3, FROZEN_BLOCK_COUNT, 4))


def _model_map():
    if not _PART0.exists():
        pytest.skip(f"local pinned Unsloth shard not found: {_PART0}")
    parts = discover_gguf_files(UNSLOTH_ROOT)
    infos = [GGUFReader(part).info for part in parts]
    return build_qwen4_exp_gguf_tensor_map(infos)


def _layer_quant_triple(m, layer_id: int) -> tuple[str, str, str]:
    layer = m.layer(layer_id)
    return (
        layer.tensor("expert_gate").tensor.ggml_type_name,
        layer.tensor("expert_up").tensor.ggml_type_name,
        layer.tensor("expert_down").tensor.ggml_type_name,
    )


@pytest.mark.skipif(not _PART0.exists(), reason="pinned Unsloth shard not present")
def test_qwen4_exp_moe_inventory_block_and_layer_types() -> None:
    m = _model_map()
    assert len(m.layers) == FROZEN_BLOCK_COUNT
    # Frozen layer type pattern: qsa every 4th layer (offset 3), gdn otherwise.
    for lid, layer in enumerate(m.layers):
        expected = QSA if lid in FROZEN_QSA_LAYERS else GDN
        assert layer.layer_type == expected, f"layer {lid} type drift"


@pytest.mark.skipif(not _PART0.exists(), reason="pinned Unsloth shard not present")
def test_qwen4_exp_moe_inventory_expert_quant_map() -> None:
    m = _model_map()
    q5_1_down = 0
    q8_0_down = 0
    for lid in range(FROZEN_BLOCK_COUNT):
        triple = _layer_quant_triple(m, lid)
        expected = FROZEN_EXPERT_QUANT_MAP.get(lid)
        if expected is not None:
            assert triple == expected, (
                f"layer {lid} expert quant drift: {triple} != {expected}"
            )
        else:
            assert triple == ("Q4_K", "Q4_K", "Q5_1"), (
                f"layer {lid} expert quant drift: {triple} != Q4_K/Q4_K/Q5_1"
            )
        down = triple[2]
        if down == "Q5_1":
            q5_1_down += 1
        elif down == "Q8_0":
            q8_0_down += 1
    assert q5_1_down == 43
    assert q8_0_down == 5
    # Exactly one layer (2) has a Q5_K gate/up pair.
    q5_k_gate = [lid for lid in range(FROZEN_BLOCK_COUNT)
                 if _layer_quant_triple(m, lid)[0] == "Q5_K"]
    assert q5_k_gate == [2]


@pytest.mark.skipif(not _PART0.exists(), reason="pinned Unsloth shard not present")
def test_qwen4_exp_moe_inventory_shapes_and_owner() -> None:
    m = _model_map()
    cfg = m.config
    assert cfg.block_count == FROZEN_BLOCK_COUNT
    assert cfg.expert_count == FROZEN_EXPERTS
    assert cfg.expert_feed_forward_length == FROZEN_FFN
    assert cfg.hidden_size == FROZEN_HIDDEN
    # Every layer owns router + shared expert slots (common to all layers).
    for lid in range(FROZEN_BLOCK_COUNT):
        layer = m.layer(lid)
        for slot in (
            "router",
            "shared_gate",
            "shared_up",
            "shared_down",
            "shared_expert_gate",
            "expert_gate",
            "expert_up",
            "expert_down",
        ):
            layer.tensor(slot)  # presence only
            # Expert weights are rank-3 [experts, out, in].
            if slot in {"expert_gate", "expert_up", "expert_down"}:
                assert len(layer.tensor(slot).tensor.shape) == 3, (
                    f"{slot}@{lid} not rank-3"
                )
