from __future__ import annotations

import pytest

from hipengine.core.dtype import DType
from hipengine.loading.materialize import DeviceWeightMap
from hipengine.loading.qwen35_paro import Qwen35ParoConfig, Qwen35ParoLayerDeviceWeights
from hipengine.runtime import Qwen35ParoDecodeState, RuntimeWorkspace


class FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0xA000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(nbytes, 1) + 0x100
        self.allocations[ptr] = nbytes
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(ptr)
        self.allocations.pop(ptr, None)


def _config() -> Qwen35ParoConfig:
    return Qwen35ParoConfig(
        architecture="Qwen3_5MoeForConditionalGeneration",
        num_hidden_layers=1,
        hidden_size=4096,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
        shared_expert_intermediate_size=768,
        layer_types=("full_attention",),
        quant_method="paroquant",
    )


def _state(runtime: FakeRuntime) -> Qwen35ParoDecodeState:
    layer = Qwen35ParoLayerDeviceWeights(config=_config(), layer_id=0, weights=DeviceWeightMap({}))
    return Qwen35ParoDecodeState(
        layer_weights=layer,
        workspace=RuntimeWorkspace(runtime=runtime),
        runtime=runtime,
    )


def test_qwen35_decode_state_reserves_full_attention_split_k_scratch() -> None:
    runtime = FakeRuntime()
    state = _state(runtime)

    scratch = state.reserve_full_attention_scratch(tokens=1, num_splits=2, gated_dtype="bf16")

    assert scratch.query.shape == (1, 16, 256)
    assert scratch.key.shape == (1, 2, 256)
    assert scratch.value.shape == (1, 2, 256)
    assert scratch.gate.shape == (1, 16, 256)
    assert scratch.partial_out.shape == (16, 2, 256)
    assert scratch.partial_m.shape == (16, 2)
    assert scratch.partial_l.shape == (16, 2)
    assert scratch.attn_out.shape == (16, 256)
    assert scratch.gated_attn.shape == (1, 4096)
    assert scratch.gated_attn.dtype is DType.BF16


def test_qwen35_decode_state_reserves_moe_c1_scratch() -> None:
    runtime = FakeRuntime()
    state = _state(runtime)

    scratch = state.reserve_moe_c1_scratch(tokens=1)

    assert scratch.normed.shape == (1, 4096)
    assert scratch.router_logits.shape == (1, 128)
    assert scratch.routing_weights.shape == (1, 8)
    assert scratch.selected_experts.shape == (1, 8)
    assert scratch.selected_experts.dtype is DType.INT32
    assert scratch.gate_up.shape == (1, 8, 1536)
    assert scratch.shared_up.shape == (1, 1536)
    assert scratch.moe_out.shape == (1, 4096)


def test_qwen35_decode_state_reuses_and_replaces_named_scratch() -> None:
    runtime = FakeRuntime()
    state = _state(runtime)

    first = state.reserve_full_attention_scratch(tokens=1, num_splits=2)
    second = state.reserve_full_attention_scratch(tokens=1, num_splits=2)
    changed = state.reserve_full_attention_scratch(tokens=1, num_splits=4)

    assert second.partial_out.ptr == first.partial_out.ptr
    assert changed.partial_out.ptr != first.partial_out.ptr
    assert first.partial_out.ptr in runtime.freed


def test_qwen35_decode_state_validates_scratch_requests() -> None:
    runtime = FakeRuntime()
    state = _state(runtime)
    with pytest.raises(ValueError, match="tokens"):
        state.reserve_full_attention_scratch(tokens=0)
    with pytest.raises(ValueError, match="num_splits"):
        state.reserve_full_attention_scratch(num_splits=0)
    with pytest.raises(ValueError, match="gated_dtype"):
        state.reserve_full_attention_scratch(gated_dtype="int32")
    with pytest.raises(ValueError, match="tokens"):
        state.reserve_moe_c1_scratch(tokens=0)


def test_qwen35_decode_state_free_releases_workspace() -> None:
    runtime = FakeRuntime()
    state = _state(runtime)
    state.reserve_full_attention_scratch(tokens=1, num_splits=2)
    state.reserve_moe_c1_scratch(tokens=1)

    state.free()

    assert runtime.allocations == {}
    assert len(runtime.freed) == 16
