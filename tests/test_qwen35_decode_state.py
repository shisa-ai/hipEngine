from __future__ import annotations

import pytest

from hipengine.core.device import Device
from pathlib import Path

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.materialize import DeviceTensorAllocation, DeviceWeightMap
from hipengine.loading.qwen35_paro import Qwen35ParoConfig, Qwen35ParoLayerDeviceWeights
from hipengine.loading.safetensors import TensorInfo
from hipengine.runtime import Qwen35ParoDecodeState, RuntimeWorkspace
import hipengine.runtime.qwen35_paro as qwen_runtime


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


def _allocation(name: str, ptr: int, shape: tuple[int, ...], dtype: str) -> DeviceTensorAllocation:
    return DeviceTensorAllocation(
        name=name,
        source=TensorInfo(name=f"model.{name}", shard_path=Path("/tmp/fake.safetensors"), dtype="F16", shape=shape),
        buffer=DeviceBuffer(ptr=ptr, nbytes=1),
        tensor=Tensor.from_handle(ptr, shape, dtype, Device("hip", 0)),
    )


def _state(runtime: FakeRuntime, weights: DeviceWeightMap | None = None) -> Qwen35ParoDecodeState:
    layer = Qwen35ParoLayerDeviceWeights(config=_config(), layer_id=0, weights=weights or DeviceWeightMap({}))
    return Qwen35ParoDecodeState(
        layer_weights=layer,
        workspace=RuntimeWorkspace(runtime=runtime),
        runtime=runtime,
    )


def _prepared_moe_weights() -> DeviceWeightMap:
    prefix = "layers.0.mlp"
    experts = f"{prefix}.experts"
    return DeviceWeightMap(
        {
            f"{prefix}.router_shared_gate.weight": _allocation(
                f"{prefix}.router_shared_gate.weight", 0xB000, (129, 4096), "bf16"
            ),
            f"{experts}.stacked_gate_qweight_pack8_decode": _allocation(
                f"{experts}.stacked_gate_qweight_pack8_decode", 0xB100, (128, 96, 4096), "int32"
            ),
            f"{experts}.stacked_gate_qzeros": _allocation(f"{experts}.stacked_gate_qzeros", 0xB200, (128, 32, 96), "int32"),
            f"{experts}.stacked_gate_scales": _allocation(f"{experts}.stacked_gate_scales", 0xB300, (128, 32, 96), "fp16"),
            f"{experts}.stacked_up_qweight_pack8_decode": _allocation(
                f"{experts}.stacked_up_qweight_pack8_decode", 0xB400, (128, 96, 4096), "int32"
            ),
            f"{experts}.stacked_up_qzeros": _allocation(f"{experts}.stacked_up_qzeros", 0xB500, (128, 32, 96), "int32"),
            f"{experts}.stacked_up_scales": _allocation(f"{experts}.stacked_up_scales", 0xB600, (128, 32, 96), "fp16"),
            f"{experts}.stacked_down_qweight_pack8_decode": _allocation(
                f"{experts}.stacked_down_qweight_pack8_decode", 0xB700, (128, 512, 768), "int32"
            ),
            f"{experts}.stacked_down_qzeros": _allocation(f"{experts}.stacked_down_qzeros", 0xB800, (128, 6, 512), "int32"),
            f"{experts}.stacked_down_scales": _allocation(f"{experts}.stacked_down_scales", 0xB900, (128, 6, 512), "fp16"),
            f"{experts}.down_weight_pairs": _allocation(f"{experts}.down_weight_pairs", 0xBA00, (6, 128), "int16"),
            f"{experts}.down_weight_theta": _allocation(f"{experts}.down_weight_theta", 0xBB00, (6, 64), "bf16"),
            f"{experts}.down_weight_channel_scales": _allocation(
                f"{experts}.down_weight_channel_scales", 0xBC00, (768,), "bf16"
            ),
            f"{prefix}.shared_expert.gate_up_weight_w8a16": _allocation(
                f"{prefix}.shared_expert.gate_up_weight_w8a16", 0xBD00, (1536, 4096), "int8"
            ),
            f"{prefix}.shared_expert.gate_up_weight_w8a16_scale": _allocation(
                f"{prefix}.shared_expert.gate_up_weight_w8a16_scale", 0xBE00, (1536,), "fp32"
            ),
            f"{prefix}.shared_expert.down_weight_w8a16": _allocation(
                f"{prefix}.shared_expert.down_weight_w8a16", 0xBF00, (4096, 768), "int8"
            ),
            f"{prefix}.shared_expert.down_weight_w8a16_scale": _allocation(
                f"{prefix}.shared_expert.down_weight_w8a16_scale", 0xC000, (4096,), "fp32"
            ),
        }
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
    assert scratch.down_input.shape == (1, 8, 768)
    assert scratch.down_out.shape == (1, 8, 4096)
    assert scratch.shared_up.shape == (1, 1536)
    assert scratch.shared_intermediate.shape == (1, 768)
    assert scratch.shared_out.shape == (1, 4096)
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


def test_qwen35_decode_state_projects_pack8_with_normalized_weight_prefix(monkeypatch) -> None:
    runtime = FakeRuntime()
    prefix = "layers.0.self_attn.o_proj"
    weights = DeviceWeightMap(
        {
            f"{prefix}.qweight": _allocation(f"{prefix}.qweight", 0xB000, (32, 512), "int32"),
            f"{prefix}.qzeros": _allocation(f"{prefix}.qzeros", 0xB100, (1, 32), "int32"),
            f"{prefix}.scales": _allocation(f"{prefix}.scales", 0xB200, (32, 128), "fp16"),
        }
    )
    state = _state(runtime, weights)
    x = Tensor.from_handle(0xC000, (1, 4096), "bf16", Device("hip", 0))
    out = Tensor.from_handle(0xC100, (1, 4096), "bf16", Device("hip", 0))
    calls = []

    def fake_gemv(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_strided_bf16", fake_gemv)

    result = state.project_pack8_bf16(x, out, weight_prefix=f"model.{prefix}", rows=1, group_size=128)

    assert result is out
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (0xC000, 0xB000, 0xB100, 0xB200, 0xC100, 1, 4096, 32, 128)
    assert kwargs == {"threads": 128, "library": None, "runtime": runtime}


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _spans() -> KVLiveSpans:
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0xD000, (2,), "int32"),
        live_counts=_tensor(0xD100, (1,), "int64"),
        max_live_count=1,
        storage_dtype="bf16",
    )


def test_qwen35_decode_state_appends_kv_with_scratch_pointers(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime)
    scratch = state.reserve_full_attention_scratch(tokens=1, num_splits=2)
    key_cache = _tensor(0xE000, (2, 256, 2, 256), "bf16")
    value_cache = _tensor(0xF000, (2, 256, 2, 256), "bf16")
    calls = []

    def fake_append(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(qwen_runtime, "qwen35_write_paged_kv_mixed_value_bf16_spans", fake_append)

    state.append_full_attention_kv(scratch, key_cache=key_cache, value_cache=value_cache, spans=_spans())

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:4] == (scratch.key.ptr, scratch.value.ptr, key_cache.ptr, value_cache.ptr)
    assert args[5:] == (256, 2, 256)
    assert kwargs == {"library": None, "runtime": runtime}


def test_qwen35_decode_state_decodes_gqa_gate_with_scratch_pointers(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime)
    scratch = state.reserve_full_attention_scratch(tokens=1, num_splits=2)
    key_cache = _tensor(0xE000, (2, 256, 2, 256), "bf16")
    value_cache = _tensor(0xF000, (2, 256, 2, 256), "bf16")
    calls = []

    def fake_decode(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(qwen_runtime, "qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans", fake_decode)

    out = state.decode_full_attention_gqa_gate_bf16(
        scratch,
        key_cache=key_cache,
        value_cache=value_cache,
        spans=_spans(),
        chunk_size=256,
        num_splits=2,
    )

    assert out is scratch.gated_attn
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:8] == (
        scratch.query.ptr,
        key_cache.ptr,
        value_cache.ptr,
        scratch.gate.ptr,
        scratch.gated_attn.ptr,
        scratch.partial_out.ptr,
        scratch.partial_m.ptr,
        scratch.partial_l.ptr,
    )
    assert args[9:18] == (256, 2, 256, 16, 2, 256, 256, 1, 256 ** -0.5)
    assert kwargs == {"library": None, "runtime": runtime}


def test_qwen35_decode_state_routes_moe_topk_shared(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    hidden = _tensor(0xCA00, (1, 4096), "bf16")
    calls = []

    def fake_router(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(qwen_runtime, "qwen35_router_topk_shared_out_bf16", fake_router)

    selected, weights = state.route_moe_topk_shared_bf16(hidden, scratch)

    assert selected is scratch.selected_experts
    assert weights is scratch.routing_weights
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (
        hidden.ptr,
        0xB000,
        scratch.router_logits.ptr,
        scratch.selected_experts.ptr,
        scratch.routing_weights.ptr,
        1,
        4096,
        129,
        128,
        8,
    )
    assert kwargs == {"threads": 512, "library": None, "runtime": runtime}


def test_qwen35_decode_state_activates_and_rotates_moe_down(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    calls = []

    def fake_silu_rotate(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_bf16", fake_silu_rotate)

    out = state.activate_rotate_moe_down_bf16(scratch)

    assert out is scratch.down_input
    args, kwargs = calls[0]
    assert args == (scratch.gate_up.ptr, 0xBA00, 0xBB00, 0xBC00, scratch.down_input.ptr, 8, 768, 128, 6)
    assert kwargs == {"library": None, "runtime": runtime}


def test_qwen35_decode_state_selected_moe_gate_up_and_down(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    hidden = _tensor(0xCA00, (1, 4096), "bf16")
    gate_calls = []
    down_calls = []

    def fake_gate(*args, **kwargs):
        gate_calls.append((args, kwargs))

    def fake_down(*args, **kwargs):
        down_calls.append((args, kwargs))

    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_bf16", fake_gate)
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_bf16", fake_down)

    gate_up = state.selected_moe_gate_up_pack8_bf16(hidden, scratch)
    down = state.selected_moe_down_pack8_bf16(scratch.down_input, scratch)

    assert gate_up is scratch.gate_up
    assert down is scratch.down_out
    gate_args, gate_kwargs = gate_calls[0]
    assert gate_args == (
        hidden.ptr,
        scratch.selected_experts.ptr,
        0xB100,
        0xB200,
        0xB300,
        0xB400,
        0xB500,
        0xB600,
        scratch.gate_up.ptr,
        1,
        8,
        4096,
        96,
        96,
        128,
        128,
    )
    assert gate_kwargs == {"threads": 128, "library": None, "runtime": runtime}
    down_args, down_kwargs = down_calls[0]
    assert down_args == (
        scratch.down_input.ptr,
        scratch.selected_experts.ptr,
        0xB700,
        0xB800,
        0xB900,
        scratch.down_out.ptr,
        8,
        768,
        512,
        128,
        128,
    )
    assert down_kwargs == {"threads": 128, "library": None, "runtime": runtime}


def test_qwen35_decode_state_runs_shared_expert_w8a16(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    hidden = _tensor(0xCA00, (1, 4096), "bf16")
    linear_calls = []
    silu_calls = []

    def fake_linear(*args, **kwargs):
        linear_calls.append((args, kwargs))

    def fake_silu(*args, **kwargs):
        silu_calls.append((args, kwargs))

    monkeypatch.setattr(qwen_runtime, "w8a16_linear_bf16_lowp_out", fake_linear)
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_out_bf16", fake_silu)

    out = state.shared_expert_w8a16_bf16(hidden, scratch)

    assert out is scratch.shared_out
    assert linear_calls[0][0] == (hidden.ptr, 0xBD00, 0xBE00, scratch.shared_up.ptr, 1, 4096, 1536)
    assert linear_calls[0][1] == {"threads": 64, "library": None, "runtime": runtime}
    assert silu_calls[0][0] == (scratch.shared_up.ptr, scratch.shared_intermediate.ptr, 1, 768)
    assert silu_calls[0][1] == {"library": None, "runtime": runtime}
    assert linear_calls[1][0] == (scratch.shared_intermediate.ptr, 0xBF00, 0xC000, scratch.shared_out.ptr, 1, 768, 4096)
    assert linear_calls[1][1] == {"threads": 64, "library": None, "runtime": runtime}


def test_qwen35_decode_state_combines_moe_shared_residual(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    shared = _tensor(0xCB00, (1, 4096), "bf16")
    residual = _tensor(0xCC00, (1, 4096), "bf16")
    calls = []

    def fake_combine(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(qwen_runtime, "weighted_sum_shared_gate_combine_residual_out_bf16_f32w", fake_combine)

    out = state.combine_moe_c1_shared_residual_bf16(scratch, shared=shared, residual=residual)

    assert out is scratch.moe_out
    args, kwargs = calls[0]
    assert args == (
        scratch.down_out.ptr,
        scratch.routing_weights.ptr,
        shared.ptr,
        scratch.router_logits.ptr + 128 * 4,
        residual.ptr,
        scratch.moe_out.ptr,
        8,
        4096,
    )
    assert kwargs == {"threads": 256, "library": None, "runtime": runtime}
    with pytest.raises(ValueError, match="tokens=1"):
        state.combine_moe_c1_shared_residual_bf16(scratch, shared=shared, residual=residual, tokens=2)


def test_qwen35_decode_state_runs_moe_c1_chain_in_parent_order(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    hidden = _tensor(0xCA00, (1, 4096), "bf16")
    residual = _tensor(0xCC00, (1, 4096), "bf16")
    order = []

    monkeypatch.setattr(qwen_runtime, "qwen35_router_topk_shared_out_bf16", lambda *a, **k: order.append("router"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_bf16", lambda *a, **k: order.append("gate_up"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_bf16", lambda *a, **k: order.append("silu_rotate"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_bf16", lambda *a, **k: order.append("down"))
    monkeypatch.setattr(qwen_runtime, "w8a16_linear_bf16_lowp_out", lambda *a, **k: order.append("w8a16"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_out_bf16", lambda *a, **k: order.append("shared_silu"))
    monkeypatch.setattr(
        qwen_runtime,
        "weighted_sum_shared_gate_combine_residual_out_bf16_f32w",
        lambda *a, **k: order.append("combine"),
    )

    out = state.run_moe_c1_bf16(hidden, residual, scratch=scratch)

    assert out is scratch.moe_out
    assert order == ["router", "gate_up", "silu_rotate", "down", "w8a16", "shared_silu", "w8a16", "combine"]
    with pytest.raises(ValueError, match="tokens=1"):
        state.run_moe_c1_bf16(hidden, residual, tokens=2)


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
    assert len(runtime.freed) == 20
