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
        rms_norm_eps=1.0e-6,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
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

    assert scratch.attn_input.shape == (1, 4096)
    assert scratch.q_rot.shape == (1, 4096)
    assert scratch.q_proj_key.shape == (1, 8704)
    assert scratch.q_proj.shape == (1, 8192)
    assert scratch.key_bf16.shape == (1, 512)
    assert scratch.query_raw.shape == (1, 16, 256)
    assert scratch.key_raw.shape == (1, 2, 256)
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
    assert scratch.o_rot.shape == (1, 4096)
    assert scratch.o_proj.shape == (1, 4096)


def test_qwen35_decode_state_reserves_moe_c1_scratch() -> None:
    runtime = FakeRuntime()
    state = _state(runtime)

    scratch = state.reserve_moe_c1_scratch(tokens=1)

    assert scratch.normed.shape == (1, 4096)
    assert scratch.residual.shape == (1, 4096)
    assert scratch.router_logits.shape == (1, 129)
    assert scratch.routing_weights.shape == (1, 8)
    assert scratch.selected_experts.shape == (1, 8)
    assert scratch.selected_experts.dtype is DType.INT64
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
            f"{prefix}.qweight": _allocation(f"{prefix}.qweight", 0xB000, (4096, 512), "int32"),
            f"{prefix}.qzeros": _allocation(f"{prefix}.qzeros", 0xB100, (32, 512), "int32"),
            f"{prefix}.scales": _allocation(f"{prefix}.scales", 0xB200, (32, 4096), "bf16"),
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
    assert args == (0xC000, 0xB000, 0xB100, 0xB200, 0xC100, 1, 4096, 512, 128)
    assert kwargs == {"threads": 128, "stream": 0, "library": None, "runtime": runtime}


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _spans() -> KVLiveSpans:
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0xD000, (2,), "int32"),
        live_counts=_tensor(0xD100, (1,), "int64"),
        max_live_count=1,
        storage_dtype="bf16",
    )


def _linear_weights() -> DeviceWeightMap:
    prefix = "layers.0.linear_attn"
    return DeviceWeightMap(
        {
            "layers.0.input_layernorm.weight": _allocation("layers.0.input_layernorm.weight", 0x9010, (4096,), "bf16"),
            "layers.0.post_attention_layernorm.weight": _allocation(
                "layers.0.post_attention_layernorm.weight", 0x9020, (4096,), "bf16"
            ),
            f"{prefix}.in_proj_qkv.pairs": _allocation(f"{prefix}.in_proj_qkv.pairs", 0x9100, (8, 4096), "int16"),
            f"{prefix}.in_proj_qkv.theta": _allocation(f"{prefix}.in_proj_qkv.theta", 0x9200, (8, 2048), "bf16"),
            f"{prefix}.in_proj_qkv.channel_scales": _allocation(
                f"{prefix}.in_proj_qkv.channel_scales", 0x9300, (1, 4096), "bf16"
            ),
            f"{prefix}.in_proj_z.pairs": _allocation(f"{prefix}.in_proj_z.pairs", 0x9400, (8, 4096), "int16"),
            f"{prefix}.in_proj_z.theta": _allocation(f"{prefix}.in_proj_z.theta", 0x9500, (8, 2048), "bf16"),
            f"{prefix}.in_proj_z.channel_scales": _allocation(
                f"{prefix}.in_proj_z.channel_scales", 0x9600, (1, 4096), "bf16"
            ),
            f"{prefix}.out_proj.pairs": _allocation(f"{prefix}.out_proj.pairs", 0x9650, (8, 4096), "int16"),
            f"{prefix}.out_proj.theta": _allocation(f"{prefix}.out_proj.theta", 0x9660, (8, 2048), "bf16"),
            f"{prefix}.out_proj.channel_scales": _allocation(
                f"{prefix}.out_proj.channel_scales", 0x9670, (1, 4096), "bf16"
            ),
            f"{prefix}.in_proj_qkv.qweight": _allocation(f"{prefix}.in_proj_qkv.qweight", 0x9700, (4096, 1024), "int32"),
            f"{prefix}.in_proj_qkv.qweight_pack8_decode": _allocation(
                f"{prefix}.in_proj_qkv.qweight_pack8_decode", 0x9710, (1024, 4096), "int32"
            ),
            f"{prefix}.in_proj_qkv.qzeros": _allocation(f"{prefix}.in_proj_qkv.qzeros", 0x9800, (32, 1024), "int32"),
            f"{prefix}.in_proj_qkv.scales": _allocation(f"{prefix}.in_proj_qkv.scales", 0x9900, (32, 8192), "bf16"),
            f"{prefix}.in_proj_z.qweight": _allocation(f"{prefix}.in_proj_z.qweight", 0x9A00, (4096, 512), "int32"),
            f"{prefix}.in_proj_z.qweight_pack8_decode": _allocation(
                f"{prefix}.in_proj_z.qweight_pack8_decode", 0x9A10, (512, 4096), "int32"
            ),
            f"{prefix}.in_proj_z.qzeros": _allocation(f"{prefix}.in_proj_z.qzeros", 0x9B00, (32, 512), "int32"),
            f"{prefix}.in_proj_z.scales": _allocation(f"{prefix}.in_proj_z.scales", 0x9C00, (32, 4096), "bf16"),
            f"{prefix}.out_proj.qweight": _allocation(f"{prefix}.out_proj.qweight", 0x9C10, (4096, 512), "int32"),
            f"{prefix}.out_proj.qzeros": _allocation(f"{prefix}.out_proj.qzeros", 0x9C20, (32, 512), "int32"),
            f"{prefix}.out_proj.scales": _allocation(f"{prefix}.out_proj.scales", 0x9C30, (32, 4096), "bf16"),
            f"{prefix}.in_proj_a.weight": _allocation(f"{prefix}.in_proj_a.weight", 0x9D00, (32, 4096), "bf16"),
            f"{prefix}.in_proj_b.weight": _allocation(f"{prefix}.in_proj_b.weight", 0x9E00, (32, 4096), "bf16"),
            f"{prefix}.conv1d.weight": _allocation(f"{prefix}.conv1d.weight", 0x9F00, (8192, 1, 4), "fp32"),
            f"{prefix}.dt_bias": _allocation(f"{prefix}.dt_bias", 0xA100, (32,), "fp32"),
            f"{prefix}.A_log": _allocation(f"{prefix}.A_log", 0xA200, (32,), "fp32"),
            f"{prefix}.norm.weight": _allocation(f"{prefix}.norm.weight", 0xA300, (128,), "fp32"),
        }
    )


def _full_attention_weights() -> DeviceWeightMap:
    prefix = "layers.0.self_attn"
    return DeviceWeightMap(
        {
            "layers.0.input_layernorm.weight": _allocation("layers.0.input_layernorm.weight", 0x8100, (4096,), "bf16"),
            "layers.0.post_attention_layernorm.weight": _allocation(
                "layers.0.post_attention_layernorm.weight", 0x8110, (4096,), "bf16"
            ),
            f"{prefix}.q_norm.weight": _allocation(f"{prefix}.q_norm.weight", 0x8120, (256,), "bf16"),
            f"{prefix}.k_norm.weight": _allocation(f"{prefix}.k_norm.weight", 0x8130, (256,), "bf16"),
            f"{prefix}.q_proj.pairs": _allocation(f"{prefix}.q_proj.pairs", 0x8200, (8, 4096), "int16"),
            f"{prefix}.q_proj.theta": _allocation(f"{prefix}.q_proj.theta", 0x8210, (8, 2048), "bf16"),
            f"{prefix}.q_proj.channel_scales": _allocation(f"{prefix}.q_proj.channel_scales", 0x8220, (1, 4096), "bf16"),
            f"{prefix}.q_proj.qweight": _allocation(f"{prefix}.q_proj.qweight", 0x8230, (4096, 1024), "int32"),
            f"{prefix}.q_proj.qweight_pack8_decode": _allocation(
                f"{prefix}.q_proj.qweight_pack8_decode", 0x8238, (1024, 4096), "int32"
            ),
            f"{prefix}.q_proj.qzeros": _allocation(f"{prefix}.q_proj.qzeros", 0x8240, (32, 1024), "int32"),
            f"{prefix}.q_proj.scales": _allocation(f"{prefix}.q_proj.scales", 0x8250, (32, 8192), "bf16"),
            f"{prefix}.k_proj.pairs": _allocation(f"{prefix}.k_proj.pairs", 0x8300, (8, 4096), "int16"),
            f"{prefix}.k_proj.theta": _allocation(f"{prefix}.k_proj.theta", 0x8310, (8, 2048), "bf16"),
            f"{prefix}.k_proj.channel_scales": _allocation(f"{prefix}.k_proj.channel_scales", 0x8320, (1, 4096), "bf16"),
            f"{prefix}.k_proj.qweight": _allocation(f"{prefix}.k_proj.qweight", 0x8330, (4096, 64), "int32"),
            f"{prefix}.k_proj.qweight_pack8_decode": _allocation(
                f"{prefix}.k_proj.qweight_pack8_decode", 0x8338, (64, 4096), "int32"
            ),
            f"{prefix}.k_proj.qzeros": _allocation(f"{prefix}.k_proj.qzeros", 0x8340, (32, 64), "int32"),
            f"{prefix}.k_proj.scales": _allocation(f"{prefix}.k_proj.scales", 0x8350, (32, 512), "bf16"),
            f"{prefix}.v_proj.pairs": _allocation(f"{prefix}.v_proj.pairs", 0x8400, (8, 4096), "int16"),
            f"{prefix}.v_proj.theta": _allocation(f"{prefix}.v_proj.theta", 0x8410, (8, 2048), "bf16"),
            f"{prefix}.v_proj.channel_scales": _allocation(f"{prefix}.v_proj.channel_scales", 0x8420, (1, 4096), "bf16"),
            f"{prefix}.v_proj.qweight": _allocation(f"{prefix}.v_proj.qweight", 0x8430, (4096, 64), "int32"),
            f"{prefix}.v_proj.qzeros": _allocation(f"{prefix}.v_proj.qzeros", 0x8440, (32, 64), "int32"),
            f"{prefix}.v_proj.scales": _allocation(f"{prefix}.v_proj.scales", 0x8450, (32, 512), "bf16"),
            f"{prefix}.o_proj.pairs": _allocation(f"{prefix}.o_proj.pairs", 0x8500, (8, 4096), "int16"),
            f"{prefix}.o_proj.theta": _allocation(f"{prefix}.o_proj.theta", 0x8510, (8, 2048), "bf16"),
            f"{prefix}.o_proj.channel_scales": _allocation(f"{prefix}.o_proj.channel_scales", 0x8520, (1, 4096), "bf16"),
            f"{prefix}.o_proj.qweight": _allocation(f"{prefix}.o_proj.qweight", 0x8530, (4096, 512), "int32"),
            f"{prefix}.o_proj.qzeros": _allocation(f"{prefix}.o_proj.qzeros", 0x8540, (32, 512), "int32"),
            f"{prefix}.o_proj.scales": _allocation(f"{prefix}.o_proj.scales", 0x8550, (32, 4096), "bf16"),
        }
    )


def test_qwen35_decode_state_reserves_linear_attention_scratch() -> None:
    runtime = FakeRuntime()
    state = _state(runtime)

    scratch = state.reserve_linear_attention_scratch(tokens=1)

    assert scratch.attn_input.shape == (1, 4096)
    assert scratch.qkv_z.shape == (1, 12288)
    assert scratch.qkv.shape == (1, 8192)
    assert scratch.z.shape == (1, 4096)
    assert scratch.qkv_f32.shape == (1, 8192)
    assert scratch.ab.shape == (1, 64)
    assert scratch.a.shape == (1, 32)
    assert scratch.b.shape == (1, 32)
    assert scratch.conv_out.dtype is DType.FP32
    assert scratch.prefill_query.shape == (1, 32, 128)
    assert scratch.prefill_key.shape == (1, 32, 128)
    assert scratch.prefill_value.shape == (1, 32, 128)
    assert scratch.prefill_beta.shape == (1, 32)
    assert scratch.prefill_decay.shape == (1, 32)
    assert scratch.recurrent_out.shape == (1, 4096)
    assert scratch.recurrent_bf16.shape == (1, 4096)
    assert scratch.out_rot.shape == (1, 4096)
    assert scratch.out_proj.shape == (1, 4096)


def test_qwen35_decode_state_runs_linear_attention_state_chain(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    hidden = _tensor(0xC000, (1, 4096), "bf16")
    conv_state = _tensor(0xC100, (8192, 4), "fp32")
    recurrent_state = _tensor(0xC200, (32, 128, 128), "fp32")
    scratch = state.reserve_linear_attention_scratch(tokens=1)
    calls = []

    def record(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "paro_rotate2_bf16", record("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_strided_bf16", record("pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", record("dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "dense_gemv_out_bf16", record("dense"))
    monkeypatch.setattr(qwen_runtime, "dense_dual_gemv_out_bf16", record("dense_dual"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_conv_decode_bf16", record("conv"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16", record("gdn"))

    out = state.run_linear_attention_state_bf16(
        hidden,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        scratch=scratch,
    )

    assert out is scratch.recurrent_out
    assert [name for name, _, _ in calls] == ["rotate2", "dual_pack8", "dense_dual", "conv", "gdn"]
    rotate_args = calls[0][1]
    assert rotate_args[:3] == (0xC000, scratch.qkv_rot.ptr, scratch.z_rot.ptr)
    assert rotate_args[9:] == (1, 4096, 128, 8)
    assert calls[1][1][:9] == (scratch.qkv_rot.ptr, scratch.z_rot.ptr, 0x9710, 0x9800, 0x9900, 0x9A10, 0x9B00, 0x9C00, scratch.qkv_z.ptr)
    assert calls[1][1][9:] == (1, 4096, 1024, 512, 128)
    assert calls[2][1][:8] == (0xC000, 0x9D00, 0x9E00, scratch.ab.ptr, 1, 4096, 32, 32)
    assert calls[3][1] == (scratch.qkv.ptr, 0xC100, 0x9F00, scratch.conv_out.ptr, 8192, 4)
    assert calls[4][1] == (
        scratch.conv_out.ptr,
        scratch.z.ptr,
        scratch.a.ptr,
        scratch.b.ptr,
        0xA100,
        0xA200,
        0xA300,
        0xC200,
        scratch.recurrent_out.ptr,
        1.0e-6,
        16,
        32,
        128,
        128,
    )


def test_qwen35_decode_state_runs_linear_attention_prefill_state_chain(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    hidden = _tensor(0xC000, (4, 4096), "bf16")
    conv_state = _tensor(0xC100, (8192, 4), "fp32")
    recurrent_state = _tensor(0xC200, (32, 128, 128), "fp32")
    scratch = state.reserve_linear_attention_scratch(tokens=4)
    calls = []

    def record(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "paro_rotate2_bf16", record("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", record("dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "dense_dual_gemv_out_bf16", record("dense_dual"))
    monkeypatch.setattr(qwen_runtime, "bf16_to_f32", record("cast_qkv"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_conv_prefill_f32", record("conv_prefill"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_prefill_prepare_f32_bf16", record("prepare"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_prefill_recurrent_k2_f32", record("gdn_k2"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_prefill_rmsnorm_gate_bf16", record("rms_gate"))

    out = state.run_linear_attention_prefill_state_bf16(
        hidden,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        scratch=scratch,
        tokens=4,
    )

    assert out is scratch.recurrent_bf16
    assert [name for name, _, _ in calls] == [
        "rotate2",
        "dual_pack8",
        "dense_dual",
        "cast_qkv",
        "conv_prefill",
        "prepare",
        "gdn_k2",
        "rms_gate",
    ]
    assert calls[3][1] == (scratch.qkv.ptr, scratch.qkv_f32.ptr, 4 * 8192)
    assert calls[4][1] == (scratch.qkv_f32.ptr, conv_state.ptr, 0x9F00, scratch.conv_out.ptr, 4, 8192, 4)
    assert calls[5][1] == (
        scratch.conv_out.ptr,
        scratch.a.ptr,
        scratch.b.ptr,
        0xA100,
        0xA200,
        scratch.prefill_query.ptr,
        scratch.prefill_key.ptr,
        scratch.prefill_value.ptr,
        scratch.prefill_beta.ptr,
        scratch.prefill_decay.ptr,
        4,
        16,
        32,
        128,
        128,
    )
    assert calls[6][1] == (
        scratch.prefill_query.ptr,
        scratch.prefill_key.ptr,
        scratch.prefill_value.ptr,
        scratch.prefill_beta.ptr,
        scratch.prefill_decay.ptr,
        recurrent_state.ptr,
        scratch.recurrent_out.ptr,
        4,
        32,
        128,
        128,
    )
    assert calls[7][1] == (scratch.recurrent_out.ptr, scratch.z.ptr, 0xA300, scratch.recurrent_bf16.ptr, 1.0e-6, 4, 32, 128)


def test_qwen35_decode_state_projects_linear_attention_prefill_out(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    scratch = state.reserve_linear_attention_scratch(tokens=4)
    calls = []

    def record(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", record("rotate1"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_strided_bf16", record("pack8"))

    out = state.project_linear_attention_prefill_out_bf16(scratch, tokens=4)

    assert out is scratch.out_proj
    assert [name for name, _, _ in calls] == ["rotate1", "pack8"]
    assert calls[0][1] == (scratch.recurrent_bf16.ptr, scratch.out_rot.ptr, 0x9650, 0x9660, 0x9670, 4, 4096, 128, 8)
    assert calls[1][1][:5] == (scratch.out_rot.ptr, 0x9C10, 0x9C20, 0x9C30, scratch.out_proj.ptr)
    assert calls[1][1][5:] == (4, 4096, 512, 128)


def test_qwen35_decode_state_runs_linear_attention_prefill_out_proj_chain(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    hidden = _tensor(0xC000, (4, 4096), "bf16")
    conv_state = _tensor(0xC100, (8192, 4), "fp32")
    recurrent_state = _tensor(0xC200, (32, 128, 128), "fp32")
    scratch = state.reserve_linear_attention_scratch(tokens=4)
    order = []

    monkeypatch.setattr(qwen_runtime, "paro_rotate2_bf16", lambda *a, **k: order.append("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", lambda *a, **k: order.append("dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "dense_dual_gemv_out_bf16", lambda *a, **k: order.append("dense_dual"))
    monkeypatch.setattr(qwen_runtime, "bf16_to_f32", lambda *a, **k: order.append("cast_qkv"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_conv_prefill_f32", lambda *a, **k: order.append("conv_prefill"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_prefill_prepare_f32_bf16", lambda *a, **k: order.append("prepare"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_prefill_recurrent_k2_f32", lambda *a, **k: order.append("gdn_k2"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_prefill_rmsnorm_gate_bf16", lambda *a, **k: order.append("rms_gate"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", lambda *a, **k: order.append("rotate1"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_strided_bf16", lambda *a, **k: order.append("pack8"))

    out = state.run_linear_attention_prefill_out_proj_bf16(
        hidden,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        scratch=scratch,
        tokens=4,
    )

    assert out is scratch.out_proj
    assert order == [
        "rotate2",
        "dual_pack8",
        "dense_dual",
        "cast_qkv",
        "conv_prefill",
        "prepare",
        "gdn_k2",
        "rms_gate",
        "rotate1",
        "pack8",
    ]
    with pytest.raises(ValueError, match="tokens >= linear_conv_kernel_dim"):
        state.run_linear_attention_prefill_out_proj_bf16(
            hidden,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            scratch=scratch,
            tokens=2,
        )


def test_qwen35_decode_state_projects_linear_attention_out(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    scratch = state.reserve_linear_attention_scratch(tokens=1)
    calls = []

    def record(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "f32_to_bf16", record("cast"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", record("rotate1"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_strided_bf16", record("pack8"))

    out = state.project_linear_attention_out_bf16(scratch)

    assert out is scratch.out_proj
    assert [name for name, _, _ in calls] == ["cast", "rotate1", "pack8"]
    assert calls[0][1] == (scratch.recurrent_out.ptr, scratch.recurrent_bf16.ptr, 4096)
    assert calls[0][2] == {"stream": 0, "library": None, "runtime": runtime}
    assert calls[1][1] == (scratch.recurrent_bf16.ptr, scratch.out_rot.ptr, 0x9650, 0x9660, 0x9670, 1, 4096, 128, 8)
    assert calls[1][2] == {"stream": 0, "library": None, "runtime": runtime}
    assert calls[2][1][:5] == (scratch.out_rot.ptr, 0x9C10, 0x9C20, 0x9C30, scratch.out_proj.ptr)
    assert calls[2][1][5:] == (1, 4096, 512, 128)
    assert calls[2][2] == {"threads": 128, "stream": 0, "library": None, "runtime": runtime}


def test_qwen35_decode_state_runs_linear_attention_out_proj_chain(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    hidden = _tensor(0xC000, (1, 4096), "bf16")
    conv_state = _tensor(0xC100, (8192, 4), "fp32")
    recurrent_state = _tensor(0xC200, (32, 128, 128), "fp32")
    scratch = state.reserve_linear_attention_scratch(tokens=1)
    order = []

    monkeypatch.setattr(qwen_runtime, "paro_rotate2_bf16", lambda *a, **k: order.append("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_strided_bf16", lambda *a, **k: order.append("pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", lambda *a, **k: order.append("dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "dense_gemv_out_bf16", lambda *a, **k: order.append("dense"))
    monkeypatch.setattr(qwen_runtime, "dense_dual_gemv_out_bf16", lambda *a, **k: order.append("dense_dual"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_conv_decode_bf16", lambda *a, **k: order.append("conv"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16", lambda *a, **k: order.append("gdn"))
    monkeypatch.setattr(qwen_runtime, "f32_to_bf16", lambda *a, **k: order.append("cast"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", lambda *a, **k: order.append("rotate1"))

    out = state.run_linear_attention_out_proj_bf16(
        hidden,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        scratch=scratch,
    )

    assert out is scratch.out_proj
    assert order == ["rotate2", "dual_pack8", "dense_dual", "conv", "gdn", "cast", "rotate1", "pack8"]
    with pytest.raises(ValueError, match="tokens=1"):
        state.run_linear_attention_out_proj_bf16(hidden, conv_state=conv_state, recurrent_state=recurrent_state, tokens=2)


def test_qwen35_decode_state_runs_linear_attention_moe_layer_chain(monkeypatch) -> None:
    runtime = FakeRuntime()
    weights = DeviceWeightMap({**_linear_weights().tensors, **_prepared_moe_weights().tensors})
    state = _state(runtime, weights)
    hidden = _tensor(0xC000, (1, 4096), "bf16")
    conv_state = _tensor(0xC100, (8192, 4), "fp32")
    recurrent_state = _tensor(0xC200, (32, 128, 128), "fp32")
    linear_scratch = state.reserve_linear_attention_scratch(tokens=1)
    moe_scratch = state.reserve_moe_c1_scratch(tokens=1)
    calls = []

    def record(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "paro_rmsnorm_out_bf16", record("input_norm"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate2_bf16", record("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_strided_bf16", record("pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", record("dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "dense_gemv_out_bf16", record("dense"))
    monkeypatch.setattr(qwen_runtime, "dense_dual_gemv_out_bf16", record("dense_dual"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_conv_decode_bf16", record("conv"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16", record("gdn"))
    monkeypatch.setattr(qwen_runtime, "f32_to_bf16", record("cast"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", record("rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_add_rmsnorm_out_bf16", record("post_norm"))
    monkeypatch.setattr(qwen_runtime, "qwen35_router_topk_shared_out_bf16", record("router"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_bf16", record("gate_up"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_bf16", record("silu_rotate"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_bf16", record("down"))
    monkeypatch.setattr(qwen_runtime, "w8a16_linear_bf16_lowp_out", record("w8a16"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_out_bf16", record("shared_silu"))
    monkeypatch.setattr(qwen_runtime, "weighted_sum_shared_gate_combine_residual_out_bf16_f32w", record("combine"))

    out = state.run_linear_attention_moe_c1_layer_bf16(
        hidden,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        linear_scratch=linear_scratch,
        moe_scratch=moe_scratch,
    )

    assert out is moe_scratch.moe_out
    assert [name for name, _, _ in calls] == [
        "input_norm",
        "rotate2",
        "dual_pack8",
        "dense_dual",
        "conv",
        "gdn",
        "cast",
        "rotate1",
        "pack8",
        "post_norm",
        "router",
        "gate_up",
        "silu_rotate",
        "down",
        "w8a16",
        "shared_silu",
        "w8a16",
        "combine",
    ]
    assert calls[0][1] == (hidden.ptr, 0x9010, linear_scratch.attn_input.ptr, 1, 4096, 1.0e-6)
    assert calls[9][1] == (
        hidden.ptr,
        linear_scratch.out_proj.ptr,
        0x9020,
        moe_scratch.normed.ptr,
        moe_scratch.residual.ptr,
        1,
        4096,
        1.0e-6,
    )
    with pytest.raises(ValueError, match="tokens=1"):
        state.run_linear_attention_moe_c1_layer_bf16(hidden, conv_state=conv_state, recurrent_state=recurrent_state, tokens=2)


def test_qwen35_decode_state_runs_full_attention_moe_layer_chain(monkeypatch) -> None:
    runtime = FakeRuntime()
    weights = DeviceWeightMap({**_full_attention_weights().tensors, **_prepared_moe_weights().tensors})
    state = _state(runtime, weights)
    hidden = _tensor(0xC000, (1, 4096), "bf16")
    key_cache = _tensor(0xE000, (1, 256, 2, 256), "bf16")
    value_cache = _tensor(0xF000, (1, 256, 2, 256), "bf16")
    cos_table = _tensor(0xD200, (4, 256), "fp32")
    sin_table = _tensor(0xD300, (4, 256), "fp32")
    position = _tensor(0xD400, (1,), "int64")
    attn = state.reserve_full_attention_scratch(tokens=1, num_splits=1)
    moe = state.reserve_moe_c1_scratch(tokens=1)
    calls = []

    def record(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "paro_rmsnorm_out_bf16", record("input_norm"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate3_bf16", record("rotate3"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_strided_bf16", record("pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", record("dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "bf16_to_f32", record("bf16_to_f32"))
    monkeypatch.setattr(qwen_runtime, "qwen35_head_rmsnorm_partial_rotary_position_f32_bf16", record("head_rotary"))
    monkeypatch.setattr(qwen_runtime, "qwen35_write_paged_kv_mixed_value_bf16_spans", record("kv"))
    monkeypatch.setattr(qwen_runtime, "qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans", record("attention"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", record("rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_add_rmsnorm_out_bf16", record("post_norm"))
    monkeypatch.setattr(qwen_runtime, "qwen35_router_topk_shared_out_bf16", record("router"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_bf16", record("gate_up"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_bf16", record("silu_rotate"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_bf16", record("down"))
    monkeypatch.setattr(qwen_runtime, "w8a16_linear_bf16_lowp_out", record("w8a16"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_out_bf16", record("shared_silu"))
    monkeypatch.setattr(qwen_runtime, "weighted_sum_shared_gate_combine_residual_out_bf16_f32w", record("combine"))

    out = state.run_full_attention_moe_c1_layer_bf16(
        hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        append_spans=_spans(),
        decode_spans=_spans(),
        cos_table=cos_table,
        sin_table=sin_table,
        position=position,
        max_positions=4,
        attention_scratch=attn,
        moe_scratch=moe,
    )

    assert out is moe.moe_out
    assert [name for name, _, _ in calls] == [
        "input_norm",
        "rotate3",
        "dual_pack8",
        "pack8",
        "bf16_to_f32",
        "bf16_to_f32",
        "head_rotary",
        "kv",
        "attention",
        "rotate1",
        "pack8",
        "post_norm",
        "router",
        "gate_up",
        "silu_rotate",
        "down",
        "w8a16",
        "shared_silu",
        "w8a16",
        "combine",
    ]
    assert calls[1][1][:4] == (attn.attn_input.ptr, attn.q_rot.ptr, attn.k_rot.ptr, attn.v_rot.ptr)
    assert calls[2][1][:9] == (attn.q_rot.ptr, attn.k_rot.ptr, 0x8238, 0x8240, 0x8250, 0x8338, 0x8340, 0x8350, attn.q_proj_key.ptr)
    assert calls[4][1] == (attn.q_proj.ptr, attn.query_raw.ptr, 4096)
    assert calls[5][1] == (attn.key_bf16.ptr, attn.key_raw.ptr, 512)
    assert calls[6][1][:9] == (attn.query_raw.ptr, attn.key_raw.ptr, 0x8120, 0x8130, 0xD200, 0xD300, 0xD400, attn.query.ptr, attn.key.ptr)
    assert calls[8][1][3] == attn.q_proj.ptr + 4096 * DType.BF16.itemsize
    assert calls[9][1][:5] == (attn.gated_attn.ptr, attn.o_rot.ptr, 0x8500, 0x8510, 0x8520)
    assert calls[11][1][:5] == (hidden.ptr, attn.o_proj.ptr, 0x8110, moe.normed.ptr, moe.residual.ptr)
    with pytest.raises(ValueError, match="tokens=1"):
        state.run_full_attention_moe_c1_layer_bf16(
            hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            append_spans=_spans(),
            decode_spans=_spans(),
            cos_table=cos_table,
            sin_table=sin_table,
            position=position,
            max_positions=4,
            tokens=2,
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
    assert kwargs == {"stream": 0, "library": None, "runtime": runtime}


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
    assert kwargs == {"stream": 0, "library": None, "runtime": runtime}


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
    assert kwargs == {"threads": 512, "stream": 0, "library": None, "runtime": runtime}


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
    assert kwargs == {"stream": 0, "library": None, "runtime": runtime}


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
    assert gate_kwargs == {"threads": 128, "stream": 0, "library": None, "runtime": runtime}
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
    assert down_kwargs == {"threads": 128, "stream": 0, "library": None, "runtime": runtime}


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
    assert linear_calls[0][1] == {"threads": 64, "stream": 0, "library": None, "runtime": runtime}
    assert silu_calls[0][0] == (scratch.shared_up.ptr, scratch.shared_intermediate.ptr, 1, 768)
    assert silu_calls[0][1] == {"stream": 0, "library": None, "runtime": runtime}
    assert linear_calls[1][0] == (scratch.shared_intermediate.ptr, 0xBF00, 0xC000, scratch.shared_out.ptr, 1, 768, 4096)
    assert linear_calls[1][1] == {"threads": 64, "stream": 0, "library": None, "runtime": runtime}


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
    assert kwargs == {"threads": 256, "stream": 0, "library": None, "runtime": runtime}
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
    assert len(runtime.freed) == 30
