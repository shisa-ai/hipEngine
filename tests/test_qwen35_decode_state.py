from __future__ import annotations

import pytest

from hipengine.core.device import Device
from pathlib import Path
from types import SimpleNamespace

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata
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
        self.memsets: list[tuple[int, int, int]] = []
        self.copies: list[tuple[int, int, int, int]] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(nbytes, 1) + 0x100
        self.allocations[ptr] = nbytes
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(ptr)
        self.allocations.pop(ptr, None)

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        assert dst in self.allocations
        assert value in {0, 0xFF}
        assert nbytes <= self.allocations[dst]
        self.memsets.append((dst, value, nbytes))

    def memset_async(self, dst: int, value: int, nbytes: int, stream: int) -> None:
        self.memset(dst, value, nbytes)

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind, stream: int) -> None:
        self.copies.append((int(dst), int(src), int(nbytes), int(stream)))


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
            f"{experts}.gate_up_weight_pairs": _allocation(f"{experts}.gate_up_weight_pairs", 0xB9A0, (32, 128), "int16"),
            f"{experts}.gate_up_weight_theta": _allocation(f"{experts}.gate_up_weight_theta", 0xB9B0, (32, 64), "fp16"),
            f"{experts}.gate_up_weight_channel_scales": _allocation(
                f"{experts}.gate_up_weight_channel_scales", 0xB9C0, (4096,), "fp16"
            ),
            f"{experts}.down_weight_pairs": _allocation(f"{experts}.down_weight_pairs", 0xBA00, (6, 128), "int16"),
            f"{experts}.down_weight_theta": _allocation(f"{experts}.down_weight_theta", 0xBB00, (6, 64), "fp16"),
            f"{experts}.down_weight_channel_scales": _allocation(
                f"{experts}.down_weight_channel_scales", 0xBC00, (768,), "fp16"
            ),
            # Packed W4 PARO shared-expert tensors (gate_proj/up_proj into
            # hidden=4096, shared_int=768; down_proj reverse).
            f"{prefix}.shared_expert.gate_proj.qweight": _allocation(
                f"{prefix}.shared_expert.gate_proj.qweight", 0xBD00, (4096, 96), "int32"
            ),
            f"{prefix}.shared_expert.gate_proj.qweight_pack8_decode": _allocation(
                f"{prefix}.shared_expert.gate_proj.qweight_pack8_decode", 0xBD10, (96, 4096), "int32"
            ),
            f"{prefix}.shared_expert.gate_proj.qzeros": _allocation(
                f"{prefix}.shared_expert.gate_proj.qzeros", 0xBD20, (32, 96), "int32"
            ),
            f"{prefix}.shared_expert.gate_proj.scales": _allocation(
                f"{prefix}.shared_expert.gate_proj.scales", 0xBD30, (32, 768), "fp16"
            ),
            f"{prefix}.shared_expert.gate_proj.theta": _allocation(
                f"{prefix}.shared_expert.gate_proj.theta", 0xBD40, (32, 64), "fp16"
            ),
            f"{prefix}.shared_expert.gate_proj.pairs": _allocation(
                f"{prefix}.shared_expert.gate_proj.pairs", 0xBD50, (32, 128), "int16"
            ),
            f"{prefix}.shared_expert.gate_proj.channel_scales": _allocation(
                f"{prefix}.shared_expert.gate_proj.channel_scales", 0xBD60, (4096,), "fp16"
            ),
            f"{prefix}.shared_expert.up_proj.qweight": _allocation(
                f"{prefix}.shared_expert.up_proj.qweight", 0xBE00, (4096, 96), "int32"
            ),
            f"{prefix}.shared_expert.up_proj.qweight_pack8_decode": _allocation(
                f"{prefix}.shared_expert.up_proj.qweight_pack8_decode", 0xBE10, (96, 4096), "int32"
            ),
            f"{prefix}.shared_expert.up_proj.qzeros": _allocation(
                f"{prefix}.shared_expert.up_proj.qzeros", 0xBE20, (32, 96), "int32"
            ),
            f"{prefix}.shared_expert.up_proj.scales": _allocation(
                f"{prefix}.shared_expert.up_proj.scales", 0xBE30, (32, 768), "fp16"
            ),
            f"{prefix}.shared_expert.up_proj.theta": _allocation(
                f"{prefix}.shared_expert.up_proj.theta", 0xBE40, (32, 64), "fp16"
            ),
            f"{prefix}.shared_expert.up_proj.pairs": _allocation(
                f"{prefix}.shared_expert.up_proj.pairs", 0xBE50, (32, 128), "int16"
            ),
            f"{prefix}.shared_expert.up_proj.channel_scales": _allocation(
                f"{prefix}.shared_expert.up_proj.channel_scales", 0xBE60, (4096,), "fp16"
            ),
            f"{prefix}.shared_expert.down_proj.qweight": _allocation(
                f"{prefix}.shared_expert.down_proj.qweight", 0xBF00, (768, 512), "int32"
            ),
            f"{prefix}.shared_expert.down_proj.qweight_pack8_decode": _allocation(
                f"{prefix}.shared_expert.down_proj.qweight_pack8_decode", 0xBF10, (512, 768), "int32"
            ),
            f"{prefix}.shared_expert.down_proj.qzeros": _allocation(
                f"{prefix}.shared_expert.down_proj.qzeros", 0xBF20, (6, 512), "int32"
            ),
            f"{prefix}.shared_expert.down_proj.scales": _allocation(
                f"{prefix}.shared_expert.down_proj.scales", 0xBF30, (6, 4096), "fp16"
            ),
            f"{prefix}.shared_expert.down_proj.theta": _allocation(
                f"{prefix}.shared_expert.down_proj.theta", 0xBF40, (6, 64), "fp16"
            ),
            f"{prefix}.shared_expert.down_proj.pairs": _allocation(
                f"{prefix}.shared_expert.down_proj.pairs", 0xBF50, (6, 128), "int16"
            ),
            f"{prefix}.shared_expert.down_proj.channel_scales": _allocation(
                f"{prefix}.shared_expert.down_proj.channel_scales", 0xBF60, (768,), "fp16"
            ),
        }
    )


def _legacy_prepared_moe_weights() -> DeviceWeightMap:
    prefix = "layers.0.mlp"
    tensors = {
        name: allocation
        for name, allocation in _prepared_moe_weights().tensors.items()
        if not name.startswith(f"{prefix}.shared_expert.")
    }
    shared = f"{prefix}.shared_expert"
    tensors.update(
        {
            f"{shared}.gate_up_weight_w8a16": _allocation(f"{shared}.gate_up_weight_w8a16", 0xBD00, (1536, 4096), "int8"),
            f"{shared}.gate_up_weight_w8a16_scale": _allocation(
                f"{shared}.gate_up_weight_w8a16_scale", 0xBD10, (1536,), "fp32"
            ),
            f"{shared}.down_weight_w8a16": _allocation(f"{shared}.down_weight_w8a16", 0xBE00, (4096, 768), "int8"),
            f"{shared}.down_weight_w8a16_scale": _allocation(
                f"{shared}.down_weight_w8a16_scale", 0xBE10, (4096,), "fp32"
            ),
        }
    )
    return DeviceWeightMap(tensors)


def test_qwen35_decode_state_reserves_full_attention_split_k_scratch() -> None:
    runtime = FakeRuntime()
    state = _state(runtime)

    scratch = state.reserve_full_attention_scratch(tokens=1, num_splits=2, gated_dtype="bf16")

    assert scratch.attn_input.shape == (1, 4096)
    assert scratch.q_rot.shape == (1, 4096)
    assert scratch.rotate_fuse_barrier.shape == (2,)
    assert scratch.rotate_fuse_barrier.dtype is DType.INT32
    assert scratch.q_proj_key.shape == (1, 8704)
    assert scratch.q_proj.shape == (1, 8192)
    assert scratch.key_bf16.shape == (1, 512)
    assert scratch.query_raw.shape == (1, 16, 256)
    assert scratch.key_raw.shape == (1, 2, 256)
    assert scratch.query.shape == (1, 16, 256)
    assert scratch.key.shape == (1, 2, 256)
    assert scratch.value.shape == (1, 2, 256)
    assert scratch.kv_proj is None
    assert scratch.gate.shape == (1, 16, 256)
    assert scratch.partial_out.shape == (16, 2, 256)
    assert scratch.partial_m.shape == (16, 2)
    assert scratch.partial_l.shape == (16, 2)
    assert scratch.attn_out.shape == (16, 256)
    assert scratch.gated_attn.shape == (1, 4096)
    assert scratch.gated_attn.dtype is DType.BF16
    assert scratch.o_rot.shape == (1, 4096)
    assert scratch.o_proj.shape == (1, 4096)


def test_qwen35_decode_state_reserves_full_attention_kv_fused_scratch(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED", "1")
    runtime = FakeRuntime()
    state = _state(runtime)

    scratch = state.reserve_full_attention_scratch(tokens=1, num_splits=1, activation_dtype="fp16")

    assert scratch.kv_proj is not None
    assert scratch.kv_proj.shape == (1, 1024)
    assert scratch.key_bf16.ptr == scratch.kv_proj.ptr
    assert scratch.value.ptr == scratch.kv_proj.ptr + 512 * DType.FP16.itemsize
    assert scratch.value.shape == (1, 2, 256)


def test_qwen35_decode_state_projects_full_attention_qkv_fp16_tokens_split_layout(monkeypatch) -> None:
    # Pin the byte/bit-exact weight-amortized W4 variants off so this dispatch
    # test deterministically exercises the baseline projection orchestration.
    # Variants are covered by test_paro_awq_output_tiled_gemv.py (C3.0c
    # output-tiled) and test_qwen35_paro_marlin_k_multi_row.py (M12.6 multi-row).
    monkeypatch.setenv("HIPENGINE_W4_MULTI_ROW_PACK8", "0")
    monkeypatch.setattr(qwen_runtime, "_PACK8_OUTPUT_TILED_ROWS", frozenset())
    # Restore the multi-token prefill changeover so tokens=2 routes to the fused
    # prefill split this test asserts (M7.C small-batch decode otherwise claims
    # tokens<=7 for the decode-style GEMV path).
    monkeypatch.setenv("HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD", "1")
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    scratch = state.reserve_full_attention_scratch(tokens=2, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    calls = []

    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_dual_pack8_transposed_fp16",
        lambda *args, **kwargs: calls.append(("dual", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_pack8_transposed_fp16",
        lambda *args, **kwargs: calls.append(("single_transposed", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "awq_fusedw4_prefill_fp16",
        lambda *args, **kwargs: calls.append(("fusedw4", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "awq_fusedw4_prefill_dual_fp16",
        lambda *args, **kwargs: calls.append(("fusedw4_dual", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "awq_fusedw4_prefill_strided_fp16",
        lambda *args, **kwargs: calls.append(("fusedw4_strided", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_pack8_strided_fp16",
        lambda *args, **kwargs: calls.append(("single_strided", args, kwargs)),
    )

    q_proj, key, value = state.project_full_attention_qkv_fp16(scratch, tokens=2)

    assert q_proj is scratch.q_proj
    assert key is scratch.key_bf16
    assert value is scratch.value
    assert [kind for kind, _args, _kwargs in calls] == ["fusedw4_dual", "fusedw4_strided"]
    assert calls[0][1][8] == scratch.q_proj.ptr
    assert calls[0][1][9] == scratch.key_bf16.ptr
    assert calls[1][1][4] == scratch.value.ptr


def test_qwen35_decode_state_projects_full_attention_qkv_fp16_with_kv_fusion(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED", "1")
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    scratch = state.reserve_full_attention_scratch(tokens=1, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    calls = []

    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_pack8_strided_fp16",
        lambda *args, **kwargs: calls.append(("single_q", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_dual_pack8_transposed_fp16",
        lambda *args, **kwargs: calls.append(("dual_kv", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_dual_pack8_transposed_rotate_staged_fp16",
        lambda *args, **kwargs: calls.append(("unexpected_rotate_fused", args, kwargs)),
    )

    q_proj, key, value = state.project_full_attention_qkv_fp16(scratch, tokens=1)

    assert q_proj is scratch.q_proj
    assert key is scratch.key_bf16
    assert value is scratch.value
    assert [kind for kind, _args, _kwargs in calls] == ["single_q", "dual_kv"]
    assert calls[0][1][:5] == (scratch.q_rot.ptr, 0x8230, 0x8240, 0x8250, scratch.q_proj.ptr)
    assert calls[1][1][:9] == (
        scratch.k_rot.ptr,
        scratch.v_rot.ptr,
        0x8338,
        0x8340,
        0x8350,
        0x8438,
        0x8440,
        0x8450,
        scratch.key_bf16.ptr,
    )
    assert calls[1][1][9:14] == (1, 4096, 64, 64, 128)


def test_qwen35_decode_state_projects_linear_qkv_z_fp16_with_fused_rotation_when_deferred(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_PARO_ROTATE_DUAL_PACK8_FUSED", "1")
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    scratch = state.reserve_linear_attention_scratch(tokens=1, activation_dtype="fp16")
    calls = []

    monkeypatch.setattr(qwen_runtime, "paro_rotate2_fp16", lambda *args, **kwargs: calls.append(("unexpected_rotate2", args, kwargs)))
    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_dual_pack8_transposed_rotate_staged_fp16",
        lambda *args, **kwargs: calls.append(("fused", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_dual_pack8_transposed_fp16",
        lambda *args, **kwargs: calls.append(("unexpected_dual", args, kwargs)),
    )

    state.rotate_linear_attention_inputs_fp16(scratch.attn_input, scratch, tokens=1)
    qkv, z = state.project_linear_attention_qkv_z_fp16(scratch, tokens=1)

    assert qkv is scratch.qkv
    assert z is scratch.z
    assert [kind for kind, _args, _kwargs in calls] == ["fused"]
    args = calls[0][1]
    assert args[0] == scratch.attn_input.ptr
    assert args[1] == scratch.qkv_rot.ptr
    assert args[2] == scratch.z_rot.ptr
    assert args[15] == scratch.qkv_z.ptr
    assert args[16] == scratch.rotate_fuse_barrier.ptr


def test_qwen35_decode_state_projects_linear_qkv_z_fp16_batch_gemv(monkeypatch) -> None:
    # Pin the byte/bit-exact weight-amortized W4 variants off so force_gemv routes
    # to the baseline per-row transposed GEMV (output-tiled handles rows in {2,4,8}
    # by default). Variants are covered by their own kernel tests.
    monkeypatch.setenv("HIPENGINE_W4_MULTI_ROW_PACK8", "0")
    monkeypatch.setattr(qwen_runtime, "_PACK8_OUTPUT_TILED_ROWS", frozenset())
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    scratch = state.reserve_linear_attention_scratch(tokens=2, activation_dtype="fp16")
    calls = []

    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_pack8_transposed_fp16",
        lambda *args, **kwargs: calls.append(("single", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "awq_fusedw4_prefill_dual_fp16",
        lambda *args, **kwargs: calls.append(("unexpected_fused", args, kwargs)),
    )

    qkv, z = state.project_linear_attention_qkv_z_fp16(scratch, tokens=2, force_gemv=True)

    assert qkv is scratch.qkv
    assert z is scratch.z
    assert [kind for kind, _args, _kwargs in calls] == ["single", "single"]
    assert calls[0][1][:5] == (scratch.qkv_rot.ptr, 0x9710, 0x9800, 0x9900, scratch.qkv.ptr)
    assert calls[0][1][5] == 2
    assert calls[0][1][6] == scratch.qkv_rot.shape[-1]
    assert calls[0][1][8] == 128
    assert calls[1][1][:5] == (scratch.z_rot.ptr, 0x9A10, 0x9B00, 0x9C00, scratch.z.ptr)
    assert calls[1][1][5] == 2
    assert calls[1][1][6] == scratch.z_rot.shape[-1]
    assert calls[1][1][8] == 128
    # gemv_awq_pack8_transposed_fp16 defaults to 128 threads; the force_gemv
    # branch (and the small-batch decode singles) use that kernel default.
    assert calls[0][2]["threads"] == 128
    assert calls[1][2]["threads"] == 128


def test_qwen35_decode_state_prepare_full_attention_qkv_fp16_tokens_uses_vector_positions(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    scratch = state.reserve_full_attention_scratch(tokens=2, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    cos_table = _tensor(0xD200, (4, 256), "fp32")
    sin_table = _tensor(0xD300, (4, 256), "fp32")
    positions = _tensor(0xD400, (2,), "int64")
    calls = []

    monkeypatch.setattr(qwen_runtime, "qwen35_split_qgate_fp16", lambda *args, **kwargs: calls.append(("split", args, kwargs)))
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_split_qgate_fp16_key_f32",
        lambda *args, **kwargs: calls.append(("split_key", args, kwargs)),
    )
    monkeypatch.setattr(qwen_runtime, "fp16_to_f32", lambda *args, **kwargs: calls.append(("cast", args, kwargs)))
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_head_rmsnorm_partial_rotary_position_f32_bf16",
        lambda *args, **kwargs: calls.append(("scalar_rotary", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16",
        lambda *args, **kwargs: calls.append(("vector_rotary", args, kwargs)),
    )

    query, key, value, gate = state.prepare_full_attention_qkv_fp16(
        scratch,
        cos_table=cos_table,
        sin_table=sin_table,
        position=positions,
        max_positions=4,
        tokens=2,
    )

    assert query is scratch.query
    assert key is scratch.key
    assert value is scratch.value
    assert gate is scratch.gate
    assert [kind for kind, _args, _kwargs in calls] == ["split", "cast", "vector_rotary"]
    assert calls[1][1][0] == scratch.key_bf16.ptr
    assert calls[1][1][1] == scratch.key_raw.ptr
    assert calls[1][1][2] == 2 * state.config.num_key_value_heads * state.config.head_dim
    assert calls[2][1][6] == positions.ptr
    assert calls[2][1][10] == 2


def test_qwen35_decode_state_prepare_full_attention_qkv_fp16_split_key_fuse_can_opt_in(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    scratch = state.reserve_full_attention_scratch(tokens=2, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    cos_table = _tensor(0xD200, (4, 256), "fp32")
    sin_table = _tensor(0xD300, (4, 256), "fp32")
    positions = _tensor(0xD400, (2,), "int64")
    calls = []

    monkeypatch.setenv("HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED", "1")
    monkeypatch.setattr(qwen_runtime, "qwen35_split_qgate_fp16", lambda *args, **kwargs: calls.append(("split", args, kwargs)))
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_split_qgate_fp16_key_f32",
        lambda *args, **kwargs: calls.append(("split_key", args, kwargs)),
    )
    monkeypatch.setattr(qwen_runtime, "fp16_to_f32", lambda *args, **kwargs: calls.append(("cast", args, kwargs)))
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_head_rmsnorm_partial_rotary_position_f32_bf16",
        lambda *args, **kwargs: calls.append(("scalar_rotary", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16",
        lambda *args, **kwargs: calls.append(("vector_rotary", args, kwargs)),
    )

    state.prepare_full_attention_qkv_fp16(
        scratch,
        cos_table=cos_table,
        sin_table=sin_table,
        position=positions,
        max_positions=4,
        tokens=2,
    )

    assert [kind for kind, _args, _kwargs in calls] == ["split_key", "vector_rotary"]
    assert calls[0][1][0] == scratch.q_proj.ptr
    assert calls[0][1][1] == scratch.key_bf16.ptr
    assert calls[0][1][2] == scratch.query_raw.ptr
    assert calls[0][1][3] == scratch.key_raw.ptr
    assert calls[0][1][4] == scratch.gate.ptr
    assert calls[0][1][5:9] == (
        2,
        state.config.num_attention_heads,
        state.config.num_key_value_heads,
        state.config.head_dim,
    )


def test_qwen35_decode_state_decode_batch_full_attention_can_force_per_row_context(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    hidden = _tensor(0xD000, (2, 4096), "fp16")
    scratch = state.reserve_full_attention_scratch(tokens=2, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    moe_scratch = state.reserve_moe_grouped_prefill_scratch(tokens=2, activation_dtype="fp16")
    batch_key_cache = _tensor(0xE000, (4, 256, 2, 256), "bf16")
    batch_value_cache = _tensor(0xF000, (4, 256, 2, 256), "bf16")
    batch_spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2, 4), "int32"),
        live_counts=_tensor(0x2000, (2,), "int64"),
        max_live_count=8,
        storage_dtype="bf16",
    )
    row_contexts = (
        (
            _tensor(0xE100, (1, 256, 2, 256), "bf16"),
            _tensor(0xF100, (1, 256, 2, 256), "bf16"),
            KVLiveSpans.paged_uniform(
                block_table=_tensor(0x1100, (1, 4), "int32"),
                live_counts=_tensor(0x2100, (1,), "int64"),
                max_live_count=5,
                storage_dtype="bf16",
            ),
        ),
        (
            _tensor(0xE200, (1, 256, 2, 256), "bf16"),
            _tensor(0xF200, (1, 256, 2, 256), "bf16"),
            KVLiveSpans.paged_uniform(
                block_table=_tensor(0x1200, (1, 4), "int32"),
                live_counts=_tensor(0x2200, (1,), "int64"),
                max_live_count=8,
                storage_dtype="bf16",
            ),
        ),
    )
    calls: list[tuple] = []

    monkeypatch.setattr(state, "input_rmsnorm_fp16", lambda *args, **kwargs: calls.append(("input_norm", None, None)) or scratch.attn_input)
    monkeypatch.setattr(
        state,
        "prepare_full_attention_qkv_fp16_decode_rows",
        lambda *args, **kwargs: calls.append(("prepare", None, None)) or (scratch.query, scratch.key, scratch.value, scratch.gate),
    )
    monkeypatch.setattr(state, "append_full_attention_kv_fp16_decode_batch", lambda *args, **kwargs: calls.append(("append", None, None)))
    monkeypatch.setattr(
        state,
        "decode_full_attention_context_gate_fp16_batch",
        lambda *args, **kwargs: pytest.fail("unexpected batch context path"),
    )

    monkeypatch.setattr(
        state,
        "decode_full_attention_context_gate_fp16",
        lambda *args, **kwargs: pytest.fail("unexpected fused per-row context+gate path"),
    )

    def fake_row_context(query_ptr, key_ptr, value_ptr, out_ptr, spans, max_live_count, *args, **kwargs):
        calls.append(("row_context", int(out_ptr), int(spans.max_live_count), int(max_live_count), int(key_ptr), int(value_ptr)))

    def fake_batch_gate(context_ptr, gate_ptr, gated_ptr, elements, **kwargs):
        calls.append(("batch_gate", int(context_ptr), int(gate_ptr), int(gated_ptr), int(elements)))

    monkeypatch.setattr(qwen_runtime, "qwen35_paged_full_attn_decode_context_bf16_spans", fake_row_context)
    monkeypatch.setattr(qwen_runtime, "qwen35_full_attn_gate_mul_fp16", fake_batch_gate)
    monkeypatch.setattr(state, "project_full_attention_o_fp16", lambda *args, **kwargs: calls.append(("o_proj", None, None)) or scratch.o_proj)
    monkeypatch.setattr(
        state,
        "post_attention_add_rmsnorm_fp16",
        lambda *args, **kwargs: calls.append(("post_norm", None, None)) or (moe_scratch.normed, moe_scratch.residual),
    )
    monkeypatch.setattr(state, "run_moe_grouped_compact_fp16", lambda *args, **kwargs: calls.append(("grouped_moe", None, None)) or moe_scratch.moe_out)

    out = state.run_full_attention_moe_decode_batch_layer_fp16(
        hidden,
        key_cache=batch_key_cache,
        value_cache=batch_value_cache,
        append_spans=batch_spans,
        decode_spans=batch_spans,
        cos_table=_tensor(0x4000, (8, 4), "fp32"),
        sin_table=_tensor(0x5000, (8, 4), "fp32"),
        positions=_tensor(0x6000, (2,), "int64"),
        max_positions=8,
        attention_scratch=scratch,
        moe_scratch=moe_scratch,
        tokens=2,
        force_per_row_context=True,
        per_row_contexts=row_contexts,
    )

    assert out is moe_scratch.moe_out
    context_row_nbytes = state.config.num_attention_heads * state.config.head_dim * DType.FP32.itemsize
    assert calls == [
        ("input_norm", None, None),
        ("prepare", None, None),
        ("append", None, None),
        ("row_context", scratch.attn_out.ptr, 5, 5, 0xE100, 0xF100),
        ("row_context", scratch.attn_out.ptr, 8, 8, 0xE200, 0xF200),
        ("batch_gate", scratch.query_raw.ptr, scratch.gate.ptr, scratch.gated_attn.ptr, 2 * state.config.num_attention_heads * state.config.head_dim),
        ("o_proj", None, None),
        ("post_norm", None, None),
        ("grouped_moe", None, None),
    ]
    assert runtime.copies == [
        (scratch.query_raw.ptr, scratch.attn_out.ptr, context_row_nbytes, 0),
        (scratch.query_raw.ptr + context_row_nbytes, scratch.attn_out.ptr, context_row_nbytes, 0),
    ]


def test_qwen35_decode_state_prefill_full_attention_fp16_calls_native_kernel(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    scratch = state.reserve_full_attention_scratch(tokens=2, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    key_cache = _tensor(0xE000, (1, 256, 2, 256), "bf16")
    value_cache = _tensor(0xF000, (1, 256, 2, 256), "bf16")
    spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2,), "int32"),
        live_counts=_tensor(0x2000, (2,), "int64"),
        max_live_count=2,
        storage_dtype="bf16",
        row_positions=_tensor(0x3000, (2,), "int64"),
        span_role="prefill",
    )
    calls = []

    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    out = state.prefill_full_attention_gqa_gate_fp16(
        scratch,
        key_cache=key_cache,
        value_cache=value_cache,
        spans=spans,
        rows=2,
    )

    assert out is scratch.gated_attn
    assert len(calls) == 1
    args, _kwargs = calls[0]
    assert args[0] == scratch.query.ptr
    assert args[3] == scratch.gate.ptr
    assert args[4] == scratch.gated_attn.ptr
    assert args[6] == 2
    assert args[7] == spans.max_live_count
    assert args[12] == scratch.gate.shape[-1]


def test_qwen35_decode_state_prefill_full_attention_int8_calls_resolved_kernel(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    scratch = state.reserve_full_attention_scratch(tokens=2, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    key_cache = _tensor(0xE000, (1, 256, 2, 256), "int8")
    value_cache = _tensor(0xF000, (1, 256, 2, 256), "int8")
    scale_metadata = KVScaleMetadata(
        k_scale=_tensor(0x11000, (1, 256, 2), "fp16"),
        v_scale=_tensor(0x12000, (1, 256, 2), "fp16"),
        scale_dtype=DType.FP16,
    )
    spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2, 1), "int32"),
        live_counts=_tensor(0x2000, (2,), "int64"),
        max_live_count=2,
        storage_dtype="int8_per_token_head",
        row_positions=_tensor(0x3000, (2,), "int64"),
        span_role="prefill",
        scale_metadata=scale_metadata,
    )
    calls = []

    def fake_resolve(**kwargs):
        assert kwargs["backend"] == "hip_gfx1100"
        assert kwargs["spans"] is spans
        assert kwargs["kind"].value == "gqa_gate_fp16"

        def fake_prefill(*args, **inner_kwargs):
            calls.append((args, inner_kwargs))

        return fake_prefill

    monkeypatch.setattr(qwen_runtime, "resolve_paged_attn_prefill", fake_resolve)

    out = state.prefill_full_attention_int8_gqa_gate_fp16(
        scratch,
        key_cache=key_cache,
        value_cache=value_cache,
        spans=spans,
        rows=2,
    )

    assert out is scratch.gated_attn
    assert len(calls) == 1
    args, _kwargs = calls[0]
    assert args[0] == scratch.query.ptr
    assert args[1] == key_cache.ptr
    assert args[2] == value_cache.ptr
    assert args[3] == scale_metadata.k_scale.ptr
    assert args[4] == scale_metadata.v_scale.ptr
    assert args[5] == scratch.gate.ptr
    assert args[6] == scratch.gated_attn.ptr
    assert args[7] is spans
    assert args[8] == 2
    assert args[9] == spans.max_live_count
    assert args[14] == scratch.gate.shape[-1]


def test_qwen35_decode_state_run_full_attention_prefill_int8_uses_direct_path(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    hidden = _tensor(0xD000, (2, 4096), "fp16")
    scratch = state.reserve_full_attention_scratch(tokens=2, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    moe_scratch = state.reserve_moe_grouped_prefill_scratch(tokens=2, activation_dtype="fp16")
    key_cache = _tensor(0xE000, (1, 256, 2, 256), "int8")
    value_cache = _tensor(0xF000, (1, 256, 2, 256), "int8")
    scale_metadata = KVScaleMetadata(
        k_scale=_tensor(0x11000, (1, 256, 2), "fp16"),
        v_scale=_tensor(0x12000, (1, 256, 2), "fp16"),
        scale_dtype=DType.FP16,
    )
    spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2, 1), "int32"),
        live_counts=_tensor(0x2000, (2,), "int64"),
        max_live_count=2,
        storage_dtype="int8_per_token_head",
        row_positions=_tensor(0x3000, (2,), "int64"),
        span_role="prefill",
        scale_metadata=scale_metadata,
    )
    calls: list[str] = []

    monkeypatch.setattr(state, "input_rmsnorm_fp16", lambda *args, **kwargs: calls.append("input_norm") or scratch.attn_input)
    monkeypatch.setattr(state, "rotate_full_attention_inputs_fp16", lambda *args, **kwargs: calls.append("rotate") or (scratch.q_rot, scratch.k_rot, scratch.v_rot))
    monkeypatch.setattr(state, "project_full_attention_qkv_fp16", lambda *args, **kwargs: calls.append("qkv") or (scratch.q_proj, scratch.key_bf16, scratch.value))
    monkeypatch.setattr(state, "prepare_full_attention_qkv_fp16", lambda *args, **kwargs: calls.append("rope") or (scratch.query, scratch.key, scratch.value, scratch.gate))
    monkeypatch.setattr(state, "append_full_attention_kv_fp16_batch", lambda *args, **kwargs: pytest.fail("unexpected BF16 append"))
    monkeypatch.setattr(state, "append_full_attention_kv_int8_per_token_head_fp16_batch", lambda *args, **kwargs: calls.append("append_int8"))
    monkeypatch.setattr(state, "prefill_full_attention_gqa_gate_fp16", lambda *args, **kwargs: pytest.fail("unexpected BF16 prefill attention"))
    monkeypatch.setattr(state, "prefill_full_attention_int8_gqa_gate_fp16", lambda *args, **kwargs: calls.append("prefill_int8") or scratch.gated_attn)
    monkeypatch.setattr(state, "project_full_attention_o_fp16", lambda *args, **kwargs: calls.append("o_proj") or scratch.o_proj)
    monkeypatch.setattr(state, "post_attention_add_rmsnorm_fp16", lambda *args, **kwargs: calls.append("post_norm") or (moe_scratch.normed, moe_scratch.residual))
    monkeypatch.setattr(state, "run_moe_grouped_compact_fp16", lambda *args, **kwargs: calls.append("grouped_moe") or moe_scratch.moe_out)

    out = state.run_full_attention_moe_prefill_layer_fp16(
        hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        append_spans=spans,
        prefill_spans=spans,
        cos_table=_tensor(0x4000, (8, 4), "fp32"),
        sin_table=_tensor(0x5000, (8, 4), "fp32"),
        positions=spans.row_positions,
        max_positions=8,
        attention_scratch=scratch,
        moe_scratch=moe_scratch,
        tokens=2,
    )

    assert out is moe_scratch.moe_out
    assert calls == ["input_norm", "rotate", "qkv", "rope", "append_int8", "prefill_int8", "o_proj", "post_norm", "grouped_moe"]


def test_qwen35_decode_state_run_full_attention_prefill_fp16_uses_grouped_moe(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    hidden = _tensor(0xD000, (2, 4096), "fp16")
    scratch = state.reserve_full_attention_scratch(tokens=2, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    moe_scratch = state.reserve_moe_grouped_prefill_scratch(tokens=2, activation_dtype="fp16")
    key_cache = _tensor(0xE000, (1, 256, 2, 256), "bf16")
    value_cache = _tensor(0xF000, (1, 256, 2, 256), "bf16")
    spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2, 1), "int32"),
        live_counts=_tensor(0x2000, (2,), "int64"),
        max_live_count=2,
        storage_dtype="bf16",
        row_positions=_tensor(0x3000, (2,), "int64"),
        span_role="prefill",
    )
    calls: list[str] = []

    monkeypatch.setattr(state, "input_rmsnorm_fp16", lambda *args, **kwargs: calls.append("input_norm") or scratch.attn_input)
    monkeypatch.setattr(state, "rotate_full_attention_inputs_fp16", lambda *args, **kwargs: calls.append("rotate") or (scratch.q_rot, scratch.k_rot, scratch.v_rot))
    monkeypatch.setattr(state, "project_full_attention_qkv_fp16", lambda *args, **kwargs: calls.append("qkv") or (scratch.q_proj, scratch.key_bf16, scratch.value))
    monkeypatch.setattr(state, "prepare_full_attention_qkv_fp16", lambda *args, **kwargs: calls.append("rope") or (scratch.query, scratch.key, scratch.value, scratch.gate))
    monkeypatch.setattr(state, "append_full_attention_kv_fp16_batch", lambda *args, **kwargs: calls.append("append"))
    monkeypatch.setattr(state, "prefill_full_attention_gqa_gate_fp16", lambda *args, **kwargs: calls.append("prefill_attn") or scratch.gated_attn)
    monkeypatch.setattr(state, "project_full_attention_o_fp16", lambda *args, **kwargs: calls.append("o_proj") or scratch.o_proj)
    monkeypatch.setattr(state, "post_attention_add_rmsnorm_fp16", lambda *args, **kwargs: calls.append("post_norm") or (moe_scratch.normed, moe_scratch.residual))
    monkeypatch.setattr(state, "run_moe_grouped_compact_fp16", lambda *args, **kwargs: calls.append("grouped_moe") or moe_scratch.moe_out)

    out = state.run_full_attention_moe_prefill_layer_fp16(
        hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        append_spans=spans,
        prefill_spans=spans,
        cos_table=_tensor(0x4000, (8, 4), "fp32"),
        sin_table=_tensor(0x5000, (8, 4), "fp32"),
        positions=spans.row_positions,
        max_positions=8,
        attention_scratch=scratch,
        moe_scratch=moe_scratch,
        tokens=2,
    )

    assert out is moe_scratch.moe_out
    assert calls == ["input_norm", "rotate", "qkv", "rope", "append", "prefill_attn", "o_proj", "post_norm", "grouped_moe"]


def test_qwen35_decode_state_aotriton_prefill_reuses_attention_query_buffer(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    hidden = _tensor(0xD000, (2, 4096), "fp16")
    scratch = state.reserve_full_attention_scratch(
        tokens=2,
        num_splits=1,
        activation_dtype="fp16",
        gated_dtype="fp16",
        query_dtype="bf16",
    )
    moe_scratch = state.reserve_moe_grouped_prefill_scratch(tokens=2, activation_dtype="fp16")
    key_cache = _tensor(0xE000, (1, 256, 2, 256), "bf16")
    value_cache = _tensor(0xF000, (1, 256, 2, 256), "bf16")
    spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2, 1), "int32"),
        live_counts=_tensor(0x2000, (2,), "int64"),
        max_live_count=2,
        storage_dtype="bf16",
        row_positions=_tensor(0x3000, (2,), "int64"),
        span_role="prefill",
    )
    cu_q = _tensor(0x6000, (2,), "int32")
    cu_k = _tensor(0x7000, (2,), "int32")
    seen: dict[str, Tensor] = {}
    calls: list[str] = []

    monkeypatch.setattr(state, "input_rmsnorm_fp16", lambda *args, **kwargs: calls.append("input_norm") or scratch.attn_input)
    monkeypatch.setattr(state, "rotate_full_attention_inputs_fp16", lambda *args, **kwargs: calls.append("rotate") or (scratch.q_rot, scratch.k_rot, scratch.v_rot))
    monkeypatch.setattr(state, "project_full_attention_qkv_fp16", lambda *args, **kwargs: calls.append("qkv") or (scratch.q_proj, scratch.key_bf16, scratch.value))

    def fake_prepare(*args, **kwargs):
        calls.append("rope")
        seen["query_bf16"] = kwargs["query_bf16_out"]
        return kwargs["query_bf16_out"], scratch.key, scratch.value, scratch.gate

    monkeypatch.setattr(state, "prepare_full_attention_qkv_fp16", fake_prepare)
    monkeypatch.setattr(state, "append_full_attention_kv_fp16_batch", lambda *args, **kwargs: calls.append("append"))

    def fake_aotriton(*args, **kwargs):
        calls.append("aotriton")
        assert kwargs["query_bf16"].ptr == seen["query_bf16"].ptr
        return kwargs["attn_bf16_out"]

    monkeypatch.setattr(state, "prefill_full_attention_aotriton_varlen_gqa_bf16", fake_aotriton)
    monkeypatch.setattr(state, "project_full_attention_o_bf16_attn_gate_fp16", lambda *args, **kwargs: calls.append("o_proj") or scratch.o_proj)
    monkeypatch.setattr(state, "post_attention_add_rmsnorm_fp16", lambda *args, **kwargs: calls.append("post_norm") or (moe_scratch.normed, moe_scratch.residual))
    monkeypatch.setattr(state, "run_moe_grouped_compact_fp16", lambda *args, **kwargs: calls.append("grouped_moe") or moe_scratch.moe_out)

    out = state.run_full_attention_moe_prefill_layer_fp16(
        hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        append_spans=spans,
        prefill_spans=spans,
        cos_table=_tensor(0x4000, (8, 4), "fp32"),
        sin_table=_tensor(0x5000, (8, 4), "fp32"),
        positions=spans.row_positions,
        max_positions=8,
        attention_scratch=scratch,
        moe_scratch=moe_scratch,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        aotriton_attention=True,
        aotriton_kv_rows=2,
        tokens=2,
    )

    assert out is moe_scratch.moe_out
    assert seen["query_bf16"].ptr == scratch.query.ptr
    assert seen["query_bf16"].dtype is DType.BF16
    assert "attn.aotriton_q_bf16" not in state.workspace.names
    assert calls == ["input_norm", "rotate", "qkv", "rope", "append", "aotriton", "o_proj", "post_norm", "grouped_moe"]


def test_qwen35_decode_state_run_full_attention_prefill_fp16_can_force_c1_moe(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    hidden = _tensor(0xD000, (2, 4096), "fp16")
    scratch = state.reserve_full_attention_scratch(tokens=2, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    moe_scratch = state.reserve_moe_c1_scratch(tokens=2, activation_dtype="fp16")
    key_cache = _tensor(0xE000, (1, 256, 2, 256), "bf16")
    value_cache = _tensor(0xF000, (1, 256, 2, 256), "bf16")
    spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2, 1), "int32"),
        live_counts=_tensor(0x2000, (2,), "int64"),
        max_live_count=2,
        storage_dtype="bf16",
        row_positions=_tensor(0x3000, (2,), "int64"),
        span_role="prefill",
    )
    calls: list[str] = []

    monkeypatch.setenv("HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS", "999999")
    monkeypatch.setattr(state, "reserve_moe_c1_scratch", lambda *args, **kwargs: calls.append("reserve_c1") or moe_scratch)
    monkeypatch.setattr(state, "input_rmsnorm_fp16", lambda *args, **kwargs: calls.append("input_norm") or scratch.attn_input)
    monkeypatch.setattr(state, "rotate_full_attention_inputs_fp16", lambda *args, **kwargs: calls.append("rotate") or (scratch.q_rot, scratch.k_rot, scratch.v_rot))
    monkeypatch.setattr(state, "project_full_attention_qkv_fp16", lambda *args, **kwargs: calls.append("qkv") or (scratch.q_proj, scratch.key_bf16, scratch.value))
    monkeypatch.setattr(state, "prepare_full_attention_qkv_fp16", lambda *args, **kwargs: calls.append("rope") or (scratch.query, scratch.key, scratch.value, scratch.gate))
    monkeypatch.setattr(state, "append_full_attention_kv_fp16_batch", lambda *args, **kwargs: calls.append("append"))
    monkeypatch.setattr(state, "prefill_full_attention_gqa_gate_fp16", lambda *args, **kwargs: calls.append("prefill_attn") or scratch.gated_attn)
    monkeypatch.setattr(state, "project_full_attention_o_fp16", lambda *args, **kwargs: calls.append("o_proj") or scratch.o_proj)
    monkeypatch.setattr(state, "post_attention_add_rmsnorm_fp16", lambda *args, **kwargs: calls.append("post_norm") or (moe_scratch.normed, moe_scratch.residual))
    monkeypatch.setattr(state, "run_moe_c1_fp16", lambda *args, **kwargs: calls.append("c1_moe") or moe_scratch.moe_out)
    monkeypatch.setattr(state, "run_moe_grouped_compact_fp16", lambda *args, **kwargs: pytest.fail("unexpected grouped MoE"))

    out = state.run_full_attention_moe_prefill_layer_fp16(
        hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        append_spans=spans,
        prefill_spans=spans,
        cos_table=_tensor(0x4000, (8, 4), "fp32"),
        sin_table=_tensor(0x5000, (8, 4), "fp32"),
        positions=spans.row_positions,
        max_positions=8,
        attention_scratch=scratch,
        tokens=2,
    )

    assert out is moe_scratch.moe_out
    assert calls == ["reserve_c1", "input_norm", "rotate", "qkv", "rope", "append", "prefill_attn", "o_proj", "post_norm", "c1_moe"]


def test_qwen35_decode_state_append_full_attention_kv_fp16_batch_calls_prompt_writer(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _full_attention_weights())
    scratch = state.reserve_full_attention_scratch(tokens=2, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    key_cache = _tensor(0xE000, (1, 256, 2, 256), "bf16")
    value_cache = _tensor(0xF000, (1, 256, 2, 256), "bf16")
    spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2,), "int32"),
        live_counts=_tensor(0x2000, (2,), "int64"),
        max_live_count=2,
        storage_dtype="bf16",
        row_positions=_tensor(0x3000, (2,), "int64"),
        span_role="prefill",
    )
    calls = []

    monkeypatch.setattr(
        qwen_runtime,
        "resolve_paged_kv_write",
        lambda **_kwargs: lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    state.append_full_attention_kv_fp16_batch(
        scratch,
        key_cache=key_cache,
        value_cache=value_cache,
        spans=spans,
        rows=2,
    )

    assert len(calls) == 1
    args, _kwargs = calls[0]
    assert args[0] == scratch.key.ptr
    assert args[1] == scratch.value.ptr
    assert args[4] is spans
    assert args[5] == 2


def test_qwen35_decode_state_reserves_moe_c1_scratch() -> None:
    runtime = FakeRuntime()
    state = _state(runtime)

    scratch = state.reserve_moe_c1_scratch(tokens=1)

    assert scratch.normed.shape == (1, 4096)
    assert scratch.residual.shape == (1, 4096)
    assert scratch.gate_up_input.shape == (1, 4096)
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


def test_qwen35_decode_state_projects_marlin_k_fp16_decode(monkeypatch) -> None:
    runtime = FakeRuntime()
    prefix = "layers.0.self_attn.o_proj"
    weights = DeviceWeightMap(
        {
            f"{prefix}.qweight_mk": _allocation(f"{prefix}.qweight_mk", 0xB000, (512, 32, 128), "int32"),
            f"{prefix}.qzeros_mk": _allocation(f"{prefix}.qzeros_mk", 0xB100, (512, 32), "int32"),
            f"{prefix}.scales_mk": _allocation(f"{prefix}.scales_mk", 0xB200, (512, 32, 8), "fp16"),
            f"{prefix}.qweight_pack8_decode": _allocation(
                f"{prefix}.qweight_pack8_decode", 0xB000, (512, 4096), "int32"
            ),
            f"{prefix}.qzeros": _allocation(f"{prefix}.qzeros", 0xB300, (32, 512), "int32"),
            f"{prefix}.scales": _allocation(f"{prefix}.scales", 0xB400, (32, 4096), "fp16"),
        }
    )
    state = _state(runtime, weights)
    x = Tensor.from_handle(0xC000, (1, 4096), "fp16", Device("hip", 0))
    out = Tensor.from_handle(0xC100, (1, 4096), "fp16", Device("hip", 0))
    calls = []

    monkeypatch.setattr(qwen_runtime, "gemv_paro_marlin_k_fma_fp16", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_fp16", lambda *a, **k: pytest.fail("unexpected pack8 fallback"))

    result = state.project_pack8_fp16(x, out, weight_prefix=f"model.{prefix}", rows=1, group_size=128)

    assert result is out
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (0xC000, 0xB000, 0xB100, 0xB200, 0xC100, 1, 4096, 512, 128)
    assert kwargs == {"threads": 128, "stream": 0, "library": None, "runtime": runtime}


def test_qwen35_decode_state_preserves_pack8_view_for_marlin_prefill(monkeypatch) -> None:
    # Pin the byte/bit-exact weight-amortized W4 variants off so rows>1 routes to
    # the baseline fused-prefill path (M12.6 multi-row otherwise claims rows 2-8 at
    # safe sites like single_full_o). Variants are covered by their own kernel tests.
    monkeypatch.setenv("HIPENGINE_W4_MULTI_ROW_PACK8", "0")
    monkeypatch.setattr(qwen_runtime, "_PACK8_OUTPUT_TILED_ROWS", frozenset())
    runtime = FakeRuntime()
    prefix = "layers.0.self_attn.o_proj"
    weights = DeviceWeightMap(
        {
            f"{prefix}.qweight_mk": _allocation(f"{prefix}.qweight_mk", 0xB000, (512, 32, 128), "int32"),
            f"{prefix}.qzeros_mk": _allocation(f"{prefix}.qzeros_mk", 0xB100, (512, 32), "int32"),
            f"{prefix}.scales_mk": _allocation(f"{prefix}.scales_mk", 0xB200, (512, 32, 8), "fp16"),
            f"{prefix}.qweight_pack8_decode": _allocation(
                f"{prefix}.qweight_pack8_decode", 0xB000, (512, 4096), "int32"
            ),
            f"{prefix}.qzeros": _allocation(f"{prefix}.qzeros", 0xB300, (32, 512), "int32"),
            f"{prefix}.scales": _allocation(f"{prefix}.scales", 0xB400, (32, 4096), "fp16"),
        }
    )
    state = _state(runtime, weights)
    x = Tensor.from_handle(0xC000, (4, 4096), "fp16", Device("hip", 0))
    out = Tensor.from_handle(0xC100, (4, 4096), "fp16", Device("hip", 0))
    calls = []

    monkeypatch.setattr(qwen_runtime, "awq_fusedw4_prefill_fp16", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(qwen_runtime, "gemv_paro_marlin_k_fma_fp16", lambda *a, **k: pytest.fail("unexpected Marlin rows>1"))

    result = state.project_pack8_fp16(x, out, weight_prefix=prefix, rows=4, group_size=128, stream=0x55)

    assert result is out
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (0xC000, 0xB000, 0xB300, 0xB400, 0xC100, 4, 4096, 512, 128)
    assert kwargs == {"stream": 0x55, "library": None, "runtime": runtime}


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _spans() -> KVLiveSpans:
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0xD000, (2,), "int32"),
        live_counts=_tensor(0xD100, (1,), "int64"),
        max_live_count=1,
        storage_dtype="bf16",
    )


def _int8_spans() -> KVLiveSpans:
    scales = KVScaleMetadata(
        k_scale=_tensor(0xD200, (2, 256, 2), "fp16"),
        v_scale=_tensor(0xD400, (2, 256, 2), "fp16"),
    )
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0xD000, (2,), "int32"),
        live_counts=_tensor(0xD100, (1,), "int64"),
        max_live_count=1,
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        scale_metadata=scales,
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
            f"{prefix}.v_proj.qweight_pack8_decode": _allocation(
                f"{prefix}.v_proj.qweight_pack8_decode", 0x8438, (64, 4096), "int32"
            ),
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
    assert scratch.rotate_fuse_barrier.shape == (2,)
    assert scratch.rotate_fuse_barrier.dtype is DType.INT32
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
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_bf16", record("single_pack8"))
    monkeypatch.setattr(qwen_runtime, "dense_dual_gemv_out_bf16", record("dense_dual"))
    monkeypatch.setattr(qwen_runtime, "dense_gemv_out_bf16", record("dense"))
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
        "single_pack8",
        "single_pack8",
        "dense",
        "dense",
        "cast_qkv",
        "conv_prefill",
        "prepare",
        "gdn_k2",
        "rms_gate",
    ]
    assert calls[1][1][:5] == (scratch.qkv_rot.ptr, 0x9710, 0x9800, 0x9900, scratch.qkv.ptr)
    assert calls[1][1][5:] == (4, 4096, 1024, 128)
    assert calls[2][1][:5] == (scratch.z_rot.ptr, 0x9A10, 0x9B00, 0x9C00, scratch.z.ptr)
    assert calls[2][1][5:] == (4, 4096, 512, 128)
    assert calls[3][1] == (0xC000, 0x9D00, scratch.a.ptr, 4, 4096, 32)
    assert calls[4][1] == (0xC000, 0x9E00, scratch.b.ptr, 4, 4096, 32)
    assert calls[5][1] == (scratch.qkv.ptr, scratch.qkv_f32.ptr, 4 * 8192)
    assert calls[6][1] == (scratch.qkv_f32.ptr, conv_state.ptr, 0x9F00, scratch.conv_out.ptr, 4, 8192, 4)
    assert calls[7][1] == (
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
    assert calls[8][1] == (
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
    assert calls[9][1] == (scratch.recurrent_out.ptr, scratch.z.ptr, 0xA300, scratch.recurrent_bf16.ptr, 1.0e-6, 4, 32, 128)


def test_qwen35_decode_state_uses_rocblas_for_linear_ab_fp16_prefill(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    hidden = _tensor(0xC000, (4, 4096), "fp16")
    scratch = state.reserve_linear_attention_scratch(tokens=4, activation_dtype="fp16")
    calls = []

    def fake_rocblas(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setenv("HIPENGINE_LINEAR_AB_PREFILL_ROCBLAS_MIN_TOKENS", "4")
    monkeypatch.setattr(qwen_runtime, "rocblas_gemm_ex_rowmajor_nt_fp16_compute_f32", fake_rocblas)
    monkeypatch.setattr(qwen_runtime, "dense_gemv_out_fp16", lambda *a, **k: pytest.fail("unexpected GEMV fallback"))

    out = state.project_linear_attention_ab_fp16(hidden, scratch, tokens=4, stream=0x55)

    assert out == (scratch.a, scratch.b)
    assert len(calls) == 2
    assert calls[0][0] == (hidden.ptr, 0x9D00, scratch.a.ptr)
    assert calls[1][0] == (hidden.ptr, 0x9E00, scratch.b.ptr)
    for _, kwargs in calls:
        assert kwargs == {"rows": 4, "in_features": 4096, "out_features": 32, "stream": 0x55}


def test_qwen35_decode_state_uses_separate_dual_for_linear_ab_fp16_prefill(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    hidden = _tensor(0xC000, (4, 4096), "fp16")
    scratch = state.reserve_linear_attention_scratch(tokens=4, activation_dtype="fp16")
    calls = []

    def fake_dual(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setenv("HIPENGINE_LINEAR_AB_DUAL_SEPARATE", "1")
    monkeypatch.setattr(qwen_runtime, "dense_dual_gemv_separate_out_fp16", fake_dual)
    monkeypatch.setattr(qwen_runtime, "dense_gemv_out_fp16", lambda *a, **k: pytest.fail("unexpected single GEMV fallback"))
    monkeypatch.setattr(qwen_runtime, "dense_gemv_out_fp16_wmma", lambda *a, **k: pytest.fail("unexpected WMMA fallback"))

    out = state.project_linear_attention_ab_fp16(hidden, scratch, tokens=4, stream=0x55)

    assert out == (scratch.a, scratch.b)
    assert calls == [
        (
            (
                hidden.ptr,
                0x9D00,
                0x9E00,
                scratch.a.ptr,
                scratch.b.ptr,
                4,
                4096,
                32,
                32,
            ),
            {"threads": 64, "stream": 0x55, "library": None, "runtime": runtime},
        )
    ]


def test_qwen35_decode_state_projects_linear_attention_prefill_out(monkeypatch) -> None:
    # Pin the byte/bit-exact weight-amortized W4 variants off so rows in {2,4,8}
    # route to the baseline strided pack8 GEMV (output-tiled handles them by
    # default). Variants are covered by their own kernel tests.
    monkeypatch.setenv("HIPENGINE_W4_MULTI_ROW_PACK8", "0")
    monkeypatch.setattr(qwen_runtime, "_PACK8_OUTPUT_TILED_ROWS", frozenset())
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
    # Pin the byte/bit-exact weight-amortized W4 variants off so the prefill chain
    # routes to the baseline pack8 GEMVs (output-tiled handles rows in {2,4,8} by
    # default). Variants are covered by their own kernel tests.
    monkeypatch.setenv("HIPENGINE_W4_MULTI_ROW_PACK8", "0")
    monkeypatch.setattr(qwen_runtime, "_PACK8_OUTPUT_TILED_ROWS", frozenset())
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    hidden = _tensor(0xC000, (4, 4096), "bf16")
    conv_state = _tensor(0xC100, (8192, 4), "fp32")
    recurrent_state = _tensor(0xC200, (32, 128, 128), "fp32")
    scratch = state.reserve_linear_attention_scratch(tokens=4)
    order = []

    monkeypatch.setattr(qwen_runtime, "paro_rotate2_bf16", lambda *a, **k: order.append("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", lambda *a, **k: order.append("dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_bf16", lambda *a, **k: order.append("single_pack8"))
    monkeypatch.setattr(qwen_runtime, "dense_dual_gemv_out_bf16", lambda *a, **k: order.append("dense_dual"))
    monkeypatch.setattr(qwen_runtime, "dense_gemv_out_bf16", lambda *a, **k: order.append("dense"))
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
        "single_pack8",
        "single_pack8",
        "dense",
        "dense",
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
    # Pin the byte/bit-exact weight-amortized W4 variants off so the c=1 and c=4
    # chains route to the baseline pack8 GEMVs (output-tiled handles rows in {2,4,8}
    # by default). Variants are covered by their own kernel tests.
    monkeypatch.setenv("HIPENGINE_W4_MULTI_ROW_PACK8", "0")
    monkeypatch.setattr(qwen_runtime, "_PACK8_OUTPUT_TILED_ROWS", frozenset())
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
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_rotate_staged_bf16", record("fused_dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_bf16", record("single_pack8"))
    monkeypatch.setattr(qwen_runtime, "dense_gemv_out_bf16", record("dense"))
    monkeypatch.setattr(qwen_runtime, "dense_dual_gemv_out_bf16", record("dense_dual"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_conv_decode_bf16", record("conv"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16", record("gdn"))
    monkeypatch.setattr(qwen_runtime, "bf16_to_f32", record("cast_qkv"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_conv_prefill_f32", record("conv_prefill"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_prefill_prepare_f32_bf16", record("prepare"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_prefill_recurrent_k2_f32", record("gdn_k2"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_prefill_rmsnorm_gate_bf16", record("rms_gate"))
    monkeypatch.setattr(qwen_runtime, "f32_to_bf16", record("cast"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", record("rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_add_rmsnorm_out_bf16", record("post_norm"))
    monkeypatch.setattr(qwen_runtime, "qwen35_router_topk_shared_out_bf16", record("router"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_bf16", record("gate_up"))
    monkeypatch.setattr(qwen_runtime, "gemm_awq_selected_dual_pack8_wmma_compact_bf16", record("gate_up_wmma"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_bf16", record("silu_rotate"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_bf16", record("down"))
    monkeypatch.setattr(qwen_runtime, "gemm_awq_selected_pack8_wmma_compact_bf16", record("down_wmma"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_out_bf16", record("shared_silu"))
    monkeypatch.setattr(qwen_runtime, "weighted_sum_shared_gate_combine_residual_out_bf16_f32w", record("combine"))
    monkeypatch.setattr(qwen_runtime, "weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w", record("combine_batch"))
    monkeypatch.setattr(qwen_runtime, "qwen35_moe_group_count", record("group_count"))
    monkeypatch.setattr(qwen_runtime, "qwen35_moe_group_prefix", record("group_prefix"))
    monkeypatch.setattr(qwen_runtime, "qwen35_moe_wmma_tile_map", record("tile_map"))
    monkeypatch.setattr(qwen_runtime, "qwen35_moe_group_scatter_gather_lowp", record("scatter_gather"))
    monkeypatch.setattr(qwen_runtime, "weighted_lanes_sum_out_bf16_f32w", record("weighted_lanes"))
    monkeypatch.setattr(qwen_runtime, "shared_gate_combine_residual_batch_out_bf16", record("shared_batch"))

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
        "rotate1",
        "gate_up",
        "silu_rotate",
        "down",
        # Shared expert W4 PARO: fused rotate2 for gate/up, dual GEMV (packed gate||up),
        # fused silu*mul+down-rotate, single GEMV.
        "rotate2",
        "dual_pack8",
        "silu_rotate",
        "single_pack8",
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

    calls.clear()
    batch_hidden = _tensor(0xD000, (4, 4096), "bf16")
    batch_linear = state.reserve_linear_attention_scratch(tokens=4)
    batch_moe = state.reserve_moe_grouped_prefill_scratch(tokens=4)
    batch_out = state.run_linear_attention_moe_c1_layer_bf16(
        batch_hidden,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        linear_scratch=batch_linear,
        moe_scratch=batch_moe,
        tokens=4,
    )
    assert batch_out is batch_moe.moe_out
    assert [name for name, _, _ in calls] == [
        "input_norm",
        "rotate2",
        "single_pack8",
        "single_pack8",
        "dense",
        "dense",
        "cast_qkv",
        "conv_prefill",
        "prepare",
        "gdn_k2",
        "rms_gate",
        "rotate1",
        "pack8",
        "post_norm",
        "router",
        "group_count",
        "group_prefix",
        "tile_map",
        "scatter_gather",
        "rotate1",
        "gate_up",
        "silu_rotate",
        "down",
        "weighted_lanes",
        # Shared expert W4 PARO (BF16 prefill: no fused W4 prefill kernel exists,
        # falls back to the batched dual/single GEMV path, with fused rotate2 and silu+rotate).
        "rotate2",
        "dual_pack8",
        "silu_rotate",
        "single_pack8",
        "shared_batch",
    ]


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
    monkeypatch.setattr(qwen_runtime, "paro_rotate2_bf16", record("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_strided_bf16", record("pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", record("dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_rotate_staged_bf16", record("fused_dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_bf16", record("single_pack8"))
    monkeypatch.setattr(qwen_runtime, "qwen35_split_qgate_bf16", record("split_qgate"))
    monkeypatch.setattr(qwen_runtime, "bf16_to_f32", record("bf16_to_f32"))
    monkeypatch.setattr(qwen_runtime, "qwen35_head_rmsnorm_partial_rotary_position_f32_bf16", record("head_rotary"))
    monkeypatch.setattr(qwen_runtime, "resolve_paged_kv_write", lambda **_kwargs: record("kv"))
    monkeypatch.setattr(qwen_runtime, "qwen35_full_attn_decode_context_bf16", record("dense_attention_context"))
    monkeypatch.setattr(qwen_runtime, "qwen35_paged_full_attn_decode_context_bf16_spans", record("attention_context"))
    monkeypatch.setattr(qwen_runtime, "qwen35_full_attn_gate_mul_bf16", record("attention_gate"))
    monkeypatch.setattr(qwen_runtime, "qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans", record("attention"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", record("rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_add_rmsnorm_out_bf16", record("post_norm"))
    monkeypatch.setattr(qwen_runtime, "qwen35_router_topk_shared_out_bf16", record("router"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_bf16", record("gate_up"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_bf16", record("silu_rotate"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_bf16", record("down"))
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
        "split_qgate",
        "bf16_to_f32",
        "head_rotary",
        "kv",
        "dense_attention_context",
        "attention_gate",
        "rotate1",
        "pack8",
        "post_norm",
        "router",
        "rotate1",
        "gate_up",
        "silu_rotate",
        "down",
        # Shared expert W4 PARO.
        "rotate2",
        "dual_pack8",
        "silu_rotate",
        "single_pack8",
        "combine",
    ]
    assert calls[1][1][:4] == (attn.attn_input.ptr, attn.q_rot.ptr, attn.k_rot.ptr, attn.v_rot.ptr)
    assert calls[2][1][:9] == (attn.q_rot.ptr, attn.k_rot.ptr, 0x8238, 0x8240, 0x8250, 0x8338, 0x8340, 0x8350, attn.q_proj_key.ptr)
    assert calls[4][1] == (attn.q_proj.ptr, attn.query_raw.ptr, attn.gate.ptr, 1, 16, 256)
    assert calls[5][1] == (attn.key_bf16.ptr, attn.key_raw.ptr, 512)
    assert calls[6][1][:9] == (attn.query_raw.ptr, attn.key_raw.ptr, 0x8120, 0x8130, 0xD200, 0xD300, 0xD400, attn.query.ptr, attn.key.ptr)
    assert calls[8][1][:5] == (attn.query.ptr, key_cache.ptr, value_cache.ptr, attn.attn_out.ptr, 0xD100)
    assert calls[8][1][5:10] == (1, 16, 2, 256, 256 ** -0.5)
    assert calls[9][1][:4] == (attn.attn_out.ptr, attn.gate.ptr, attn.gated_attn.ptr, 4096)
    assert calls[10][1][:5] == (attn.gated_attn.ptr, attn.o_rot.ptr, 0x8500, 0x8510, 0x8520)
    assert calls[12][1][:5] == (hidden.ptr, attn.o_proj.ptr, 0x8110, moe.normed.ptr, moe.residual.ptr)
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

    monkeypatch.setattr(qwen_runtime, "resolve_paged_kv_write", lambda **_kwargs: fake_append)

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


def test_qwen35_decode_state_adaptive_split_gate_uses_warp_then_gqa(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime)
    scratch = state.reserve_full_attention_scratch(tokens=1, num_splits=64, activation_dtype="fp16", gated_dtype="fp16")
    key_cache = _tensor(0xE000, (256, 256, 2, 256), "bf16")
    value_cache = _tensor(0xF000, (256, 256, 2, 256), "bf16")
    calls = []

    monkeypatch.delenv("HIPENGINE_PAGED_ATTN_GQA_GROUPED_CTX", raising=False)
    monkeypatch.delenv("NANOVLLM_AMD_PAGED_ATTN_GQA_GROUPED_CTX", raising=False)
    monkeypatch.setattr(qwen_runtime, "qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans", lambda *a, **k: calls.append(("warp", a, k)))
    monkeypatch.setattr(qwen_runtime, "qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans", lambda *a, **k: calls.append(("gqa", a, k)))

    state.decode_full_attention_split_gate_fp16(
        scratch,
        key_cache=key_cache,
        value_cache=value_cache,
        spans=_spans(),
        chunk_size=256,
        num_splits=16,
    )
    state.decode_full_attention_split_gate_fp16(
        scratch,
        key_cache=key_cache,
        value_cache=value_cache,
        spans=_spans(),
        chunk_size=256,
        num_splits=64,
    )

    assert [item[0] for item in calls] == ["warp", "gqa"]
    assert calls[0][1][9:18] == (256, 16, 256, 16, 2, 256, 256, 1, 256 ** -0.5)
    assert calls[1][1][9:18] == (256, 64, 256, 16, 2, 256, 256, 1, 256 ** -0.5)


def test_qwen35_decode_state_int8_split_decode_uses_registry_metadata(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime)
    scratch = state.reserve_full_attention_scratch(tokens=1, num_splits=2, activation_dtype="fp16", gated_dtype="fp16")
    key_cache = _tensor(0xE000, (2, 256, 2, 256), "int8")
    value_cache = _tensor(0xF000, (2, 256, 2, 256), "int8")
    spans = _int8_spans()
    calls = []

    def fake_decode(*args, **kwargs):
        calls.append(("decode", args, kwargs))

    def fake_resolve(**kwargs):
        calls.append(("resolve", kwargs))
        return fake_decode

    monkeypatch.setattr(qwen_runtime, "resolve_paged_attn_decode", fake_resolve)

    out = state.decode_full_attention_split_gate_fp16(
        scratch,
        key_cache=key_cache,
        value_cache=value_cache,
        spans=spans,
        chunk_size=256,
        num_splits=2,
    )

    assert out is scratch.gated_attn
    assert calls[0] == (
        "resolve",
        {
            "backend": "hip_gfx1100",
            "spans": spans,
            "kind": qwen_runtime.PagedAttnDecodeKind.GQA_SPLITK_GATE_FP16,
        },
    )
    args, kwargs = calls[1][1], calls[1][2]
    assert args[:10] == (
        scratch.query.ptr,
        key_cache.ptr,
        value_cache.ptr,
        spans.scale_metadata.k_scale.ptr,
        spans.scale_metadata.v_scale.ptr,
        scratch.gate.ptr,
        scratch.gated_attn.ptr,
        scratch.partial_out.ptr,
        scratch.partial_m.ptr,
        scratch.partial_l.ptr,
    )
    assert args[11:20] == (256, 2, 256, 16, 2, 256, 256, 1, 256 ** -0.5)
    assert kwargs == {"stream": 0, "library": None, "runtime": runtime}


def test_qwen35_decode_state_split_decode_threshold_defaults_to_1024(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", raising=False)
    monkeypatch.delenv("NANOVLLM_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", raising=False)

    assert not qwen_runtime._use_full_attention_split_decode(512)
    assert qwen_runtime._use_full_attention_split_decode(1024)

    monkeypatch.setenv("HIPENGINE_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "4096")
    assert not qwen_runtime._use_full_attention_split_decode(1024)
    assert qwen_runtime._use_full_attention_split_decode(4096)


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


def test_qwen35_decode_state_routes_moe_topk_shared_coop_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_PARO_ROUTER_TOPK_COOP", "1")
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    hidden = _tensor(0xCA00, (1, 4096), "bf16")
    calls = []

    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_router_topk_shared_out_bf16",
        lambda *args, **kwargs: calls.append(("unexpected", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_router_topk_shared_coop_out_bf16",
        lambda *args, **kwargs: calls.append(("coop", args, kwargs)),
    )

    selected, weights = state.route_moe_topk_shared_bf16(hidden, scratch)

    assert selected is scratch.selected_experts
    assert weights is scratch.routing_weights
    assert [kind for kind, _args, _kwargs in calls] == ["coop"]
    args, kwargs = calls[0][1], calls[0][2]
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



def test_qwen35_decode_state_routes_prefill_sigmoid_only_for_legacy_fp16(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED", "1")
    runtime = FakeRuntime()
    hidden = _tensor(0xCA00, (2, 4096), "fp16")
    calls = []

    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_router_topk_shared_out_fp16",
        lambda *args, **kwargs: calls.append(("raw", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_router_topk_shared_sigmoid_out_fp16",
        lambda *args, **kwargs: calls.append(("sigmoid", args, kwargs)),
    )

    legacy = _state(runtime, _legacy_prepared_moe_weights())
    legacy_scratch = legacy.reserve_moe_grouped_prefill_scratch(tokens=2, activation_dtype="fp16")
    selected, weights = legacy.route_moe_topk_shared_fp16(hidden, legacy_scratch, tokens=2, stream=0x77)
    assert selected is legacy_scratch.selected_experts
    assert weights is legacy_scratch.routing_weights

    packed = _state(runtime, _prepared_moe_weights())
    packed_scratch = packed.reserve_moe_grouped_prefill_scratch(tokens=2, activation_dtype="fp16")
    packed.route_moe_topk_shared_fp16(hidden, packed_scratch, tokens=2, stream=0x78)

    assert [kind for kind, _args, _kwargs in calls] == ["sigmoid", "raw"]
    legacy_args, legacy_kwargs = calls[0][1], calls[0][2]
    assert legacy_args == (
        hidden.ptr,
        0xB000,
        legacy_scratch.router_logits.ptr,
        legacy_scratch.selected_experts.ptr,
        legacy_scratch.routing_weights.ptr,
        2,
        4096,
        129,
        128,
        8,
    )
    assert legacy_kwargs == {"threads": 256, "stream": 0x77, "library": None, "runtime": runtime}



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
    rotate_calls = []
    gate_calls = []
    down_calls = []

    def fake_rotate(*args, **kwargs):
        rotate_calls.append((args, kwargs))

    def fake_gate(*args, **kwargs):
        gate_calls.append((args, kwargs))

    def fake_down(*args, **kwargs):
        down_calls.append((args, kwargs))

    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", fake_rotate)
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_bf16", fake_gate)
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_bf16", fake_down)

    gate_up = state.selected_moe_gate_up_pack8_bf16(hidden, scratch)
    down = state.selected_moe_down_pack8_bf16(scratch.down_input, scratch)

    assert gate_up is scratch.gate_up
    assert down is scratch.down_out
    rotate_args, rotate_kwargs = rotate_calls[0]
    assert rotate_args == (hidden.ptr, scratch.gate_up_input.ptr, 0xB9A0, 0xB9B0, 0xB9C0, 1, 4096, 128, 32)
    assert rotate_kwargs == {"stream": 0, "library": None, "runtime": runtime}
    gate_args, gate_kwargs = gate_calls[0]
    assert gate_args == (
        scratch.gate_up_input.ptr,
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


def test_qwen35_decode_state_selected_moe_fp16_uses_staged_keyed_rotate(monkeypatch) -> None:
    qwen_runtime._reset_shared_rotate_fuse_barrier_state()
    monkeypatch.setenv("HIPENGINE_SELECTED_MOE_STAGED_ROTATE", "1")
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=2, activation_dtype="fp16")
    hidden = _tensor(0xCA00, (2, 4096), "fp16")
    calls = []

    def record(label):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "paro_rotate1_fp16", record("unexpected_rotate1"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_fp16", record("unexpected_selected_dual"))
    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16",
        record("staged_keyed"),
    )

    out1 = state.selected_moe_gate_up_pack8_fp16(hidden, scratch, tokens=2)
    out2 = state.selected_moe_gate_up_pack8_fp16(hidden, scratch, tokens=2)

    assert out1 is scratch.gate_up
    assert out2 is scratch.gate_up
    labels = [label for label, _args, _kwargs in calls]
    assert labels == ["staged_keyed", "staged_keyed"]
    first_args = calls[0][1]
    second_args = calls[1][1]
    assert first_args[0:3] == (hidden.ptr, scratch.gate_up_input.ptr, scratch.selected_experts.ptr)
    assert first_args[12:16] == (scratch.gate_up.ptr, scratch.shared_rotate_fuse_barrier.ptr, 2, 16)
    # 4096 hidden / group 128 -> 32 groups; one staged rotation and two rows
    # => 64 rotate blocks per selected-gate launch.
    assert first_args[22:24] == (64, 1)
    assert second_args[22:24] == (128, 2)
    barrier_memsets = [m for m in runtime.memsets if m[0] == scratch.shared_rotate_fuse_barrier.ptr]
    assert barrier_memsets == [(scratch.shared_rotate_fuse_barrier.ptr, 0, 8)]


def test_qwen35_decode_state_selected_moe_down_fp16_uses_staged_keyed_kernel(monkeypatch) -> None:
    qwen_runtime._reset_shared_rotate_fuse_barrier_state()
    monkeypatch.delenv("HIPENGINE_SELECTED_MOE_DOWN_STAGED", raising=False)
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=2, activation_dtype="fp16")
    calls = []

    def record(label):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_fp16", record("unexpected_silu_rotate"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_fp16", record("unexpected_down"))
    monkeypatch.setattr(
        qwen_runtime,
        "gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16",
        record("staged_down"),
    )

    out1 = state.selected_moe_activate_down_pack8_fp16(scratch, tokens=2)
    out2 = state.selected_moe_activate_down_pack8_fp16(scratch, tokens=2)

    assert out1 is scratch.down_out
    assert out2 is scratch.down_out
    labels = [label for label, _args, _kwargs in calls]
    assert labels == ["staged_down", "staged_down"]
    first_args = calls[0][1]
    second_args = calls[1][1]
    assert first_args[0:3] == (scratch.gate_up.ptr, scratch.down_input.ptr, scratch.selected_experts.ptr)
    assert first_args[9:12] == (scratch.down_out.ptr, scratch.shared_rotate_fuse_barrier.ptr, 16)
    # 2 tokens * top_k 8 = 16 selected rows; 768 intermediate / group 128
    # -> 6 groups, so the staged activation/rotation phase has 96 blocks.
    assert first_args[17:19] == (96, 1)
    assert second_args[17:19] == (192, 2)
    barrier_memsets = [m for m in runtime.memsets if m[0] == scratch.shared_rotate_fuse_barrier.ptr]
    assert barrier_memsets == [(scratch.shared_rotate_fuse_barrier.ptr, 0, 8)]


def test_qwen35_decode_state_runs_shared_expert_paro_w4_bf16(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    hidden = _tensor(0xCA00, (1, 4096), "bf16")
    calls = []

    def record(label):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", record("unexpected_rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate2_bf16", record("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", record("dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_bf16", record("single_pack8"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_bf16", record("silu_rotate"))

    out = state.shared_expert_paro_w4_bf16(hidden, scratch)

    assert out is scratch.shared_out
    assert [label for label, _a, _k in calls] == [
        "rotate2",  # gate_proj + up_proj input rotations in one launch
        "dual_pack8",  # gate_proj + up_proj GEMV with separate inputs, packed gate||up output
        "silu_rotate",  # fused silu(gate) * up + down_proj input rotation
        "single_pack8",  # down_proj GEMV -> shared_out
    ]
    # rotate2 reads hidden, writes shared gate/up inputs, and uses both projection parameter sets.
    rotate_args = calls[0][1]
    assert rotate_args[:3] == (hidden.ptr, scratch.shared_gate_input.ptr, scratch.shared_up_input.ptr)
    # dual GEMV consumes both rotated inputs, writes the packed shared_up buffer.
    dual_args = calls[1][1]
    assert dual_args[0] == scratch.shared_gate_input.ptr
    assert dual_args[1] == scratch.shared_up_input.ptr
    assert dual_args[8] == scratch.shared_up.ptr
    # fused silu+rotate reads packed shared_up and writes shared_down_input.
    silu_args = calls[2][1]
    assert silu_args[:5] == (
        scratch.shared_up.ptr,
        state.tensor("layers.0.mlp.shared_expert.down_proj.pairs").ptr,
        state.tensor("layers.0.mlp.shared_expert.down_proj.theta").ptr,
        state.tensor("layers.0.mlp.shared_expert.down_proj.channel_scales").ptr,
        scratch.shared_down_input.ptr,
    )
    # down GEMV reads shared_down_input, writes shared_out.
    down_args = calls[3][1]
    assert down_args[0] == scratch.shared_down_input.ptr
    assert down_args[4] == scratch.shared_out.ptr


def test_qwen35_decode_state_runs_shared_expert_paro_w4_fp16_large_prefill_uses_fused_w4(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=16, activation_dtype="fp16")
    hidden = _tensor(0xCA00, (16, 4096), "fp16")
    calls = []

    def record(label):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "paro_rotate1_fp16", record("rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate2_fp16", record("rotate2"))
    monkeypatch.setattr(qwen_runtime, "awq_fusedw4_prefill_dual_fp16", record("fusedw4_dual"))
    monkeypatch.setattr(qwen_runtime, "awq_fusedw4_prefill_fp16", record("fusedw4_single"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_separate_out_fp16", record("silu_separate"))
    # Small decode batches use the GEMV path; larger prefill still uses fused W4.
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_fp16", record("unexpected_dual_gemv"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_fp16", record("unexpected_single_gemv"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_out_fp16", record("unexpected_silu_dual"))

    out = state.shared_expert_paro_w4_fp16(hidden, scratch, tokens=16)

    assert out is scratch.shared_out
    labels = [label for label, _a, _k in calls]
    assert labels == [
        "rotate2",  # gate/up input rotations
        "fusedw4_dual",  # fused W4 prefill dual (separate inputs, separate outputs)
        "silu_separate",  # silu(gate) * up from two separate buffers
        "rotate1",  # down input rotation still uses rotate1 for FP16 prefill
        "fusedw4_single",  # fused W4 prefill single down
    ]
    fused_dual = calls[1][1]
    assert fused_dual[0] == scratch.shared_gate_input.ptr
    assert fused_dual[1] == scratch.shared_up_input.ptr
    assert fused_dual[8] == scratch.shared_gate_out.ptr
    assert fused_dual[9] == scratch.shared_up_out.ptr
    silu_args = calls[2][1]
    assert silu_args == (
        scratch.shared_gate_out.ptr,
        scratch.shared_up_out.ptr,
        scratch.shared_intermediate.ptr,
        16,
        768,
    )


def test_qwen35_decode_state_shared_expert_fp16_fused_rotate_uses_keyed_barrier(monkeypatch) -> None:
    qwen_runtime._reset_shared_rotate_fuse_barrier_state()
    monkeypatch.setenv("HIPENGINE_SHARED_EXPERT_FUSED_ROTATE", "1")
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=2, activation_dtype="fp16")
    hidden = _tensor(0xCA00, (2, 4096), "fp16")
    calls = []

    def record(label):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))
        return fake

    monkeypatch.setattr(qwen_runtime, "paro_rotate2_fp16", record("unexpected_rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_fp16", record("unexpected_dual_gemv"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_rotate_staged_fp16", record("unexpected_non_keyed"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_rotate_staged_keyed_fp16", record("keyed"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_fp16", record("silu_rotate"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_fp16", record("single_pack8"))

    out1 = state.shared_expert_paro_w4_fp16(hidden, scratch, tokens=2)
    out2 = state.shared_expert_paro_w4_fp16(hidden, scratch, tokens=2)
    sibling_layer = Qwen35ParoLayerDeviceWeights(
        config=_config(), layer_id=0, weights=_prepared_moe_weights()
    )
    sibling = Qwen35ParoDecodeState(layer_weights=sibling_layer, workspace=state.workspace, runtime=runtime)
    sibling_scratch = sibling.reserve_moe_c1_scratch(tokens=2, activation_dtype="fp16")
    out3 = sibling.shared_expert_paro_w4_fp16(hidden, sibling_scratch, tokens=2)

    assert out1 is scratch.shared_out
    assert out2 is scratch.shared_out
    assert out3 is sibling_scratch.shared_out
    labels = [label for label, _args, _kwargs in calls]
    assert labels == [
        "keyed", "silu_rotate", "single_pack8",
        "keyed", "silu_rotate", "single_pack8",
        "keyed", "silu_rotate", "single_pack8",
    ]
    keyed_first = calls[0][1]
    keyed_second = calls[3][1]
    keyed_third = calls[6][1]
    # 4096 hidden / group 128 -> 32 groups; two rotations and two rows => 128
    # staged rotate blocks per launch.  The keyed API accumulates counts and
    # epochs instead of resetting the barrier each launch.
    assert keyed_first[23:25] == (128, 1)
    assert keyed_second[23:25] == (256, 2)
    assert keyed_third[23:25] == (384, 3)
    assert sibling_scratch.shared_rotate_fuse_barrier.ptr == scratch.shared_rotate_fuse_barrier.ptr
    barrier_memsets = [m for m in runtime.memsets if m[0] == scratch.shared_rotate_fuse_barrier.ptr]
    assert barrier_memsets == [(scratch.shared_rotate_fuse_barrier.ptr, 0, 8)]


def test_qwen35_decode_state_dispatches_legacy_shared_expert_w8a16_bf16(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _legacy_prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    hidden = _tensor(0xCA00, (1, 4096), "bf16")
    calls = []

    def record(label):
        def inner(*args, **kwargs):
            calls.append((label, args, kwargs))
        return inner

    monkeypatch.setattr(qwen_runtime, "w8a16_linear_bf16_lowp_out", record("w8a16_linear"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_out_bf16", record("silu"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", record("unexpected_w4"))

    out = state.shared_expert_bf16(hidden, scratch)

    assert out is scratch.shared_out
    assert [kind for kind, _args, _kwargs in calls] == ["w8a16_linear", "silu", "w8a16_linear"]
    assert calls[0][1][0] == hidden.ptr
    assert calls[0][1][1] == state.tensor("layers.0.mlp.shared_expert.gate_up_weight_w8a16").ptr
    assert calls[2][1][1] == state.tensor("layers.0.mlp.shared_expert.down_weight_w8a16").ptr


def test_qwen35_decode_state_runs_grouped_moe_fp16_paro_w4_shared_then_combine(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_grouped_prefill_scratch(tokens=2, activation_dtype="fp16")
    hidden = _tensor(0xD000, (2, 4096), "fp16")
    residual = _tensor(0xD100, (2, 4096), "fp16")
    calls = []

    monkeypatch.setattr(state, "_prepare_grouped_moe_prefill_metadata", lambda *args, **kwargs: 16)
    for name, label in [
        ("qwen35_router_topk_shared_out_fp16", "router"),
        ("paro_rotate1_fp16", "rotate1"),
        ("paro_rotate2_fp16", "rotate2"),
        ("gemm_awq_selected_dual_pack8_wmma_compact_fp16", "unexpected_gate_up_wmma"),
        ("gemv_awq_selected_dual_pack8_transposed_fp16", "gate_up_gemv"),
        ("silu_mul_dual_rotate_out_fp16", "silu_rotate"),
        ("gemm_awq_selected_pack8_wmma_compact_fp16", "unexpected_down_wmma"),
        ("gemv_awq_selected_pack8_transposed_fp16", "down_gemv"),
        ("weighted_lanes_sum_out_fp16_f32w", "weighted_lanes"),
        # Shared expert W4 PARO small-batch decode: GEMV dual + fused silu/rotate + GEMV single.
        ("gemv_awq_dual_pack8_transposed_fp16", "shared_dual_gemv"),
        ("awq_fusedw4_prefill_dual_fp16", "unexpected_shared_fusedw4_dual"),
        ("silu_mul_separate_out_fp16", "unexpected_shared_silu_separate"),
        ("awq_fusedw4_prefill_fp16", "unexpected_shared_fusedw4_single"),
        ("gemv_awq_pack8_transposed_fp16", "shared_single_gemv"),
        # Split combine (no fused W8A16+combine).
        ("shared_gate_combine_residual_batch_out_fp16", "shared_combine_batch"),
    ]:
        monkeypatch.setattr(qwen_runtime, name, lambda *args, label=label, **kwargs: calls.append((label, args, kwargs)))

    out = state.run_moe_grouped_compact_fp16(hidden, residual, scratch=scratch, tokens=2)

    assert out is scratch.moe_out
    assert [kind for kind, _args, _kwargs in calls] == [
        "router",
        "rotate1",  # routed gate_up rotate
        "gate_up_gemv",
        "silu_rotate",
        "down_gemv",
        "weighted_lanes",
        # Shared expert W4 PARO chain (small decode batches use GEMV).
        "rotate2",  # shared_expert gate/up input rotations
        "shared_dual_gemv",
        "silu_rotate",  # shared_expert fused silu + down input rotation
        "shared_single_gemv",
        "shared_combine_batch",
    ]
    shared_gate_logits_ptr = scratch.router_logits.ptr + 128 * 4
    final_combine = calls[-1][1]
    assert final_combine == (
        scratch.selected_out.ptr,
        scratch.shared_out.ptr,
        shared_gate_logits_ptr,
        residual.ptr,
        scratch.moe_out.ptr,
        2,
        4096,
        129,
    )


def test_qwen35_decode_state_uses_token_tiled_legacy_shared_gate_up_prefill(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _legacy_prepared_moe_weights())
    scratch = state.reserve_moe_grouped_prefill_scratch(tokens=4, activation_dtype="fp16")
    hidden = _tensor(0xD000, (4, 4096), "fp16")
    calls = []

    def fake_tiled(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setenv("HIPENGINE_SHARED_GATE_UP_PREFILL_TOKEN_TILE", "4")
    monkeypatch.setenv("HIPENGINE_SHARED_GATE_UP_PREFILL_MIN_TOKENS", "2")
    monkeypatch.setattr(qwen_runtime, "w8a16_shared_gate_up_silu_fp16_token_tiled", fake_tiled)
    monkeypatch.setattr(qwen_runtime, "w8a16_shared_gate_up_silu_fp16", lambda *a, **k: pytest.fail("unexpected fallback"))

    out = state.shared_expert_gate_up_silu_fp16(hidden, scratch, tokens=4, stream=0x55)

    assert out is scratch.shared_intermediate
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (
        hidden.ptr,
        state.tensor("layers.0.mlp.shared_expert.gate_up_weight_w8a16").ptr,
        state.tensor("layers.0.mlp.shared_expert.gate_up_weight_w8a16_scale").ptr,
        scratch.shared_intermediate.ptr,
        4,
        4096,
        768,
    )
    assert kwargs == {"token_tile": 4, "threads": 64, "stream": 0x55, "library": None, "runtime": runtime}


def test_qwen35_decode_state_uses_token_tiled_legacy_shared_down_prefill(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _legacy_prepared_moe_weights())
    scratch = state.reserve_moe_grouped_prefill_scratch(tokens=4, activation_dtype="fp16")
    residual = _tensor(0xD100, (4, 4096), "fp16")
    calls = []

    def fake_sigmoid(*args, **kwargs):
        calls.append(("sigmoid", args, kwargs))

    def fake_tiled(*args, **kwargs):
        calls.append(("tiled", args, kwargs))

    monkeypatch.setenv("HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_TOKEN_TILE", "4")
    monkeypatch.setattr(qwen_runtime, "w8a16_shared_gate_sigmoid_fp32", fake_sigmoid)
    monkeypatch.setattr(qwen_runtime, "w8a16_shared_down_combine_residual_fp16_token_tiled", fake_tiled)
    monkeypatch.setattr(qwen_runtime, "w8a16_shared_down_combine_residual_fp16", lambda *a, **k: pytest.fail("unexpected fallback"))

    out = state.shared_expert_down_combine_residual_fp16(scratch, residual, tokens=4, stream=0x55)

    assert out is scratch.moe_out
    assert [kind for kind, _args, _kwargs in calls] == ["sigmoid", "tiled"]
    shared_gate_logits_ptr = scratch.router_logits.ptr + 128 * 4
    sigmoid_args, sigmoid_kwargs = calls[0][1], calls[0][2]
    assert sigmoid_args == (shared_gate_logits_ptr, shared_gate_logits_ptr, 4, 129)
    assert sigmoid_kwargs == {"threads": 128, "stream": 0x55, "library": None, "runtime": runtime}
    tiled_args, tiled_kwargs = calls[1][1], calls[1][2]
    assert tiled_args == (
        scratch.shared_intermediate.ptr,
        state.tensor("layers.0.mlp.shared_expert.down_weight_w8a16").ptr,
        state.tensor("layers.0.mlp.shared_expert.down_weight_w8a16_scale").ptr,
        scratch.selected_out.ptr,
        shared_gate_logits_ptr,
        residual.ptr,
        scratch.moe_out.ptr,
        4,
        4096,
        768,
        129,
    )
    assert tiled_kwargs == {"token_tile": 4, "threads": 64, "stream": 0x55, "library": None, "runtime": runtime}


def test_qwen35_decode_state_skips_legacy_shared_gate_sigmoid_when_router_fused(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _legacy_prepared_moe_weights())
    scratch = state.reserve_moe_grouped_prefill_scratch(tokens=4, activation_dtype="fp16")
    residual = _tensor(0xD100, (4, 4096), "fp16")
    calls = []

    monkeypatch.setenv("HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_TOKEN_TILE", "4")
    monkeypatch.setattr(qwen_runtime, "w8a16_shared_gate_sigmoid_fp32", lambda *a, **k: pytest.fail("unexpected sigmoid"))
    monkeypatch.setattr(
        qwen_runtime,
        "w8a16_shared_down_combine_residual_fp16_token_tiled",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(qwen_runtime, "w8a16_shared_down_combine_residual_fp16", lambda *a, **k: pytest.fail("unexpected fallback"))

    out = state.shared_expert_down_combine_residual_fp16(
        scratch,
        residual,
        tokens=4,
        shared_gate_already_sigmoid=True,
        stream=0x56,
    )

    assert out is scratch.moe_out
    assert len(calls) == 1
    args, kwargs = calls[0]
    shared_gate_logits_ptr = scratch.router_logits.ptr + 128 * 4
    assert args[:11] == (
        scratch.shared_intermediate.ptr,
        state.tensor("layers.0.mlp.shared_expert.down_weight_w8a16").ptr,
        state.tensor("layers.0.mlp.shared_expert.down_weight_w8a16_scale").ptr,
        scratch.selected_out.ptr,
        shared_gate_logits_ptr,
        residual.ptr,
        scratch.moe_out.ptr,
        4,
        4096,
        768,
        129,
    )
    assert kwargs == {"token_tile": 4, "threads": 64, "stream": 0x56, "library": None, "runtime": runtime}


def test_qwen35_decode_state_runs_grouped_moe_fp16_legacy_w8a16_shared_fused_combine(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _legacy_prepared_moe_weights())
    scratch = state.reserve_moe_grouped_prefill_scratch(tokens=2, activation_dtype="fp16")
    hidden = _tensor(0xD000, (2, 4096), "fp16")
    residual = _tensor(0xD100, (2, 4096), "fp16")
    calls = []

    monkeypatch.setattr(state, "_prepare_grouped_moe_prefill_metadata", lambda *args, **kwargs: 16)
    for name, label in [
        ("qwen35_router_topk_shared_out_fp16", "router"),
        ("paro_rotate1_fp16", "rotate1"),
        ("gemm_awq_selected_dual_pack8_wmma_compact_fp16", "unexpected_gate_up_wmma"),
        ("gemv_awq_selected_dual_pack8_transposed_fp16", "gate_up_gemv"),
        ("silu_mul_dual_rotate_out_fp16", "silu_rotate"),
        ("gemm_awq_selected_pack8_wmma_compact_fp16", "unexpected_down_wmma"),
        ("gemv_awq_selected_pack8_transposed_fp16", "down_gemv"),
        ("weighted_lanes_sum_out_fp16_f32w", "weighted_lanes"),
        ("w8a16_shared_gate_up_silu_fp16", "legacy_gate_up_silu"),
        ("w8a16_shared_gate_sigmoid_fp32", "legacy_shared_gate_sigmoid"),
        ("w8a16_shared_down_combine_residual_fp16", "unexpected_legacy_down_fallback"),
        ("w8a16_shared_down_combine_residual_fp16_token_tiled", "legacy_down_combine_tiled"),
        ("shared_gate_combine_residual_batch_out_fp16", "unexpected_split_combine"),
        ("awq_fusedw4_prefill_dual_fp16", "unexpected_w4"),
    ]:
        monkeypatch.setattr(qwen_runtime, name, lambda *args, label=label, **kwargs: calls.append((label, args, kwargs)))

    out = state.run_moe_grouped_compact_fp16(hidden, residual, scratch=scratch, tokens=2)

    assert out is scratch.moe_out
    assert [kind for kind, _args, _kwargs in calls] == [
        "router",
        "rotate1",
        "gate_up_gemv",
        "silu_rotate",
        "down_gemv",
        "weighted_lanes",
        "legacy_gate_up_silu",
        "legacy_shared_gate_sigmoid",
        "legacy_down_combine_tiled",
    ]
    final_combine = calls[-1][1]
    shared_gate_logits_ptr = scratch.router_logits.ptr + 128 * 4
    assert final_combine[:7] == (
        scratch.shared_intermediate.ptr,
        state.tensor("layers.0.mlp.shared_expert.down_weight_w8a16").ptr,
        state.tensor("layers.0.mlp.shared_expert.down_weight_w8a16_scale").ptr,
        scratch.selected_out.ptr,
        shared_gate_logits_ptr,
        residual.ptr,
        scratch.moe_out.ptr,
    )


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
    monkeypatch.setattr(qwen_runtime, "weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w", fake_combine)

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

    batch_scratch = state.reserve_moe_c1_scratch(tokens=2)
    batch_shared = _tensor(0xCD00, (2, 4096), "bf16")
    batch_residual = _tensor(0xCE00, (2, 4096), "bf16")
    batch_out = state.combine_moe_c1_shared_residual_bf16(
        batch_scratch,
        shared=batch_shared,
        residual=batch_residual,
        tokens=2,
    )
    assert batch_out is batch_scratch.moe_out
    batch_args, batch_kwargs = calls[1]
    assert batch_args == (
        batch_scratch.down_out.ptr,
        batch_scratch.routing_weights.ptr,
        batch_shared.ptr,
        batch_scratch.router_logits.ptr + 128 * 4,
        batch_residual.ptr,
        batch_scratch.moe_out.ptr,
        2,
        8,
        4096,
        129,
    )
    assert batch_kwargs == {"threads": 256, "stream": 0, "library": None, "runtime": runtime}


def test_qwen35_decode_state_runs_moe_c1_chain_in_parent_order(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    hidden = _tensor(0xCA00, (1, 4096), "bf16")
    residual = _tensor(0xCC00, (1, 4096), "bf16")
    order = []

    monkeypatch.setattr(qwen_runtime, "qwen35_router_topk_shared_out_bf16", lambda *a, **k: order.append("router"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", lambda *a, **k: order.append("rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate2_bf16", lambda *a, **k: order.append("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_bf16", lambda *a, **k: order.append("gate_up"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_bf16", lambda *a, **k: order.append("silu_rotate"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_bf16", lambda *a, **k: order.append("down"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", lambda *a, **k: order.append("shared_dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_bf16", lambda *a, **k: order.append("shared_single_pack8"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_out_bf16", lambda *a, **k: order.append("shared_silu"))
    monkeypatch.setattr(
        qwen_runtime,
        "weighted_sum_shared_gate_combine_residual_out_bf16_f32w",
        lambda *a, **k: order.append("combine"),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w",
        lambda *a, **k: order.append("combine_batch"),
    )

    out = state.run_moe_c1_bf16(hidden, residual, scratch=scratch)

    assert out is scratch.moe_out
    # routed: router + (rotate + dual GEMV + silu_rotate + single GEMV)
    # shared: rotate2 gate/up + dual GEMV (packed) + fused silu*mul/down-rotate + single GEMV
    # combine
    assert order == [
        "router", "rotate1", "gate_up", "silu_rotate", "down",
        "rotate2", "shared_dual_pack8", "silu_rotate", "shared_single_pack8",
        "combine",
    ]

    batch_scratch = state.reserve_moe_c1_scratch(tokens=2)
    batch_hidden = _tensor(0xD000, (2, 4096), "bf16")
    batch_residual = _tensor(0xD100, (2, 4096), "bf16")
    order.clear()
    batch_out = state.run_moe_c1_bf16(batch_hidden, batch_residual, scratch=batch_scratch, tokens=2)
    assert batch_out is batch_scratch.moe_out
    assert order == [
        "router", "rotate1", "gate_up", "silu_rotate", "down",
        "rotate2", "shared_dual_pack8", "silu_rotate", "shared_single_pack8",
        "combine_batch",
    ]


def test_qwen35_decode_state_runs_moe_c1_bf16_legacy_w8a16_chain(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _legacy_prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1)
    hidden = _tensor(0xCA00, (1, 4096), "bf16")
    residual = _tensor(0xCC00, (1, 4096), "bf16")
    order = []

    monkeypatch.setattr(qwen_runtime, "qwen35_router_topk_shared_out_bf16", lambda *a, **k: order.append("router"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_bf16", lambda *a, **k: order.append("rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate2_bf16", lambda *a, **k: order.append("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_bf16", lambda *a, **k: order.append("gate_up"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_bf16", lambda *a, **k: order.append("silu_rotate"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_bf16", lambda *a, **k: order.append("down"))
    monkeypatch.setattr(qwen_runtime, "w8a16_linear_bf16_lowp_out", lambda *a, **k: order.append("shared_w8a16"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_out_bf16", lambda *a, **k: order.append("shared_silu"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_bf16", lambda *a, **k: order.append("unexpected_w4_dual"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_bf16", lambda *a, **k: order.append("unexpected_w4_single"))
    monkeypatch.setattr(
        qwen_runtime,
        "weighted_sum_shared_gate_combine_residual_out_bf16_f32w",
        lambda *a, **k: order.append("combine"),
    )

    out = state.run_moe_c1_bf16(hidden, residual, scratch=scratch)

    assert out is scratch.moe_out
    assert order == ["router", "rotate1", "gate_up", "silu_rotate", "down", "shared_w8a16", "shared_silu", "shared_w8a16", "combine"]


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
    # 33 attention/moe scratch tensors (including rotate_fuse_barrier and the
    # always-reserved MoE shared_rotate_fuse_barrier) + 5 shared-expert scratch
    # tensors (shared_gate_input, shared_up_input, shared_gate_out, shared_up_out,
    # shared_down_input).
    assert len(runtime.freed) == 38


def test_qwen35_decode_state_reserves_parent_mixed_fp16_scratch() -> None:
    runtime = FakeRuntime()
    state = _state(runtime)

    attn = state.reserve_full_attention_scratch(tokens=1, num_splits=2, activation_dtype="fp16", gated_dtype="fp16")
    linear = state.reserve_linear_attention_scratch(tokens=1, activation_dtype="fp16")
    moe = state.reserve_moe_c1_scratch(tokens=1, activation_dtype="fp16")

    assert attn.attn_input.dtype is DType.FP16
    assert attn.q_proj.dtype is DType.FP16
    assert attn.key_bf16.dtype is DType.FP16
    assert attn.value.dtype is DType.FP16
    assert attn.query.dtype is DType.FP32
    assert attn.key.dtype is DType.FP32
    assert linear.qkv.dtype is DType.FP16
    assert linear.recurrent_bf16.dtype is DType.FP16
    assert moe.normed.dtype is DType.FP16
    assert moe.gate_up.dtype is DType.FP16
    assert moe.moe_out.dtype is DType.FP16


def test_qwen35_decode_state_runs_linear_attention_fp16_out_proj_chain(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    hidden = _tensor(0xC000, (1, 4096), "fp16")
    conv_state = _tensor(0xC100, (8192, 4), "fp32")
    recurrent_state = _tensor(0xC200, (32, 128, 128), "fp32")
    scratch = state.reserve_linear_attention_scratch(tokens=1, activation_dtype="fp16")
    order = []

    monkeypatch.setattr(qwen_runtime, "paro_rotate2_fp16", lambda *a, **k: order.append("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_fp16", lambda *a, **k: order.append("dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "dense_dual_gemv_out_fp16", lambda *a, **k: order.append("dense_dual"))
    monkeypatch.setattr(qwen_runtime, "qwen35_linear_attn_conv_decode_fp16", lambda *a, **k: order.append("conv"))
    monkeypatch.setattr(qwen_runtime, "qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16", lambda *a, **k: order.append("gdn"))
    monkeypatch.setattr(qwen_runtime, "f32_to_fp16", lambda *a, **k: order.append("cast"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_fp16", lambda *a, **k: order.append("rotate1"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_strided_fp16", lambda *a, **k: order.append("pack8"))

    out = state.run_linear_attention_out_proj_fp16(
        hidden,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        scratch=scratch,
    )

    assert out is scratch.out_proj
    assert order == ["rotate2", "dual_pack8", "dense_dual", "conv", "gdn", "cast", "rotate1", "pack8"]


def test_qwen35_decode_state_can_fuse_linear_attention_cast_rotate(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    scratch = state.reserve_linear_attention_scratch(tokens=2, activation_dtype="fp16")
    order = []

    monkeypatch.delenv("HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED", raising=False)
    monkeypatch.setattr(qwen_runtime, "f32_to_fp16", lambda *a, **k: order.append("unexpected_cast"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_fp16", lambda *a, **k: order.append("unexpected_rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_f32_to_fp16", lambda *a, **k: order.append("cast_rotate"))
    for name in (
        "gemv_awq_pack8_strided_fp16",
        "gemv_awq_pack8_output_tiled_fp16",
        "gemv_awq_pack8_multi_row_strided_fp16",
        "awq_fusedw4_prefill_strided_fp16",
        "gemv_awq_pack8_transposed_fp16",
        "gemv_awq_pack8_output_tiled_transposed_fp16",
        "gemv_awq_pack8_multi_row_transposed_fp16",
        "awq_fusedw4_prefill_fp16",
    ):
        monkeypatch.setattr(qwen_runtime, name, lambda *a, **k: order.append("pack8"))

    out = state.project_linear_attention_out_fp16(scratch, tokens=2)

    assert out is scratch.out_proj
    assert order == ["cast_rotate", "pack8"]


def test_qwen35_decode_state_can_opt_out_of_linear_attention_cast_rotate_fusion(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    scratch = state.reserve_linear_attention_scratch(tokens=2, activation_dtype="fp16")
    order = []

    monkeypatch.setenv("HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED", "0")
    monkeypatch.setattr(qwen_runtime, "f32_to_fp16", lambda *a, **k: order.append("cast"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_fp16", lambda *a, **k: order.append("rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_f32_to_fp16", lambda *a, **k: order.append("unexpected_cast_rotate"))
    for name in (
        "gemv_awq_pack8_strided_fp16",
        "gemv_awq_pack8_output_tiled_fp16",
        "gemv_awq_pack8_multi_row_strided_fp16",
        "awq_fusedw4_prefill_strided_fp16",
        "gemv_awq_pack8_transposed_fp16",
        "gemv_awq_pack8_output_tiled_transposed_fp16",
        "gemv_awq_pack8_multi_row_transposed_fp16",
        "awq_fusedw4_prefill_fp16",
    ):
        monkeypatch.setattr(qwen_runtime, name, lambda *a, **k: order.append("pack8"))

    out = state.project_linear_attention_out_fp16(scratch, tokens=2)

    assert out is scratch.out_proj
    assert order == ["cast", "rotate1", "pack8"]


def test_qwen35_w4_dual_output_tiled_split_prefill_default_and_optout(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL", raising=False)
    monkeypatch.delenv("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_SITES", raising=False)

    assert qwen_runtime._w4_dual_output_tiled_split_site_eligible("shared_gate_up", 2, 4096, 128)
    assert not qwen_runtime._w4_dual_output_tiled_split_site_eligible("dense_gate_up", 2, 4096, 128)

    monkeypatch.setenv("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL", "0")

    assert not qwen_runtime._w4_dual_output_tiled_split_site_eligible("shared_gate_up", 2, 4096, 128)


def test_qwen35_decode_state_selected_output_uses_prefill_lowp_after_segment_state(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _linear_weights())
    hidden = _tensor(0xC000, (2, 4096), "fp16")
    conv_state = _tensor(0xC100, (2, 8192, 4), "fp32")
    recurrent_state = _tensor(0xC200, (2, 32, 128, 128), "fp32")
    cu_seqlens = _tensor(0xC300, (3,), "int32")
    state_indices = _tensor(0xC400, (2,), "int64")
    scratch = state.reserve_linear_attention_scratch(tokens=2, activation_dtype="fp16")
    state_pairs = ((conv_state, recurrent_state), (conv_state, recurrent_state))
    calls = []

    def fake_state_segments(*args, **kwargs):
        calls.append(("state_segments", kwargs["force_selected_c1_state"]))
        assert args == (hidden,)
        assert kwargs["scratch"] is scratch
        assert kwargs["tokens"] == 2
        return scratch.recurrent_bf16

    def fake_decode_rows_out(*args, **kwargs):
        calls.append(("decode_rows_out", kwargs["tokens"]))
        return scratch.out_proj

    def fake_prefill_rows_out(*args, **kwargs):
        calls.append(("prefill_rows_out", kwargs["tokens"]))
        return scratch.out_proj

    def fake_decode_order_out(*args, **kwargs):
        calls.append(("decode_order_out", kwargs["tokens"]))
        return scratch.out_proj

    monkeypatch.setattr(state, "run_linear_attention_prefill_state_segments_fp16", fake_state_segments)
    monkeypatch.setattr(state, "project_linear_attention_decode_rows_out_fp16", fake_decode_rows_out)
    monkeypatch.setattr(state, "project_linear_attention_prefill_rows_out_fp16", fake_prefill_rows_out)
    monkeypatch.setattr(state, "project_linear_attention_out_fp16", fake_decode_order_out)

    out = state.run_linear_attention_prefill_out_proj_segments_fp16(
        hidden,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        segments=2,
        scratch=scratch,
        tokens=2,
        force_selected_c1_out=True,
    )

    assert out is scratch.out_proj
    assert calls == [("state_segments", False), ("prefill_rows_out", 2)]

    calls.clear()
    out = state.run_linear_attention_prefill_out_proj_segments_fp16(
        hidden,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        segments=2,
        scratch=scratch,
        tokens=2,
        force_selected_c1_state=True,
        selected_c1_state_pairs=state_pairs,
        force_selected_c1_out=True,
    )

    assert out is scratch.out_proj
    assert calls == [("state_segments", True), ("decode_rows_out", 2)]

    calls.clear()
    out = state.run_linear_attention_prefill_out_proj_segments_fp16(
        hidden,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        segments=2,
        scratch=scratch,
        tokens=2,
        decode_order_state=True,
    )

    assert out is scratch.out_proj
    assert calls == [("state_segments", False), ("decode_order_out", 2)]


def test_qwen35_decode_state_runs_moe_c1_fp16_chain_in_parent_order(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime, _prepared_moe_weights())
    scratch = state.reserve_moe_c1_scratch(tokens=1, activation_dtype="fp16")
    hidden = _tensor(0xCA00, (1, 4096), "fp16")
    residual = _tensor(0xCC00, (1, 4096), "fp16")
    order = []

    monkeypatch.setattr(qwen_runtime, "qwen35_router_topk_shared_out_fp16", lambda *a, **k: order.append("router"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate1_fp16", lambda *a, **k: order.append("rotate1"))
    monkeypatch.setattr(qwen_runtime, "paro_rotate2_fp16", lambda *a, **k: order.append("rotate2"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_dual_pack8_transposed_fp16", lambda *a, **k: order.append("gate_up"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_rotate_out_fp16", lambda *a, **k: order.append("silu_rotate"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_selected_pack8_transposed_fp16", lambda *a, **k: order.append("down"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_dual_pack8_transposed_fp16", lambda *a, **k: order.append("shared_dual_pack8"))
    monkeypatch.setattr(qwen_runtime, "gemv_awq_pack8_transposed_fp16", lambda *a, **k: order.append("shared_single_pack8"))
    monkeypatch.setattr(qwen_runtime, "silu_mul_dual_out_fp16", lambda *a, **k: order.append("unexpected_shared_silu"))
    monkeypatch.setattr(
        qwen_runtime,
        "weighted_sum_shared_gate_combine_residual_out_fp16_f32w",
        lambda *a, **k: order.append("combine"),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w",
        lambda *a, **k: order.append("combine_batch"),
    )

    out = state.run_moe_c1_fp16(hidden, residual, scratch=scratch)

    assert out is scratch.moe_out
    assert order == [
        "router", "rotate1", "gate_up", "silu_rotate", "down",
        "rotate2", "shared_dual_pack8", "silu_rotate", "shared_single_pack8",
        "combine",
    ]


def test_qwen35_decode_state_runs_full_attention_fp16_pre_moe_chain(monkeypatch) -> None:
    runtime = FakeRuntime()
    weights = DeviceWeightMap({**_full_attention_weights().tensors, **_prepared_moe_weights().tensors})
    state = _state(runtime, weights)
    hidden = _tensor(0xC000, (1, 4096), "fp16")
    key_cache = _tensor(0xE000, (1, 256, 2, 256), "bf16")
    value_cache = _tensor(0xF000, (1, 256, 2, 256), "bf16")
    cos_table = _tensor(0xD200, (4, 256), "fp32")
    sin_table = _tensor(0xD300, (4, 256), "fp32")
    position = _tensor(0xD400, (1,), "int64")
    attn = state.reserve_full_attention_scratch(tokens=1, num_splits=1, activation_dtype="fp16", gated_dtype="fp16")
    moe = state.reserve_moe_c1_scratch(tokens=1, activation_dtype="fp16")
    order = []
    monkeypatch.setattr(qwen_runtime, "resolve_paged_kv_write", lambda **_kwargs: lambda *a, **k: order.append("kv"))

    for name, label in [
        ("paro_rmsnorm_out_fp16", "input_norm"),
        ("paro_rotate3_fp16", "rotate3"),
        ("gemv_awq_dual_pack8_transposed_fp16", "dual_pack8"),
        ("gemv_awq_dual_pack8_transposed_rotate_staged_fp16", "fused_dual_pack8"),
        ("gemv_awq_pack8_strided_fp16", "pack8"),
        ("qwen35_split_qgate_fp16", "split_qgate"),
        ("fp16_to_f32", "fp16_to_f32"),
        ("qwen35_head_rmsnorm_partial_rotary_position_f32_bf16", "head_rotary"),
        ("qwen35_full_attn_decode_context_bf16", "dense_attention_context"),
        ("qwen35_full_attn_gate_mul_fp16", "attention_gate"),
        ("paro_rotate1_fp16", "rotate1"),
        ("paro_rotate2_fp16", "rotate2"),
        ("paro_add_rmsnorm_out_fp16", "post_norm"),
        ("qwen35_router_topk_shared_out_fp16", "router"),
        ("gemv_awq_selected_dual_pack8_transposed_fp16", "gate_up"),
        ("silu_mul_dual_rotate_out_fp16", "silu_rotate"),
        ("gemv_awq_selected_pack8_transposed_fp16", "down"),
        ("gemv_awq_pack8_transposed_fp16", "shared_single_pack8"),
        ("silu_mul_dual_out_fp16", "shared_silu"),
        ("weighted_sum_shared_gate_combine_residual_out_fp16_f32w", "combine"),
    ]:
        monkeypatch.setattr(qwen_runtime, name, lambda *a, label=label, **k: order.append(label))

    out = state.run_full_attention_moe_c1_layer_fp16(
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
    assert order == [
        "input_norm",
        "rotate3",
        "dual_pack8",  # attention QKV
        "pack8",
        "split_qgate",
        "fp16_to_f32",
        "head_rotary",
        "kv",
        "dense_attention_context",
        "attention_gate",
        "rotate1",
        "pack8",
        "post_norm",
        "router",
        "rotate1",
        "gate_up",
        "silu_rotate",
        "down",
        # Shared expert W4 PARO chain. "dual_pack8" here is the same kernel as
        # the attention QKV dual GEMV; the runtime reuses gemv_awq_dual_pack8_
        # transposed_fp16 for the dense shared expert gate||up.
        "rotate2",
        "dual_pack8",
        "silu_rotate",
        "shared_single_pack8",
        "combine",
    ]


def test_qwen35_decode_state_int8_prefill_append_converts_fp16_value_and_passes_scales(monkeypatch) -> None:
    runtime = FakeRuntime()
    state = _state(runtime)
    device = Device("hip", 0)
    rows = 2
    kv_shape = (rows, state.config.num_key_value_heads, state.config.head_dim)
    scratch = SimpleNamespace(
        key=Tensor.from_handle(0x1000, kv_shape, DType.FP32, device),
        value=Tensor.from_handle(0x2000, kv_shape, DType.FP16, device),
        key_raw=Tensor.from_handle(0x3000, kv_shape, DType.FP32, device),
    )
    scales = KVScaleMetadata(
        k_scale=Tensor.from_handle(0x4000, (1, 256, state.config.num_key_value_heads), DType.FP16, device),
        v_scale=Tensor.from_handle(0x5000, (1, 256, state.config.num_key_value_heads), DType.FP16, device),
    )
    spans = KVLiveSpans.paged_uniform(
        block_table=Tensor.from_handle(0x6000, (1,), DType.INT32, device),
        live_counts=Tensor.from_handle(0x7000, (rows,), DType.INT64, device),
        max_live_count=rows - 1,
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        scale_metadata=scales,
    )
    key_cache = Tensor.from_handle(0x8000, (1, 256, state.config.num_key_value_heads, state.config.head_dim), DType.INT8, device)
    value_cache = Tensor.from_handle(0x9000, key_cache.shape, DType.INT8, device)
    calls = []

    def fake_fp16_to_f32(src, dst, count, **kwargs):
        calls.append(("cast", src, dst, count, kwargs["stream"]))

    def fake_writer(key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, k_scale_ptr, v_scale_ptr, spans_arg, rows_arg, *args, **kwargs):
        calls.append(
            (
                "writer",
                key_ptr,
                value_ptr,
                key_cache_ptr,
                value_cache_ptr,
                k_scale_ptr,
                v_scale_ptr,
                spans_arg,
                rows_arg,
                kwargs["stream"],
            )
        )

    monkeypatch.setattr(qwen_runtime, "fp16_to_f32", fake_fp16_to_f32)
    monkeypatch.setattr(qwen_runtime, "resolve_paged_kv_write", lambda **_kwargs: fake_writer)

    state.append_full_attention_kv_int8_per_token_head_fp16_batch(
        scratch,
        key_cache=key_cache,
        value_cache=value_cache,
        spans=spans,
        rows=rows,
        library={},
        stream=7,
    )

    assert calls[0] == ("cast", scratch.value.ptr, scratch.key_raw.ptr, rows * state.config.num_key_value_heads * state.config.head_dim, 7)
    assert calls[1] == (
        "writer",
        scratch.key.ptr,
        scratch.key_raw.ptr,
        key_cache.ptr,
        value_cache.ptr,
        scales.k_scale.ptr,
        scales.v_scale.ptr,
        spans,
        rows,
        7,
    )
