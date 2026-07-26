from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import hipengine.runtime.laguna_moe as laguna_moe_module
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.registry import KernelKey
from hipengine.loading.laguna_gguf import (
    SLIDING_ATTENTION,
    SPARSE_MOE,
    laguna_gguf_config_from_metadata,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf_materialize import (
    LagunaGGUFResidentLayerWeights,
    _materialize_spec,
    _spec_for_tensor,
    materialize_laguna_gguf_weights,
)
from hipengine.models.laguna import LAGUNA_GGUF
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime.laguna_moe import (
    allocate_laguna_moe_scratch,
    resolve_laguna_dense_q4_prefill_mode,
    resolve_laguna_group_compact_mode,
    resolve_laguna_moe_plan,
    resolve_laguna_router_logits_mode,
    resolve_laguna_selected_down_mode,
    resolve_laguna_selected_gate_up_mode,
    run_laguna_moe_c1,
    run_laguna_moe_rows,
    validate_laguna_moe_layer,
)
from tests._gguf_synthetic_weights import make_q4_k_weight, make_q6_k_weight
from tests._laguna_synthetic import make_laguna_info, tensor_info


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()
LAGUNA_Q2_MODEL = Path("/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf")


def _f32_to_bf16_u16(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.float32)
    bits = value.view(np.uint32).copy()
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(value.shape)


def _bf16_u16_to_f32(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.uint16)
    return (value.astype(np.uint32) << 16).view(np.float32).reshape(value.shape).copy()


def _bf16_round(array: np.ndarray) -> np.ndarray:
    return _bf16_u16_to_f32(_f32_to_bf16_u16(array))


def _silu(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    positive = value >= 0
    sigmoid = np.empty_like(value)
    sigmoid[positive] = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-value[positive]).astype(np.float32)
    )
    numerator = np.exp(value[~positive]).astype(np.float32)
    sigmoid[~positive] = numerator / (np.float32(1.0) + numerator)
    return (value * sigmoid).astype(np.float32)


def _max_softmax_kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    ref -= ref.max(axis=-1, keepdims=True)
    cand -= cand.max(axis=-1, keepdims=True)
    ref_logp = ref - np.log(np.exp(ref).sum(axis=-1, keepdims=True))
    cand_logp = cand - np.log(np.exp(cand).sum(axis=-1, keepdims=True))
    return float(
        np.max(np.sum(np.exp(ref_logp) * (ref_logp - cand_logp), axis=-1))
    )


def _read_bf16(buffer, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.zeros(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(bits), buffer, bits.nbytes)
    return _bf16_u16_to_f32(bits)


def _read_array(buffer, dtype, shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), buffer, out.nbytes)
    return out


def test_laguna_model_moe_plan_resolves_production_contract_on_gfx1151() -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    plan = resolve_laguna_moe_plan(config, backend="hip_gfx1151")

    assert (plan.hidden_size, plan.expert_count, plan.top_k) == (3_072, 256, 10)
    assert (plan.expert_ffn_size, plan.shared_ffn_size) == (1_024, 1_024)
    assert plan.routed_scaling_factor == pytest.approx(2.5)
    assert plan.q6_qmicro
    assert plan.q6_compact_activation
    assert plan.q6_half_row_activation
    assert plan.q6_skip_padded_activation
    assert plan.q6_qmicro_permute
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        q6_qmicro=False,
    ).q6_qmicro
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        q6_compact_activation=False,
    ).q6_compact_activation
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        q6_compact_activation=False,
    ).q6_half_row_activation
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        q6_half_row_activation=False,
    ).q6_skip_padded_activation
    assert resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        q6_skip_padded_activation=True,
    ).q6_skip_padded_activation
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        q6_skip_padded_activation=False,
    ).q6_skip_padded_activation
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        q6_qmicro_permute=False,
    ).q6_qmicro_permute
    assert resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        q6_half_row_activation=True,
    ).q6_half_row_activation
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        q6_half_row_activation=False,
    ).q6_half_row_activation
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
    ).q6_qmicro
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
    ).q6_compact_activation
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
    ).q6_half_row_activation
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
    ).q6_skip_padded_activation
    assert not resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
    ).q6_qmicro_permute
    with pytest.raises(
        ValueError,
        match="half-row activation staging requires compact activation",
    ):
        resolve_laguna_moe_plan(
            config,
            backend="hip_gfx1151",
            q6_compact_activation=False,
            q6_half_row_activation=True,
        )
    with pytest.raises(
        ValueError,
        match="skipping padded activation rows requires half-row staging",
    ):
        resolve_laguna_moe_plan(
            config,
            backend="hip_gfx1151",
            q6_half_row_activation=False,
            q6_skip_padded_activation=True,
        )
    with pytest.raises(
        ValueError,
        match="qmicro permute decode requires padded-row skipping",
    ):
        resolve_laguna_moe_plan(
            config,
            backend="hip_gfx1151",
            q6_skip_padded_activation=False,
            q6_qmicro_permute=True,
        )
    assert plan.router_select_key.layer == "laguna_sigmoid_router_topk"
    assert set(plan.router_logits_keys) == {
        "token_tile_4",
        "token_tile_8",
        "token_tile_16",
    }
    assert plan.router_logits_keys["token_tile_8"] == KernelKey(
        "hip_gfx1151",
        "router_logits",
        "f32",
        "bf16_hidden_token_tile_8",
    )
    assert (plan.routed_sum_rows_key.layer, plan.routed_sum_rows_key.variant) == (
        "weighted_sum",
        "laguna_rows",
    )
    assert plan.selected_gate_up_key.quant == "gguf_q4_k_t16_v1"
    assert plan.selected_gate_up_prefill_key == KernelKey(
        "hip_gfx1151",
        "moe_linear",
        "gguf_q4_k_t16_v1",
        "selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out",
    )
    assert plan.activation_quant_key == KernelKey(
        "hip_gfx1151", "activation_quant", "q8_1_ds4x3", "bf16"
    )
    assert plan.grouped_compact_source_rows_key.variant == (
        "active_experts_source_rows"
    )
    assert plan.grouped_compact_parallel_key.variant == "active_experts_parallel"
    assert plan.grouped_compact_source_rows_parallel_key.variant == (
        "active_experts_source_rows_parallel"
    )
    assert plan.mmq_tile_map_key == KernelKey(
        "hip_gfx1151", "moe_mmq_tile_map", "generic", "tile32"
    )
    assert plan.mmq64_tile_map_key == KernelKey(
        "hip_gfx1151", "moe_mmq_tile_map", "generic", "tile64"
    )
    assert plan.selected_dual_silu_key == KernelKey(
        "hip_gfx1151", "silu_mul_dual", "bf16", "out"
    )
    assert plan.fused_selected_silu_pack_key == KernelKey(
        "hip_gfx1151",
        "silu_mul_dual+activation_quant",
        "q8_1_ds4x3_f32",
        "bf16",
    )
    assert set(plan.selected_gate_up_keys) == {
        "gguf_q4_k_t16_v1",
        "gguf_iq2_xs",
        "gguf_iq3_xxs",
    }
    assert set(plan.selected_down_keys) == {
        "gguf_q4_k_t16_v1",
        "gguf_q6_k_t16_v1",
        "gguf_iq3_xxs",
        "gguf_iq4_xs",
    }
    assert plan.selected_down_keys["gguf_q6_k_t16_v1"] == plan.selected_down_key
    assert set(plan.grouped_smallm_down_keys) == {
        "gguf_q4_k_t16_v1",
        "gguf_q6_k_t16_v1",
    }
    assert plan.grouped_weighted_sum_shared_add_key == KernelKey(
        "hip_gfx1151", "weighted_lanes_sum+shared_add", "bf16", "out"
    )
    assert plan.grouped_prefix_active_key == KernelKey(
        "hip_gfx1151", "moe_group_prefix", "generic", "active_experts"
    )
    assert set(plan.selected_weighted_down_keys) == {"gguf_iq3_xxs"}
    assert plan.selected_weighted_down_keys["gguf_iq3_xxs"].variant == (
        "selected_weighted_down_gemv_decode_bf16_bf16_out"
    )
    assert all(key.backend == "hip_gfx1151" for key in plan.kernel_keys)

    sparse_sequence = LAGUNA_GGUF.decode_layer_sequence(
        attention_kind="sliding_attention",
        mlp_kind="sparse_moe",
    )
    assert "laguna_sigmoid_router_topk" in sparse_sequence
    assert "selected_expert_mlp" in sparse_sequence
    assert "laguna_shared_expert" in sparse_sequence
    assert "laguna_routed_shared_combine" in sparse_sequence


def test_laguna_dense_q4_prefill_mode_is_explicit_and_fail_closed() -> None:
    assert resolve_laguna_dense_q4_prefill_mode("hip_gfx1100") == "retained"
    assert resolve_laguna_dense_q4_prefill_mode("hip_gfx1151") == "wmma_pack8"
    assert (
        resolve_laguna_dense_q4_prefill_mode("hip_gfx1151", "retained")
        == "retained"
    )
    assert (
        resolve_laguna_dense_q4_prefill_mode("hip_gfx1151", "wmma_pack8")
        == "wmma_pack8"
    )
    with pytest.raises(ValueError, match="dense/shared Q4"):
        resolve_laguna_dense_q4_prefill_mode("hip_gfx1151", "rowtile8")


def test_laguna_mmq_selected_down_reuses_exact_grouped_combine() -> None:
    should_fuse = laguna_moe_module._should_fuse_grouped_weighted_sum
    assert should_fuse(
        selected_down_mode="mmq64x64_d4_f32_q6_wavecols_direct_q4",
        tokens=512,
        use_mmq_down=True,
    )
    assert should_fuse(
        selected_down_mode="grouped_smallm_fused",
        tokens=3,
        use_mmq_down=False,
    )
    assert not should_fuse(
        selected_down_mode="grouped_smallm",
        tokens=512,
        use_mmq_down=False,
    )
    assert not should_fuse(
        selected_down_mode="direct",
        tokens=512,
        use_mmq_down=False,
    )


def test_laguna_mmq_selected_silu_pack_fusion_is_shape_qualified() -> None:
    should_fuse = laguna_moe_module._should_fuse_selected_silu_pack
    production = {
        "selected_gate_up_mode": (
            "mmq128x32_d8_f32_wavecols_direct_doublebuf"
        ),
        "selected_down_mode": (
            "mmq64x64_d4_f32_q6_wavecols_direct_q4"
        ),
    }
    assert should_fuse(requested=True, **production)
    assert not should_fuse(requested=False, **production)
    assert not should_fuse(
        requested=True,
        selected_gate_up_mode="direct",
        selected_down_mode=production["selected_down_mode"],
    )
    assert not should_fuse(
        requested=True,
        selected_gate_up_mode=production["selected_gate_up_mode"],
        selected_down_mode="direct",
    )
    assert not should_fuse(
        requested=True,
        selected_gate_up_mode="mmq64x32_d4x3_f32",
        selected_down_mode="mmq64x32_d4x3_f32",
    )


def test_laguna_group_compact_mode_is_backend_qualified() -> None:
    assert resolve_laguna_group_compact_mode("hip_gfx1100") == "serial"
    assert resolve_laguna_group_compact_mode("hip_gfx1151") == "parallel"
    assert resolve_laguna_group_compact_mode("hip_gfx1151", "serial") == "serial"
    assert (
        resolve_laguna_group_compact_mode("hip_gfx1151", "parallel")
        == "parallel"
    )
    with pytest.raises(ValueError, match="group compact"):
        resolve_laguna_group_compact_mode("hip_gfx1151", "atomic")


def test_laguna_selected_down_default_is_backend_qualified() -> None:
    assert resolve_laguna_selected_down_mode("hip_gfx1100") == "direct"
    assert (
        resolve_laguna_selected_down_mode("hip_gfx1151")
        == "mmq64x64_d4_f32_q6_wavecols_direct_q4"
    )
    assert resolve_laguna_selected_down_mode("hip_gfx1151", "direct") == "direct"
    assert (
        resolve_laguna_selected_down_mode("hip_gfx1151", "adaptive_grouped_smallm")
        == "adaptive_grouped_smallm"
    )
    assert (
        resolve_laguna_selected_down_mode(
            "hip_gfx1151", "adaptive_grouped_smallm_fused"
        )
        == "adaptive_grouped_smallm_fused"
    )
    assert (
        resolve_laguna_selected_down_mode(
            "hip_gfx1151",
            "mmq64x32_d4_f32_rowvec",
        )
        == "mmq64x32_d4_f32_rowvec"
    )
    assert (
        resolve_laguna_selected_down_mode(
            "hip_gfx1151",
            "mmq64x32_d4_f32_wavecols_q4",
        )
        == "mmq64x32_d4_f32_wavecols_q4"
    )
    assert (
        resolve_laguna_selected_down_mode(
            "hip_gfx1151",
            "mmq64x32_d4_f32_wavecols_direct_q4",
        )
        == "mmq64x32_d4_f32_wavecols_direct_q4"
    )
    for rejected in ("wmma16_down", "adaptive_wmma16_down", "invalid"):
        with pytest.raises(ValueError, match="unsupported Laguna selected-down mode"):
            resolve_laguna_selected_down_mode("hip_gfx1151", rejected)


def test_laguna_selected_gate_up_default_is_backend_qualified() -> None:
    assert resolve_laguna_selected_gate_up_mode("hip_gfx1100") == "direct"
    assert (
        resolve_laguna_selected_gate_up_mode("hip_gfx1151")
        == "mmq128x32_d8_f32_wavecols_direct_doublebuf"
    )
    assert (
        resolve_laguna_selected_gate_up_mode("hip_gfx1151", "direct")
        == "direct"
    )
    assert (
        resolve_laguna_selected_gate_up_mode("hip_gfx1151", "mmq32_d4x3")
        == "mmq32_d4x3"
    )
    assert (
        resolve_laguna_selected_gate_up_mode(
            "hip_gfx1151", "mmq128x32_d8_f32_rowvec"
        )
        == "mmq128x32_d8_f32_rowvec"
    )
    assert (
        resolve_laguna_selected_gate_up_mode(
            "hip_gfx1151", "mmq128x32_d8_f32_wavecols_direct"
        )
        == "mmq128x32_d8_f32_wavecols_direct"
    )
    assert (
        resolve_laguna_selected_gate_up_mode(
            "hip_gfx1151",
            "mmq128x32_d8_f32_wavecols_direct_doublebuf",
        )
        == "mmq128x32_d8_f32_wavecols_direct_doublebuf"
    )
    with pytest.raises(ValueError, match="unsupported Laguna selected gate/up mode"):
        resolve_laguna_selected_gate_up_mode("hip_gfx1151", "unsafe")


def test_laguna_router_logits_mode_is_exact_and_explicit() -> None:
    assert (
        resolve_laguna_router_logits_mode("hip_gfx1100")
        == "token_tile_4"
    )
    assert (
        resolve_laguna_router_logits_mode("hip_gfx1151")
        == "token_tile_8"
    )
    for mode in ("token_tile_4", "token_tile_8", "token_tile_16"):
        assert resolve_laguna_router_logits_mode("hip_gfx1151", mode) == mode
    with pytest.raises(ValueError, match="router-logits"):
        resolve_laguna_router_logits_mode("hip_gfx1151", "unsafe")


def test_laguna_moe_plan_rejects_qwen_softmax_or_unnormalized_contracts() -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    with pytest.raises(ValueError, match="sigmoid"):
        resolve_laguna_moe_plan(
            replace(config, expert_gating_func="softmax"),
            backend="hip_gfx1151",
        )
    with pytest.raises(ValueError, match="normalized"):
        resolve_laguna_moe_plan(
            replace(config, expert_weights_norm=False),
            backend="hip_gfx1151",
        )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "down_qtype",
    (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q6_K),
)
def test_laguna_unfused_moe_matches_production_shape_quant_oracle(
    down_qtype: GGMLQuantizationType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep production H/F/top-k and nontrivial rank-3 strides while reducing the
    # synthetic expert inventory; the separate router test covers all 256 IDs.
    config = replace(
        laguna_gguf_config_from_metadata(make_laguna_info()),
        expert_count=11,
        expert_used_count=10,
    )
    plan = resolve_laguna_moe_plan(config, backend="hip_gfx1151")
    h, e, k, f = plan.hidden_size, plan.expert_count, plan.top_k, plan.expert_ffn_size
    selected_order = np.asarray([10, 0, 9, 1, 8, 2, 7, 3, 6, 4], dtype=np.int64)

    router = np.zeros((e, h), dtype=np.float32)
    correction = np.zeros(e, dtype=np.float32)
    correction[selected_order] = np.linspace(1.0, 0.1, k, dtype=np.float32)
    q4_base = make_q4_k_weight(f, h)
    down_base = (
        make_q4_k_weight(h, f)
        if down_qtype == GGMLQuantizationType.Q4_K
        else make_q6_k_weight(h, f)
    )
    gate_experts = np.stack([np.roll(q4_base, 7 * expert, axis=0) for expert in range(e)], axis=0)
    up_experts = np.stack(
        [np.roll(q4_base, 11 * expert + 3, axis=0) for expert in range(e)], axis=0
    )
    down_experts = np.stack(
        [np.roll(down_base, 13 * expert + 5, axis=0) for expert in range(e)], axis=0
    )
    shared_gate = np.roll(q4_base, 17, axis=0).copy()
    shared_up = np.roll(q4_base, 29, axis=0).copy()
    shared_down = np.roll(down_base, 31, axis=0).copy()
    payloads = {
        "ffn_gate_inp": (router, GGMLQuantizationType.F32, (e, h)),
        "exp_probs_b": (correction, GGMLQuantizationType.F32, (e,)),
        "ffn_gate_exps": (gate_experts, GGMLQuantizationType.Q4_K, (e, f, h)),
        "ffn_up_exps": (up_experts, GGMLQuantizationType.Q4_K, (e, f, h)),
        "ffn_down_exps": (down_experts, down_qtype, (e, h, f)),
        "ffn_gate_shexp": (shared_gate, GGMLQuantizationType.Q4_K, (f, h)),
        "ffn_up_shexp": (shared_up, GGMLQuantizationType.Q4_K, (f, h)),
        "ffn_down_shexp": (shared_down, down_qtype, (h, f)),
    }

    resident = {}
    scratch = None
    bulk_scratch = None
    router_tile8_scratch = None
    grouped_scratch = None
    fused_scratch = None
    mmq_scratch = None
    mmq_parallel_scratch = None
    down_rowvec_scratch = None
    concurrent_scratch = None
    concurrent_stream = 0
    concurrent_input_ready = 0
    concurrent_output_ready = 0
    hidden_buffer = None
    bulk_hidden_buffer = None
    try:
        for slot, (array, qtype, shape) in payloads.items():
            source = tensor_info(f"synthetic.{slot}", shape, qtype)
            resident[slot] = _materialize_spec(
                _spec_for_tensor(f"layers.1.{slot}", source),
                _ArrayReader(source.name, array),
                device=None,
                runtime=_runtime(),
                backend="hip_gfx1151",
            )
        layer = LagunaGGUFResidentLayerWeights(
            layer_id=1,
            attention_type=SLIDING_ATTENTION,
            mlp_type=SPARSE_MOE,
            weights=MappingProxyType(resident),
        )
        validate_laguna_moe_layer(layer, plan)
        assert layer.weight("ffn_gate_exps").spec.source.byte_shape == (
            e,
            f,
            (h // 256) * 144,
        )
        down_block_bytes = 144 if down_qtype == GGMLQuantizationType.Q4_K else 210
        down_tile_bytes = 2_368 if down_qtype == GGMLQuantizationType.Q4_K else 3_360
        assert layer.weight("ffn_down_exps").spec.source.byte_shape == (
            e,
            h,
            (f // 256) * down_block_bytes,
        )
        assert layer.weight("ffn_gate_exps").allocation("tiles").tensor.shape == (
            e,
            f // 16,
            h // 256,
            2_368,
        )
        assert layer.weight("ffn_down_exps").allocation("tiles").tensor.shape == (
            e,
            h // 16,
            f // 256,
            down_tile_bytes,
        )

        rng = np.random.default_rng(1138)
        hidden_bits = _f32_to_bf16_u16(rng.normal(0.0, 2.0e-4, size=(1, h)).astype(np.float32))
        hidden = _bf16_u16_to_f32(hidden_bits)
        hidden_buffer = malloc(hidden_bits.nbytes)
        copy_host_to_device(hidden_buffer, host_array_ptr(hidden_bits), hidden_bits.nbytes)
        scratch = allocate_laguna_moe_scratch(plan)
        output_buffer = run_laguna_moe_c1(hidden_buffer.ptr, layer, scratch)

        actual_selected = _read_array(scratch.selected_experts, np.int64, (k,))
        routing = _read_array(scratch.routing_weights, np.float32, (k,))
        scaled = _read_array(scratch.scaled_routing_weights, np.float32, (k,))
        actual = _read_bf16(output_buffer, (1, h))
        routed_actual = _read_bf16(scratch.routed_output, (1, h))
        shared_actual = _read_bf16(scratch.shared_output, (1, h))

        np.testing.assert_array_equal(actual_selected, selected_order)
        np.testing.assert_allclose(routing, np.full(k, 0.1, dtype=np.float32), atol=1.0e-7)
        np.testing.assert_allclose(scaled, np.full(k, 0.25, dtype=np.float32), atol=2.0e-7)
        np.testing.assert_allclose(routing.sum(), 1.0, atol=2.0e-7)
        np.testing.assert_allclose(scaled.sum(), 2.5, atol=5.0e-7)

        route_outputs = np.empty((k, h), dtype=np.float32)
        for route, expert in enumerate(selected_order.tolist()):
            gate = _bf16_round(
                gguf_quant_gemv(hidden, gate_experts[expert], GGMLQuantizationType.Q4_K)
            )
            up = _bf16_round(gguf_quant_gemv(hidden, up_experts[expert], GGMLQuantizationType.Q4_K))
            intermediate = _bf16_round(_silu(gate) * up)
            route_outputs[route] = _bf16_round(
                gguf_quant_gemv(
                    intermediate,
                    down_experts[expert],
                    down_qtype,
                )
            )[0]
        routed_expected = _bf16_round(
            np.sum(route_outputs * scaled[:, None], axis=0, dtype=np.float32)[None, :]
        )
        shared_gate_out = _bf16_round(
            gguf_quant_gemv(hidden, shared_gate, GGMLQuantizationType.Q4_K)
        )
        shared_up_out = _bf16_round(gguf_quant_gemv(hidden, shared_up, GGMLQuantizationType.Q4_K))
        shared_intermediate = _bf16_round(_silu(shared_gate_out) * shared_up_out)
        shared_expected = _bf16_round(
            gguf_quant_gemv(
                shared_intermediate,
                shared_down,
                down_qtype,
            )
        )
        expected = _bf16_round(routed_expected + shared_expected)

        assert np.isfinite(actual).all()
        for candidate, reference in (
            (routed_actual, routed_expected),
            (shared_actual, shared_expected),
            (actual, expected),
        ):
            relative_l2 = float(
                np.linalg.norm(candidate.astype(np.float64) - reference.astype(np.float64))
                / max(np.linalg.norm(reference.astype(np.float64)), 1.0e-12)
            )
            assert relative_l2 <= 0.02
        # The independent shared branch is not multiplied by routed scale.
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(actual),
            _f32_to_bf16_u16(_bf16_round(routed_actual + shared_actual)),
        )

        bulk_hidden_bits = np.concatenate(
            (
                hidden_bits,
                _f32_to_bf16_u16(hidden * np.float32(0.75)),
                _f32_to_bf16_u16(hidden * np.float32(-0.5)),
            ),
            axis=0,
        )
        bulk_hidden_buffer = malloc(bulk_hidden_bits.nbytes)
        copy_host_to_device(
            bulk_hidden_buffer,
            host_array_ptr(bulk_hidden_bits),
            bulk_hidden_bits.nbytes,
        )
        bulk_scratch = allocate_laguna_moe_scratch(plan, max_rows=3)
        bulk_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            bulk_scratch,
            rows=3,
        )
        bulk_actual = _read_bf16(bulk_output, (3, h))
        router_tile8_scratch = allocate_laguna_moe_scratch(
            plan,
            max_rows=3,
        )
        router_tile8_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            router_tile8_scratch,
            rows=3,
            router_logits_mode="token_tile_8",
        )
        np.testing.assert_array_equal(
            _read_bf16(router_tile8_output, (3, h)),
            bulk_actual,
        )
        np.testing.assert_array_equal(
            _read_array(
                router_tile8_scratch.selected_experts,
                np.int64,
                (3, k),
            ),
            _read_array(
                bulk_scratch.selected_experts,
                np.int64,
                (3, k),
            ),
        )
        np.testing.assert_array_equal(
            _read_array(
                router_tile8_scratch.scaled_routing_weights,
                np.float32,
                (3, k),
            ),
            _read_array(
                bulk_scratch.scaled_routing_weights,
                np.float32,
                (3, k),
            ),
        )
        grouped_scratch = allocate_laguna_moe_scratch(plan, max_rows=3)
        grouped_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            grouped_scratch,
            rows=3,
            selected_down_mode="grouped_smallm",
        )
        grouped_actual = _read_bf16(grouped_output, (3, h))
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(grouped_actual),
            _f32_to_bf16_u16(bulk_actual),
        )
        fused_scratch = allocate_laguna_moe_scratch(plan, max_rows=3)
        fused_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            fused_scratch,
            rows=3,
            selected_down_mode="grouped_smallm_fused",
        )
        fused_actual = _read_bf16(fused_output, (3, h))
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(fused_actual),
            _f32_to_bf16_u16(grouped_actual),
        )
        monkeypatch.setattr(laguna_moe_module, "_SELECTED_MMQ32_MIN_ROWS", 3)
        mmq_scratch = allocate_laguna_moe_scratch(plan, max_rows=3)
        mmq_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            mmq_scratch,
            rows=3,
            selected_down_mode="grouped_smallm_fused",
            selected_gate_up_mode="mmq32_d4x3",
        )
        mmq_actual = _read_bf16(mmq_output, (3, h))
        assert np.isfinite(mmq_actual).all()
        assert _max_softmax_kl(grouped_actual, mmq_actual) <= 0.05
        assert float(
            np.mean(
                np.argmax(grouped_actual, axis=-1)
                == np.argmax(mmq_actual, axis=-1)
            )
        ) >= 0.9
        mmq_parallel_scratch = allocate_laguna_moe_scratch(
            plan,
            max_rows=3,
        )
        mmq_parallel_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            mmq_parallel_scratch,
            rows=3,
            selected_down_mode="grouped_smallm_fused",
            selected_gate_up_mode="mmq32_d4x3",
            group_compact_mode="parallel",
        )
        mmq_parallel_actual = _read_bf16(
            mmq_parallel_output,
            (3, h),
        )
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(mmq_parallel_actual),
            _f32_to_bf16_u16(mmq_actual),
        )
        down_rowvec_scratch = allocate_laguna_moe_scratch(
            plan,
            max_rows=3,
        )
        down_baseline_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            down_rowvec_scratch,
            rows=3,
            selected_gate_up_mode="mmq128x32_d8_f32_rowvec",
            selected_down_mode="mmq64x32_d4_f32",
        )
        down_baseline_actual = _read_bf16(
            down_baseline_output,
            (3, h),
        )
        down_rowvec_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            down_rowvec_scratch,
            rows=3,
            selected_gate_up_mode="mmq128x32_d8_f32_rowvec",
            selected_down_mode="mmq64x32_d4_f32_rowvec",
        )
        down_rowvec_actual = _read_bf16(
            down_rowvec_output,
            (3, h),
        )
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(down_rowvec_actual),
            _f32_to_bf16_u16(down_baseline_actual),
        )
        down_wavecols_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            down_rowvec_scratch,
            rows=3,
            selected_gate_up_mode="mmq128x32_d8_f32_rowvec",
            selected_down_mode="mmq64x32_d4_f32_wavecols_q4",
        )
        down_wavecols_actual = _read_bf16(
            down_wavecols_output,
            (3, h),
        )
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(down_wavecols_actual),
            _f32_to_bf16_u16(down_rowvec_actual),
        )
        down_wavecols_direct_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            down_rowvec_scratch,
            rows=3,
            selected_gate_up_mode="mmq128x32_d8_f32_rowvec",
            selected_down_mode="mmq64x32_d4_f32_wavecols_direct_q4",
        )
        down_wavecols_direct_actual = _read_bf16(
            down_wavecols_direct_output,
            (3, h),
        )
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(down_wavecols_direct_actual),
            _f32_to_bf16_u16(down_wavecols_actual),
        )
        down_q6_rows64_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            down_rowvec_scratch,
            rows=3,
            selected_gate_up_mode="mmq128x32_d8_f32_rowvec",
            selected_down_mode="mmq64x64_d4_f32_q6_wavecols_direct_q4",
        )
        down_q6_rows64_actual = _read_bf16(
            down_q6_rows64_output,
            (3, h),
        )
        with monkeypatch.context() as unfused_combine:
            unfused_combine.setattr(
                laguna_moe_module,
                "_should_fuse_grouped_weighted_sum",
                lambda **_: False,
            )
            down_q6_rows64_unfused_output = run_laguna_moe_rows(
                bulk_hidden_buffer.ptr,
                layer,
                down_rowvec_scratch,
                rows=3,
                selected_gate_up_mode="mmq128x32_d8_f32_rowvec",
                selected_down_mode="mmq64x64_d4_f32_q6_wavecols_direct_q4",
            )
            down_q6_rows64_unfused_actual = _read_bf16(
                down_q6_rows64_unfused_output,
                (3, h),
            )
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(down_q6_rows64_actual),
            _f32_to_bf16_u16(down_q6_rows64_unfused_actual),
        )
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(down_q6_rows64_actual),
            _f32_to_bf16_u16(down_wavecols_direct_actual),
        )
        fused_silu_pack_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            down_rowvec_scratch,
            rows=3,
            selected_gate_up_mode=(
                "mmq128x32_d8_f32_wavecols_direct_doublebuf"
            ),
            selected_down_mode=(
                "mmq64x64_d4_f32_q6_wavecols_direct_q4"
            ),
            fuse_selected_silu_pack=True,
        )
        fused_silu_pack_actual = _read_bf16(
            fused_silu_pack_output,
            (3, h),
        )
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(fused_silu_pack_actual),
            _f32_to_bf16_u16(down_q6_rows64_actual),
        )
        runtime = _runtime()
        concurrent_scratch = allocate_laguna_moe_scratch(
            plan,
            max_rows=3,
        )
        least_priority, greatest_priority = runtime.stream_priority_range()
        concurrent_stream = runtime.stream_create(
            nonblocking=True,
            priority=(
                least_priority
                if least_priority != greatest_priority
                else None
            ),
        )
        concurrent_input_ready = runtime.event_create(flags=0x2)
        concurrent_output_ready = runtime.event_create(flags=0x2)
        concurrent_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            concurrent_scratch,
            rows=3,
            selected_gate_up_mode=(
                "mmq128x32_d8_f32_wavecols_direct_doublebuf"
            ),
            selected_down_mode=(
                "mmq64x64_d4_f32_q6_wavecols_direct_q4"
            ),
            fuse_selected_silu_pack=True,
            shared_stream=concurrent_stream,
            shared_input_ready_event=concurrent_input_ready,
            shared_output_ready_event=concurrent_output_ready,
            runtime=runtime,
        )
        runtime.device_synchronize()
        concurrent_actual = _read_bf16(
            concurrent_output,
            (3, h),
        )
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(concurrent_actual),
            _f32_to_bf16_u16(fused_silu_pack_actual),
        )
        after_router_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            concurrent_scratch,
            rows=3,
            selected_gate_up_mode=(
                "mmq128x32_d8_f32_wavecols_direct_doublebuf"
            ),
            selected_down_mode=(
                "mmq64x64_d4_f32_q6_wavecols_direct_q4"
            ),
            fuse_selected_silu_pack=True,
            shared_after_router=True,
            shared_stream=concurrent_stream,
            shared_input_ready_event=concurrent_input_ready,
            shared_output_ready_event=concurrent_output_ready,
            runtime=runtime,
        )
        runtime.device_synchronize()
        after_router_actual = _read_bf16(
            after_router_output,
            (3, h),
        )
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(after_router_actual),
            _f32_to_bf16_u16(fused_silu_pack_actual),
        )
        serial_actual = np.empty_like(bulk_actual)
        for row in range(3):
            copy_host_to_device(
                hidden_buffer,
                host_array_ptr(bulk_hidden_bits[row]),
                bulk_hidden_bits[row].nbytes,
            )
            serial_output = run_laguna_moe_c1(hidden_buffer.ptr, layer, scratch)
            serial_actual[row] = _read_bf16(serial_output, (1, h))[0]
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(bulk_actual),
            _f32_to_bf16_u16(serial_actual),
        )
        assert bulk_scratch.max_rows == 3
        assert bulk_scratch.selected_experts.nbytes == 3 * k * np.dtype(np.int64).itemsize
        assert grouped_scratch.grouped_active_experts.nbytes == e * np.dtype(np.int64).itemsize
        assert grouped_scratch.grouped_sorted_lanes.nbytes == 3 * k * np.dtype(np.int64).itemsize
    finally:
        runtime = _runtime()
        if concurrent_stream:
            runtime.stream_destroy(concurrent_stream)
        if concurrent_output_ready:
            runtime.event_destroy(concurrent_output_ready)
        if concurrent_input_ready:
            runtime.event_destroy(concurrent_input_ready)
        if concurrent_scratch is not None:
            concurrent_scratch.free()
        if router_tile8_scratch is not None:
            router_tile8_scratch.free()
        if mmq_parallel_scratch is not None:
            mmq_parallel_scratch.free()
        if down_rowvec_scratch is not None:
            down_rowvec_scratch.free()
        if mmq_scratch is not None:
            mmq_scratch.free()
        if fused_scratch is not None:
            fused_scratch.free()
        if grouped_scratch is not None:
            grouped_scratch.free()
        if bulk_scratch is not None:
            bulk_scratch.free()
        if scratch is not None:
            scratch.free()
        if bulk_hidden_buffer is not None:
            free(bulk_hidden_buffer)
        if hidden_buffer is not None:
            free(hidden_buffer)
        for weight in reversed(tuple(resident.values())):
            weight.free()


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.skipif(
    not LAGUNA_Q2_MODEL.exists(),
    reason=f"local Laguna Q2 XL fixture not found: {LAGUNA_Q2_MODEL}",
)
@pytest.mark.parametrize(
    ("layer_id", "gate_quant", "down_quant", "shared_gate_quant", "shared_down_quant"),
    (
        (1, "gguf_iq2_xs", "gguf_iq3_xxs", "gguf_q5_k", "gguf_q6_k"),
        (47, "gguf_iq3_xxs", "gguf_iq4_xs", "gguf_q6_k", "gguf_q8_0"),
    ),
)
def test_laguna_q2_xl_actual_sparse_layer_matches_quant_oracle(
    layer_id: int,
    gate_quant: str,
    down_quant: str,
    shared_gate_quant: str,
    shared_down_quant: str,
) -> None:
    runtime = _runtime()
    reader = GGUFReader(LAGUNA_Q2_MODEL)
    slots = tuple(
        f"layers.{layer_id}.{slot}"
        for slot in (
            "ffn_gate_inp",
            "exp_probs_b",
            "ffn_gate_exps",
            "ffn_up_exps",
            "ffn_down_exps",
            "ffn_gate_shexp",
            "ffn_up_shexp",
            "ffn_down_shexp",
        )
    )
    resident = None
    scratch = None
    bulk_scratch = None
    hidden_buffer = None
    bulk_hidden_buffer = None
    try:
        resident = materialize_laguna_gguf_weights(
            reader,
            selected_slots=slots,
            context_length=4_096,
            available_bytes=runtime.mem_get_info()[0],
            safety_reserve_nbytes=4 * 2**30,
            backend="hip_gfx1100",
            runtime=runtime,
        )
        plan = resolve_laguna_moe_plan(resident.config, backend="hip_gfx1100")
        layer = resident.layer(layer_id)
        validate_laguna_moe_layer(layer, plan)
        assert layer.weight("ffn_gate_exps").spec.quant_key == gate_quant
        assert layer.weight("ffn_down_exps").spec.quant_key == down_quant
        assert layer.weight("ffn_gate_shexp").spec.quant_key == shared_gate_quant
        assert layer.weight("ffn_down_shexp").spec.quant_key == shared_down_quant

        h, k = plan.hidden_size, plan.top_k
        rng = np.random.default_rng(233 + layer_id)
        hidden_bits = _f32_to_bf16_u16(
            rng.normal(0.0, 2.0e-4, size=(1, h)).astype(np.float32)
        )
        hidden = _bf16_u16_to_f32(hidden_bits)
        hidden_buffer = malloc(hidden_bits.nbytes, runtime=runtime)
        copy_host_to_device(
            hidden_buffer,
            host_array_ptr(hidden_bits),
            hidden_bits.nbytes,
            runtime=runtime,
        )
        scratch = allocate_laguna_moe_scratch(plan, runtime=runtime)
        output_buffer = run_laguna_moe_c1(
            hidden_buffer.ptr,
            layer,
            scratch,
            runtime=runtime,
        )
        runtime.device_synchronize()

        selected = _read_array(scratch.selected_experts, np.int64, (k,))
        scaled = _read_array(scratch.scaled_routing_weights, np.float32, (k,))
        actual = _read_bf16(output_buffer, (1, h))
        routed_actual = _read_bf16(scratch.routed_output, (1, h))
        shared_actual = _read_bf16(scratch.shared_output, (1, h))

        gate_tensor = reader.tensor_info(f"blk.{layer_id}.ffn_gate_exps.weight")
        up_tensor = reader.tensor_info(f"blk.{layer_id}.ffn_up_exps.weight")
        down_tensor = reader.tensor_info(f"blk.{layer_id}.ffn_down_exps.weight")
        gate_experts = np.asarray(reader.tensor_data(gate_tensor.name))
        up_experts = np.asarray(reader.tensor_data(up_tensor.name))
        down_experts = np.asarray(reader.tensor_data(down_tensor.name))
        route_outputs = np.empty((k, h), dtype=np.float32)
        for route, expert in enumerate(selected.tolist()):
            gate = _bf16_round(
                gguf_quant_gemv(
                    hidden,
                    gate_experts[expert],
                    GGMLQuantizationType(gate_tensor.ggml_type),
                )
            )
            up = _bf16_round(
                gguf_quant_gemv(
                    hidden,
                    up_experts[expert],
                    GGMLQuantizationType(up_tensor.ggml_type),
                )
            )
            intermediate = _bf16_round(_silu(gate) * up)
            route_outputs[route] = _bf16_round(
                gguf_quant_gemv(
                    intermediate,
                    down_experts[expert],
                    GGMLQuantizationType(down_tensor.ggml_type),
                )
            )[0]
        routed_expected = _bf16_round(
            np.sum(route_outputs * scaled[:, None], axis=0, dtype=np.float32)[None, :]
        )

        shared_gate_tensor = reader.tensor_info(f"blk.{layer_id}.ffn_gate_shexp.weight")
        shared_up_tensor = reader.tensor_info(f"blk.{layer_id}.ffn_up_shexp.weight")
        shared_down_tensor = reader.tensor_info(f"blk.{layer_id}.ffn_down_shexp.weight")
        shared_gate = _bf16_round(
            gguf_quant_gemv(
                hidden,
                np.asarray(reader.tensor_data(shared_gate_tensor.name)),
                GGMLQuantizationType(shared_gate_tensor.ggml_type),
            )
        )
        shared_up = _bf16_round(
            gguf_quant_gemv(
                hidden,
                np.asarray(reader.tensor_data(shared_up_tensor.name)),
                GGMLQuantizationType(shared_up_tensor.ggml_type),
            )
        )
        shared_intermediate = _bf16_round(_silu(shared_gate) * shared_up)
        shared_expected = _bf16_round(
            gguf_quant_gemv(
                shared_intermediate,
                np.asarray(reader.tensor_data(shared_down_tensor.name)),
                GGMLQuantizationType(shared_down_tensor.ggml_type),
            )
        )
        expected = _bf16_round(routed_expected + shared_expected)
        for candidate, reference in (
            (routed_actual, routed_expected),
            (shared_actual, shared_expected),
            (actual, expected),
        ):
            relative_l2 = float(
                np.linalg.norm(candidate.astype(np.float64) - reference.astype(np.float64))
                / max(np.linalg.norm(reference.astype(np.float64)), 1.0e-12)
            )
            assert relative_l2 <= 0.02
        assert np.isfinite(actual).all()

        bulk_hidden_bits = np.concatenate(
            (
                hidden_bits,
                _f32_to_bf16_u16(hidden * np.float32(0.75)),
                _f32_to_bf16_u16(hidden * np.float32(-0.5)),
            ),
            axis=0,
        )
        bulk_hidden_buffer = malloc(bulk_hidden_bits.nbytes, runtime=runtime)
        copy_host_to_device(
            bulk_hidden_buffer,
            host_array_ptr(bulk_hidden_bits),
            bulk_hidden_bits.nbytes,
            runtime=runtime,
        )
        bulk_scratch = allocate_laguna_moe_scratch(plan, max_rows=3, runtime=runtime)
        bulk_output = run_laguna_moe_rows(
            bulk_hidden_buffer.ptr,
            layer,
            bulk_scratch,
            rows=3,
            runtime=runtime,
        )
        runtime.device_synchronize()
        bulk_actual = _read_bf16(bulk_output, (3, h))
        serial_actual = np.empty_like(bulk_actual)
        for row in range(3):
            copy_host_to_device(
                hidden_buffer,
                host_array_ptr(bulk_hidden_bits[row]),
                bulk_hidden_bits[row].nbytes,
                runtime=runtime,
            )
            serial_output = run_laguna_moe_c1(
                hidden_buffer.ptr,
                layer,
                scratch,
                runtime=runtime,
            )
            runtime.device_synchronize()
            serial_actual[row] = _read_bf16(serial_output, (1, h))[0]
        np.testing.assert_array_equal(
            _f32_to_bf16_u16(bulk_actual),
            _f32_to_bf16_u16(serial_actual),
        )
    finally:
        if bulk_scratch is not None:
            bulk_scratch.free(runtime=runtime)
        if scratch is not None:
            scratch.free(runtime=runtime)
        if bulk_hidden_buffer is not None:
            free(bulk_hidden_buffer, runtime=runtime)
        if hidden_buffer is not None:
            free(hidden_buffer, runtime=runtime)
        if resident is not None:
            resident.free(runtime=runtime)


class _ArrayReader:
    def __init__(self, name: str, array: np.ndarray) -> None:
        self.name = name
        self.array = array

    def tensor_data(self, name: str) -> np.ndarray:
        assert name == self.name
        return self.array


def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()
