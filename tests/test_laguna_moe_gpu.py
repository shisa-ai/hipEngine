from __future__ import annotations

import ctypes
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.loading.laguna_gguf import (
    SLIDING_ATTENTION,
    SPARSE_MOE,
    laguna_gguf_config_from_metadata,
)
from hipengine.loading.laguna_gguf_materialize import (
    LagunaGGUFResidentLayerWeights,
    _materialize_spec,
    _spec_for_tensor,
)
from hipengine.models.laguna import LAGUNA_GGUF
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime.laguna_moe import (
    allocate_laguna_moe_scratch,
    resolve_laguna_moe_plan,
    run_laguna_moe_c1,
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
    assert plan.router_select_key.layer == "laguna_sigmoid_router_topk"
    assert plan.selected_gate_up_key.quant == "gguf_q4_k_t16_v1"
    assert plan.selected_down_key.quant == "gguf_q6_k_t16_v1"
    assert all(key.backend == "hip_gfx1151" for key in plan.kernel_keys)

    sparse_sequence = LAGUNA_GGUF.decode_layer_sequence(
        attention_kind="sliding_attention",
        mlp_kind="sparse_moe",
    )
    assert "laguna_sigmoid_router_topk" in sparse_sequence
    assert "selected_expert_mlp" in sparse_sequence
    assert "laguna_shared_expert" in sparse_sequence
    assert "laguna_routed_shared_combine" in sparse_sequence


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
def test_laguna_unfused_moe_matches_production_shape_quant_oracle() -> None:
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
    q6_base = make_q6_k_weight(h, f)
    gate_experts = np.stack([np.roll(q4_base, 7 * expert, axis=0) for expert in range(e)], axis=0)
    up_experts = np.stack(
        [np.roll(q4_base, 11 * expert + 3, axis=0) for expert in range(e)], axis=0
    )
    down_experts = np.stack(
        [np.roll(q6_base, 13 * expert + 5, axis=0) for expert in range(e)], axis=0
    )
    shared_gate = np.roll(q4_base, 17, axis=0).copy()
    shared_up = np.roll(q4_base, 29, axis=0).copy()
    shared_down = np.roll(q6_base, 31, axis=0).copy()
    payloads = {
        "ffn_gate_inp": (router, GGMLQuantizationType.F32, (e, h)),
        "exp_probs_b": (correction, GGMLQuantizationType.F32, (e,)),
        "ffn_gate_exps": (gate_experts, GGMLQuantizationType.Q4_K, (e, f, h)),
        "ffn_up_exps": (up_experts, GGMLQuantizationType.Q4_K, (e, f, h)),
        "ffn_down_exps": (down_experts, GGMLQuantizationType.Q6_K, (e, h, f)),
        "ffn_gate_shexp": (shared_gate, GGMLQuantizationType.Q4_K, (f, h)),
        "ffn_up_shexp": (shared_up, GGMLQuantizationType.Q4_K, (f, h)),
        "ffn_down_shexp": (shared_down, GGMLQuantizationType.Q6_K, (h, f)),
    }

    resident = {}
    scratch = None
    hidden_buffer = None
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
        assert layer.weight("ffn_down_exps").spec.source.byte_shape == (
            e,
            h,
            (f // 256) * 210,
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
            3_360,
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
                    GGMLQuantizationType.Q6_K,
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
                GGMLQuantizationType.Q6_K,
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
    finally:
        if scratch is not None:
            scratch.free()
        if hidden_buffer is not None:
            free(hidden_buffer)
        for weight in reversed(tuple(resident.values())):
            weight.free()


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
