"""WPF-H6Q source-default promotion contract."""

from __future__ import annotations

from dataclasses import replace

import pytest

import hipengine.runtime.laguna_moe as laguna_moe_module
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime.laguna_gguf_runner import (
    LagunaPrefillChunkPolicy,
    LagunaPrefillScratchPlan,
)
from hipengine.runtime.laguna_moe import (
    laguna_moe_scratch_nbytes,
    resolve_laguna_moe_plan,
)
from tests._laguna_synthetic import make_laguna_info

_QUANT = "gguf_iq3_xxs"
_BASELINE_IQ3 = "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out"
_H5J_IQ4 = "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out"
_H5Q_IQ3 = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "resident_rowbatch8_bf16_bf16_out"
)
_H5Z_IQ3 = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_rowbatch8_bf16_bf16_out"
)
_H6D_IQ3 = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out"
)
_H6F_IQ3 = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_paired_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6I_IQ3 = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6P_IQ3 = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_triple_output_rowbatch8_bf16_bf16_out"
)
_H6Q_IQ3 = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_compact_shuffle_loop_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6C_IQ3_GATE_UP = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_rowbatch4_bf16_bf16_out"
)
_H6L_IQ2_GATE_UP = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_pair16_rowbatch16_bf16_bf16_out"
)
_ACTIVE_EXPERT_ABI = "grouped_raw_iq_active_experts"
_PRODUCTION_MOE_SCRATCH_BYTES = 104_370_208
_PRODUCTION_WORKSPACE_BYTES = 161_120_256
_PRODUCTION_TOTAL_SCRATCH_BYTES = 600_141_856


def _prefill_scratch(config, plan) -> LagunaPrefillScratchPlan:
    return LagunaPrefillScratchPlan.build(
        config,
        plan,
        policy=LagunaPrefillChunkPolicy.resolve(
            context_length=4096,
            matrix_rows=512,
            attention_rows=128,
        ),
        use_q5_f32_ordered=True,
        use_q5_activation_tile_k_row=True,
    )


def test_h6q_source_default_promotes_only_iq3_selected_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    rollback_variants = {
        _QUANT: _H6P_IQ3,
        "gguf_iq4_xs": _H5J_IQ4,
    }
    production_variants = {**rollback_variants, _QUANT: _H6Q_IQ3}
    qualified_abis = {
        _H5Q_IQ3: _ACTIVE_EXPERT_ABI,
        _H5Z_IQ3: _ACTIVE_EXPERT_ABI,
        _H6D_IQ3: _ACTIVE_EXPERT_ABI,
        _H6F_IQ3: _ACTIVE_EXPERT_ABI,
        _H6I_IQ3: _ACTIVE_EXPERT_ABI,
        _H6P_IQ3: _ACTIVE_EXPERT_ABI,
        _H6Q_IQ3: _ACTIVE_EXPERT_ABI,
    }

    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == production_variants
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == qualified_abis
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    package_default = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert package_default.grouped_exact_down_keys[_QUANT].variant == _H6Q_IQ3
    package_route = package_default.grouped_exact_down_routes[_QUANT]
    assert package_route.abi == _ACTIVE_EXPERT_ABI
    assert package_route.allocation_name == "raw"
    assert package_route.library_key == "grouped_iq_prefill"
    assert package_default.grouped_exact_down_keys["gguf_iq4_xs"].variant == (
        _H5J_IQ4
    )
    assert package_default.grouped_special_gate_up_keys[
        (47, "gguf_iq3_xxs")
    ].variant == _H6C_IQ3_GATE_UP
    assert package_default.grouped_pair16_gate_up_keys[
        "gguf_iq2_xs"
    ].variant == _H6L_IQ2_GATE_UP
    package_moe_scratch = laguna_moe_scratch_nbytes(
        package_default,
        max_rows=512,
    )
    package_scratch = _prefill_scratch(config, package_default)
    assert package_moe_scratch == _PRODUCTION_MOE_SCRATCH_BYTES
    assert package_scratch.q5_f32_ordered_nbytes == _PRODUCTION_WORKSPACE_BYTES
    assert package_scratch.total_nbytes == _PRODUCTION_TOTAL_SCRATCH_BYTES

    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_GROUPED_IQ_DOWN_VARIANTS",
        rollback_variants,
    )
    rollback = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert rollback.grouped_exact_down_keys[_QUANT].variant == _H6P_IQ3
    rollback_route = rollback.grouped_exact_down_routes[_QUANT]
    assert rollback_route.abi == _ACTIVE_EXPERT_ABI
    assert rollback_route.allocation_name == "raw"
    assert rollback_route.library_key == "grouped_iq_prefill"
    assert rollback.grouped_exact_down_keys["gguf_iq4_xs"].variant == _H5J_IQ4
    assert rollback.grouped_special_gate_up_keys[
        (47, "gguf_iq3_xxs")
    ].variant == _H6C_IQ3_GATE_UP
    assert rollback.grouped_pair16_gate_up_keys[
        "gguf_iq2_xs"
    ].variant == _H6L_IQ2_GATE_UP
    assert laguna_moe_scratch_nbytes(rollback, max_rows=512) == package_moe_scratch
    rollback_scratch = _prefill_scratch(config, rollback)
    assert (
        rollback_scratch.q5_f32_ordered_nbytes
        == package_scratch.q5_f32_ordered_nbytes
    )
    assert rollback_scratch.total_nbytes == package_scratch.total_nbytes
    assert len(rollback.kernel_keys) == len(package_default.kernel_keys)

    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_GROUPED_IQ_DOWN_VARIANTS",
        production_variants,
    )
    for wrong_config in (
        replace(config, hidden_size=1024),
        replace(config, expert_feed_forward_length=2048),
        replace(config, expert_count=255),
    ):
        wrong_shape = resolve_laguna_moe_plan(
            wrong_config,
            backend="hip_gfx1100",
        )
        assert wrong_shape.grouped_exact_down_keys[_QUANT].variant == _BASELINE_IQ3
        wrong_route = wrong_shape.grouped_exact_down_routes[_QUANT]
        assert wrong_route.abi == "grouped_raw_iq"
        assert wrong_route.allocation_name == "raw"
        assert wrong_route.library_key == "grouped_iq_prefill"

    original_is_registered = laguna_moe_module.is_registered
    monkeypatch.setattr(
        laguna_moe_module,
        "is_registered",
        lambda key: key.variant != _H6Q_IQ3 and original_is_registered(key),
    )
    registration_miss = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert registration_miss.grouped_exact_down_keys[_QUANT].variant == _BASELINE_IQ3
    assert registration_miss.grouped_exact_down_routes[_QUANT].abi == "grouped_raw_iq"

    gfx1151 = resolve_laguna_moe_plan(config, backend="hip_gfx1151")
    assert gfx1151.grouped_exact_down_keys[_QUANT].variant == _BASELINE_IQ3
    gfx1151_route = gfx1151.grouped_exact_down_routes[_QUANT]
    assert gfx1151_route.abi == "grouped_raw_iq"
    assert gfx1151_route.allocation_name == "raw"
    assert gfx1151_route.library_key == "grouped_iq_prefill"
