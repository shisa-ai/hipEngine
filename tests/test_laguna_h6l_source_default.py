from __future__ import annotations

from dataclasses import replace

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

_CAPABILITY = "LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANTS"
_ABI_CAPABILITY = "LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANT_ABIS"
_QUANT = "gguf_iq2_xs"
_ABI = "grouped_raw_iq_dual_silu"
_ROLLBACK_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "pair16_rowbatch8_bf16_bf16_out"
)
_H6L_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_pair16_rowbatch16_bf16_bf16_out"
)
_H6C_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_rowbatch4_bf16_bf16_out"
)
_H6I_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
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


def test_h6l_source_default_promotes_only_iq2_pair16_gate_up(
    monkeypatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    rollback_variants = {_QUANT: _ROLLBACK_VARIANT}
    production_variants = {_QUANT: _H6L_VARIANT}
    qualified_abis = {
        _ROLLBACK_VARIANT: _ABI,
        _H6L_VARIANT: _ABI,
    }

    assert getattr(hip_gfx1100, _CAPABILITY) == production_variants
    assert getattr(hip_gfx1100, _ABI_CAPABILITY) == qualified_abis
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert not hasattr(hip_gfx1151, _ABI_CAPABILITY)

    package_default = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    key = package_default.grouped_pair16_gate_up_keys[_QUANT]
    route = package_default.grouped_pair16_gate_up_routes[_QUANT]
    assert key.variant == _H6L_VARIANT
    assert route.abi == _ABI
    assert route.allocation_name == "raw"
    assert route.library_key == "grouped_iq_prefill"
    assert package_default.grouped_special_gate_up_keys[(47, "gguf_iq3_xxs")].variant == (
        _H6C_VARIANT
    )
    assert package_default.grouped_exact_down_keys["gguf_iq3_xxs"].variant == (
        _H6I_VARIANT
    )
    production_moe_scratch = laguna_moe_scratch_nbytes(
        package_default,
        max_rows=512,
    )
    production_scratch = _prefill_scratch(config, package_default)
    assert production_moe_scratch == _PRODUCTION_MOE_SCRATCH_BYTES
    assert production_scratch.q5_f32_ordered_nbytes == _PRODUCTION_WORKSPACE_BYTES
    assert production_scratch.total_nbytes == _PRODUCTION_TOTAL_SCRATCH_BYTES

    monkeypatch.setattr(hip_gfx1100, _CAPABILITY, rollback_variants)
    rollback = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert rollback.grouped_pair16_gate_up_keys[_QUANT].variant == _ROLLBACK_VARIANT
    rollback_route = rollback.grouped_pair16_gate_up_routes[_QUANT]
    assert rollback_route.abi == _ABI
    assert rollback_route.allocation_name == "raw"
    assert rollback_route.library_key == "grouped_iq_prefill"
    assert rollback.grouped_special_gate_up_keys[(47, "gguf_iq3_xxs")].variant == (
        _H6C_VARIANT
    )
    assert rollback.grouped_exact_down_keys["gguf_iq3_xxs"].variant == _H6I_VARIANT
    assert laguna_moe_scratch_nbytes(rollback, max_rows=512) == production_moe_scratch
    rollback_scratch = _prefill_scratch(config, rollback)
    assert rollback_scratch.q5_f32_ordered_nbytes == production_scratch.q5_f32_ordered_nbytes
    assert rollback_scratch.total_nbytes == production_scratch.total_nbytes
    assert len(rollback.kernel_keys) == len(package_default.kernel_keys)

    monkeypatch.setattr(hip_gfx1100, _CAPABILITY, production_variants)
    for wrong_config in (
        replace(config, hidden_size=1024),
        replace(config, expert_feed_forward_length=2048),
        replace(config, expert_count=255),
    ):
        wrong_shape = resolve_laguna_moe_plan(wrong_config, backend="hip_gfx1100")
        assert wrong_shape.grouped_pair16_gate_up_keys[_QUANT].variant == (
            _ROLLBACK_VARIANT
        )
        assert wrong_shape.grouped_pair16_gate_up_routes[_QUANT].abi == _ABI

    original_is_registered = laguna_moe_module.is_registered
    monkeypatch.setattr(
        laguna_moe_module,
        "is_registered",
        lambda candidate: (
            candidate.variant != _H6L_VARIANT
            and original_is_registered(candidate)
        ),
    )
    registration_miss = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert registration_miss.grouped_pair16_gate_up_keys[_QUANT].variant == (
        _ROLLBACK_VARIANT
    )
    assert registration_miss.grouped_pair16_gate_up_routes[_QUANT].abi == _ABI

    gfx1151 = resolve_laguna_moe_plan(config, backend="hip_gfx1151")
    assert gfx1151.grouped_pair16_gate_up_keys[_QUANT].variant == _ROLLBACK_VARIANT
    gfx1151_route = gfx1151.grouped_pair16_gate_up_routes[_QUANT]
    assert gfx1151_route.abi == _ABI
    assert gfx1151_route.allocation_name == "raw"
    assert gfx1151_route.library_key == "grouped_iq_prefill"
