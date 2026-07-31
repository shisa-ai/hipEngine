from __future__ import annotations

from dataclasses import replace

import hipengine.runtime.laguna_moe as laguna_moe_module
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime.laguna_moe import (
    laguna_moe_scratch_nbytes,
    resolve_laguna_moe_plan,
)
from tests._laguna_synthetic import make_laguna_info


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
_ACTIVE_EXPERT_ABI = "grouped_raw_iq_active_experts"
_H6C_ROLE = "layer47_iq3_k3072_n1024_e256"
_H6C_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_rowbatch4_bf16_bf16_out"
)
_PRODUCTION_MOE_SCRATCH_BYTES = 104_370_208


def test_h6f_source_default_is_retained_as_h6i_rollback_and_fail_closed(
    monkeypatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    rollback_variants = {
        "gguf_iq3_xxs": _H6F_IQ3,
        "gguf_iq4_xs": _H5J_IQ4,
    }
    production_variants = {**rollback_variants, "gguf_iq3_xxs": _H6I_IQ3}
    qualified_abis = {
        _H5Q_IQ3: _ACTIVE_EXPERT_ABI,
        _H5Z_IQ3: _ACTIVE_EXPERT_ABI,
        _H6D_IQ3: _ACTIVE_EXPERT_ABI,
        _H6F_IQ3: _ACTIVE_EXPERT_ABI,
        _H6I_IQ3: _ACTIVE_EXPERT_ABI,
        _H6P_IQ3: _ACTIVE_EXPERT_ABI,
    }

    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == production_variants
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == qualified_abis
    assert hip_gfx1100.LAGUNA_GROUPED_GATE_UP_ROLE_VARIANTS == {
        _H6C_ROLE: _H6C_VARIANT
    }
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    package_default = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    key = package_default.grouped_exact_down_keys["gguf_iq3_xxs"]
    route = package_default.grouped_exact_down_routes["gguf_iq3_xxs"]
    assert key.variant == _H6I_IQ3
    assert route.abi == _ACTIVE_EXPERT_ABI
    assert route.allocation_name == "raw"
    assert route.library_key == "grouped_iq_prefill"
    assert package_default.grouped_exact_down_keys["gguf_iq4_xs"].variant == (
        _H5J_IQ4
    )
    assert package_default.grouped_exact_down_routes["gguf_iq4_xs"].abi == (
        "grouped_raw_iq"
    )
    production_scratch = laguna_moe_scratch_nbytes(package_default, max_rows=512)
    assert production_scratch == _PRODUCTION_MOE_SCRATCH_BYTES

    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_GROUPED_IQ_DOWN_VARIANTS",
        rollback_variants,
    )
    rollback = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert rollback.grouped_exact_down_keys["gguf_iq3_xxs"].variant == _H6F_IQ3
    rollback_route = rollback.grouped_exact_down_routes["gguf_iq3_xxs"]
    assert rollback_route.abi == _ACTIVE_EXPERT_ABI
    assert rollback_route.allocation_name == "raw"
    assert rollback_route.library_key == "grouped_iq_prefill"
    assert laguna_moe_scratch_nbytes(rollback, max_rows=512) == production_scratch

    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_GROUPED_IQ_DOWN_VARIANTS",
        production_variants,
    )
    wrong_shape = resolve_laguna_moe_plan(
        replace(config, expert_feed_forward_length=2048),
        backend="hip_gfx1100",
    )
    assert wrong_shape.grouped_exact_down_keys["gguf_iq3_xxs"].variant == (
        _BASELINE_IQ3
    )
    assert wrong_shape.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        "grouped_raw_iq"
    )

    original_is_registered = laguna_moe_module.is_registered
    monkeypatch.setattr(
        laguna_moe_module,
        "is_registered",
        lambda key: key.variant != _H6I_IQ3 and original_is_registered(key),
    )
    registration_miss = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert registration_miss.grouped_exact_down_keys["gguf_iq3_xxs"].variant == (
        _BASELINE_IQ3
    )
    assert registration_miss.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        "grouped_raw_iq"
    )

    gfx1151 = resolve_laguna_moe_plan(config, backend="hip_gfx1151")
    assert gfx1151.grouped_exact_down_keys["gguf_iq3_xxs"].variant == _BASELINE_IQ3
    assert gfx1151.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        "grouped_raw_iq"
    )
