from __future__ import annotations

from dataclasses import replace

import hipengine.runtime.laguna_moe as laguna_moe_module
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime.laguna_moe import resolve_laguna_moe_plan
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


def test_h5z_source_default_is_retained_under_h6f_source_and_fail_closed(
    monkeypatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    h5q_variants = {
        "gguf_iq3_xxs": _H5Q_IQ3,
        "gguf_iq4_xs": _H5J_IQ4,
    }
    h5z_variants = {**h5q_variants, "gguf_iq3_xxs": _H5Z_IQ3}
    production_variants = {**h5q_variants, "gguf_iq3_xxs": _H6I_IQ3}
    production_abis = {
        _H5Q_IQ3: _ACTIVE_EXPERT_ABI,
        _H5Z_IQ3: _ACTIVE_EXPERT_ABI,
        _H6D_IQ3: _ACTIVE_EXPERT_ABI,
        _H6F_IQ3: _ACTIVE_EXPERT_ABI,
        _H6I_IQ3: _ACTIVE_EXPERT_ABI,
        _H6P_IQ3: _ACTIVE_EXPERT_ABI,
    }

    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == production_variants
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == production_abis
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    package_default = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert package_default.grouped_exact_down_keys["gguf_iq3_xxs"].variant == (
        _H6I_IQ3
    )
    assert package_default.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        _ACTIVE_EXPERT_ABI
    )
    assert package_default.grouped_exact_down_keys["gguf_iq4_xs"].variant == (
        _H5J_IQ4
    )
    assert package_default.grouped_exact_down_routes["gguf_iq4_xs"].abi == (
        "grouped_raw_iq"
    )

    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_GROUPED_IQ_DOWN_VARIANTS",
        h5z_variants,
    )
    rollback = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert rollback.grouped_exact_down_keys["gguf_iq3_xxs"].variant == _H5Z_IQ3
    assert rollback.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        _ACTIVE_EXPERT_ABI
    )

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
    assert gfx1151.grouped_exact_down_keys["gguf_iq3_xxs"].variant == (
        _BASELINE_IQ3
    )
    assert gfx1151.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        "grouped_raw_iq"
    )
