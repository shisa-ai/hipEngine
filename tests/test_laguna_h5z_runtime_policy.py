from __future__ import annotations

from dataclasses import replace

import hipengine.runtime.laguna_moe as laguna_moe_module
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime.laguna_moe import resolve_laguna_moe_plan
from tests._laguna_synthetic import make_laguna_info


_H5J_IQ3 = (
    "selected_grouped_prefill_compact_k1024_resident_"
    "rowbatch8_bf16_bf16_out"
)
_H5J_IQ4 = "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out"
_H5Q_IQ3 = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "resident_rowbatch8_bf16_bf16_out"
)
_H5Z_IQ3 = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_rowbatch8_bf16_bf16_out"
)
_ACTIVE_EXPERT_ABI = "grouped_raw_iq_active_experts"


def test_h5z_runtime_capability_is_default_off_bounded_and_fail_closed(
    monkeypatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    production_variants = {
        "gguf_iq3_xxs": _H5Q_IQ3,
        "gguf_iq4_xs": _H5J_IQ4,
    }
    candidate_variants = {**production_variants, "gguf_iq3_xxs": _H5Z_IQ3}
    qualified_abis = {
        _H5Q_IQ3: _ACTIVE_EXPERT_ABI,
        _H5Z_IQ3: _ACTIVE_EXPERT_ABI,
    }

    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == production_variants
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == qualified_abis
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    package_default = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert package_default.grouped_exact_down_keys["gguf_iq3_xxs"].variant == (
        _H5Q_IQ3
    )
    assert package_default.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        _ACTIVE_EXPERT_ABI
    )

    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_GROUPED_IQ_DOWN_VARIANTS",
        candidate_variants,
    )
    candidate = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert candidate.grouped_exact_down_keys["gguf_iq3_xxs"].variant == _H5Z_IQ3
    assert candidate.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        _ACTIVE_EXPERT_ABI
    )
    assert candidate.grouped_exact_down_keys["gguf_iq4_xs"].variant == _H5J_IQ4
    assert candidate.grouped_exact_down_routes["gguf_iq4_xs"].abi == (
        "grouped_raw_iq"
    )

    wrong_shape = resolve_laguna_moe_plan(
        replace(config, expert_feed_forward_length=2048),
        backend="hip_gfx1100",
    )
    assert wrong_shape.grouped_exact_down_keys["gguf_iq3_xxs"].variant == _H5J_IQ3
    assert wrong_shape.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        "grouped_raw_iq"
    )

    original_is_registered = laguna_moe_module.is_registered
    monkeypatch.setattr(
        laguna_moe_module,
        "is_registered",
        lambda key: key.variant != _H5Z_IQ3 and original_is_registered(key),
    )
    registration_miss = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert registration_miss.grouped_exact_down_keys["gguf_iq3_xxs"].variant == (
        _H5J_IQ3
    )
    assert registration_miss.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        "grouped_raw_iq"
    )

    gfx1151 = resolve_laguna_moe_plan(config, backend="hip_gfx1151")
    assert gfx1151.grouped_exact_down_keys["gguf_iq3_xxs"].variant == _H5J_IQ3
    assert gfx1151.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        "grouped_raw_iq"
    )
