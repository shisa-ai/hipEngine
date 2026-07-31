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


_CAPABILITY = "LAGUNA_GROUPED_GATE_UP_ROLE_VARIANTS"
_ABI_CAPABILITY = "LAGUNA_GROUPED_GATE_UP_VARIANT_ABIS"
_ROLE = "layer47_iq3_k3072_n1024_e256"
_QUANT = "gguf_iq3_xxs"
_ROUTE_KEY = (47, _QUANT)
_H6C_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_rowbatch4_bf16_bf16_out"
)
_H6C_ABI = "grouped_raw_iq_dual_silu"
_H6Q_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_compact_shuffle_loop_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H5J_VARIANT = "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out"
_IQ2_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_pair16_rowbatch16_bf16_bf16_out"
)
_PRODUCTION_ROLES = {_ROLE: _H6C_VARIANT}


def test_h6c_source_default_promotes_only_layer47_iq3_role(
    monkeypatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())

    assert getattr(hip_gfx1100, _CAPABILITY) == _PRODUCTION_ROLES
    assert getattr(hip_gfx1100, _ABI_CAPABILITY) == {_H6C_VARIANT: _H6C_ABI}
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert not hasattr(hip_gfx1151, _ABI_CAPABILITY)

    package_default = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    route = package_default.grouped_special_gate_up_routes[_ROUTE_KEY]
    assert package_default.grouped_special_gate_up_keys[_ROUTE_KEY].variant == (
        _H6C_VARIANT
    )
    assert route.abi == _H6C_ABI
    assert route.allocation_name == "raw"
    assert route.library_key == "grouped_iq_prefill"
    assert package_default.grouped_exact_down_keys[_QUANT].variant == _H6Q_VARIANT
    assert package_default.grouped_exact_down_keys["gguf_iq4_xs"].variant == (
        _H5J_VARIANT
    )
    assert package_default.grouped_pair16_gate_up_keys["gguf_iq2_xs"].variant == (
        _IQ2_VARIANT
    )
    production_scratch = laguna_moe_scratch_nbytes(package_default, max_rows=512)

    monkeypatch.setattr(hip_gfx1100, _CAPABILITY, {})
    rollback = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert rollback.grouped_special_gate_up_keys == {}
    assert rollback.grouped_special_gate_up_routes == {}
    assert rollback.grouped_exact_down_keys[_QUANT].variant == _H6Q_VARIANT
    assert rollback.grouped_pair16_gate_up_keys["gguf_iq2_xs"].variant == _IQ2_VARIANT
    assert laguna_moe_scratch_nbytes(rollback, max_rows=512) == production_scratch

    monkeypatch.setattr(hip_gfx1100, _CAPABILITY, _PRODUCTION_ROLES)
    wrong_shape = resolve_laguna_moe_plan(
        replace(config, expert_feed_forward_length=2048),
        backend="hip_gfx1100",
    )
    assert wrong_shape.grouped_special_gate_up_routes == {}

    original_is_registered = laguna_moe_module.is_registered
    monkeypatch.setattr(
        laguna_moe_module,
        "is_registered",
        lambda key: key.variant != _H6C_VARIANT and original_is_registered(key),
    )
    registration_miss = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert registration_miss.grouped_special_gate_up_routes == {}

    gfx1151 = resolve_laguna_moe_plan(config, backend="hip_gfx1151")
    assert gfx1151.grouped_special_gate_up_routes == {}
