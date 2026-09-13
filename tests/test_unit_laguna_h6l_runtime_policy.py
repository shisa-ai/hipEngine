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

_CAPABILITY = "LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANTS"
_ABI_CAPABILITY = "LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANT_ABIS"
_QUANT = "gguf_iq2_xs"
_ABI = "grouped_raw_iq_dual_silu"
_CONTROL_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_pair16_rowbatch8_bf16_bf16_out"
)
_H6L_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_k3072_n1024_e256_pair16_"
    "rowbatch16_bf16_bf16_out"
)
_H6C_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_rowbatch4_bf16_bf16_out"
)
_H6T_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_dpp_peer_exchange_fused_add_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_PRODUCTION_MOE_SCRATCH_BYTES = 104_370_976
_PRODUCTION_WORKSPACE_BYTES = 161_120_256
_PRODUCTION_TOTAL_SCRATCH_BYTES = 600_142_624


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


def test_h6l_runtime_capability_is_source_default_bounded_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    source_variants = {_QUANT: _H6L_VARIANT}
    rollback_variants = {_QUANT: _CONTROL_VARIANT}
    qualified_abis = {
        _CONTROL_VARIANT: _ABI,
        _H6L_VARIANT: _ABI,
    }

    assert getattr(hip_gfx1100, _CAPABILITY) == source_variants
    assert getattr(hip_gfx1100, _ABI_CAPABILITY) == qualified_abis
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert not hasattr(hip_gfx1151, _ABI_CAPABILITY)

    package_default = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert package_default.grouped_pair16_gate_up_keys[_QUANT].variant == (
        _H6L_VARIANT
    )
    package_route = package_default.grouped_pair16_gate_up_routes[_QUANT]
    assert package_route.abi == _ABI
    assert package_route.allocation_name == "raw"
    assert package_route.library_key == "grouped_iq_prefill"
    assert package_default.grouped_special_gate_up_keys[(47, "gguf_iq3_xxs")].variant == (
        _H6C_VARIANT
    )
    assert package_default.grouped_exact_down_keys["gguf_iq3_xxs"].variant == (
        _H6T_VARIANT
    )
    package_moe_scratch = laguna_moe_scratch_nbytes(package_default, max_rows=512)
    package_scratch = _prefill_scratch(config, package_default)
    assert package_moe_scratch == _PRODUCTION_MOE_SCRATCH_BYTES
    assert package_scratch.q5_f32_ordered_nbytes == _PRODUCTION_WORKSPACE_BYTES
    assert package_scratch.total_nbytes == _PRODUCTION_TOTAL_SCRATCH_BYTES

    monkeypatch.setattr(hip_gfx1100, _CAPABILITY, rollback_variants)
    rollback = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert rollback.grouped_pair16_gate_up_keys[_QUANT].variant == _CONTROL_VARIANT
    rollback_route = rollback.grouped_pair16_gate_up_routes[_QUANT]
    assert rollback_route.abi == _ABI
    assert rollback_route.allocation_name == "raw"
    assert rollback_route.library_key == "grouped_iq_prefill"
    assert rollback.grouped_special_gate_up_keys[(47, "gguf_iq3_xxs")].variant == (
        _H6C_VARIANT
    )
    assert rollback.grouped_exact_down_keys["gguf_iq3_xxs"].variant == _H6T_VARIANT
    assert laguna_moe_scratch_nbytes(rollback, max_rows=512) == package_moe_scratch
    rollback_scratch = _prefill_scratch(config, rollback)
    assert rollback_scratch.q5_f32_ordered_nbytes == package_scratch.q5_f32_ordered_nbytes
    assert rollback_scratch.total_nbytes == package_scratch.total_nbytes
    assert len(rollback.kernel_keys) == len(package_default.kernel_keys)

    monkeypatch.setattr(hip_gfx1100, _CAPABILITY, source_variants)
    for wrong_config in (
        replace(config, hidden_size=1024),
        replace(config, expert_feed_forward_length=2048),
        replace(config, expert_count=255),
    ):
        wrong_shape = resolve_laguna_moe_plan(
            wrong_config,
            backend="hip_gfx1100",
        )
        assert wrong_shape.grouped_pair16_gate_up_keys[_QUANT].variant == (
            _CONTROL_VARIANT
        )
        wrong_route = wrong_shape.grouped_pair16_gate_up_routes[_QUANT]
        assert wrong_route.abi == _ABI
        assert wrong_route.allocation_name == "raw"
        assert wrong_route.library_key == "grouped_iq_prefill"

    original_is_registered = laguna_moe_module.is_registered
    monkeypatch.setattr(
        laguna_moe_module,
        "is_registered",
        lambda key: key.variant != _H6L_VARIANT and original_is_registered(key),
    )
    registration_miss = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert registration_miss.grouped_pair16_gate_up_keys[_QUANT].variant == (
        _CONTROL_VARIANT
    )
    assert registration_miss.grouped_pair16_gate_up_routes[_QUANT].abi == _ABI

    gfx1151 = resolve_laguna_moe_plan(config, backend="hip_gfx1151")
    assert gfx1151.grouped_pair16_gate_up_keys[_QUANT].variant == _CONTROL_VARIANT
    gfx1151_route = gfx1151.grouped_pair16_gate_up_routes[_QUANT]
    assert gfx1151_route.abi == _ABI
    assert gfx1151_route.allocation_name == "raw"
    assert gfx1151_route.library_key == "grouped_iq_prefill"

    monkeypatch.setattr(laguna_moe_module, "is_registered", original_is_registered)
    for malformed, message in (
        (17, "must be a mapping"),
        ({"unknown": _H6L_VARIANT}, "unsupported quant"),
        ({_QUANT: ""}, "non-empty variants"),
    ):
        monkeypatch.setattr(hip_gfx1100, _CAPABILITY, malformed)
        with pytest.raises(ValueError, match=message):
            resolve_laguna_moe_plan(config, backend="hip_gfx1100")

    monkeypatch.setattr(hip_gfx1100, _CAPABILITY, source_variants)
    monkeypatch.setattr(
        hip_gfx1100,
        _ABI_CAPABILITY,
        {_CONTROL_VARIANT: _ABI, _H6L_VARIANT: "unsupported"},
    )
    with pytest.raises(ValueError, match="unsupported variant ABI"):
        resolve_laguna_moe_plan(config, backend="hip_gfx1100")
