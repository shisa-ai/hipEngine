from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

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
_H6F_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_paired_output_"
    "rowbatch8_bf16_bf16_out"
)
_IQ2_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "pair16_rowbatch8_bf16_bf16_out"
)


def _buffer(ptr: int) -> SimpleNamespace:
    return SimpleNamespace(ptr=ptr)


def _record(calls: list[tuple[str, tuple, dict]], name: str):
    return lambda *args, **kwargs: calls.append((name, args, kwargs))


def _weight(quant: str, ptr: int) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(quant_key=quant),
        allocation=lambda _name: SimpleNamespace(tensor=_buffer(ptr)),
    )


def _layer(
    layer_id: int,
    *,
    gate_quant: str,
    up_quant: str | None = None,
) -> SimpleNamespace:
    weights = {
        "ffn_down_exps": _weight(_QUANT, 20),
        "ffn_gate_exps": _weight(gate_quant, 21),
        "ffn_up_exps": _weight(up_quant or gate_quant, 22),
    }
    return SimpleNamespace(layer_id=layer_id, weight=weights.__getitem__)


def _dispatch_fixture(
    calls: list[tuple[str, tuple, dict]],
) -> tuple[SimpleNamespace, SimpleNamespace]:
    candidate_route = SimpleNamespace(
        function=_record(calls, "h6c"),
        allocation_name="raw",
        library_key="grouped_iq_prefill",
    )
    iq2_route = SimpleNamespace(
        function=_record(calls, "iq2"),
        allocation_name="raw",
        library_key="grouped_iq_prefill",
    )
    plan = SimpleNamespace(
        grouped_exact_down_routes={_QUANT: object()},
        grouped_special_gate_up_routes={_ROUTE_KEY: candidate_route},
        grouped_pair16_gate_up_routes={"gguf_iq2_xs": iq2_route},
        grouped_compact_source_rows=_record(calls, "compact"),
        grouped_compact_source_rows_parallel=_record(calls, "compact_parallel"),
        grouped_gather=_record(calls, "gather"),
        expert_count=256,
        top_k=10,
        hidden_size=3072,
        expert_ffn_size=1024,
    )
    scratch = SimpleNamespace(
        plan=plan,
        selected_experts=_buffer(1),
        scaled_routing_weights=_buffer(2),
        grouped_expert_start=_buffer(3),
        grouped_active_experts=_buffer(4),
        grouped_active_count=_buffer(5),
        grouped_sorted_lanes=_buffer(6),
        grouped_lane_to_row=_buffer(7),
        grouped_sorted_weights=_buffer(8),
        expert_down=_buffer(9),
        expert_gate=_buffer(10),
        expert_intermediate=_buffer(11),
    )
    return plan, scratch


def _launch(
    layer: SimpleNamespace,
    scratch: SimpleNamespace,
    *,
    tokens: int = 512,
) -> bool:
    return laguna_moe_module._launch_selected_gate_up_grouped_exact(
        13,
        layer,
        scratch,
        tokens=tokens,
        lanes=tokens * 10,
        group_compact_mode="serial",
        pair16=True,
        stream=14,
        runtime=SimpleNamespace(),
        libraries=None,
    )


def test_h6c_runtime_capability_is_source_default_bounded_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    production_roles = {_ROLE: _H6C_VARIANT}

    assert getattr(hip_gfx1100, _CAPABILITY) == production_roles
    assert getattr(hip_gfx1100, _ABI_CAPABILITY) == {_H6C_VARIANT: _H6C_ABI}
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert not hasattr(hip_gfx1151, _ABI_CAPABILITY)

    package_default = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert package_default.grouped_special_gate_up_keys[_ROUTE_KEY].variant == (
        _H6C_VARIANT
    )
    route = package_default.grouped_special_gate_up_routes[_ROUTE_KEY]
    assert route.abi == _H6C_ABI
    assert route.allocation_name == "raw"
    assert route.library_key == "grouped_iq_prefill"
    assert package_default.grouped_exact_down_keys[_QUANT].variant == _H6F_VARIANT
    assert package_default.grouped_pair16_gate_up_keys["gguf_iq2_xs"].variant == (
        _IQ2_VARIANT
    )

    monkeypatch.setattr(hip_gfx1100, _CAPABILITY, {})
    rollback = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert rollback.grouped_special_gate_up_keys == {}
    assert rollback.grouped_special_gate_up_routes == {}
    assert rollback.grouped_exact_down_keys[_QUANT].variant == _H6F_VARIANT
    assert rollback.grouped_pair16_gate_up_keys["gguf_iq2_xs"].variant == (
        _IQ2_VARIANT
    )

    monkeypatch.setattr(hip_gfx1100, _CAPABILITY, production_roles)
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


def test_h6c_runtime_dispatches_only_layer47_iq3_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    _, scratch = _dispatch_fixture(calls)
    monkeypatch.setattr(
        laguna_moe_module,
        "_launch_selected_gate_up",
        _record(calls, "route_major"),
    )

    assert _launch(_layer(47, gate_quant=_QUANT), scratch)
    assert [name for name, _, _ in calls] == ["compact", "gather", "h6c"]
    _, gather_args, _ = calls[1]
    assert gather_args[:3] == (13, 6, 9)
    assert gather_args[3:] == (5120 * 3072, 512, 10, 3072)
    _, candidate_args, candidate_kwargs = calls[2]
    assert candidate_args[:5] == (9, 3, 21, 22, 10)
    assert candidate_kwargs["compact_rows"] == 5120
    assert candidate_kwargs["in_features"] == 3072
    assert candidate_kwargs["out_features"] == 1024
    assert candidate_kwargs["num_experts"] == 256


def test_h6c_runtime_preserves_c1_other_layer_wrong_quant_and_iq2_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    _, scratch = _dispatch_fixture(calls)
    monkeypatch.setattr(
        laguna_moe_module,
        "_launch_selected_gate_up",
        _record(calls, "route_major"),
    )

    for layer, tokens, expected in (
        (_layer(47, gate_quant=_QUANT), 1, ["compact", "route_major", "gather"]),
        (_layer(46, gate_quant=_QUANT), 512, ["compact", "route_major", "gather"]),
        (
            _layer(47, gate_quant=_QUANT, up_quant="gguf_iq2_xs"),
            512,
            ["compact", "route_major", "gather"],
        ),
        (_layer(46, gate_quant="gguf_iq2_xs"), 512, ["compact", "gather", "iq2"]),
    ):
        calls.clear()
        assert _launch(layer, scratch, tokens=tokens)
        assert [name for name, _, _ in calls] == expected


def test_h6c_runtime_policy_rejects_malformed_metadata_and_adds_no_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    baseline = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    baseline_nbytes = laguna_moe_scratch_nbytes(baseline, max_rows=512)

    for malformed, message in (
        (17, "must be a mapping"),
        ({"unknown": _H6C_VARIANT}, "unsupported role"),
        ({_ROLE: ""}, "non-empty variants"),
    ):
        monkeypatch.setattr(hip_gfx1100, _CAPABILITY, malformed, raising=False)
        with pytest.raises(ValueError, match=message):
            resolve_laguna_moe_plan(config, backend="hip_gfx1100")

    monkeypatch.setattr(
        hip_gfx1100,
        _CAPABILITY,
        {_ROLE: _H6C_VARIANT},
        raising=False,
    )
    monkeypatch.setattr(
        hip_gfx1100,
        _ABI_CAPABILITY,
        {_H6C_VARIANT: "unsupported"},
        raising=False,
    )
    with pytest.raises(ValueError, match="unsupported variant ABI"):
        resolve_laguna_moe_plan(config, backend="hip_gfx1100")

    monkeypatch.setattr(
        hip_gfx1100,
        _ABI_CAPABILITY,
        {_H6C_VARIANT: _H6C_ABI},
        raising=False,
    )
    candidate = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert laguna_moe_scratch_nbytes(candidate, max_rows=512) == baseline_nbytes
