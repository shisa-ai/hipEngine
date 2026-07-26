"""Default-off runtime contract for Laguna's certified IQ4 weighted composite."""

from __future__ import annotations

from dataclasses import replace
import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_moe
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from tests._laguna_synthetic import make_laguna_info


def test_iq4_weighted_runtime_capability_is_gfx1100_default_off() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert gfx1100.LAGUNA_IQ4_WEIGHTED_COMPOSITE is False
    assert not hasattr(gfx1151, "LAGUNA_IQ4_WEIGHTED_COMPOSITE")


def test_iq4_weighted_plan_is_explicit_shape_scoped_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    default = laguna_moe.resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    candidate = laguna_moe.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_iq4_weighted_composite=True,
    )
    rollback = laguna_moe.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_iq4_weighted_composite=False,
    )

    assert default.use_iq4_weighted_composite is False
    assert set(default.selected_weighted_down_keys) == {"gguf_iq3_xxs"}
    assert candidate.use_iq4_weighted_composite is True
    assert candidate.selected_weighted_down_keys["gguf_iq4_xs"] == KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_iq4_xs",
        "selected_weighted_down_gemv_decode_bf16_bf16_out",
    )
    assert candidate.selected_weighted_down_routes["gguf_iq4_xs"].abi == (
        "raw_iq_weighted"
    )
    assert rollback.use_iq4_weighted_composite is False
    assert set(rollback.selected_weighted_down_keys) == {"gguf_iq3_xxs"}

    top8 = laguna_moe.resolve_laguna_moe_plan(
        replace(config, expert_used_count=8),
        backend="hip_gfx1100",
        use_iq4_weighted_composite=True,
    )
    k1280 = laguna_moe.resolve_laguna_moe_plan(
        replace(config, expert_feed_forward_length=1280),
        backend="hip_gfx1100",
        use_iq4_weighted_composite=True,
    )
    gfx1151 = laguna_moe.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        use_iq4_weighted_composite=True,
    )
    assert not top8.use_iq4_weighted_composite
    assert not k1280.use_iq4_weighted_composite
    assert not gfx1151.use_iq4_weighted_composite
    assert "gguf_iq4_xs" not in top8.selected_weighted_down_keys
    assert "gguf_iq4_xs" not in k1280.selected_weighted_down_keys
    assert "gguf_iq4_xs" not in gfx1151.selected_weighted_down_keys

    monkeypatch.setattr(
        laguna_moe,
        "is_registered",
        lambda key: not (
            key.quant == "gguf_iq4_xs"
            and key.variant == "selected_weighted_down_gemv_decode_bf16_bf16_out"
        ),
    )
    missing = laguna_moe.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_iq4_weighted_composite=True,
    )
    assert not missing.use_iq4_weighted_composite
    assert "gguf_iq4_xs" not in missing.selected_weighted_down_keys


def test_existing_weighted_dispatch_owns_only_c1_iq4_route() -> None:
    launches: list[tuple[tuple, dict]] = []

    def candidate(*args, **kwargs):
        launches.append((args, kwargs))

    route = SimpleNamespace(
        function=candidate,
        abi="raw_iq_weighted",
        allocation_name="raw",
        library_key="selected_down_iq",
    )
    plan = SimpleNamespace(
        c1_selected_down_routes={},
        selected_weighted_down_routes={"gguf_iq4_xs": route},
        top_k=10,
        expert_count=256,
        expert_ffn_size=1024,
        hidden_size=3072,
    )
    scratch = SimpleNamespace(
        plan=plan,
        expert_intermediate=SimpleNamespace(ptr=11),
        selected_experts=SimpleNamespace(ptr=12),
        scaled_routing_weights=SimpleNamespace(ptr=13),
        routed_output=SimpleNamespace(ptr=14),
    )
    allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=15))
    weight = SimpleNamespace(
        spec=SimpleNamespace(quant_key="gguf_iq4_xs"),
        allocation=lambda name: allocation if name == "raw" else None,
    )
    layer = SimpleNamespace(weight=lambda name: weight if name == "ffn_down_exps" else None)

    assert laguna_moe._launch_weighted_selected_down(
        layer,
        scratch,
        tokens=1,
        stream=0,
        runtime=None,
        libraries=None,
    )
    assert len(launches) == 1
    _, kwargs = launches[0]
    assert kwargs["tokens"] == 1
    assert kwargs["top_k"] == 10
    assert kwargs["in_features"] == 1024
    assert kwargs["out_features"] == 3072

    assert not laguna_moe._launch_weighted_selected_down(
        layer,
        scratch,
        tokens=2,
        stream=0,
        runtime=None,
        libraries=None,
    )
    assert len(launches) == 1


def test_iq4_weighted_session_and_benchmark_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    assert "use_iq4_weighted_composite" in inspect.signature(
        LagunaGGUFResidentSession
    ).parameters
    assert "use_iq4_weighted_composite" in inspect.signature(
        laguna_moe.resolve_laguna_moe_plan
    ).parameters

    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert benchmark._parse_args().enable_iq4_weighted_composite is False
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-iq4-weighted-composite"],
    )
    assert benchmark._parse_args().enable_iq4_weighted_composite is True
