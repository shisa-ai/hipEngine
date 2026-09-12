"""Runtime-rejection contract for Laguna's certified IQ4 weighted composite."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_moe
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from tests._laguna_synthetic import make_laguna_info

_CANDIDATE_KEY = KernelKey(
    "hip_gfx1100",
    "moe_linear",
    "gguf_iq4_xs",
    "selected_weighted_down_gemv_decode_bf16_bf16_out",
)


def test_iq4_weighted_runtime_owner_is_removed_but_primitive_remains() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert not hasattr(gfx1100, "LAGUNA_IQ4_WEIGHTED_COMPOSITE")
    assert not hasattr(gfx1151, "LAGUNA_IQ4_WEIGHTED_COMPOSITE")
    assert not hasattr(laguna_moe, "resolve_laguna_iq4_weighted_composite")
    assert "use_iq4_weighted_composite" not in inspect.signature(
        LagunaGGUFResidentSession
    ).parameters
    assert "use_iq4_weighted_composite" not in inspect.signature(
        laguna_moe.resolve_laguna_moe_plan
    ).parameters
    assert is_registered(_CANDIDATE_KEY)
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            _CANDIDATE_KEY.layer,
            _CANDIDATE_KEY.quant,
            _CANDIDATE_KEY.variant,
        )
    )


def test_laguna_plan_retains_split_iq4_runtime_route() -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    plan = laguna_moe.resolve_laguna_moe_plan(config, backend="hip_gfx1100")

    assert not hasattr(plan, "use_iq4_weighted_composite")
    assert set(plan.selected_weighted_down_keys) == {"gguf_iq3_xxs"}
    assert "gguf_iq4_xs" not in plan.selected_weighted_down_routes
    assert not plan.c1_selected_down_keys


def test_certified_weighted_primitive_stays_callable_only_via_explicit_route() -> None:
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
    layer = SimpleNamespace(
        weight=lambda name: weight if name == "ffn_down_exps" else None
    )

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


def test_iq4_weighted_benchmark_opt_in_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-iq4-weighted-composite"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
