from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner_module
from hipengine.runtime import laguna_moe as moe_module
from scripts import laguna_target_ar_bench as benchmark
from tests._laguna_synthetic import make_laguna_info

_KEY = KernelKey(
    "hip_gfx1100",
    "weighted_sum+moe_tail",
    "bf16",
    "laguna_top10_routed_hidden_out",
)


def _buffer(ptr: int) -> SimpleNamespace:
    return SimpleNamespace(ptr=ptr)


def test_weighted_hidden_runtime_selection_is_removed_but_primitive_remains() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_WEIGHTED_HIDDEN_SPLIT",
        None,
    ) is None
    assert not hasattr(moe_module, "resolve_laguna_weighted_hidden_split")
    assert "use_weighted_hidden_split" not in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters
    assert "use_weighted_hidden_split" not in inspect.signature(
        moe_module.resolve_laguna_moe_plan
    ).parameters

    config = laguna_gguf_config_from_metadata(make_laguna_info())
    plan = moe_module.resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert not hasattr(plan, "weighted_hidden_split_key")
    assert not hasattr(plan, "weighted_hidden_split")
    assert _KEY not in plan.kernel_keys
    assert is_registered(_KEY)
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            _KEY.layer,
            _KEY.quant,
            _KEY.variant,
        )
    )


def test_weighted_hidden_rejection_restores_c1_weighted_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class StopRouted(Exception):
        pass

    def routed_sum(*_args, **_kwargs) -> None:
        calls.append("routed_sum")
        raise StopRouted

    plan = SimpleNamespace(
        backend="hip_gfx1100",
        hidden_size=3_072,
        expert_count=256,
        top_k=10,
        shared_ffn_size=1_024,
        routed_scaling_factor=2.5,
        router_logits=lambda *_args, **_kwargs: None,
        router_select=lambda *_args, **_kwargs: None,
        routed_sum=routed_sum,
    )
    scratch = SimpleNamespace(
        plan=plan,
        max_rows=1,
        router_logits=_buffer(10),
        routing_scores=_buffer(11),
        selection_scores=_buffer(12),
        selected_experts=_buffer(13),
        routing_weights=_buffer(14),
        scaled_routing_weights=_buffer(15),
        expert_down=_buffer(16),
        routed_output=_buffer(17),
    )
    weights = {
        name: SimpleNamespace(
            allocation=lambda _kind, ptr=100 + index: SimpleNamespace(
                tensor=_buffer(ptr)
            )
        )
        for index, name in enumerate(("ffn_gate_inp", "exp_probs_b"))
    }
    layer = SimpleNamespace(weight=lambda name: weights[name])
    monkeypatch.setattr(moe_module, "validate_laguna_moe_layer", lambda *_args: None)
    monkeypatch.setattr(
        moe_module,
        "_launch_selected_gate_up",
        lambda *_args, **_kwargs: calls.append("selected_gate_up"),
    )

    def selected_down_composite(*_args, **kwargs) -> bool:
        calls.append(f"natural_parallel={kwargs['use_natural_parallel']}")
        return False

    monkeypatch.setattr(
        moe_module,
        "_launch_weighted_selected_down",
        selected_down_composite,
    )
    monkeypatch.setattr(
        moe_module,
        "_launch_selected_down",
        lambda *_args, **_kwargs: calls.append("selected_down"),
    )

    with pytest.raises(StopRouted):
        moe_module.run_laguna_moe_c1_components(
            1,
            layer,
            scratch,
            use_selected_down_natural_parallel_decode=True,
        )
    assert calls == [
        "selected_gate_up",
        "natural_parallel=True",
        "selected_down",
        "routed_sum",
    ]
    assert "defer_routed_sum" not in inspect.signature(
        moe_module.run_laguna_moe_c1_components
    ).parameters
    assert "defer_weighted_sum" not in inspect.signature(
        moe_module._launch_weighted_selected_down
    ).parameters


def test_route_parallel_weighted_down_owns_the_reducer_boundary() -> None:
    calls: list[tuple] = []

    def fused(*args, **kwargs) -> None:
        calls.append((*args, kwargs))

    route = SimpleNamespace(
        function=fused,
        abi="t16_natural_weighted",
        allocation_name="tiles",
        library_key="selected_down",
    )
    plan = SimpleNamespace(
        top_k=10,
        expert_count=256,
        expert_ffn_size=1_024,
        hidden_size=3_072,
        natural_parallel_weighted_selected_down_routes={
            "gguf_q6_k_t16_v1": route
        },
    )
    scratch = SimpleNamespace(
        plan=plan,
        expert_intermediate=_buffer(1),
        selected_experts=_buffer(2),
        expert_down=_buffer(3),
        scaled_routing_weights=_buffer(4),
        routed_output=_buffer(5),
        selected_down_completion=_buffer(6),
    )
    weight = SimpleNamespace(
        spec=SimpleNamespace(quant_key="gguf_q6_k_t16_v1"),
        allocation=lambda _name: SimpleNamespace(tensor=_buffer(7)),
    )
    layer = SimpleNamespace(weight=lambda _name: weight)

    assert moe_module._launch_weighted_selected_down(
        layer,
        scratch,
        tokens=1,
        stream=11,
        runtime=None,
        libraries=None,
        use_natural=True,
        use_natural_parallel=True,
        use_natural_parallel_weighted=True,
    )
    assert len(calls) == 1
    args, kwargs = calls[0][:-1], calls[0][-1]
    assert args[:7] == (1, 2, 7, 3, 4, 5, 6)
    assert args[7:] == (10, 10, 256, 1_024, 3_072)
    assert kwargs["stream"] == 11


def test_q4_paircoeff_weighted_down_selects_the_exact_sibling() -> None:
    calls: list[str] = []

    def retained(*_args, **_kwargs) -> None:
        calls.append("retained")

    def paircoeff(*_args, **_kwargs) -> None:
        calls.append("paircoeff")

    route_kwargs = {
        "abi": "t16_natural_weighted",
        "allocation_name": "tiles",
        "library_key": "selected_down",
    }
    plan = SimpleNamespace(
        top_k=10,
        expert_count=256,
        expert_ffn_size=1_024,
        hidden_size=3_072,
        natural_parallel_weighted_selected_down_routes={
            "gguf_q4_k_t16_v1": SimpleNamespace(
                function=retained,
                **route_kwargs,
            )
        },
        natural_parallel_paircoeff_weighted_selected_down_routes={
            "gguf_q4_k_t16_v1": SimpleNamespace(
                function=paircoeff,
                **route_kwargs,
            )
        },
    )
    scratch = SimpleNamespace(
        plan=plan,
        expert_intermediate=_buffer(1),
        selected_experts=_buffer(2),
        expert_down=_buffer(3),
        scaled_routing_weights=_buffer(4),
        routed_output=_buffer(5),
        selected_down_completion=_buffer(6),
    )
    weight = SimpleNamespace(
        spec=SimpleNamespace(quant_key="gguf_q4_k_t16_v1"),
        allocation=lambda _name: SimpleNamespace(tensor=_buffer(7)),
    )
    layer = SimpleNamespace(weight=lambda _name: weight)

    assert moe_module._launch_weighted_selected_down(
        layer,
        scratch,
        tokens=1,
        stream=0,
        runtime=None,
        libraries=None,
        use_natural=True,
        use_natural_parallel=True,
        use_natural_parallel_weighted=True,
        use_q4_paircoeff_weighted=True,
    )
    assert calls == ["paircoeff"]


def test_weighted_hidden_cli_is_removed_after_clean_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-weighted-hidden-split"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
