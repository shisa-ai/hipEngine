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
    monkeypatch.setattr(
        moe_module,
        "_launch_weighted_selected_down",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        moe_module,
        "_launch_selected_down",
        lambda *_args, **_kwargs: calls.append("selected_down"),
    )

    with pytest.raises(StopRouted):
        moe_module.run_laguna_moe_c1_components(1, layer, scratch)
    assert calls == ["selected_gate_up", "selected_down", "routed_sum"]
    assert "defer_routed_sum" not in inspect.signature(
        moe_module.run_laguna_moe_c1_components
    ).parameters
    assert "defer_weighted_sum" not in inspect.signature(
        moe_module._launch_weighted_selected_down
    ).parameters


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
