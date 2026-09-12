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

_VARIANT = "correction_bias_compact_wave32"
_CONTROL_VARIANT = "correction_bias"


def _key(backend: str, variant: str) -> KernelKey:
    return KernelKey(
        backend,
        "laguna_sigmoid_router_topk",
        "f32",
        variant,
    )


def _buffer(ptr: int) -> SimpleNamespace:
    return SimpleNamespace(ptr=ptr)


def test_compact_wave32_runtime_selection_is_removed_but_primitive_remains() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_ROUTER_SELECTOR_COMPACT_WAVE32",
        None,
    ) is None
    assert not hasattr(moe_module, "resolve_laguna_router_selector_compact_wave32")
    assert "use_router_selector_compact_wave32" not in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters
    assert "use_router_selector_compact_wave32" not in inspect.signature(
        moe_module.resolve_laguna_moe_plan
    ).parameters

    config = laguna_gguf_config_from_metadata(make_laguna_info())
    plan = moe_module.resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert plan.router_select_key == _key("hip_gfx1100", _CONTROL_VARIANT)
    assert not hasattr(plan, "c1_router_select_key")
    assert not hasattr(plan, "c1_router_select")
    assert is_registered(_key("hip_gfx1100", _VARIANT))
    assert not is_registered(_key("hip_gfx1151", _VARIANT))


def test_compact_wave32_rejection_restores_control_for_c1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class StopSelection(Exception):
        pass

    def control(*_args, **_kwargs) -> None:
        calls.append("control")
        raise StopSelection

    plan = SimpleNamespace(
        hidden_size=3_072,
        expert_count=256,
        top_k=10,
        shared_ffn_size=1_024,
        routed_scaling_factor=2.5,
        router_logits=lambda *_args, **_kwargs: None,
        router_select=control,
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

    with pytest.raises(StopSelection):
        moe_module.run_laguna_moe_c1_components(1, layer, scratch)
    assert calls == ["control"]


def test_compact_wave32_cli_is_removed_after_clean_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-router-selector-compact-wave32"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
