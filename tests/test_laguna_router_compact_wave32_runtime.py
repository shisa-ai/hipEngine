from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey
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


def test_compact_wave32_runtime_owner_is_explicit_default_off_and_gfx1100_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = getattr(
        moe_module,
        "resolve_laguna_router_selector_compact_wave32",
        None,
    )
    assert callable(resolver), "runtime selector resolver must be present"
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_ROUTER_SELECTOR_COMPACT_WAVE32",
        None,
    ) is False
    assert backend_package_capability(
        "hip_gfx1151",
        "LAGUNA_ROUTER_SELECTOR_COMPACT_WAVE32",
        None,
    ) is None
    assert not resolver("hip_gfx1100")
    assert resolver("hip_gfx1100", True)
    assert not resolver("hip_gfx1100", False)
    assert not resolver("hip_gfx1151", True)

    config = laguna_gguf_config_from_metadata(make_laguna_info())
    default = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
    )
    candidate = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_router_selector_compact_wave32=True,
    )
    unsupported = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        use_router_selector_compact_wave32=True,
    )
    assert default.router_select_key == default.c1_router_select_key == _key(
        "hip_gfx1100",
        _CONTROL_VARIANT,
    )
    assert candidate.router_select_key == _key("hip_gfx1100", _CONTROL_VARIANT)
    assert candidate.c1_router_select_key == _key("hip_gfx1100", _VARIANT)
    assert candidate.router_select is default.router_select
    assert candidate.c1_router_select is not candidate.router_select
    assert unsupported.router_select_key == unsupported.c1_router_select_key == _key(
        "hip_gfx1151",
        _CONTROL_VARIANT,
    )

    original_is_registered = moe_module.is_registered
    monkeypatch.setattr(
        moe_module,
        "is_registered",
        lambda key: False if key == _key("hip_gfx1100", _VARIANT) else original_is_registered(key),
    )
    missing = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_router_selector_compact_wave32=True,
    )
    assert missing.router_select_key == missing.c1_router_select_key == _key(
        "hip_gfx1100",
        _CONTROL_VARIANT,
    )


def test_compact_wave32_runtime_owner_changes_only_c1_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class StopSelection(Exception):
        pass

    def select(name: str):
        def launch(*_args, **_kwargs) -> None:
            calls.append(name)
            raise StopSelection

        return launch

    plan = SimpleNamespace(
        hidden_size=3_072,
        expert_count=256,
        top_k=10,
        shared_ffn_size=1_024,
        routed_scaling_factor=2.5,
        router_logits=lambda *_args, **_kwargs: None,
        router_select=select("control"),
        c1_router_select=select("compact_wave32"),
    )
    scratch = SimpleNamespace(
        plan=plan,
        max_rows=2,
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
    assert calls == ["compact_wave32"]

    calls.clear()
    with pytest.raises(StopSelection):
        moe_module.run_laguna_moe_rows(1, layer, scratch, rows=2)
    assert calls == ["control"]


def test_compact_wave32_session_and_cli_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "use_router_selector_compact_wave32" in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters

    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert not benchmark._parse_args().enable_router_selector_compact_wave32

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-router-selector-compact-wave32"],
    )
    assert benchmark._parse_args().enable_router_selector_compact_wave32
