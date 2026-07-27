"""Default-off runtime contract for Laguna's exact router projection tree."""

from __future__ import annotations

import inspect

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_moe
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from tests._laguna_synthetic import make_laguna_info

_LAYER = "router_logits"
_QUANT = "f32"
_CANDIDATE_VARIANT = "bf16_hidden_wave0_tree"
_RETAINED_VARIANT = "bf16_hidden"
_CANDIDATE_KEY = KernelKey(
    "hip_gfx1100",
    _LAYER,
    _QUANT,
    _CANDIDATE_VARIANT,
)
_RETAINED_KEY = KernelKey(
    "hip_gfx1100",
    _LAYER,
    _QUANT,
    _RETAINED_VARIANT,
)


def _config():
    return laguna_gguf_config_from_metadata(make_laguna_info())


def test_router_projection_wave0_tree_capability_is_gfx1100_default_off() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert gfx1100.LAGUNA_ROUTER_PROJECTION_WAVE0_TREE is False
    assert not hasattr(gfx1151, "LAGUNA_ROUTER_PROJECTION_WAVE0_TREE")


def test_router_projection_wave0_tree_resolver_is_exact_key_and_backend_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_candidate = laguna_moe.resolve_laguna_router_projection_wave0_tree
    assert not resolve_candidate("hip_gfx1100")
    assert resolve_candidate("hip_gfx1100", True)
    assert not resolve_candidate("hip_gfx1100", False)
    assert not resolve_candidate("hip_gfx1151", True)

    original_is_registered = laguna_moe.is_registered
    monkeypatch.setattr(
        laguna_moe,
        "is_registered",
        lambda key: key != _CANDIDATE_KEY and original_is_registered(key),
    )
    assert not resolve_candidate("hip_gfx1100", True)


def test_router_projection_wave0_tree_plan_is_c1_only_and_retains_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = laguna_moe.resolve_laguna_moe_plan(
        _config(),
        backend="hip_gfx1100",
    )
    candidate = laguna_moe.resolve_laguna_moe_plan(
        _config(),
        backend="hip_gfx1100",
        use_router_projection_wave0_tree=True,
    )
    rollback = laguna_moe.resolve_laguna_moe_plan(
        _config(),
        backend="hip_gfx1100",
        use_router_projection_wave0_tree=False,
    )
    unsupported = laguna_moe.resolve_laguna_moe_plan(
        _config(),
        backend="hip_gfx1151",
        use_router_projection_wave0_tree=True,
    )

    assert default.router_logits_key == _RETAINED_KEY
    assert default.c1_router_logits_key is None
    assert default.c1_router_logits is None
    assert candidate.router_logits_key == _RETAINED_KEY
    assert candidate.c1_router_logits_key == _CANDIDATE_KEY
    assert candidate.c1_router_logits is not None
    assert _CANDIDATE_KEY in candidate.kernel_keys
    assert rollback.router_logits_key == _RETAINED_KEY
    assert rollback.c1_router_logits_key is None
    assert rollback.c1_router_logits is None
    assert unsupported.router_logits_key.variant == _RETAINED_VARIANT
    assert unsupported.c1_router_logits_key is None
    assert unsupported.c1_router_logits is None

    c1_source = inspect.getsource(laguna_moe.run_laguna_moe_c1_components)
    rows_source = inspect.getsource(laguna_moe.run_laguna_moe_rows)
    assert 'getattr(plan, "c1_router_logits", None) or plan.router_logits' in c1_source
    assert "c1_router_logits" not in rows_source

    original_is_registered = laguna_moe.is_registered
    monkeypatch.setattr(
        laguna_moe,
        "is_registered",
        lambda key: key != _CANDIDATE_KEY and original_is_registered(key),
    )
    key_miss = laguna_moe.resolve_laguna_moe_plan(
        _config(),
        backend="hip_gfx1100",
        use_router_projection_wave0_tree=True,
    )
    assert key_miss.router_logits_key == _RETAINED_KEY
    assert key_miss.c1_router_logits_key is None
    assert key_miss.c1_router_logits is None


def test_router_projection_wave0_tree_session_and_benchmark_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    option = "use_router_projection_wave0_tree"
    assert option in inspect.signature(LagunaGGUFResidentSession).parameters
    assert option in inspect.signature(laguna_moe.resolve_laguna_moe_plan).parameters

    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert benchmark._parse_args().enable_router_projection_wave0_tree is False
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-router-projection-wave0-tree"],
    )
    assert benchmark._parse_args().enable_router_projection_wave0_tree is True
