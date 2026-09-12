"""gfx1151 opt-in contract for Laguna's router projection wave-0 tree."""

from __future__ import annotations

import inspect

import pytest

from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_moe
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from tests._laguna_synthetic import make_laguna_info

_GFX1100_CANDIDATE_KEY = KernelKey(
    "hip_gfx1100",
    "router_logits",
    "f32",
    "bf16_hidden_wave0_tree",
)
_GFX1151_CANDIDATE_KEY = KernelKey(
    "hip_gfx1151",
    "router_logits",
    "f32",
    "bf16_hidden_wave0_tree",
)
_RETAINED_VARIANT = "bf16_hidden"


def test_router_projection_wave0_tree_is_gfx1151_default_on() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151
    from hipengine.kernels.hip_gfx1100.moe.router import (
        register_qwen35_router_kernels,
    )

    register_qwen35_router_kernels()
    assert not hasattr(gfx1100, "LAGUNA_ROUTER_PROJECTION_WAVE0_TREE")
    assert gfx1151.LAGUNA_ROUTER_PROJECTION_WAVE0_TREE is True
    assert "use_router_projection_wave0_tree" in inspect.signature(
        LagunaGGUFResidentSession
    ).parameters
    assert not hasattr(
        LagunaGGUFResidentSession,
        "set_router_projection_wave0_tree",
    )
    assert is_registered(_GFX1100_CANDIDATE_KEY)
    gfx1151.register_gfx1151_kernels(replace=True)
    assert is_registered(_GFX1151_CANDIDATE_KEY)


def test_laguna_plan_exposes_exact_router_projection_candidate() -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    plan = laguna_moe.resolve_laguna_moe_plan(config, backend="hip_gfx1151")

    assert plan.router_logits_key.variant == _RETAINED_VARIANT
    assert plan.router_logits_key in plan.kernel_keys
    assert plan.router_logits is not None
    assert plan.router_logits_functions["wave0_tree"] is not None
    assert plan.router_select_key.variant == "correction_bias"
    assert plan.router_select_key in plan.kernel_keys
    assert plan.router_select is not None


def test_router_projection_wave0_tree_benchmark_compare_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_long_context_profile as benchmark

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        [
            "laguna_long_context_profile.py",
            "--compare-router-projection-wave0-tree",
        ],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
