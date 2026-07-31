"""Runtime contract for Laguna's gfx1151 exact wave-0 RMS-tree tail."""

from __future__ import annotations

import inspect

import pytest

from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from tests._laguna_synthetic import make_laguna_info

_CANDIDATE_KEY = KernelKey(
    "hip_gfx1100",
    "moe_tail+next_rmsnorm",
    "bf16",
    "laguna_aggregate_wave0_tree_gguf_f32_weight_out",
)
_RETAINED_VARIANT = "laguna_aggregate_gguf_f32_weight_out"


def test_wave0_tree_runtime_owner_is_gfx1151_only() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
        register_paro_combine_kernels,
    )

    register_paro_combine_kernels()
    gfx1151.register_backend_kernels()
    assert not hasattr(gfx1100, "LAGUNA_MOE_TAIL_WAVE0_TREE")
    assert gfx1151.LAGUNA_MOE_TAIL_WAVE0_TREE
    assert not hasattr(runner, "resolve_laguna_moe_tail_wave0_tree")
    assert "use_moe_tail_wave0_tree" in inspect.signature(
        LagunaGGUFResidentSession
    ).parameters
    assert "use_moe_tail_wave0_tree" in inspect.signature(
        runner.resolve_laguna_eager_kernel_plan
    ).parameters
    assert is_registered(_CANDIDATE_KEY)
    assert is_registered(
        KernelKey(
            "hip_gfx1151",
            _CANDIDATE_KEY.layer,
            _CANDIDATE_KEY.quant,
            _CANDIDATE_KEY.variant,
        )
    )


def test_laguna_plan_retains_d9_and_registered_unfused_rollback() -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    plan = runner.resolve_laguna_eager_kernel_plan(config, backend="hip_gfx1100")
    assert plan.moe_tail_next_rmsnorm is not None
    assert plan.moe_tail_next_rmsnorm_key.variant == _RETAINED_VARIANT
    assert plan.moe_tail_next_rmsnorm_key in plan.kernel_keys

    unfused = runner.resolve_laguna_eager_kernel_plan(
        config,
        backend="hip_gfx1100",
        use_moe_tail_next_rmsnorm=False,
    )
    assert unfused.moe_tail_next_rmsnorm is None
    assert unfused.moe_tail_next_rmsnorm_key not in unfused.kernel_keys


def test_gfx1151_plan_selects_wave0_tree_with_scalar_rollback() -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    candidate = runner.resolve_laguna_eager_kernel_plan(
        config,
        backend="hip_gfx1151",
        use_moe_tail_wave0_tree=True,
    )
    assert candidate.moe_tail_next_rmsnorm is not None
    assert candidate.moe_tail_next_rmsnorm_key.variant == _CANDIDATE_KEY.variant

    rollback = runner.resolve_laguna_eager_kernel_plan(
        config,
        backend="hip_gfx1151",
        use_moe_tail_wave0_tree=False,
    )
    assert rollback.moe_tail_next_rmsnorm is not None
    assert rollback.moe_tail_next_rmsnorm_key.variant == _RETAINED_VARIANT


def test_wave0_tree_benchmark_opt_in_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-moe-tail-wave0-tree"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
