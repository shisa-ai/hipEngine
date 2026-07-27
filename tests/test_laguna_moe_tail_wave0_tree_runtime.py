"""Default-off runtime contract for Laguna's exact wave-0 RMS-tree tail."""

from __future__ import annotations

import inspect

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from tests._laguna_synthetic import make_laguna_info

_CANDIDATE_VARIANT = "laguna_aggregate_wave0_tree_gguf_f32_weight_out"
_RETAINED_VARIANT = "laguna_aggregate_gguf_f32_weight_out"
_CANDIDATE_KEY = KernelKey(
    "hip_gfx1100",
    "moe_tail+next_rmsnorm",
    "bf16",
    _CANDIDATE_VARIANT,
)


def _config():
    return laguna_gguf_config_from_metadata(make_laguna_info())


def test_wave0_tree_runtime_capability_is_gfx1100_default_off() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert gfx1100.LAGUNA_MOE_TAIL_WAVE0_TREE is False
    assert not hasattr(gfx1151, "LAGUNA_MOE_TAIL_WAVE0_TREE")


def test_wave0_tree_resolver_is_explicit_backend_and_exact_key_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_candidate = runner.resolve_laguna_moe_tail_wave0_tree
    assert not resolve_candidate("hip_gfx1100")
    assert resolve_candidate("hip_gfx1100", True)
    assert not resolve_candidate("hip_gfx1100", False)
    assert not resolve_candidate("hip_gfx1151", True)

    original_is_registered = runner.is_registered
    monkeypatch.setattr(
        runner,
        "is_registered",
        lambda key: key != _CANDIDATE_KEY and original_is_registered(key),
    )
    assert not resolve_candidate("hip_gfx1100", True)


def test_wave0_tree_plan_selects_only_candidate_or_retained_d9() -> None:
    default = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
    )
    candidate = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
        use_moe_tail_wave0_tree=True,
    )
    rollback = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
        use_moe_tail_wave0_tree=False,
    )
    unfused = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
        use_moe_tail_next_rmsnorm=False,
        use_moe_tail_wave0_tree=True,
    )
    unsupported = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1151",
        use_moe_tail_wave0_tree=True,
    )

    assert default.moe_tail_next_rmsnorm_key.variant == _RETAINED_VARIANT
    assert default.moe_tail_next_rmsnorm is not None
    assert candidate.moe_tail_next_rmsnorm_key == _CANDIDATE_KEY
    assert candidate.moe_tail_next_rmsnorm is not None
    assert rollback.moe_tail_next_rmsnorm_key.variant == _RETAINED_VARIANT
    assert rollback.moe_tail_next_rmsnorm is not None
    assert unfused.moe_tail_next_rmsnorm is None
    assert unfused.moe_tail_next_rmsnorm_key not in unfused.kernel_keys
    assert unsupported.moe_tail_next_rmsnorm is None
    assert unsupported.moe_tail_next_rmsnorm_key.variant == _RETAINED_VARIANT


def test_wave0_tree_session_and_benchmark_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    option = "use_moe_tail_wave0_tree"
    assert option in inspect.signature(LagunaGGUFResidentSession).parameters
    assert option in inspect.signature(runner.resolve_laguna_eager_kernel_plan).parameters

    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert benchmark._parse_args().enable_moe_tail_wave0_tree is False
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-moe-tail-wave0-tree"],
    )
    assert benchmark._parse_args().enable_moe_tail_wave0_tree is True
