"""Runtime-rejection contract for the exact global-head wave-0 tree."""

from __future__ import annotations

import inspect

import pytest

from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from tests._laguna_synthetic import make_laguna_info

_LAYER = "head_rmsnorm+partial_rotary+kv_write"
_QUANT = "laguna_f32_weight"
_CANDIDATE_VARIANT = "global_wave0_tree_f32_bf16_spans"
_RETAINED_GLOBAL_VARIANT = "global_f32_bf16_spans"
_RETAINED_SWA_VARIANT = "swa_f32_bf16_spans"
_CANDIDATE_KEY = KernelKey("hip_gfx1100", _LAYER, _QUANT, _CANDIDATE_VARIANT)


def _config():
    return laguna_gguf_config_from_metadata(make_laguna_info())


def test_global_head_wave0_tree_runtime_owner_is_removed_but_primitive_remains() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        register_laguna_kv_attention_kernels,
    )

    register_laguna_kv_attention_kernels()
    assert not hasattr(gfx1100, "LAGUNA_GLOBAL_HEAD_WAVE0_TREE")
    assert not hasattr(gfx1151, "LAGUNA_GLOBAL_HEAD_WAVE0_TREE")
    assert not hasattr(runner, "resolve_laguna_global_head_wave0_tree")
    assert "use_global_head_wave0_tree" not in inspect.signature(
        LagunaGGUFResidentSession
    ).parameters
    assert "use_global_head_wave0_tree" not in inspect.signature(
        runner.resolve_laguna_eager_kernel_plan
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


def test_laguna_plan_retains_current_p4_global_swa_and_unfused_rollback() -> None:
    plan = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
        use_head_kv_fusion=True,
    )
    assert plan.global_head_kv_key.variant == _RETAINED_GLOBAL_VARIANT
    assert plan.swa_head_kv_key.variant == _RETAINED_SWA_VARIANT
    assert plan.global_head_kv is not None and plan.swa_head_kv is not None
    assert plan.global_head_kv_key in plan.kernel_keys
    assert plan.swa_head_kv_key in plan.kernel_keys

    unfused = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
        use_head_kv_fusion=False,
    )
    assert unfused.global_head_kv is None and unfused.swa_head_kv is None
    assert unfused.global_head_kv_key not in unfused.kernel_keys
    assert unfused.swa_head_kv_key not in unfused.kernel_keys


def test_global_head_wave0_tree_session_and_benchmark_opt_in_are_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    option = "use_global_head_wave0_tree"
    assert option not in inspect.signature(LagunaGGUFResidentSession).parameters
    assert option not in inspect.signature(runner.resolve_laguna_eager_kernel_plan).parameters

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-global-head-wave0-tree"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
