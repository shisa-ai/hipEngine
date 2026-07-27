"""Default-off runtime contract for the exact global-head wave-0 tree."""

from __future__ import annotations

import inspect

import pytest

from hipengine.kernels.registry import KernelKey
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


def test_global_head_wave0_tree_runtime_capability_is_gfx1100_default_off() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert gfx1100.LAGUNA_GLOBAL_HEAD_WAVE0_TREE is False
    assert not hasattr(gfx1151, "LAGUNA_GLOBAL_HEAD_WAVE0_TREE")


def test_global_head_wave0_tree_resolver_is_exact_key_and_backend_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_candidate = runner.resolve_laguna_global_head_wave0_tree
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


def test_global_head_wave0_tree_plan_changes_global_only_and_retains_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
        use_head_kv_fusion=True,
    )
    candidate = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
        use_head_kv_fusion=True,
        use_global_head_wave0_tree=True,
    )
    rollback = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
        use_head_kv_fusion=True,
        use_global_head_wave0_tree=False,
    )
    unfused = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
        use_head_kv_fusion=False,
        use_global_head_wave0_tree=True,
    )
    unsupported = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1151",
        use_head_kv_fusion=True,
        use_global_head_wave0_tree=True,
    )

    assert default.global_head_kv_key.variant == _RETAINED_GLOBAL_VARIANT
    assert default.swa_head_kv_key.variant == _RETAINED_SWA_VARIANT
    assert default.global_head_kv is not None and default.swa_head_kv is not None
    assert candidate.global_head_kv_key == _CANDIDATE_KEY
    assert candidate.swa_head_kv_key.variant == _RETAINED_SWA_VARIANT
    assert candidate.global_head_kv is not None and candidate.swa_head_kv is not None
    assert candidate.global_head_kv_key in candidate.kernel_keys
    assert candidate.swa_head_kv_key in candidate.kernel_keys
    assert rollback.global_head_kv_key.variant == _RETAINED_GLOBAL_VARIANT
    assert rollback.swa_head_kv_key.variant == _RETAINED_SWA_VARIANT
    assert rollback.global_head_kv is not None and rollback.swa_head_kv is not None
    assert unfused.global_head_kv is None and unfused.swa_head_kv is None
    assert unsupported.global_head_kv is None and unsupported.swa_head_kv is None

    original_is_registered = runner.is_registered
    monkeypatch.setattr(
        runner,
        "is_registered",
        lambda key: key != _CANDIDATE_KEY and original_is_registered(key),
    )
    key_miss = runner.resolve_laguna_eager_kernel_plan(
        _config(),
        backend="hip_gfx1100",
        use_head_kv_fusion=True,
        use_global_head_wave0_tree=True,
    )
    assert key_miss.global_head_kv_key.variant == _RETAINED_GLOBAL_VARIANT
    assert key_miss.swa_head_kv_key.variant == _RETAINED_SWA_VARIANT
    assert key_miss.global_head_kv is not None and key_miss.swa_head_kv is not None


def test_global_head_wave0_tree_session_and_benchmark_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    option = "use_global_head_wave0_tree"
    assert option in inspect.signature(LagunaGGUFResidentSession).parameters
    assert option in inspect.signature(runner.resolve_laguna_eager_kernel_plan).parameters

    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert benchmark._parse_args().enable_global_head_wave0_tree is False
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-global-head-wave0-tree"],
    )
    assert benchmark._parse_args().enable_global_head_wave0_tree is True
