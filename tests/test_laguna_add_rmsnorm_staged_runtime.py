from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner_module
from scripts import laguna_target_ar_bench as benchmark
from tests._laguna_synthetic import make_laguna_info

_CANDIDATE_VARIANT = "bf16_out_staged_f32_local256"
_CONTROL_VARIANT = "bf16_out"


def _key(backend: str, variant: str) -> KernelKey:
    return KernelKey(
        backend,
        "add_rmsnorm",
        "gguf_f32_weight",
        variant,
    )


def test_staged_add_rmsnorm_runtime_capability_is_default_off_and_gfx1100_only() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_ADD_RMSNORM_STAGED_F32",
        None,
    ) is False
    assert backend_package_capability(
        "hip_gfx1151",
        "LAGUNA_ADD_RMSNORM_STAGED_F32",
        None,
    ) is None
    resolver = getattr(runner_module, "resolve_laguna_add_rmsnorm_staged_f32", None)
    assert callable(resolver)
    assert not resolver("hip_gfx1100")
    assert resolver("hip_gfx1100", True)
    assert not resolver("hip_gfx1100", False)
    assert not resolver("hip_gfx1151", True)
    assert "use_add_rmsnorm_staged_f32" in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters


def test_staged_add_rmsnorm_plan_is_c1_only_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    control_key = _key("hip_gfx1100", _CONTROL_VARIANT)
    candidate_key = _key("hip_gfx1100", _CANDIDATE_VARIANT)

    default = runner_module.resolve_laguna_eager_kernel_plan(
        config,
        backend="hip_gfx1100",
    )
    candidate = runner_module.resolve_laguna_eager_kernel_plan(
        config,
        backend="hip_gfx1100",
        use_add_rmsnorm_staged_f32=True,
    )
    unsupported = runner_module.resolve_laguna_eager_kernel_plan(
        config,
        backend="hip_gfx1151",
        use_add_rmsnorm_staged_f32=True,
    )

    assert default.add_rmsnorm_key == control_key
    assert default.c1_add_rmsnorm_key == control_key
    assert default.c1_add_rmsnorm is default.add_rmsnorm
    assert candidate.add_rmsnorm_key == control_key
    assert candidate.c1_add_rmsnorm_key == candidate_key
    assert candidate.c1_add_rmsnorm is not candidate.add_rmsnorm
    assert candidate_key not in default.kernel_keys
    assert candidate_key in candidate.kernel_keys
    assert unsupported.add_rmsnorm_key == _key("hip_gfx1151", _CONTROL_VARIANT)
    assert unsupported.c1_add_rmsnorm_key == unsupported.add_rmsnorm_key
    assert unsupported.c1_add_rmsnorm is unsupported.add_rmsnorm

    original_is_registered = runner_module.is_registered
    monkeypatch.setattr(
        runner_module,
        "is_registered",
        lambda key: False if key == candidate_key else original_is_registered(key),
    )
    missing = runner_module.resolve_laguna_eager_kernel_plan(
        config,
        backend="hip_gfx1100",
        use_add_rmsnorm_staged_f32=True,
    )
    assert missing.c1_add_rmsnorm_key == control_key
    assert missing.c1_add_rmsnorm is missing.add_rmsnorm
    assert candidate_key not in missing.kernel_keys

    rows_source = inspect.getsource(runner_module.LagunaGGUFResidentSession._run_layer_rows)
    c1_source = inspect.getsource(runner_module.LagunaGGUFResidentSession._run_layer)
    assert "self.kernel_plan.add_rmsnorm(" in rows_source
    assert "self.kernel_plan.c1_add_rmsnorm(" in c1_source


def test_staged_add_rmsnorm_session_and_cli_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert not benchmark._parse_args().enable_add_rmsnorm_staged_f32

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-add-rmsnorm-staged-f32"],
    )
    args = benchmark._parse_args()
    assert args.enable_add_rmsnorm_staged_f32

    captured: dict[str, object] = {}

    def session_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(benchmark, "LagunaGGUFResidentSession", session_factory)
    owner = SimpleNamespace(weights=object(), runtime=object())
    benchmark._session(owner, args)
    assert captured["use_add_rmsnorm_staged_f32"] is True
