from __future__ import annotations

import inspect

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner_module
from scripts import laguna_target_ar_bench as benchmark
from tests._laguna_synthetic import make_laguna_info

_CANDIDATE_KEY = KernelKey(
    "hip_gfx1100",
    "add_rmsnorm",
    "gguf_f32_weight",
    "bf16_out_staged_f32_local256",
)


def test_staged_add_rmsnorm_runtime_selection_is_removed_but_primitive_remains() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_ADD_RMSNORM_STAGED_F32",
        None,
    ) is None
    assert backend_package_capability(
        "hip_gfx1151",
        "LAGUNA_ADD_RMSNORM_STAGED_F32",
        None,
    ) is None
    assert not hasattr(runner_module, "resolve_laguna_add_rmsnorm_staged_f32")
    assert "use_add_rmsnorm_staged_f32" not in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters
    assert "use_add_rmsnorm_staged_f32" not in inspect.signature(
        runner_module.resolve_laguna_eager_kernel_plan
    ).parameters

    config = laguna_gguf_config_from_metadata(make_laguna_info())
    plan = runner_module.resolve_laguna_eager_kernel_plan(
        config,
        backend="hip_gfx1100",
    )
    assert not hasattr(plan, "c1_add_rmsnorm_key")
    assert not hasattr(plan, "c1_add_rmsnorm")
    assert _CANDIDATE_KEY not in plan.kernel_keys
    assert is_registered(_CANDIDATE_KEY)
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            _CANDIDATE_KEY.layer,
            _CANDIDATE_KEY.quant,
            _CANDIDATE_KEY.variant,
        )
    )


def test_staged_add_rmsnorm_rejection_restores_control_for_rows_and_c1() -> None:
    rows_source = inspect.getsource(runner_module.LagunaGGUFResidentSession._run_layer_rows)
    c1_source = inspect.getsource(runner_module.LagunaGGUFResidentSession._run_layer)
    assert "self.kernel_plan.add_rmsnorm(" in rows_source
    assert "self.kernel_plan.add_rmsnorm(" in c1_source
    assert "c1_add_rmsnorm" not in rows_source
    assert "c1_add_rmsnorm" not in c1_source


def test_staged_add_rmsnorm_cli_is_removed_after_clean_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-add-rmsnorm-staged-f32"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
