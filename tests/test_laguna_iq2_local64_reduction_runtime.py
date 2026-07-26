from __future__ import annotations

import inspect

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner_module
from hipengine.runtime import laguna_moe as moe_module
from scripts import laguna_target_ar_bench as benchmark
from tests._laguna_synthetic import make_laguna_info

_RETAINED_VARIANT = "selected_dual_silu_gemv_decode_tile2_grid64_bf16_bf16_out"
_CANDIDATE_VARIANT = (
    "selected_dual_silu_gemv_decode_tile2_grid64_local64_reduce_bf16_bf16_out"
)


def _key(backend: str, variant: str) -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq2_xs", variant)


def test_iq2_local64_runtime_selection_is_removed_but_primitive_remains() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_IQ2_LOCAL64_REDUCTION",
        None,
    ) is None
    assert backend_package_capability(
        "hip_gfx1151",
        "LAGUNA_IQ2_LOCAL64_REDUCTION",
        None,
    ) is None
    assert not hasattr(runner_module, "resolve_laguna_iq2_local64_reduction")
    assert "use_iq2_local64_reduction" not in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters
    assert "use_iq2_local64_reduction" not in inspect.signature(
        moe_module.resolve_laguna_moe_plan
    ).parameters

    candidate_key = _key("hip_gfx1100", _CANDIDATE_VARIANT)
    assert is_registered(candidate_key)
    assert not is_registered(_key("hip_gfx1151", _CANDIDATE_VARIANT))


def test_iq2_local64_rejection_restores_retained_grid64_c1_owner() -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    retained_key = _key("hip_gfx1100", _RETAINED_VARIANT)
    candidate_key = _key("hip_gfx1100", _CANDIDATE_VARIANT)
    retained = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_iq2_grid64=True,
    )
    disabled = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_iq2_grid64=False,
    )

    assert retained.c1_selected_gate_up_keys["gguf_iq2_xs"] == retained_key
    assert retained.c1_selected_gate_up_routes["gguf_iq2_xs"].library_key == (
        "selected_gate_up_iq"
    )
    assert candidate_key not in retained.kernel_keys
    assert not disabled.c1_selected_gate_up_keys
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        moe_module.resolve_laguna_moe_plan(
            config,
            backend="hip_gfx1100",
            use_iq2_grid64=True,
            use_iq2_local64_reduction=True,
        )

    launch_source = inspect.getsource(moe_module._launch_selected_gate_up)
    assert "if x_rows == 1" in launch_source
    assert "plan.c1_selected_gate_up_routes" in launch_source


def test_iq2_local64_reduction_cli_is_removed_after_clean_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-iq2-local64-reduction"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
