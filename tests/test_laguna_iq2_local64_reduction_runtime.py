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

_RETAINED_VARIANT = "selected_dual_silu_gemv_decode_tile2_grid64_bf16_bf16_out"
_CANDIDATE_VARIANT = (
    "selected_dual_silu_gemv_decode_tile2_grid64_local64_reduce_bf16_bf16_out"
)


def _key(backend: str, variant: str) -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq2_xs", variant)


def test_iq2_local64_reduction_capability_is_default_off_and_gfx1100_only() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_IQ2_LOCAL64_REDUCTION",
        None,
    ) is False
    assert backend_package_capability(
        "hip_gfx1151",
        "LAGUNA_IQ2_LOCAL64_REDUCTION",
        None,
    ) is None
    resolver = getattr(runner_module, "resolve_laguna_iq2_local64_reduction", None)
    assert callable(resolver)
    assert not resolver("hip_gfx1100")
    assert resolver("hip_gfx1100", True)
    assert not resolver("hip_gfx1100", False)
    assert not resolver("hip_gfx1151", True)
    assert "use_iq2_local64_reduction" in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters


def test_iq2_local64_reduction_plan_is_c1_only_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    retained_key = _key("hip_gfx1100", _RETAINED_VARIANT)
    candidate_key = _key("hip_gfx1100", _CANDIDATE_VARIANT)

    retained = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_iq2_grid64=True,
    )
    candidate = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_iq2_grid64=True,
        use_iq2_local64_reduction=True,
    )
    disabled = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_iq2_grid64=True,
        use_iq2_local64_reduction=False,
    )
    no_grid64 = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_iq2_grid64=False,
        use_iq2_local64_reduction=True,
    )
    unsupported = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        use_iq2_grid64=False,
        use_iq2_local64_reduction=True,
    )

    assert retained.c1_selected_gate_up_keys["gguf_iq2_xs"] == retained_key
    assert disabled.c1_selected_gate_up_keys["gguf_iq2_xs"] == retained_key
    assert candidate.c1_selected_gate_up_keys["gguf_iq2_xs"] == candidate_key
    assert candidate.c1_selected_gate_up_routes["gguf_iq2_xs"].library_key == (
        "selected_gate_up_iq"
    )
    assert candidate_key not in retained.kernel_keys
    assert candidate_key in candidate.kernel_keys
    assert not no_grid64.c1_selected_gate_up_keys
    assert not unsupported.c1_selected_gate_up_keys

    original_is_registered = moe_module.is_registered
    monkeypatch.setattr(
        moe_module,
        "is_registered",
        lambda key: False if key == candidate_key else original_is_registered(key),
    )
    missing = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_iq2_grid64=True,
        use_iq2_local64_reduction=True,
    )
    assert missing.c1_selected_gate_up_keys["gguf_iq2_xs"] == retained_key
    assert candidate_key not in missing.kernel_keys

    launch_source = inspect.getsource(moe_module._launch_selected_gate_up)
    assert "if x_rows == 1" in launch_source
    assert "plan.c1_selected_gate_up_routes" in launch_source


def test_iq2_local64_reduction_session_and_cli_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert not benchmark._parse_args().enable_iq2_local64_reduction

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-iq2-local64-reduction"],
    )
    args = benchmark._parse_args()
    assert args.enable_iq2_local64_reduction

    captured: dict[str, object] = {}

    def session_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(benchmark, "LagunaGGUFResidentSession", session_factory)
    owner = SimpleNamespace(weights=object(), runtime=object())
    benchmark._session(owner, args)
    assert captured["use_iq2_local64_reduction"] is True
