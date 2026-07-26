from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.runtime import laguna_gguf_runner as runner_module
from scripts import laguna_target_ar_bench as benchmark

_CANDIDATE = "mixed_pair_reuse_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_LOCAL32 = "mixed_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_LOCAL128 = "mixed_q6_fixed_meta_pack8_gemv_decode_bf16_f32_out"


def test_pair_reuse_runtime_capability_is_default_off_and_gfx1100_only() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_MIXED_PAIR_REUSE",
        None,
    ) is False
    assert backend_package_capability(
        "hip_gfx1151",
        "LAGUNA_MIXED_PAIR_REUSE",
        None,
    ) is None
    resolver = getattr(runner_module, "resolve_laguna_mixed_pair_reuse_attention", None)
    assert callable(resolver)
    assert not resolver("hip_gfx1100")
    assert resolver("hip_gfx1100", True)
    assert not resolver("hip_gfx1100", False)
    assert not resolver("hip_gfx1151", True)
    assert "use_mixed_pair_reuse_attention" in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters


def test_pair_reuse_runtime_owner_is_candidate_first_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants: list[str] = []
    accepted = [_CANDIDATE]

    def mixed(*args, **kwargs):
        del args
        variant = kwargs["variant"]
        variants.append(variant)
        return variant == accepted[0]

    monkeypatch.setattr(runner_module, "launch_laguna_mixed_attention_projections", mixed)
    weight = SimpleNamespace(spec=SimpleNamespace(layout="gguf_raw"))

    def launch(*, enabled: bool) -> bool:
        return runner_module.launch_laguna_attention_projections(
            weight,
            weight,
            weight,
            weight,
            10,
            20,
            30,
            40,
            50,
            1,
            3072,
            6144,
            1024,
            1024,
            48,
            backend="hip_gfx1100",
            stream=0,
            libraries=SimpleNamespace(),
            runtime=None,
            use_mixed_q5_q6_attention=True,
            use_mixed_q6_fixed_meta_attention=True,
            use_mixed_local32_fixed_meta_attention=True,
            use_mixed_pair_reuse_attention=enabled,
        )

    assert launch(enabled=True)
    assert variants == [_CANDIDATE]

    variants.clear()
    accepted[0] = _LOCAL32
    assert launch(enabled=True)
    assert variants == [_CANDIDATE, _LOCAL32]

    variants.clear()
    accepted[0] = _LOCAL128
    assert launch(enabled=True)
    assert variants == [_CANDIDATE, _LOCAL32, _LOCAL128]

    variants.clear()
    accepted[0] = _LOCAL32
    assert launch(enabled=False)
    assert variants == [_LOCAL32]


def test_pair_reuse_runtime_cli_is_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert not benchmark._parse_args().enable_mixed_pair_reuse_attention

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-mixed-pair-reuse-attention"],
    )
    assert benchmark._parse_args().enable_mixed_pair_reuse_attention
