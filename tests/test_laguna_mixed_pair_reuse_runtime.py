from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime import laguna_gguf_runner as runner_module
from scripts import laguna_target_ar_bench as benchmark

_CANDIDATE = "mixed_pair_reuse_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_LOCAL32 = "mixed_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_LOCAL128 = "mixed_q6_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_QUANT = "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k"


def _key(backend: str) -> KernelKey:
    return KernelKey(backend, "attention_projection_quad", _QUANT, _CANDIDATE)


def test_pair_reuse_runtime_selection_is_removed_but_primitive_remains() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_MIXED_PAIR_REUSE",
        None,
    ) is None
    assert not hasattr(runner_module, "resolve_laguna_mixed_pair_reuse_attention")
    assert "use_mixed_pair_reuse_attention" not in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters
    assert "use_mixed_pair_reuse_attention" not in inspect.signature(
        runner_module.launch_laguna_attention_projections
    ).parameters
    assert is_registered(_key("hip_gfx1100"))
    assert not is_registered(_key("hip_gfx1151"))


def test_pair_reuse_rejection_restores_retained_projection_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants: list[str] = []
    accepted = [_LOCAL32]

    def mixed(*args, **kwargs):
        del args
        variant = kwargs["variant"]
        variants.append(variant)
        return variant == accepted[0]

    monkeypatch.setattr(runner_module, "launch_laguna_mixed_attention_projections", mixed)
    weight = SimpleNamespace(spec=SimpleNamespace(layout="gguf_raw"))

    def launch() -> bool:
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
        )

    assert launch()
    assert variants == [_LOCAL32]

    variants.clear()
    accepted[0] = _LOCAL128
    assert launch()
    assert variants == [_LOCAL32, _LOCAL128]
    assert _CANDIDATE not in variants


def test_pair_reuse_cli_is_removed_after_clean_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-mixed-pair-reuse-attention"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
