from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime import laguna_gguf_runner as runner_module
from scripts import laguna_target_ar_bench as benchmark

_OUTPUT_CANDIDATE = "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out"
_SHARED_CANDIDATE = "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out"
_MIXED_CANDIDATE = (
    "mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out"
)
_MIXED_CONTROL = "mixed_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_MIXED_FALLBACK = "mixed_q6_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_MIXED_QUANT = "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k"
_CANDIDATE_KEYS = (
    KernelKey("hip_gfx1100", "linear", "gguf_q5_k", _OUTPUT_CANDIDATE),
    KernelKey("hip_gfx1100", "linear_pair", "gguf_q5_k", _SHARED_CANDIDATE),
    KernelKey(
        "hip_gfx1100",
        "attention_projection_quad",
        _MIXED_QUANT,
        _MIXED_CANDIDATE,
    ),
)


def test_q5_swar_runtime_selections_are_removed_but_primitives_remain() -> None:
    for capability in ("LAGUNA_Q5_SWAR_OUTPUT", "LAGUNA_Q5_SWAR_PAIR"):
        assert backend_package_capability("hip_gfx1100", capability, None) is None
        assert backend_package_capability("hip_gfx1151", capability, None) is None

    assert not hasattr(runner_module, "resolve_laguna_q5_swar_output")
    assert not hasattr(runner_module, "resolve_laguna_q5_swar_pair")
    session_parameters = inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters
    assert "use_q5_swar_output" not in session_parameters
    assert "use_q5_swar_pair" not in session_parameters
    assert "q5_swar_pair_variant" not in inspect.signature(
        runner_module.launch_laguna_attention_projections
    ).parameters

    for key in _CANDIDATE_KEYS:
        assert is_registered(key)
        assert not is_registered(
            KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
        )


def test_q5_swar_rejections_restore_all_retained_projection_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants: list[str] = []
    accepted = [_MIXED_CONTROL]

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
    assert variants == [_MIXED_CONTROL]

    variants.clear()
    accepted[0] = _MIXED_FALLBACK
    assert launch()
    assert variants == [_MIXED_CONTROL, _MIXED_FALLBACK]
    assert _MIXED_CANDIDATE not in variants

    session_source = inspect.getsource(runner_module.LagunaGGUFResidentSession.__init__)
    assert "_Q5_SWAR_PAIR_OUTPUT_VARIANT" not in session_source
    assert "_Q5_SWAR_PAIR_SHARED_VARIANT" not in session_source
    assert "_MIXED_ATTENTION_Q5_SWAR_PAIR_VARIANT" not in session_source


@pytest.mark.parametrize(
    "option",
    ("--enable-q5-swar-output", "--enable-q5-swar-pair"),
)
def test_q5_swar_cli_options_are_removed_after_clean_rejection(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
) -> None:
    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py", option])
    with pytest.raises(SystemExit):
        benchmark._parse_args()
