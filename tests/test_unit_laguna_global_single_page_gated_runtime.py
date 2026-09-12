"""Runtime-rejection contract for gated single-page global attention."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime import laguna_kv
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession

_CANDIDATE_KEY = KernelKey(
    "hip_gfx1100",
    "laguna_attention_decode+attention_gate",
    "bf16",
    "global_single_page_softplus_bf16_spans",
)
_RUNTIME = Path("hipengine/runtime/laguna_kv.py")
_SYMBOL = "laguna_global_attention_decode_single_page_softplus_gate_bf16_spans"


def test_gated_single_page_runtime_owner_is_removed_but_primitive_remains() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    load_backend_kernel_package("hip_gfx1100")
    load_backend_kernel_package("hip_gfx1151")
    assert not hasattr(gfx1100, "LAGUNA_GLOBAL_SINGLE_PAGE_GATED_ATTENTION")
    assert not hasattr(gfx1151, "LAGUNA_GLOBAL_SINGLE_PAGE_GATED_ATTENTION")
    assert not hasattr(
        laguna_kv,
        "resolve_laguna_global_single_page_gated_attention",
    )
    assert is_registered(_CANDIDATE_KEY)
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            _CANDIDATE_KEY.layer,
            _CANDIDATE_KEY.quant,
            _CANDIDATE_KEY.variant,
        )
    )


def test_gated_single_page_runtime_and_session_seams_are_removed() -> None:
    option = "use_global_single_page_gated_attention"
    assert option not in inspect.signature(LagunaGGUFResidentSession).parameters
    assert option not in inspect.signature(laguna_kv.allocate_laguna_kv_cache).parameters
    assert option not in inspect.signature(laguna_kv.LagunaKVCache).parameters
    runtime_source = _RUNTIME.read_text(encoding="utf-8")
    assert _CANDIDATE_KEY.variant not in runtime_source
    assert _SYMBOL not in runtime_source


def test_gated_single_page_benchmark_opt_in_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        [
            "laguna_target_ar_bench.py",
            "--enable-global-single-page-gated-attention",
        ],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
