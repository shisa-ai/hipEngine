"""Runtime-rejection contract for Laguna single-page global attention."""

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
    "laguna_attention_decode",
    "bf16",
    "global_context_single_page_spans",
)
_RUNTIME = Path("hipengine/runtime/laguna_kv.py")


def test_global_single_page_runtime_owner_is_removed_but_primitive_remains() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    load_backend_kernel_package("hip_gfx1100")
    load_backend_kernel_package("hip_gfx1151")
    assert not hasattr(gfx1100, "LAGUNA_GLOBAL_SINGLE_PAGE_ATTENTION")
    assert not hasattr(gfx1151, "LAGUNA_GLOBAL_SINGLE_PAGE_ATTENTION")
    assert not hasattr(laguna_kv, "resolve_laguna_global_single_page_attention")
    assert is_registered(_CANDIDATE_KEY)
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            _CANDIDATE_KEY.layer,
            _CANDIDATE_KEY.quant,
            _CANDIDATE_KEY.variant,
        )
    )


def test_global_single_page_runtime_and_session_seams_are_removed() -> None:
    assert "use_global_single_page_attention" not in inspect.signature(
        LagunaGGUFResidentSession
    ).parameters
    assert "use_global_single_page_attention" not in inspect.signature(
        laguna_kv.allocate_laguna_kv_cache
    ).parameters
    assert "use_global_single_page_attention" not in inspect.signature(
        laguna_kv.LagunaKVCache
    ).parameters
    runtime_source = _RUNTIME.read_text(encoding="utf-8")
    assert _CANDIDATE_KEY.variant not in runtime_source
    assert "laguna_global_attention_decode_single_page_bf16_spans" not in runtime_source


def test_global_single_page_benchmark_opt_in_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-global-single-page-attention"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()
