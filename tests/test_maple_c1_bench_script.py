"""Unit contracts for the backend-aware Maple c1 qualification harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "maple_c1_bench.py"
_SPEC = importlib.util.spec_from_file_location("maple_c1_bench_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_maple_c1_harness_uses_backend_runner_capability(monkeypatch) -> None:
    class CudaRunner:
        pass

    monkeypatch.setattr(
        _MODULE,
        "backend_package_capability",
        lambda backend, name, default=None: (
            (lambda: CudaRunner)
            if (backend, name) == ("cuda_sm120a", "maple_runner_type")
            else default
        ),
    )

    assert _MODULE._maple_runner_type("cuda_sm120a") is CudaRunner
    assert _MODULE._maple_runner_type("hip_gfx1151") is _MODULE.MapleRunner


def test_maple_c1_cuda_attention_modes_select_and_restore() -> None:
    import hipengine.runtime.maple_cuda as runtime_module
    from hipengine.kernels.cuda_sm120a.attention.maple_attention import (
        maple_attention_decode_bf16,
        maple_attention_decode_wave32_exact_bf16,
    )

    original = runtime_module.maple_attention_decode_wave32_exact_bf16
    controller = _MODULE._CudaAttentionModes()
    try:
        controller.select("local128")
        assert (
            runtime_module.maple_attention_decode_wave32_exact_bf16
            is maple_attention_decode_bf16
        )
        controller.select("wave32")
        assert (
            runtime_module.maple_attention_decode_wave32_exact_bf16
            is maple_attention_decode_wave32_exact_bf16
        )
    finally:
        controller.close()
    assert runtime_module.maple_attention_decode_wave32_exact_bf16 is original
