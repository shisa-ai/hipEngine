"""One F32-weight router callable preserves registered and direct launch widths."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.moe import router


@pytest.mark.parametrize("threads", [None, 128, 256, 512])
def test_canonical_router_default_and_explicit_widths(monkeypatch, threads):
    calls = []
    library = object()
    runtime = object()
    monkeypatch.setattr(router, "signed_kernel_fn", lambda *args: lambda *values: calls.append(values) or 0)
    monkeypatch.setattr(router, "_check_launch", lambda *args: None)
    kwargs = {} if threads is None else {"threads": threads}
    router.qwen35_router_logits_bf16_f32w(
        1, 2, 3, 17, 2816, 128, stream=7, library=library, runtime=runtime, **kwargs,
    )
    assert calls == [(1, 2, 3, 17, 2816, 128, 256 if threads is None else threads, 7)]


def test_router_registry_has_one_canonical_f32_callable():
    from hipengine.kernels import hip_gfx1151
    from hipengine.kernels.hip_gfx1100 import moe
    from hipengine.kernels.registry import resolve

    assert inspect.signature(router.qwen35_router_logits_bf16_f32w).parameters["threads"].default == 256
    assert not hasattr(router, "qwen35_router_logits_bf16_f32w_auto_256")
    assert not hasattr(moe, "qwen35_router_logits_bf16_f32w_auto_256")
    router.register_qwen35_router_kernels(replace=True)
    hip_gfx1151.register_gfx1151_kernels(replace=True)
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        assert resolve(backend=backend, layer="router_logits", quant="f32", variant="bf16_hidden") is router.qwen35_router_logits_bf16_f32w


@pytest.mark.parametrize("tokens", [2, 7, 17])
def test_gemma_untiled_comparison_keeps_explicit_512(monkeypatch, tokens):
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_router as gemma
    from hipengine.runtime import laguna_moe

    calls = []
    scratch = SimpleNamespace(
        hidden_size=2816, num_experts=128, top_k=8, tokens=31,
        buffer=lambda name: SimpleNamespace(ptr={"prescaled": 11, "logits": 12}[name]),
    )
    monkeypatch.setattr(laguna_moe, "resolve_laguna_router_logits_mode", lambda *args, **kwargs: "token_tile_4")
    monkeypatch.setattr(gemma, "_TOKEN_TILE_16_MIN_TOKENS", 32)
    monkeypatch.setattr(gemma, "gemma4_router_prescale_bf16", lambda *args, **kwargs: None)
    monkeypatch.setattr(gemma, "qwen35_router_select", lambda *args, **kwargs: None)
    monkeypatch.setattr(gemma, "gemma4_expert_weight_scale_f32", lambda *args, **kwargs: None)
    monkeypatch.setattr(gemma, "qwen35_router_logits_bf16_f32w", lambda *args, **kwargs: calls.append((args, kwargs)))
    gemma.gemma4_router_topk_bf16(
        1, 2, 3, 4, 5, 6, tokens=tokens, hidden_size=2816, num_experts=128,
        top_k=8, scratch=scratch, stream=7,
    )
    assert calls == [((11, 3, 12, tokens, 2816, 128), {"threads": 512, "stream": 7})]
