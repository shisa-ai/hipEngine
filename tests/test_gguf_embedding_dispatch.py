from __future__ import annotations

from types import SimpleNamespace

# Import built-ins so registry keys exist before tests override them.
import hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding  # noqa: F401
import hipengine.kernels.hip_gfx1100.runtime.state  # noqa: F401
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_DENSE_BF16, LAYOUT_RAW_GGUF
from hipengine.runtime.gguf_embedding import launch_gguf_embedding, resolve_gguf_embedding_dispatch


def _fake_weight(*, layout: str, quant_key: str):
    allocations = {"raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10))}

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(layout=layout, quant_key=quant_key)

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


def test_resolve_gguf_embedding_dispatch_uses_raw_quant_or_dense_fallback() -> None:
    q6 = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k")
    q8 = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    dense = _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="gguf_q4_1")

    assert resolve_gguf_embedding_dispatch(q6).key == KernelKey(
        "hip_gfx1100", "embedding", "gguf_q6_k", "lookup_bf16_out"
    )
    assert resolve_gguf_embedding_dispatch(q8).key == KernelKey(
        "hip_gfx1100", "embedding", "gguf_q8_0", "lookup_bf16_out"
    )
    assert resolve_gguf_embedding_dispatch(dense).key == KernelKey(
        "hip_gfx1100", "embedding", "bf16", "lookup_bf16_out"
    )


def test_launch_gguf_embedding_calls_registry_kernel_with_expected_abi() -> None:
    key = KernelKey("hip_gfx1100", "embedding", "gguf_q8_0", "lookup_bf16_out")
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, fake_kernel, replace=True)
    try:
        launch_gguf_embedding(
            _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0"),
            token_ids_ptr=100,
            out_ptr=200,
            rows=2,
            hidden_size=1024,
            vocab_size=248320,
            threads=128,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        register(key, original, replace=True)

    assert calls == [((100, 10, 200, 2, 1024, 248320), {"stream": 7, "runtime": "runtime-sentinel", "threads": 128})]
