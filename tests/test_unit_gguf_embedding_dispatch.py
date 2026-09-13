from __future__ import annotations

from math import prod
from types import SimpleNamespace

# Import built-ins so registry keys exist before tests override them.
import hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding  # noqa: F401
import hipengine.kernels.hip_gfx1100.runtime.state  # noqa: F401
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    plan_qwen35_gguf_weight_spec,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime.gguf_embedding import launch_gguf_embedding, resolve_gguf_embedding_dispatch


def _fake_weight(*, layout: str, quant_key: str):
    allocations = {"raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10))}

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(layout=layout, quant_key=quant_key)

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


def test_q4_k_token_embedding_plan_keeps_dispatchable_raw_gguf() -> None:
    shape = (248320, 5120)
    n_elements = prod(shape)
    tensor = GGUFTensorInfo(
        name="token_embd.weight",
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(GGMLQuantizationType.Q4_K),
        ggml_type_name="Q4_K",
        n_elements=n_elements,
        nbytes=715161600,
        offset=0,
        data_offset=0,
        byte_shape=shape,
    )

    embedding = plan_qwen35_gguf_weight_spec(
        "root.token_embedding", tensor, decode_repack=True
    )
    dense_linear = plan_qwen35_gguf_weight_spec(
        "layers.0.ffn_gate", tensor, decode_repack=True
    )

    assert embedding.layout == LAYOUT_RAW_GGUF
    assert embedding.quant_key == "gguf_q4_k"
    assert embedding.allocation_names == ("raw",)
    assert dense_linear.layout == LAYOUT_Q4_K_PACK8


def test_resolve_gguf_embedding_dispatch_uses_raw_quant_or_dense_fallback() -> None:
    q4 = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    q6 = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k")
    q8 = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    dense = _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="gguf_q4_1")

    assert resolve_gguf_embedding_dispatch(q4).key == KernelKey(
        "hip_gfx1100", "embedding", "gguf_q4_k", "lookup_bf16_out"
    )
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
