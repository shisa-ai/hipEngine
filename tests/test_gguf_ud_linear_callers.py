"""CPU-only raw UD caller ABI checks; no device/math qualification."""
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.qwen35_gguf_materialize import _spec_for_tensor, LAYOUT_RAW_GGUF
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime import gguf_linear


@pytest.mark.parametrize("backend", ("hip_gfx1100", "hip_gfx1151"))
@pytest.mark.parametrize("quant,stride", (
    ("IQ4_XS", 136), ("IQ4_NL", 144), ("IQ3_S", 110),
    ("Q3_K", 110), ("IQ3_XXS", 98), ("IQ2_S", 82),
))
@pytest.mark.parametrize("rows", (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 28, 32, 511, 512, 513))
@pytest.mark.parametrize("output", ("bf16", "f32"))
@pytest.mark.parametrize("wmma", (False, True))
def test_raw_ud_runtime_linear_call(monkeypatch, backend, quant, stride, rows, output, wmma):
    # Use full FFN down dimensions; this test passes metadata, not weight bytes.
    n, k = 5120, 17408
    tensor = GGUFTensorInfo("blk.0.ffn_down.weight", (n, k), (k, n),
                           int(GGMLQuantizationType[quant]), quant, n*k,
                           n*(k//256)*stride, 0, 0, (n, (k//256)*stride))
    spec = _spec_for_tensor("layers.0.ffn_down", tensor, decode_repack=True)
    assert spec.layout == LAYOUT_RAW_GGUF
    assert spec.allocation_names == ("raw",) and not spec.sidecar_layouts
    allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=0x222000))
    accesses = []

    def get_allocation(name="raw"):
        accesses.append(name)
        assert name == "raw"
        return allocation

    weight = SimpleNamespace(spec=spec, backend=backend,
                             allocations={"raw": allocation}, allocation=get_allocation)
    calls, keys = [], []
    original_resolve = gguf_linear.resolve

    def capture_resolve(**kwargs):
        # Resolve the real registered leaf first; do not invent a missing route.
        assert callable(original_resolve(**kwargs))
        keys.append(KernelKey(**kwargs))
        return lambda *args, **kw: calls.append((args, kw))

    monkeypatch.setattr(gguf_linear, "resolve", capture_resolve)
    gguf_linear.clear_gguf_linear_dispatch_cache()
    variant = ("gemv" if rows == 1 else "prefill") + f"_bf16_{output}_out"
    library, runtime = object(), object()
    try:
        gguf_linear.launch_gguf_linear(
            weight, 0x111000, 0x333000, rows, k, n,
            activation_dtype="bf16", output_dtype=output,
            stream=17, runtime=runtime,
            libraries={f"{spec.quant_key}:{variant}": library},
            use_wmma_prefill=wmma, use_gemv_decode=True,
        )
        assert keys == [KernelKey(backend, "linear", spec.quant_key, variant)]
        assert calls == [((0x111000, 0x222000, 0x333000, rows, k, n),
                          {"stream": 17, "runtime": runtime, "library": library})]
        assert accesses and set(accesses) == {"raw"}
    finally:
        gguf_linear.clear_gguf_linear_dispatch_cache()
