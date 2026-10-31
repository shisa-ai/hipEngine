"""Host-only contract for bounded, parent-order T16 source-F16 projections."""
from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_f16_rocblas_prefill as source


@pytest.mark.parametrize("producer", [
    source.gguf_q4_k_t16_f16_rocblas_bf16_bf16_out,
    source.gguf_q4_k_t16_f16_rocblas_pair_bf16_bf16_out,
    source.gguf_q5_k_t16_f16_rocblas_bf16_bf16_out,
    source.gguf_q5_k_t16_f16_rocblas_pair_bf16_bf16_out,
    source.gguf_q5_k_t16_f16_rocblas_octet_bf16_bf16_out,
    source.gguf_q6_k_t16_qmicro_planar_f16_rocblas_bf16_bf16_out,
])
@pytest.mark.parametrize("cast_activation", [True, False])
def test_t16_projection_uses_parent_k16_order(monkeypatch, producer, cast_activation):
    calls = []
    monkeypatch.setattr(source, "bf16_to_fp16", lambda *a, **kw: calls.append("cast"))
    monkeypatch.setattr(source, "_launch_t16_dequantize_f16_tile", lambda *a, **kw: calls.append("dequant"))
    monkeypatch.setattr(source, "_launch_source_f16_k16", lambda *a, **kw: calls.append((a, kw)), raising=False)

    def reject(*args, **kwargs):
        pytest.fail("generic rocBLAS accumulation/publication changes the parent K16 order")

    monkeypatch.setattr(source, "get_rocblas", reject)
    blas = SimpleNamespace(gemm_ex_rowmajor_nt_fp16_compute_f32=reject,
                           gemm_ex_rowmajor_nt_fp16_compute_f16=reject)
    producer(1, 2, 3, 4, 5, 6, 517, 6144, 48,
             tile_out_features=32, stream=7, rocblas=blas,
             runtime=object(), solution_index=42, cast_activation=cast_activation)
    assert calls.count("cast") == int(cast_activation)
    assert calls.count("dequant") == 2
    launches = [c for c in calls if isinstance(c, tuple)]
    assert [c[0] for c in launches] == [(4, 5, 3, 517, 6144, 32, 48, 0),
                                        (4, 5, 3, 517, 6144, 16, 48, 32)]
    assert all(kw["stream"] == 7 for _, kw in launches)


@pytest.mark.parametrize("rows,hidden,cols,stride,offset", [
    (0, 256, 16, 16, 0), (1, 255, 16, 16, 0),
    (1, 256, 15, 32, 0), (1, 256, 16, 32, -1),
    (1, 256, 16, 32, 17),
])
def test_k16_rejects_invalid_shapes_before_loading(rows, hidden, cols, stride, offset):
    with pytest.raises(ValueError):
        source._launch_source_f16_k16(1, 2, 3, rows, hidden, cols, stride, offset)


def test_raw_q6_diagnostic_producer_keeps_its_rocblas_ownership(monkeypatch) -> None:
    """Raw-Q6 diagnostics stay a rocBLAS route; only the T16 owner changed.

    The bounded T16 K16 consumer no longer holds a resident rocBLAS handle. The
    raw-Q6 diagnostic producer must keep acquiring one and issuing its FP16
    GEMM, so removing the resident plumbing cannot silently downgrade the
    diagnostic surface it exists to provide.
    """

    calls = []

    class Handle:
        def gemm_ex_rowmajor_nt_fp16_compute_f16(self, *args, **kwargs):
            calls.append("gemm")

    monkeypatch.setattr(
        source, "get_rocblas", lambda: calls.append("handle") or Handle()
    )
    monkeypatch.setattr(
        source,
        "gguf_q6_k_dequantize_bf16_to_f16_source_fused",
        lambda *a, **kw: calls.append("dequant"),
    )
    monkeypatch.setattr(
        source, "fp16_to_bf16", lambda *a, **kw: calls.append("publish")
    )

    source.gguf_q6_k_f16_rocblas_bf16_bf16_out(
        1, 2, 3, 4, 5, 6, 17, 256, 72, stream=0, runtime=object()
    )

    assert calls == ["dequant", "handle", "gemm", "publish"]


def test_k16_forwards_raw_pointer_abi_and_runtime_errors():
    seen = []

    class Function:
        def __call__(self, *args):
            seen.append(tuple(arg.value for arg in args))
            return 7

    fn = Function()
    runtime = SimpleNamespace(check=lambda error: seen.append(error))
    source._launch_source_f16_k16(
        1, 2, 3, 517, 6144, 16, 48, 32, stream=9,
        library=SimpleNamespace(hipengine_source_f16_k16_bf16=fn), runtime=runtime)
    assert seen == [(1, 2, 3, 517, 6144, 16, 48, 32, 9), 7]
