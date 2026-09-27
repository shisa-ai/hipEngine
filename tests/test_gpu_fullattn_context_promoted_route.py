"""RED gate for the promoted full-attention context route.

`qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_promoted_spans` adds
two levers over the shipped fixed256 route:

* the per-token physical-offset table is int32 instead of int64, which lifts
  residency from 2 to 3 blocks per CU at `max_context_len` 2048, and
* the QK phase scores two tokens per iteration.

Both are intended to leave every floating-point operation unchanged, so the
contract is byte equality against the shipped route, not a KL envelope. The two
tests below pin the parts that can silently corrupt a result instead of merely
slowing it down: the int32 selection boundary, and the int64 fallback the guard
is supposed to take when the offset range does not fit.

The guard test needs the JIT-built library but launches no kernel. The device
tests skip when the process HIP runtime is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

INT32_MAX = 2**31 - 1


def test_gpu_fullattn_context_offsets_fit_int32_boundary() -> None:
    """The int32 offset table is used only when the whole range provably fits."""

    if not hip_runtime_available():
        pytest.skip("process HIP runtime unavailable")

    from hipengine.kernels.hip_gfx1100.attention import (
        qwen35_fullattn_context_offsets_fit_int32 as fits,
    )

    nkv, hd = 2, 256
    stride = nkv * hd
    exact = INT32_MAX // stride

    assert fits(exact, nkv, hd) is True
    assert fits(exact + 1, nkv, hd) is False
    assert fits(0, nkv, hd) is False
    assert fits(-1, nkv, hd) is False
    assert fits(exact, 0, hd) is False
    assert fits(exact, nkv, 0) is False
    # The production prefill launch: 4352 tokens of 2 KV heads x 256 dims.
    assert fits(4352, nkv, hd) is True


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    b = x.view(np.uint32)
    lsb = (b >> 16) & 1
    return ((b + 0x7FFF + lsb) >> 16).astype(np.uint16)


@pytest.mark.parametrize(
    "rows,nqh,nkv,hd,block_size,max_ctx",
    [
        (3, 4, 1, 256, 256, 256),
        (5, 4, 2, 256, 256, 512),
        (2, 8, 2, 256, 256, 256),
    ],
)
def test_gpu_fullattn_context_promoted_route_is_byte_identical(
    hip_test_target_arch: str,
    rows: int,
    nqh: int,
    nkv: int,
    hd: int,
    block_size: int,
    max_ctx: int,
) -> None:
    """Both capacity choices must reproduce the shipped route byte for byte."""

    if not hip_runtime_available():
        pytest.skip("process HIP runtime unavailable")

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.attention import (
        build_qwen35_paged_attn_decode,
        qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_promoted_spans,
        qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
    )
    from hipengine.kvcache import KVLiveSpans

    rt = get_hip_runtime()
    lib = build_qwen35_paged_attn_decode(load=True)
    rng = np.random.default_rng(0x9E3779B9)
    blen = (max_ctx + block_size - 1) // block_size
    scale = hd ** -0.5

    q = (rng.standard_normal((rows, nqh, hd)) * 0.3).astype(np.float32)
    k = _to_bf16_bits((rng.standard_normal((blen * block_size, nkv, hd)) * 0.3).astype(np.float32))
    v = _to_bf16_bits((rng.standard_normal((blen * block_size, nkv, hd)) * 0.3).astype(np.float32))
    bt = np.tile(np.arange(blen, dtype=np.int32), rows)
    live = np.minimum(1 + np.arange(rows, dtype=np.int64) * (max_ctx // rows), max_ctx)

    bufs = []

    def up(a):
        a = np.ascontiguousarray(a)
        b = malloc(max(a.nbytes, 4), runtime=rt)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=rt)
        return b

    qb, kb, vb, lb, bb = up(q), up(k), up(v), up(live), up(bt)
    out = malloc(rows * nqh * hd * 4, runtime=rt)
    bufs.append(out)
    nbytes = rows * nqh * hd * 4

    def device(ptr, shape, dtype):
        return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))

    spans = KVLiveSpans.paged_uniform(
        block_table=device(bb.ptr, (rows * blen,), "int32"),
        live_counts=device(lb.ptr, (rows,), "int64"),
        max_live_count=max_ctx,
        storage_dtype="bf16",
    )

    def read_out() -> np.ndarray:
        got = np.empty(rows * nqh * hd, np.float32)
        copy_device_to_host(host_array_ptr(got), out, nbytes, runtime=rt)
        return got

    try:
        qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans(
            qb.ptr, kb.ptr, vb.ptr, out.ptr, spans, rows, max_ctx, block_size,
            nqh, nkv, hd, scale, library=lib, runtime=rt,
        )
        rt.device_synchronize()
        ref = read_out()

        for label, capacity in (
            ("int32", blen * block_size),
            ("int64 fallback", 0),
            ("overflow rejected", 2**40),
        ):
            out_ref = np.zeros(rows * nqh * hd, np.float32)
            copy_host_to_device(out, host_array_ptr(out_ref), nbytes, runtime=rt)
            qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_promoted_spans(
                qb.ptr, kb.ptr, vb.ptr, out.ptr, spans, rows, max_ctx, block_size,
                nqh, nkv, hd, scale, capacity, library=lib, runtime=rt,
            )
            rt.device_synchronize()
            got = read_out()
            np.testing.assert_array_equal(
                got.view(np.uint32),
                ref.view(np.uint32),
                err_msg=f"{label} route is not byte-identical to the shipped route",
            )
    finally:
        for b in bufs:
            free(b, runtime=rt)
