"""Matched-input compact/dense attention diagnostic at the no-evict boundary."""
import numpy as np
import pytest
from tests.test_gpu_dms_compact_attn_decode_hip import (
    _device_run_splitk, _extent_buffers, _bf16_roundtrip, _hip_available,
)
from tests.test_gpu_qwen35_paged_attn_decode_direct_gate import (
    _upload, _alloc, _download, _free_all, _spans,
)


@pytest.mark.skipif(not _hip_available(), reason='HIP runtime unavailable')
@pytest.mark.parametrize('live_count', [257, 1023, 1025, 1026, 1537])
def test_no_evict_compact_and_dense_matched_inputs(hip_test_target_arch, live_count):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.attention import paged_attn_decode as dense

    heads, kv_heads, dim = 24, 4, 256
    rng = np.random.default_rng(20260926)
    q = rng.normal(size=(1, heads, dim)).astype(np.float32)
    k = rng.normal(size=(1, kv_heads, live_count, dim)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    live = np.full((1, kv_heads), live_count, np.int32)
    kb, vb, base = _extent_buffers(1, kv_heads, dim, live_count, live, k, v)
    splits = (live_count + 255) // 256
    # Same BF16 bytes, with the dense position-major cache padded by canaries.
    dk = np.full((splits * 256, kv_heads, dim), 0x7fc0, np.uint16)
    dv = dk.copy()
    dk[:live_count] = kb.reshape(kv_heads, live_count, dim).transpose(1, 0, 2)
    dv[:live_count] = vb.reshape(kv_heads, live_count, dim).transpose(1, 0, 2)
    runtime = get_hip_runtime()
    bufs = []
    with hip_target_arch_environment(hip_test_target_arch):
        compact = _device_run_splitk(q, kb, vb, base, live, dim, dim**-0.5, live_count)
        repeat = _device_run_splitk(q, kb, vb, base, live, dim, dim**-0.5, live_count)
        np.testing.assert_array_equal(compact, repeat)
        try:
            qb, kbuf, vbuf = [_upload(runtime, bufs, arr) for arr in (q, dk, dv)]
            table = _upload(runtime, bufs, np.arange(splits, dtype=np.int32))
            counts = _upload(runtime, bufs, np.array([live_count], np.int64))
            out = _alloc(runtime, bufs, heads * dim * 4)
            po = _alloc(runtime, bufs, heads * splits * dim * 4)
            pm = _alloc(runtime, bufs, heads * splits * 4)
            pl = _alloc(runtime, bufs, heads * splits * 4)
            spans = _spans(table, counts, rows=1, max_live_count=live_count, block_table_len=splits)
            library = dense.build_qwen35_paged_attn_decode(load=True)
            dense._launch_split_context(qb.ptr, kbuf.ptr, vbuf.ptr, po.ptr, pm.ptr, pl.ptr,
                spans, 256, splits, 256, heads, kv_heads, dim, dim**-0.5,
                stream=0, library=library, runtime=runtime)
            dense._launch_reduce(po.ptr, pm.ptr, pl.ptr, out.ptr, heads, splits, dim,
                                 stream=0, library=library, runtime=runtime)
            actual = _download(runtime, out, q.shape, np.float32)
        finally:
            _free_all(runtime, bufs)
    keys = _bf16_roundtrip(kb).reshape(kv_heads, live_count, dim)
    values = _bf16_roundtrip(vb).reshape(kv_heads, live_count, dim)
    reference = np.empty_like(q)
    for h in range(heads):
        scores = keys[h // 6].astype(np.float64) @ q[0, h].astype(np.float64) / np.sqrt(dim)
        p = np.exp(scores - scores.max())
        reference[0, h] = (p / p.sum()) @ values[h // 6].astype(np.float64)
    np.testing.assert_allclose(compact, reference, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-4)
    print({'live': live_count, 'compact_dense_max_abs': float(np.max(np.abs(compact-actual))),
           'compact_reference_max_abs': float(np.max(np.abs(compact-reference))),
           'dense_reference_max_abs': float(np.max(np.abs(actual-reference)))})
