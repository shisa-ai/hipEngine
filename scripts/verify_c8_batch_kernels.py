"""One-shot C8 phase-1 kernel verification: batched glue + LM-head vs sequential single-row.

GPU-gated; run under the same env as the hipEngine CUDA tests:
  HIPENGINE_RUN_CUDA_SM120A=1 HIPENGINE_CUDA_ARCH=sm_120a CUDA_VISIBLE_DEVICES=0
  PYTHONPATH=$PWD uv run --no-project python scripts/verify_c8_batch_kernels.py
"""

from __future__ import annotations

import numpy as np

from hipengine.core.cuda import get_cuda_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)

HEADS = 8
HEAD_DIM = 52
HIDDEN = 416
CAPACITY = 194
VOCAB = 36864
MAX_POS = 100000
ROTARY = 32
ROWS_PER_BLOCK = 8


def _alloc(shape, dtype, runtime, allocations):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = malloc(nbytes, runtime=runtime)
    allocations.append(buf)
    return buf


def _upload(arr, runtime, allocations):
    host = np.ascontiguousarray(arr)
    buf = _alloc(host.shape, host.dtype, runtime, allocations)
    copy_host_to_device(buf, host_array_ptr(host), runtime=runtime)
    return buf


def _download(buf, shape, dtype, runtime):
    host = np.empty(int(np.prod(shape)), dtype=dtype)
    copy_device_to_host(host_array_ptr(host), buf, runtime=runtime)
    return host.reshape(shape)


def _zero(buf, runtime):
    runtime.memset_async(buf.ptr, 0, buf.nbytes, 0)
    runtime.device_synchronize()


def main() -> None:
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        build_moonshine_glue,
        moonshine_embedding_lookup_batch_fp16,
        moonshine_embedding_lookup_fp16,
        moonshine_partial_rope_cache_append_batch_fp16,
        moonshine_partial_rope_cache_append_fp16,
    )
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        build_moonshine_lm_head,
        lm_head_argmax_scratch_elements,
        moonshine_lm_head_argmax_batch_fp16,
        moonshine_lm_head_argmax_fp16,
    )

    rng = np.random.default_rng(0xC8B47C8)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    glue = build_moonshine_glue(load=True)
    head = build_moonshine_lm_head(load=True)
    allocations: list[object] = []
    batch = 4
    try:
        # ---- batched embedding lookup ------------------------------------
        embedding = rng.normal(0.0, 0.05, size=(VOCAB, HIDDEN)).astype(np.float16)
        tokens = np.array([7, 0, 36863, 12345], dtype=np.int64)
        d_emb = _upload(embedding, runtime, allocations)
        d_tok = _upload(tokens, runtime, allocations)
        d_out = _alloc((batch, HIDDEN), np.float16, runtime, allocations)
        moonshine_embedding_lookup_batch_fp16(
            d_emb.ptr, d_tok.ptr, d_out.ptr, HIDDEN, VOCAB, batch,
            library=glue, runtime=runtime,
        )
        runtime.device_synchronize()
        batched = _download(d_out, (batch, HIDDEN), np.float16, runtime)
        single = _alloc((1, HIDDEN), np.float16, runtime, allocations)
        for row in range(batch):
            d_tok1 = _upload(tokens[row : row + 1], runtime, allocations)
            moonshine_embedding_lookup_fp16(
                d_emb.ptr, d_tok1.ptr, single.ptr, HIDDEN, VOCAB,
                library=glue, runtime=runtime,
            )
            runtime.device_synchronize()
            expected = _download(single, (1, HIDDEN), np.float16, runtime)[0]
            assert np.array_equal(batched[row], expected), f"embedding row {row}"
        print(f"embedding lookup batch: bit-exact across {batch} rows")

        # ---- batched partial-RoPE + cache append --------------------------
        positions = np.array([0, 5, 9, 193], dtype=np.int64)
        cos = rng.normal(0.0, 1.0, size=(MAX_POS, ROTARY // 2)).astype(np.float16)
        sin = rng.normal(0.0, 1.0, size=(MAX_POS, ROTARY // 2)).astype(np.float16)
        d_cos = _upload(cos, runtime, allocations)
        d_sin = _upload(sin, runtime, allocations)
        d_pos = _upload(positions, runtime, allocations)
        d_query = _upload(rng.normal(0, 0.4, (batch, HEADS * HEAD_DIM)).astype(np.float16), runtime, allocations)
        d_key = _upload(rng.normal(0, 0.4, (batch, HEADS * HEAD_DIM)).astype(np.float16), runtime, allocations)
        d_value = _upload(rng.normal(0, 0.4, (batch, HEADS * HEAD_DIM)).astype(np.float16), runtime, allocations)
        d_qout = _alloc((batch, HEADS * HEAD_DIM), np.float16, runtime, allocations)
        d_kout = _alloc((batch, HEADS * HEAD_DIM), np.float16, runtime, allocations)
        d_kcache = _alloc((batch, HEADS * CAPACITY * HEAD_DIM), np.float16, runtime, allocations)
        d_vcache = _alloc((batch, HEADS * CAPACITY * HEAD_DIM), np.float16, runtime, allocations)
        _zero(d_kcache, runtime)
        _zero(d_vcache, runtime)
        moonshine_partial_rope_cache_append_batch_fp16(
            d_query.ptr, d_key.ptr, d_value.ptr, d_cos.ptr, d_sin.ptr, d_pos.ptr,
            d_qout.ptr, d_kout.ptr, d_kcache.ptr, d_vcache.ptr,
            HEADS, HEAD_DIM, ROTARY, CAPACITY, MAX_POS, batch,
            library=glue, runtime=runtime,
        )
        runtime.device_synchronize()
        b_qout = _download(d_qout, (batch, HEADS * HEAD_DIM), np.float16, runtime)
        b_kout = _download(d_kout, (batch, HEADS * HEAD_DIM), np.float16, runtime)
        b_kcache = _download(d_kcache, (batch, HEADS * CAPACITY * HEAD_DIM), np.float16, runtime)
        b_vcache = _download(d_vcache, (batch, HEADS * CAPACITY * HEAD_DIM), np.float16, runtime)
        s_qout = _alloc((1, HEADS * HEAD_DIM), np.float16, runtime, allocations)
        s_kout = _alloc((1, HEADS * HEAD_DIM), np.float16, runtime, allocations)
        s_kcache = _alloc((1, HEADS * CAPACITY * HEAD_DIM), np.float16, runtime, allocations)
        s_vcache = _alloc((1, HEADS * CAPACITY * HEAD_DIM), np.float16, runtime, allocations)
        _zero(s_kcache, runtime)
        _zero(s_vcache, runtime)
        host_q = _download(d_query, (batch, HEADS * HEAD_DIM), np.float16, runtime)
        host_k = _download(d_key, (batch, HEADS * HEAD_DIM), np.float16, runtime)
        host_v = _download(d_value, (batch, HEADS * HEAD_DIM), np.float16, runtime)
        for row in range(batch):
            d_q1 = _upload(host_q[row : row + 1], runtime, allocations)
            d_k1 = _upload(host_k[row : row + 1], runtime, allocations)
            d_v1 = _upload(host_v[row : row + 1], runtime, allocations)
            d_p1 = _upload(positions[row : row + 1], runtime, allocations)
            _zero(s_kcache, runtime)
            _zero(s_vcache, runtime)
            moonshine_partial_rope_cache_append_fp16(
                d_q1.ptr, d_k1.ptr, d_v1.ptr, d_cos.ptr, d_sin.ptr, d_p1.ptr,
                s_qout.ptr, s_kout.ptr, s_kcache.ptr, s_vcache.ptr,
                HEADS, HEAD_DIM, ROTARY, CAPACITY, MAX_POS,
                library=glue, runtime=runtime,
            )
            runtime.device_synchronize()
            e_qout = _download(s_qout, (1, HEADS * HEAD_DIM), np.float16, runtime)[0]
            e_kout = _download(s_kout, (1, HEADS * HEAD_DIM), np.float16, runtime)[0]
            e_kcache = _download(s_kcache, (1, HEADS * CAPACITY * HEAD_DIM), np.float16, runtime)[0]
            e_vcache = _download(s_vcache, (1, HEADS * CAPACITY * HEAD_DIM), np.float16, runtime)[0]
            assert np.array_equal(b_qout[row], e_qout), f"rope qout row {row}"
            assert np.array_equal(b_kout[row], e_kout), f"rope kout row {row}"
            assert np.array_equal(b_kcache[row], e_kcache), f"rope kcache row {row}"
            assert np.array_equal(b_vcache[row], e_vcache), f"rope vcache row {row}"
        print(f"partial-ROPE+cache-append batch: bit-exact across {batch} rows (pos {positions.tolist()})")

        # ---- batched LM head ----------------------------------------------
        input_rows = rng.normal(0.0, 1.0, size=(batch, HIDDEN)).astype(np.float16)
        weight = rng.normal(0.0, 0.02, size=(VOCAB, HIDDEN)).astype(np.float16)
        num_blocks = lm_head_argmax_scratch_elements(VOCAB, ROWS_PER_BLOCK)
        d_in = _upload(input_rows, runtime, allocations)
        d_w = _upload(weight, runtime, allocations)
        d_bv = _alloc((batch, num_blocks), np.float32, runtime, allocations)
        d_bi = _alloc((batch, num_blocks), np.int64, runtime, allocations)
        d_idx = _alloc((batch,), np.int64, runtime, allocations)
        d_val = _alloc((batch,), np.float32, runtime, allocations)
        moonshine_lm_head_argmax_batch_fp16(
            d_in.ptr, d_w.ptr, d_bv.ptr, d_bi.ptr, d_idx.ptr, d_val.ptr,
            HIDDEN, VOCAB, batch, rows_per_block=ROWS_PER_BLOCK,
            library=head, runtime=runtime,
        )
        runtime.device_synchronize()
        b_idx = _download(d_idx, (batch,), np.int64, runtime)
        b_val = _download(d_val, (batch,), np.float32, runtime)
        s_idx = _alloc((1,), np.int64, runtime, allocations)
        s_val = _alloc((1,), np.float32, runtime, allocations)
        s_bv = _alloc((1, num_blocks), np.float32, runtime, allocations)
        s_bi = _alloc((1, num_blocks), np.int64, runtime, allocations)
        for row in range(batch):
            d_in1 = _upload(input_rows[row : row + 1], runtime, allocations)
            moonshine_lm_head_argmax_fp16(
                d_in1.ptr, d_w.ptr, s_bv.ptr, s_bi.ptr, s_idx.ptr, s_val.ptr,
                HIDDEN, VOCAB, rows_per_block=ROWS_PER_BLOCK,
                library=head, runtime=runtime,
            )
            runtime.device_synchronize()
            e_idx = _download(s_idx, (1,), np.int64, runtime)[0]
            e_val = _download(s_val, (1,), np.float32, runtime)[0]
            assert b_idx[row] == e_idx, f"lm-head token row {row}: {b_idx[row]} != {e_idx}"
            assert b_val[row] == e_val, f"lm-head value row {row}: {b_val[row]} != {e_val}"
        print(f"lm-head argmax batch: bit-exact tokens+values across {batch} rows")
        print("ALL C8 BATCH KERNELS BIT-EXACT vs sequential single-row")
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


if __name__ == "__main__":
    main()
