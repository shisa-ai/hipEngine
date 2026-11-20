"""GPU parity guards for empty-split elimination in batched split-K decode.

The optimization removes zeroing work from empty split blocks and bounds the
batch gate reducers by each row's valid split count (derived device-side from
live_counts). Valid-split arithmetic is unchanged, so:

- mixed-length outputs must stay within oracle tolerance,
- a row's output must be byte-identical alone vs inside a mixed batch,
- the bounded reducer must ignore poisoned (NaN) invalid partials,
- HIP graph replay must pick up changed live_counts without recapture.

The reducer-bounds test is the RED gate for the new ABI; the rest are
regression guards for an output-neutral change (perf RED lives in
scripts/probe_paged_decode_pool_invariance.py).
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest


def _has_hip() -> bool:
    try:
        from hipengine.core.hip import get_hip_runtime
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_hip(), reason="HIP runtime not available")

BLOCK = 256
HEAD_DIM = 256
SCALE = HEAD_DIM ** -0.5


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-np.asarray(x, dtype=np.float32)))


@pytest.fixture(scope="module")
def _lib(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment(hip_test_target_arch):
        return build_qwen35_paged_attn_decode(load=True, compiler_version=compiler_version)


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


def _upload(runtime, bufs, array):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    arr = np.ascontiguousarray(array)
    buf = malloc(max(arr.nbytes, 4), runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes, runtime=runtime)
    return buf


def _alloc(runtime, bufs, nbytes):
    from hipengine.core.memory import malloc

    buf = malloc(max(nbytes, 4), runtime=runtime)
    bufs.append(buf)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, arr.nbytes, runtime=runtime)
    return arr


def _free_all(runtime, bufs):
    from hipengine.core.memory import free

    for buf in reversed(bufs):
        free(buf, runtime=runtime)


def _spans(table_buf, live_buf, rows, table_len, max_live):
    from hipengine.core.device import Device
    from hipengine.core.tensor import Tensor
    from hipengine.kvcache import KVLiveSpans

    device = Device("hip", 0)
    return KVLiveSpans.paged_uniform(
        block_table=Tensor.from_handle(table_buf.ptr, (rows, table_len), "int32", device),
        live_counts=Tensor.from_handle(live_buf.ptr, (rows,), "int64", device),
        max_live_count=max_live,
        storage_dtype="bf16",
        span_role="decode",
    )


class _BatchCase:
    """One mixed-length batch wired to a real wrapper, reusable for eager/graph."""

    def __init__(self, runtime, lib, *, wrapper, q_heads, kv_heads, gate_dtype,
                 contexts, num_splits, chunk=BLOCK, seed=11, poison=False):
        from hipengine.kernels.hip_gfx1100.attention import paged_attn_decode  # noqa: F401

        self.runtime = runtime
        self.lib = lib
        self.wrapper = wrapper
        self.rows = len(contexts)
        self.contexts = np.asarray(contexts, dtype=np.int64)
        self.num_splits = num_splits
        self.chunk = chunk
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.bufs = []
        table_len = max(num_splits, (int(self.contexts.max()) + BLOCK - 1) // BLOCK)
        table_len = max(table_len, (chunk * num_splits + BLOCK - 1) // BLOCK)
        self.table_len = table_len
        rng = np.random.default_rng(seed)
        pages = self.rows * table_len
        key = _to_bf16_bits(rng.normal(0, 0.05, (pages, BLOCK, kv_heads, HEAD_DIM)).astype(np.float32))
        value = _to_bf16_bits(rng.normal(0, 0.05, (pages, BLOCK, kv_heads, HEAD_DIM)).astype(np.float32))
        self.query = rng.normal(0, 0.05, (self.rows, q_heads, HEAD_DIM)).astype(np.float32)
        gate_f32 = rng.normal(0, 0.5, (self.rows, q_heads, HEAD_DIM)).astype(np.float32)
        self.gate_f32 = gate_f32
        gate = _to_bf16_bits(gate_f32) if gate_dtype == "bf16" else gate_f32.astype(np.float16)
        table = np.zeros((self.rows, table_len), dtype=np.int32)
        for r in range(self.rows):
            table[r] = np.arange(r * table_len, (r + 1) * table_len, dtype=np.int32)
        self.key_f32 = _from_bf16_bits(key).reshape(-1, kv_heads, HEAD_DIM)
        self.value_f32 = _from_bf16_bits(value).reshape(-1, kv_heads, HEAD_DIM)
        self.key_buf = _upload(runtime, self.bufs, key)
        self.value_buf = _upload(runtime, self.bufs, value)
        self.query_buf = _upload(runtime, self.bufs, self.query)
        self.gate_buf = _upload(runtime, self.bufs, gate)
        self.table_buf = _upload(runtime, self.bufs, table)
        self.live_buf = _upload(runtime, self.bufs, self.contexts)
        self.out_buf = _alloc(runtime, self.bufs, self.rows * q_heads * HEAD_DIM * 2)
        n_part = self.rows * q_heads * num_splits
        if poison:
            nan_f32 = np.full(n_part * HEAD_DIM, np.nan, dtype=np.float32)
            self.po_buf = _upload(runtime, self.bufs, nan_f32)
            self.pm_buf = _upload(runtime, self.bufs, np.full(n_part, np.nan, dtype=np.float32))
            self.pl_buf = _upload(runtime, self.bufs, np.full(n_part, np.nan, dtype=np.float32))
        else:
            self.po_buf = _alloc(runtime, self.bufs, n_part * HEAD_DIM * 4)
            self.pm_buf = _alloc(runtime, self.bufs, n_part * 4)
            self.pl_buf = _alloc(runtime, self.bufs, n_part * 4)
        self.spans = _spans(self.table_buf, self.live_buf, self.rows, table_len,
                            int(self.contexts.max()))

    def call(self, stream=0):
        self.wrapper(
            self.query_buf.ptr, self.key_buf.ptr, self.value_buf.ptr,
            self.gate_buf.ptr, self.out_buf.ptr,
            self.po_buf.ptr, self.pm_buf.ptr, self.pl_buf.ptr,
            self.spans, self.rows, self.chunk, self.num_splits, BLOCK,
            self.q_heads, self.kv_heads, HEAD_DIM, HEAD_DIM, 1, SCALE,
            stream=stream, library=self.lib, runtime=self.runtime,
        )

    def output_bits(self):
        return _download(self.runtime, self.out_buf,
                         (self.rows, self.q_heads, HEAD_DIM), np.uint16)

    def set_contexts(self, contexts):
        from hipengine.core.memory import copy_host_to_device, host_array_ptr

        arr = np.asarray(contexts, dtype=np.int64)
        copy_host_to_device(self.live_buf, host_array_ptr(arr), arr.nbytes, runtime=self.runtime)
        self.contexts = arr

    def oracle(self):
        out = np.empty((self.rows, self.q_heads, HEAD_DIM), dtype=np.float32)
        q_per_kv = self.q_heads // self.kv_heads
        for r in range(self.rows):
            ctx = int(self.contexts[r])
            base = r * self.table_len * BLOCK
            k = self.key_f32[base:base + ctx]
            v = self.value_f32[base:base + ctx]
            for h in range(self.q_heads):
                kvh = h // q_per_kv
                scores = (k[:, kvh] @ self.query[r, h]) * np.float32(SCALE)
                w = np.exp(scores - scores.max())
                w = w / w.sum()
                out[r, h] = (w @ v[:, kvh]) * _sigmoid(self.gate_f32[r, h])
        return out

    def close(self):
        _free_all(self.runtime, self.bufs)


def _wrappers():
    from hipengine.kernels.hip_gfx1100.attention import paged_attn_decode as m

    return {
        "bf16_gqa_24q": (m.qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans, 24, 4, "bf16"),
        "bf16_gqa_16q": (m.qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans, 16, 2, "bf16"),
        "fp16_gqa_16q": (m.qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans, 16, 2, "fp16"),
        "bf16_warp_16q": (m.qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_batch_spans, 16, 2, "bf16"),
    }


@pytest.mark.parametrize("variant", list(_wrappers()))
@pytest.mark.parametrize("contexts", [
    [1025, 255, 513, 257],   # crosses 256-token split boundaries both sides
    [768, 300, 1024, 65],    # unrelated values, exact multiple included
])
def test_mixed_length_batch_matches_oracle(_lib, _runtime, variant, contexts):
    wrapper, q, kv, gate_dtype = _wrappers()[variant]
    case = _BatchCase(_runtime, _lib, wrapper=wrapper, q_heads=q, kv_heads=kv,
                      gate_dtype=gate_dtype, contexts=contexts, num_splits=8)
    try:
        case.call()
        _runtime.stream_synchronize(0)
        got = case.output_bits()
        got_f = _from_bf16_bits(got) if gate_dtype == "bf16" else got.view(np.float16).astype(np.float32)
        ref = case.oracle()
        assert np.isfinite(got_f).all()
        np.testing.assert_allclose(got_f, ref, atol=5e-3, rtol=2e-2)
    finally:
        case.close()


@pytest.mark.parametrize("variant", ["bf16_gqa_24q", "bf16_gqa_16q", "fp16_gqa_16q"])
def test_batch_composition_invariance_byte_exact(_lib, _runtime, variant):
    wrapper, q, kv, gate_dtype = _wrappers()[variant]
    contexts = [1025, 255, 513, 257]
    mixed = _BatchCase(_runtime, _lib, wrapper=wrapper, q_heads=q, kv_heads=kv,
                       gate_dtype=gate_dtype, contexts=contexts, num_splits=8, seed=22)
    try:
        mixed.call()
        _runtime.stream_synchronize(0)
        mixed_bits = mixed.output_bits()
        for r, ctx in enumerate(contexts):
            alone = _BatchCase(_runtime, _lib, wrapper=wrapper, q_heads=q, kv_heads=kv,
                               gate_dtype=gate_dtype, contexts=[ctx],
                               num_splits=(ctx + BLOCK - 1) // BLOCK, seed=22)
            try:
                # Reuse the mixed row's query/gate/K/V so inputs are identical.
                from hipengine.core.memory import copy_host_to_device, host_array_ptr

                copy_host_to_device(alone.query_buf, host_array_ptr(mixed.query[r:r + 1]),
                                    mixed.query[r:r + 1].nbytes, runtime=_runtime)
                gate_row = mixed.gate_f32[r:r + 1]
                gate_arr = (_to_bf16_bits(gate_row) if gate_dtype == "bf16"
                            else gate_row.astype(np.float16))
                copy_host_to_device(alone.gate_buf, host_array_ptr(gate_arr),
                                    gate_arr.nbytes, runtime=_runtime)
                n_pages = (ctx + BLOCK - 1) // BLOCK
                page_bytes = BLOCK * kv * HEAD_DIM * 2
                src_off = r * mixed.table_len * page_bytes
                _runtime.memcpy(alone.key_buf.ptr, mixed.key_buf.ptr + src_off,
                                n_pages * page_bytes, 3)  # D2D
                _runtime.memcpy(alone.value_buf.ptr, mixed.value_buf.ptr + src_off,
                                n_pages * page_bytes, 3)
                alone.call()
                _runtime.stream_synchronize(0)
                alone_bits = alone.output_bits()
                np.testing.assert_array_equal(
                    alone_bits[0], mixed_bits[r],
                    err_msg=f"row {r} (ctx {ctx}) differs alone vs mixed batch")
            finally:
                alone.close()
    finally:
        mixed.close()


def test_bounded_reduce_ignores_poisoned_invalid_partials(_lib, _runtime):
    """RED for the bounds ABI: invalid partials are NaN and must never be read."""
    from hipengine.kernels.hip_gfx1100.attention import paged_attn_decode as m

    rows, q_heads, num_splits, chunk = 2, 2, 4, 256
    contexts = np.array([300, 100], dtype=np.int64)  # valid splits: 2 and 1
    rng = np.random.default_rng(33)
    valid = rng.normal(0, 0.1, (rows, q_heads, num_splits, HEAD_DIM)).astype(np.float32)
    partial_out = np.full((rows, q_heads, num_splits, HEAD_DIM), np.nan, dtype=np.float32)
    partial_m = np.full((rows, q_heads, num_splits), np.nan, dtype=np.float32)
    partial_l = np.full((rows, q_heads, num_splits), np.nan, dtype=np.float32)
    expected = np.empty((rows, q_heads, HEAD_DIM), dtype=np.float32)
    gate = rng.normal(0, 0.5, (rows, q_heads, HEAD_DIM)).astype(np.float32)
    for r in range(rows):
        n_valid = (int(contexts[r]) + chunk - 1) // chunk
        partial_out[r, :, :n_valid] = valid[r, :, :n_valid]
        partial_m[r, :, :n_valid] = 0.0
        partial_l[r, :, :n_valid] = 1.0
        expected[r] = valid[r, :, :n_valid].mean(axis=1) * _sigmoid(gate[r])
    expected_bits = _to_bf16_bits(expected)

    bufs = []
    try:
        po = _upload(_runtime, bufs, partial_out)
        pm = _upload(_runtime, bufs, partial_m)
        pl = _upload(_runtime, bufs, partial_l)
        gb = _upload(_runtime, bufs, _to_bf16_bits(gate))
        live = _upload(_runtime, bufs, contexts)
        out = _alloc(_runtime, bufs, rows * q_heads * HEAD_DIM * 2)
        m._launch_gate_reduce_batch(
            m._SYMBOL_SPLIT_REDUCE_GATE_BF16_BATCH,
            po.ptr, pm.ptr, pl.ptr, gb.ptr, out.ptr,
            rows, q_heads, num_splits, HEAD_DIM, HEAD_DIM, 1,
            stream=0, library=_lib, runtime=_runtime,
            context_len_ptr=live.ptr, chunk_size=chunk,
        )
        _runtime.stream_synchronize(0)
        got = _download(_runtime, out, (rows, q_heads, HEAD_DIM), np.uint16)
        assert np.isfinite(_from_bf16_bits(got)).all()
        np.testing.assert_allclose(_from_bf16_bits(got), _from_bf16_bits(expected_bits),
                                   atol=1e-3, rtol=1e-2)
    finally:
        _free_all(_runtime, bufs)


def test_graph_replay_tracks_live_counts_device_side(_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.attention import paged_attn_decode as m

    wrapper, q, kv, gate_dtype = _wrappers()["bf16_gqa_16q"]
    contexts_a = [1025, 255, 513, 257]
    contexts_b = [257, 1025, 255, 513]
    case = _BatchCase(_runtime, _lib, wrapper=wrapper, q_heads=q, kv_heads=kv,
                      gate_dtype=gate_dtype, contexts=contexts_a, num_splits=8, seed=44)
    stream = _runtime.stream_create()
    graph = graph_exec = 0
    try:
        case.call(stream=0)
        _runtime.stream_synchronize(0)
        eager_a = case.output_bits()

        _runtime.stream_begin_capture(stream)
        case.call(stream=stream)
        graph = _runtime.stream_end_capture(stream)
        graph_exec = _runtime.graph_instantiate(graph)

        _runtime.graph_launch(graph_exec, stream)
        _runtime.stream_synchronize(stream)
        np.testing.assert_array_equal(case.output_bits(), eager_a)

        # Change only the device live_counts; replay must re-derive split bounds.
        case.set_contexts(contexts_b)
        _runtime.stream_synchronize(0)
        _runtime.graph_launch(graph_exec, stream)
        _runtime.stream_synchronize(stream)
        replay_b = case.output_bits()

        case.call(stream=0)
        _runtime.stream_synchronize(0)
        np.testing.assert_array_equal(replay_b, case.output_bits())
    finally:
        if graph_exec:
            _runtime.graph_exec_destroy(graph_exec)
        if graph:
            _runtime.graph_destroy(graph)
        _runtime.stream_destroy(stream)
        case.close()