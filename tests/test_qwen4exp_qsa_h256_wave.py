"""H256 wave mapping preserves the parent score tree and online recurrence."""
import ctypes

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.kernels.cpu_reference.qwen4_exp import qsa_sparse_gqa_attention
from hipengine.kernels.hip_gfx1100.attention import qwen4_exp_qsa as qsa
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_h256_wave_registration_and_reject_wrong_dimension():
    from types import SimpleNamespace
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    candidate = resolve(
        backend="hip_gfx1151", layer="qsa_sparse_attention",
        quant="bf16_kv", variant="strict_h256_wave_rows_spans")
    assert candidate is qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_wave_rows_f32
    assert callable(resolve(backend="hip_gfx1151", layer="qsa_sparse_attention",
                            quant="bf16_kv", variant="strict_rows_spans"))
    with pytest.raises(ValueError, match="head_dim=256"):
        candidate(1, 1, 1, 1, 1, 1,
                  SimpleNamespace(spans_mode="uniform", storage_dtype=DType.BF16),
                  rows=1, selected_stride=1, block_size=256,
                  query_heads=24, kv_heads=2, head_dim=128)


class Fixture:
    def __init__(self, rows, stride, edge=False):
        self.rows, self.stride = rows, stride
        self.runtime = get_hip_runtime()
        self.allocations = []
        rng = np.random.default_rng(506)
        capacity = 4352
        self.query = rng.normal(0, 0.4, (rows, 24, 256)).astype(np.float32)
        self.key = float_array_to_bf16_bits(rng.normal(0, 0.4, (capacity, 2, 256)).astype(np.float32))
        self.value = float_array_to_bf16_bits(rng.normal(0, 0.4, (capacity, 2, 256)).astype(np.float32))
        self.tables = np.stack([rng.permutation(17) for _ in range(rows)]).astype(np.int32)
        self.selected = np.stack([
            np.sort(rng.choice(capacity, size=stride, replace=False))
            for _ in range(rows)
        ]).astype(np.int64)
        self.counts = np.maximum(1, stride - np.arange(rows) % 4).astype(np.int32)
        if edge:
            self.counts[-1] = 0
            self.selected[0, 0] = -1
            if stride > 1:
                self.selected[0, 1] = capacity
        self.dq, self.dk, self.dv, self.ds, self.dc = [
            self.upload(v) for v in (self.query, self.key, self.value, self.selected, self.counts)]
        dt = self.upload(self.tables)
        dl = self.upload(np.full(rows, capacity-1, np.int64))
        self.spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(dt.ptr, self.tables.shape, DType.INT32, Device("hip", 0)),
            live_counts=Tensor.from_handle(dl.ptr, (rows,), DType.INT64, Device("hip", 0)),
            max_live_count=capacity-1, storage_dtype=DType.BF16)
        self.parent = self.upload(np.full(self.query.shape, 23.0, np.float32))
        self.output = self.upload(np.full(self.query.shape, 23.0, np.float32))
        self.library = qsa.build_qwen4_exp_qsa(load=True)

    def upload(self, values):
        values = np.ascontiguousarray(values)
        p = malloc(values.nbytes, runtime=self.runtime)
        self.allocations.append(p)
        copy_host_to_device(p, host_array_ptr(values), runtime=self.runtime)
        return p

    def run(self, candidate):
        fn = (qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_wave_rows_f32
              if candidate else qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32)
        target = self.output if candidate else self.parent
        fn(self.dq.ptr, self.dk.ptr, self.dv.ptr, self.ds.ptr, self.dc.ptr,
           target.ptr, self.spans, rows=self.rows, selected_stride=self.stride,
           block_size=256, query_heads=24, kv_heads=2, head_dim=256,
           library=self.library, runtime=self.runtime)
        self.runtime.device_synchronize()

    def download(self, candidate):
        out = np.empty_like(self.query)
        copy_device_to_host(host_array_ptr(out), self.output if candidate else self.parent, runtime=self.runtime)
        return out

    def close(self):
        for p in reversed(self.allocations):
            free(p, runtime=self.runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("rows,stride,edge,query_scale", [
    (1,1,False,1), (3,33,True,1), (4,2051,False,1),
    (2,257,False,128), (2,2048,False,0.00001),
])
def test_h256_wave_exact(rows, stride, edge, query_scale):
    assert hasattr(qsa, "qwen4_exp_qsa_sparse_attention_paged_bf16_h256_wave_rows_f32")
    f = Fixture(rows, stride, edge)
    try:
        if query_scale != 1:
            f.query *= np.float32(query_scale)
            copy_host_to_device(f.dq, host_array_ptr(f.query), runtime=f.runtime)
        f.run(False)
        parent = f.download(False)
        for _ in range(2):
            f.run(True)
            got = f.download(True)
            np.testing.assert_array_equal(got.view(np.uint32), parent.view(np.uint32))
        for row in range(rows):
            selected = f.selected[row, :f.counts[row]]
            selected = selected[(selected >= 0) & (selected < 4352)]
            if not selected.size:
                continue
            physical = (f.tables[row, selected // 256] * 256 + selected % 256)
            expected = qsa_sparse_gqa_attention(
                f.query[row:row+1], bf16_to_float32(f.key[physical]),
                bf16_to_float32(f.value[physical]), query_positions=[4351],
                key_positions=selected, selected_positions=[selected])
            np.testing.assert_allclose(got[row], expected[0], rtol=2e-4, atol=2e-5)
    finally:
        f.close()
