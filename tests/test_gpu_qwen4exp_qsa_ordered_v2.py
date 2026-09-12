"""Ordered three-pass v2 decode preserves the parent's exact arithmetic."""

import ctypes

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.kernels.hip_gfx1100.attention import qwen4_exp_qsa as qsa
from hipengine.loading.materialize import float_array_to_bf16_bits


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


class OrderedV2Fixture:
    """c1 decode fixture for the ordered parent and its v2 rewrite."""

    def __init__(self, selected, edge=False, query_scale=1.0, capacity=4352):
        self.selected_count = selected
        self.runtime = get_hip_runtime()
        self.allocations = []
        rng = np.random.default_rng(506)
        self.query = (rng.normal(0, 0.4, (1, 24, 256)) * query_scale).astype(np.float32)
        self.key = float_array_to_bf16_bits(
            rng.normal(0, 0.4, (capacity, 2, 256)).astype(np.float32))
        self.value = float_array_to_bf16_bits(
            rng.normal(0, 0.4, (capacity, 2, 256)).astype(np.float32))
        pages = 4352 // 256
        self.tables = rng.permutation(pages).astype(np.int32)[None, :]
        if selected <= capacity:
            self.positions = np.sort(rng.choice(capacity, size=selected, replace=False))
        else:
            self.positions = np.arange(selected) % capacity
        if edge and selected >= 4:
            self.positions = self.positions.copy()
            self.positions[0] = -1
            self.positions[1] = 10 ** 9
            self.positions[2] = 4352
        self.dq, self.dk, self.dv, self.ds = [
            self.upload(v) for v in
            (self.query, self.key, self.value, self.positions.astype(np.int64))]
        dt = self.upload(self.tables)
        dl = self.upload(np.array([capacity], dtype=np.int64))
        self.spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(
                dt.ptr, self.tables.shape, DType.INT32, Device("hip", 0)),
            live_counts=Tensor.from_handle(
                dl.ptr, (1,), DType.INT64, Device("hip", 0)),
            max_live_count=capacity, storage_dtype=DType.BF16)
        self.parent_out = self.upload(np.full(self.query.shape, 23.0, np.float32))
        self.candidate_out = self.upload(np.full(self.query.shape, 23.0, np.float32))
        self.scores = self.upload(np.zeros((24, selected), np.float32))
        self.coefficients = self.upload(np.zeros((2, 24, selected), np.float32))
        self.library = qsa.build_qwen4_exp_qsa(load=True)

    def upload(self, values):
        values = np.ascontiguousarray(values)
        p = malloc(values.nbytes, runtime=self.runtime)
        self.allocations.append(p)
        copy_host_to_device(p, host_array_ptr(values), runtime=self.runtime)
        return p

    def run(self, candidate):
        target = self.candidate_out if candidate else self.parent_out
        fn = (qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_v2_f32
              if candidate else
              qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32)
        fn(self.dq.ptr, self.dk.ptr, self.dv.ptr, self.ds.ptr,
           self.scores.ptr, self.coefficients.ptr, target.ptr, self.spans,
           selected_count=self.selected_count, block_size=256,
           query_heads=24, kv_heads=2, head_dim=256,
           library=self.library, runtime=self.runtime)
        self.runtime.device_synchronize()

    def download(self, candidate):
        out = np.empty_like(self.query)
        copy_device_to_host(
            host_array_ptr(out),
            self.candidate_out if candidate else self.parent_out,
            runtime=self.runtime)
        return out

    def close(self):
        for p in reversed(self.allocations):
            free(p, runtime=self.runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("selected,edge,query_scale", [
    (1, False, 1), (7, False, 1), (31, False, 1), (32, False, 1), (33, False, 1),
    (64, False, 1), (2048, False, 1), (2051, False, 1), (2051, True, 1),
    (2051, False, 40.0), (2051, False, 1e-3), (3000, False, 1), (4096, False, 1),
])
def test_ordered_v2_matches_parent_bitwise(selected, edge, query_scale):
    fixture = OrderedV2Fixture(selected, edge=edge, query_scale=query_scale)
    try:
        fixture.run(False)
        fixture.run(True)
        parent = fixture.download(False)
        candidate = fixture.download(True)
        assert np.array_equal(parent.view(np.uint32), candidate.view(np.uint32))
    finally:
        fixture.close()


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
def test_ordered_v2_registration_and_guards():
    from types import SimpleNamespace
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    candidate = resolve(
        backend="hip_gfx1151", layer="qsa_sparse_attention",
        quant="bf16_kv", variant="strict_ordered_three_pass_v2_spans")
    assert candidate is qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_v2_f32
    spans = SimpleNamespace(spans_mode="uniform", storage_dtype=DType.BF16)
    with pytest.raises(ValueError, match="head_dim=256"):
        candidate(1, 1, 1, 1, 1, 1, 1, spans,
                  selected_count=4, block_size=256, query_heads=24,
                  kv_heads=2, head_dim=128)
    with pytest.raises(ValueError, match="selected_count <= 4096"):
        candidate(1, 1, 1, 1, 1, 1, 1, spans,
                  selected_count=4097, block_size=256, query_heads=24,
                  kv_heads=2, head_dim=256)
