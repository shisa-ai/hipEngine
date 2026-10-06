"""EG-5 parent parity, risk membership, owned buffers and CPU smoke floor."""
from __future__ import annotations

import ctypes
import dataclasses

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
from hipengine.quant.gguf import GGMLQuantizationType
from tests.test_gpu_gguf_q4_k_selected_wmma_prefill import _build_compact_fixture
from tests.test_gpu_qwen4exp_q4_iu8_exact import _adversarial_x, _upload, _download

NAME = "gguf_q4_k_selected_dual_wmma_iu8_risk_narrow_prefill_bf16_bf16_out"
SYMBOL = "hipengine_" + NAME


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not hip_available(), reason="HIP runtime unavailable")


def bits_to_float(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def run(fixture, fn, library, runtime, allocations, capacity=None, stream=0):
    arrays = [fixture.x_host, fixture.expert_start_compact, fixture.expert_start_wmma,
              fixture.tile_expert, fixture.qweight_a, fixture.qweight_b]
    inputs = [_upload(a, runtime, allocations) for a in arrays]
    total = fixture.compact_rows * (fixture.out_features_a + fixture.out_features_b)
    capacity = total if capacity is None else capacity
    guard = 64
    output_host = np.full(total + 2 * guard, 0xA55A, dtype=np.uint16)
    index_host = np.full(capacity + 2 * guard, -17017, dtype=np.int32)
    output = _upload(output_host, runtime, allocations)
    indices = _upload(index_host, runtime, allocations)
    counter = _upload(np.zeros(1, np.int32), runtime, allocations)
    fn(*(d.ptr for d in inputs), output.ptr + guard * 2, counter.ptr, indices.ptr + guard * 4,
       capacity, 1.0, fixture.compact_rows, fixture.in_features, fixture.out_features_a,
       fixture.out_features_b, fixture.num_experts, fixture.wmma_total_rows,
       library=library, runtime=runtime, stream=stream)
    if stream:
        runtime.stream_synchronize(stream)
    pre = _download(output, output_host.shape, np.uint16, runtime)
    count = int(_download(counter, (1,), np.int32, runtime)[0])
    queue = _download(indices, index_host.shape, np.int32, runtime)
    assert 0 <= count <= total
    np.testing.assert_array_equal(pre[:guard], output_host[:guard])
    np.testing.assert_array_equal(pre[-guard:], output_host[-guard:])
    np.testing.assert_array_equal(queue[:guard], index_host[:guard])
    np.testing.assert_array_equal(queue[-guard:], index_host[-guard:])
    membership = np.sort(queue[guard:guard + min(count, capacity)])
    assert np.all((membership >= 0) & (membership < total))
    assert len(np.unique(membership)) == len(membership)
    q4.gguf_q4_k_selected_dual_sparse_exact_repair_bf16(
        inputs[0].ptr, inputs[1].ptr, inputs[4].ptr, inputs[5].ptr, output.ptr + guard * 2,
        counter.ptr, indices.ptr + guard * 4, capacity, fixture.compact_rows,
        fixture.in_features, fixture.out_features_a, fixture.out_features_b,
        fixture.num_experts, library=library, runtime=runtime, stream=stream)
    if stream:
        runtime.stream_synchronize(stream)
    post = _download(output, output_host.shape, np.uint16, runtime)
    np.testing.assert_array_equal(post[:guard], output_host[:guard])
    np.testing.assert_array_equal(post[-guard:], output_host[-guard:])
    for d, host in zip(inputs, arrays):
        np.testing.assert_array_equal(_download(d, host.shape, host.dtype, runtime), host)
    shape = (fixture.compact_rows, fixture.out_features_a + fixture.out_features_b)
    return pre[guard:-guard].reshape(shape), post[guard:-guard].reshape(shape), count, membership


@pytest.mark.parametrize("counts,k,kind,a,b", [
    ([1, 0, 16, 17, 31], 256, None, 128, 256),
    ([33, 0, 3, 64], 2560, None, 256, 128),
    ([17, 0, 5, 1], 256, "cancel", 128, 128),
    ([17, 0, 5, 1], 256, "tiny", 128, 128),
    ([17, 0, 5, 1], 256, "subtiny", 128, 128),
    ([17, 0, 5, 1], 256, "nonfinite_rows", 128, 128),
    ([17, 0, 5, 1], 256, "huge", 128, 128),
    ([17, 0, 5, 1], 256, "ties", 128, 128),
    ([17, 0, 5, 1], 256, "wide_scale", 128, 128),
])
def test_narrow_matches_wide_risk_chain_and_guards(counts, k, kind, a, b):
    candidate = getattr(q4, NAME)
    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fixture = _build_compact_fixture(counts=counts, in_features=k,
                                    out_features_a=128, out_features_b=128,
                                    dtype="bf16", seed=23)
    # The shared uint8 synthetic generator is bounded to columns<128 on
    # NumPy2. Extend by repeating valid raw weight rows, without changing
    # shared fixture code or weakening the unequal-plane address test.
    fixture = dataclasses.replace(
        fixture, out_features_a=a, out_features_b=b,
        qweight_a=np.tile(fixture.qweight_a, (1, a // 128, 1)),
        qweight_b=np.tile(fixture.qweight_b, (1, b // 128, 1)),
    )
    if kind:
        fixture = dataclasses.replace(fixture, x_host=_adversarial_x(kind, fixture.compact_rows, k, 31))
    allocations = []
    try:
        parent = run(fixture, q4.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out,
                     library, runtime, allocations)
        for _ in range(2):
            sibling = run(fixture, candidate, library, runtime, allocations)
            for old, new in zip(parent, sibling):
                np.testing.assert_array_equal(old, new)
    finally:
        for d in reversed(allocations):
            free(d, runtime=runtime)


@pytest.mark.parametrize("capacity", [0, 3, 255])
def test_bounded_capacity_preserves_count_and_output_guards(capacity):
    candidate = getattr(q4, NAME)
    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fixture = _build_compact_fixture(counts=[9, 0, 3], in_features=256,
                                    out_features_a=128, out_features_b=128, dtype="bf16", seed=23)
    fixture = dataclasses.replace(fixture, x_host=_adversarial_x("tiny", fixture.compact_rows, 256, 31))
    allocations = []
    try:
        old = run(fixture, q4.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out,
                  library, runtime, allocations, capacity=capacity)
        new = run(fixture, candidate, library, runtime, allocations, capacity=capacity)
        assert old[2] == new[2] > capacity
        # Truncated queues can differ in scheduling order. Every member must
        # be valid/unique (checked in run), and forced tiny rows queue all
        # outputs. Do not require identical *subsets* for truncated buffers.
        np.testing.assert_array_equal(old[0], new[0])
        if capacity == 0:
            np.testing.assert_array_equal(old[1], new[1])
    finally:
        for d in reversed(allocations):
            free(d, runtime=runtime)


def test_raw_int64_abi_refuses_overflow_before_touching_pointers():
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fn = getattr(library, SYMBOL)
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64, ctypes.c_float] + [ctypes.c_int64] * 6 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    for rows, k, a, b, experts, padded in [
        (500000, 2560, 128, 128, 1, 500000),
        (16, 2560, 640, 128, 3000, 16),
        (16, 2560, 128, 640, 3000, 16),
        (16, 2**32, 128, 128, 1, 16),
        (32, 256, 128, 128, 1, 16),
        (16, 256, 128, 128, 1, 48),
    ]:
        assert fn(*([1] * 9), 1, 1.0, rows, k, a, b, experts, padded, 0) == 1
    for capacity in [-1, 2**31 // 4, 2**32]:
        assert fn(*([1] * 9), capacity, 1.0, 16, 256, 128, 128, 1, 16, 0) == 1


@pytest.mark.parametrize("field,value", [
    ("expert_start_compact", [-1, 16]),
    ("expert_start_compact", [0, 2**32]),
    ("expert_start_compact", [16, 0]),
    ("expert_start_wmma", [-1, 16]),
    ("expert_start_wmma", [2**32, 16]),
    ("tile_expert", [2**32]),
])
def test_invalid_device_map_is_not_narrowed_or_published(field, value):
    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fixture = _build_compact_fixture(counts=[16], in_features=256,
                                    out_features_a=128, out_features_b=128, dtype="bf16", seed=23)
    fixture = dataclasses.replace(fixture, **{field: np.array(value, np.int64)})
    allocations = []
    try:
        pre, post, count, membership = run(fixture, getattr(q4, NAME), library, runtime, allocations)
        assert count == len(membership) == 0
        assert np.all(pre == 0xA55A)
        np.testing.assert_array_equal(pre, post)
    finally:
        for d in reversed(allocations):
            free(d, runtime=runtime)


def test_nondefault_stream_preserves_chain_ownership():
    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fixture = _build_compact_fixture(counts=[17, 0, 3], in_features=256,
                                    out_features_a=128, out_features_b=128, dtype="bf16", seed=23)
    allocations = []
    stream = runtime.stream_create()
    try:
        old = run(fixture, q4.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out,
                  library, runtime, allocations, stream=stream)
        new = run(fixture, getattr(q4, NAME), library, runtime, allocations, stream=stream)
        for a, b in zip(old, new):
            np.testing.assert_array_equal(a, b)
    finally:
        runtime.stream_destroy(stream)
        for d in reversed(allocations):
            free(d, runtime=runtime)


def test_narrow_repaired_output_passes_independent_cpu_floor():
    candidate = getattr(q4, NAME)
    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fixture = _build_compact_fixture(counts=[3, 0, 5], in_features=256,
                                    out_features_a=128, out_features_b=128, dtype="bf16", seed=23)
    allocations = []
    try:
        _, output, _, _ = run(fixture, candidate, library, runtime, allocations)
        ref = np.empty(output.shape, np.float32)
        x = bits_to_float(fixture.x_host)
        for expert in range(fixture.num_experts):
            start, end = fixture.expert_start_compact[expert:expert + 2]
            for row in range(start, end):
                ref[row, :128] = gguf_quant_gemv(x[row:row + 1], fixture.qweight_a[expert], GGMLQuantizationType.Q4_K)[0]
                ref[row, 128:] = gguf_quant_gemv(x[row:row + 1], fixture.qweight_b[expert], GGMLQuantizationType.Q4_K)[0]
        actual = bits_to_float(output)
        np.testing.assert_allclose(actual, ref, rtol=0.02, atol=0.02)
        p = np.exp(ref.astype(np.float64) - ref.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
        q = np.exp(actual.astype(np.float64) - actual.max(axis=1, keepdims=True)); q /= q.sum(axis=1, keepdims=True)
        assert np.mean(np.sum(p * np.log(p / q), axis=1)) <= 0.05
        assert np.mean(ref.argmax(axis=1) == actual.argmax(axis=1)) >= 0.90
    finally:
        for d in reversed(allocations):
            free(d, runtime=runtime)
