"""Q8 block-scale WMMA: independent raw-weight oracle and ragged ownership."""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download


def available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not available(), reason="HIP unavailable")


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("case", ["positive_basis", "signed_basis", "random", "cancellation"])
@pytest.mark.parametrize("hidden,outputs", [(64, 33), (640, 320)])
def test_blockscale_raw_oracle_and_ragged_tiles(case, hidden, outputs, guarded):
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
        gguf_q8_0_selected_grouped_blockscale_prefill_bf16_bf16_out,
        gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        gguf_q8_0_selected_grouped_gemv_bf16_bf16_out,
    )
    rng = np.random.default_rng(28147)
    counts = np.array([1, 0, 17, 33], dtype=np.int64)
    starts = np.r_[0, np.cumsum(counts)].astype(np.int64)
    padded = (counts + 15) // 16 * 16
    wmma = np.r_[0, np.cumsum(padded)].astype(np.int64)
    tiles = np.repeat(np.arange(4), padded // 16).astype(np.int64)
    rows = int(starts[-1])
    scales = rng.uniform(.001, .2, (4, outputs, hidden // 32)).astype(np.float16)
    codes = rng.integers(-127, 128, (4, outputs, hidden // 32, 32), dtype=np.int8)
    if case == "positive_basis":
        codes = np.abs(codes)
    if case == "cancellation":
        codes[..., 1::2] = codes[..., ::2]
    raw = np.empty((4, outputs, hidden // 32, 34), dtype=np.uint8)
    raw[..., :2] = scales.view(np.uint8).reshape(4, outputs, hidden // 32, 2)
    raw[..., 2:] = codes.view(np.uint8)
    x = np.zeros((rows, hidden), dtype=np.float32)
    if case == "cancellation":
        x[:, ::2], x[:, 1::2] = 1, -1
    elif case != "random":
        x[np.arange(rows), np.arange(rows) * 17 % hidden] = 1
    else:
        x[:] = rng.normal(0, .2, x.shape)
    xb = bf16(x)
    xf = (xb.astype(np.uint32) << 16).view(np.float32)
    weights = (scales.astype(np.float64)[..., None] * codes).reshape(4, outputs, hidden)
    reference = np.empty((rows, outputs), dtype=np.float64)
    cpu = np.empty((rows, outputs), dtype=np.float32)
    from hipengine.kernels.cpu_reference import gguf_q8_0_gemv
    for expert in range(4):
        lo, hi = starts[expert:expert + 2]
        reference[lo:hi] = xf[lo:hi].astype(np.float64) @ weights[expert].T
        if hi > lo:
            cpu[lo:hi] = gguf_q8_0_gemv(xf[lo:hi], raw[expert].reshape(outputs, -1))
            eps = np.finfo(np.float32).eps
            bound = hidden * eps / (1 - hidden * eps) * (
                np.abs(xf[lo:hi]).astype(np.float64) @ np.abs(weights[expert]).T)
            assert np.all(np.abs(cpu[lo:hi] - reference[lo:hi]) <= bound + eps)
    runtime = get_hip_runtime()
    allocations = []
    try:
        inputs = [_upload(a, runtime, allocations) for a in (xb, starts, wmma, tiles, raw)]
        out = _alloc(rows * outputs, np.uint16, runtime, allocations)
        parent = _alloc(rows * outputs, np.uint16, runtime, allocations)
        gguf_q8_0_selected_grouped_gemv_bf16_bf16_out(
            inputs[0].ptr, inputs[1].ptr, 0, inputs[4].ptr, parent.ptr,
            rows, rows, 4, hidden, outputs, runtime=runtime)
        runtime.device_synchronize()
        parent_bits = _download(parent, (rows, outputs), np.uint16, runtime)
        extra = {}
        fn = gguf_q8_0_selected_grouped_blockscale_prefill_bf16_bf16_out
        if guarded:
            count = _alloc(1, np.int32, runtime, allocations)
            indices = _alloc(rows * outputs, np.int32, runtime, allocations)
            extra = dict(risk_count_ptr=count.ptr, risk_indices_ptr=indices.ptr,
                         risk_capacity=rows * outputs)
            fn = gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out
        previous = None
        for _ in range(3):
            fn(
                *(a.ptr for a in inputs), out.ptr, rows, 4, hidden, outputs,
                int(wmma[-1]), runtime=runtime, **extra)
            runtime.device_synchronize()
            got = _download(out, (rows, outputs), np.uint16, runtime)
            if previous is not None:
                np.testing.assert_array_equal(got, previous)
            previous = got.copy()
            if guarded:
                np.testing.assert_array_equal(got, parent_bits)
                queued = int(_download(count, (1,), np.int32, runtime)[0])
                assert 0 <= queued <= rows * outputs
                if case == "cancellation":
                    assert queued == rows * outputs
                    queue = _download(indices, (queued,), np.int32, runtime)
                    np.testing.assert_array_equal(np.sort(queue), np.arange(queued))
            elif case == "positive_basis":
                np.testing.assert_array_equal(got, bf16(reference))
            elif case == "signed_basis":
                # Unit-scale FP32 probes exhibit tiny signed-WMMA drift;
                # retain the basis coverage without asserting RNE tie parity.
                assert np.max(np.abs(got.astype(np.int32) -
                                     bf16(reference).astype(np.int32))) <= 1
            else:
                result = (got.astype(np.uint32) << 16).view(np.float32)
                np.testing.assert_allclose(result, reference, rtol=.008, atol=.0003)
        if case == "cancellation":
            np.testing.assert_allclose(
                (got.astype(np.uint32) << 16).view(np.float32), reference, atol=.0003)
            return
        from hipengine.benchmark.execution_profiles import (
            EvaluationThresholds, RowDescriptor, compare_profile_logits,
        )
        descriptors = [RowDescriptor(
            scenario_id="blockscale-outer-floor", scenario_step=row,
            request_id=str(row), teacher_step=0, category="code",
            shape="fixture", transition="prefill",
            teacher_token_id=int(np.argmax(cpu[row]))) for row in range(rows)]
        quality = compare_profile_logits(
            cpu, (got.astype(np.uint32) << 16).view(np.float32), descriptors,
            thresholds=EvaluationThresholds(mean_kl_max=.05, p95_kl_max=.05,
                p99_kl_max=.05, max_kl_max=.05, top1_min=.9, per_scope_top1_min=.9))
        assert quality["hard_gates_passed"], quality["summary"]
    finally:
        for allocation in reversed(allocations):
            free(allocation)
