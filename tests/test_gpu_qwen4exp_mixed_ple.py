"""Mixed PLE waves match sequential request-local history and output oracles."""
import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL('libamdhip64.so')
    HIP_AVAILABLE = True
except OSError:
    HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HIP_AVAILABLE, reason='HIP runtime is not available')

from hipengine.core.memory import copy_host_to_device, free, host_array_ptr
from hipengine.runtime.qwen4_exp_batch_ple import Qwen4ExpBatchPleState
from hipengine.runtime.qwen4_exp_mixed_layout import Qwen4ExpMixedRequest, compile_mixed_layout
from hipengine.runtime.qwen4_exp_mixed_ple import Qwen4ExpMixedPleAdapter
from tests.test_gpu_qwen4exp_batch_ple import (
    BRANCHES, HIDDEN, CHANNELS, KERNEL, SLOTS, Weights,
    _build_for_detected_target, _download, _initial_history, _run_c1, _upload,
)
from hipengine.loading.materialize import float_array_to_bf16_bits


@pytest.mark.parametrize('permuted', [False, True])
def test_mixed_ple_output_and_accepted_history_match_request_oracles(permuted):
    from hipengine.core.hip import get_hip_runtime
    runtime = get_hip_runtime()
    rng = np.random.default_rng(54)
    dilation = 3
    weights = Weights(rng, runtime)
    live = Qwen4ExpBatchPleState.allocate(
        state_slots=SLOTS, branches=BRANCHES, hidden=HIDDEN,
        conv_kernel=KERNEL, dilation=dilation, runtime=runtime,
    )
    allocations = []
    adapter = None
    try:
        initial = _initial_history(rng, live.history_rows)
        copy_host_to_device(live.history, host_array_ptr(initial), runtime=runtime)
        requests = (Qwen4ExpMixedRequest(1, 3, 21, 7, 2, (3, 4)),
                    Qwen4ExpMixedRequest(2, 0, 22, 9, 5, ()),
                    Qwen4ExpMixedRequest(3, 2, 23, 4, 6, (7,)))
        layout = compile_mixed_layout(requests, physical_row_order=(3, 0, 4, 1, 5, 2) if permuted else None)
        residual = float_array_to_bf16_bits(rng.normal(0, 0.2, (layout.total_rows, BRANCHES, HIDDEN)).astype(np.float32))
        embedding = rng.normal(0, 0.2, (layout.total_rows, HIDDEN)).astype(np.float32)
        d_residual = _upload(residual, runtime, allocations)
        d_embedding = _upload(embedding, runtime, allocations)
        snapshots = {}
        expected = np.empty_like(residual)
        for request in requests:
            history = initial[request.slot].copy()
            for row in layout.rows_for_request(request.request_id):
                output, history = _run_c1(weights, residual[row.physical_row], embedding[row.physical_row],
                                          history, dilation, runtime, allocations)
                expected[row.physical_row] = output
                snapshots[(request.request_id, row.depth)] = history.copy()
        adapter = Qwen4ExpMixedPleAdapter(layout, live, owner_of={3: 21, 0: 22, 2: 23}.get)
        output = adapter.verify(
            d_residual, d_embedding, weights.device,
            norm_key_ptr=weights.d_norm_key.ptr, norm_query_ptr=weights.d_norm_query.ptr,
            norm_conv_ptr=weights.d_norm_conv.ptr, conv_weight_ptr=weights.d_conv.ptr,
        )
        actual = _download(output, expected.shape, np.uint16, runtime)
        # Projection width differs from c1, so compare at the established PLE
        # BF16 rounding boundary rather than asserting cross-width bit identity.
        from hipengine.quant.gguf import bf16_to_float32
        left, right = bf16_to_float32(actual), bf16_to_float32(expected)
        np.testing.assert_allclose(left, right, atol=0.004, rtol=0.008)
        np.testing.assert_array_equal(_download(live.history, initial.shape, np.float32, runtime), initial)
        adapter.commit((1, 0, 1))
        committed = _download(live.history, initial.shape, np.float32, runtime)
        np.testing.assert_array_equal(committed[1], initial[1])
        for request, depth in zip(requests, (1, 0, 1), strict=True):
            np.testing.assert_allclose(committed[request.slot], snapshots[(request.request_id, depth)], atol=2e-6, rtol=2e-5)
    finally:
        if adapter is not None:
            adapter.close()
        live.close()
        weights.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
