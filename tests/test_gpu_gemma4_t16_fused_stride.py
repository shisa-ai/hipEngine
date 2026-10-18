"""Fused resident T16 must use the split parent's arithmetic and expert stride."""
import numpy as np
import pytest

from tests.test_gpu_gguf_q4_k_t16_selected_wmma_prefill import (
    _build_compact_fixture, _hip_available, _run_t16_selected_dual_gpu,
)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("stream_capture", [False, True])
@pytest.mark.parametrize("counts,hidden,gate,up", [
    ([4, 0, 5], 256, 16, 16),
    ([0, 33, 1, 16], 512, 48, 32),
    ([7, 18, 0, 33], 512, 64, 16),
])
def test_fused_stride_matches_split_parent(counts, hidden, gate, up, stream_capture):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_t16_selected_prefill as leaf
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

    fixture = _build_compact_fixture(counts=counts, in_features=hidden,
                                    out_features_a=gate, out_features_b=up, dtype="bf16")
    parent = _run_t16_selected_dual_gpu(fixture, "bf16", raw_bits=True)
    tiles = repack_gguf_q4_k_tile16(np.concatenate(
        [fixture.qweight_a, fixture.qweight_b], axis=1)).tiles
    host_out = np.zeros_like(parent)
    runtime = get_hip_runtime()
    buffers = []
    stream = runtime.stream_create() if stream_capture else 0
    graph = executable = None
    tile_experts = np.concatenate([fixture.tile_expert, np.full(5, -1, dtype=np.int64)])
    try:
        for arr in [fixture.x_host, fixture.expert_start_compact, fixture.expert_start_wmma,
                    tile_experts, tiles, host_out]:
            arr = np.ascontiguousarray(arr)
            buf = malloc(arr.nbytes, runtime=runtime)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
        x, starts, padded, experts, resident, out = buffers
        library = leaf.build_gguf_q4_k_t16_selected_prefill(load=True)
        def launch():
            leaf.gguf_q4_k_t16_selected_dual_wmma_prefill_fused_bf16_bf16_out(
                x.ptr, starts.ptr, padded.ptr, experts.ptr, resident.ptr, out.ptr,
                fixture.compact_rows, hidden, gate, up, fixture.num_experts,
                fixture.wmma_total_rows + 5 * 16, runtime=runtime,
                stream=stream, library=library)
        launch()  # Warm module before recording.
        runtime.stream_synchronize(stream)
        if stream_capture:
            # Ensure a no-op graph cannot pass using the eager warmup result.
            host_out.fill(0x7fc0)  # BF16 NaN poison.
            copy_host_to_device(out, host_array_ptr(host_out), runtime=runtime)
            runtime.stream_begin_capture(stream)
            launch()
            graph = runtime.stream_end_capture(stream)
            executable = runtime.graph_instantiate(graph)
            runtime.graph_launch(executable, stream)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out, runtime=runtime)
    finally:
        if executable is not None:
            runtime.graph_exec_destroy(executable)
        if graph is not None:
            runtime.graph_destroy(graph)
        if stream:
            runtime.stream_destroy(stream)
        for buf in reversed(buffers):
            free(buf, runtime=runtime)
    np.testing.assert_array_equal(host_out, parent)
    decoded = (host_out.astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_allclose(decoded, fixture.reference, rtol=0.025, atol=0.025)
    reference = fixture.reference.astype(np.float64)
    candidate = decoded.astype(np.float64)
    def probabilities(logits):
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)
    p, q = probabilities(reference), probabilities(candidate)
    kl = np.sum(p * np.log(p / q), axis=-1)
    assert np.max(kl) <= 0.05
    assert np.mean(np.argmax(reference, axis=-1) == np.argmax(candidate, axis=-1)) >= 0.90
