from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("rows", [2, 4, 7])
@pytest.mark.parametrize("top_p,min_p", [(1.0, 0.0), (0.95, 0.0), (0.73, 0.1)])
def test_captured_native_chain_matches_independent_ar_samples(rows, top_p, min_p):
    try:
        from hipengine.core.hip import get_hip_runtime
        runtime = get_hip_runtime()
        pointer = runtime.malloc(8)
        runtime.free(pointer)
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host
    from hipengine.generation.sampling import RowSamplingState
    from hipengine.kernels.hip_gfx1100.sampling.sampler import build_sampler
    from hipengine.llm import SamplingParams
    from hipengine.runtime.native_sampler import NativeSamplerChainWorkspace, NativeSamplerWorkspace

    vocab = 1027
    logits = np.random.default_rng(11).normal(size=(rows, vocab)).astype(np.float32)
    library = build_sampler(load=True)
    params = SamplingParams(temperature=0.7, top_p=top_p, min_p=min_p, seed=17)
    inputs = malloc(logits.nbytes, runtime=runtime)
    output = malloc(rows * 4, runtime=runtime)
    ar = NativeSamplerWorkspace(runtime=runtime, vocab_size=vocab, sampler_library=library)
    chain = NativeSamplerChainWorkspace(
        runtime=runtime, vocab_size=vocab, rows=rows, sampler_library=library,
    )
    stream = runtime.stream_create()
    graph = graph_exec = None
    try:
        copy_host_to_device(inputs, logits.ctypes.data, logits.nbytes, runtime=runtime)
        chain.stage(SimpleNamespace(params=params, seed=17, step_index=0))
        runtime.stream_begin_capture(stream)
        chain.enqueue(inputs.ptr, output.ptr, stream=stream)
        graph = runtime.stream_end_capture(stream)
        graph_exec = runtime.graph_instantiate(graph)
        for step in (0, 13, 37):
            chain.stage(SimpleNamespace(params=params, seed=17, step_index=step))
            runtime.graph_launch(graph_exec, stream)
            selected = np.empty(rows, dtype=np.int32)
            runtime.device_synchronize()
            copy_device_to_host(selected.ctypes.data, output, selected.nbytes, runtime=runtime)
            expected = []
            for row in range(rows):
                state = RowSamplingState(seed=17)
                state.step_index = step + row
                expected.append(ar.sample(inputs.ptr + row * vocab * 4, params, state).token_id)
            np.testing.assert_array_equal(selected, expected)
    finally:
        if graph_exec is not None:
            runtime.graph_exec_destroy(graph_exec)
        if graph is not None:
            runtime.graph_destroy(graph)
        runtime.stream_destroy(stream)
        chain.close()
        ar.close()
        free(output, runtime=runtime)
        free(inputs, runtime=runtime)


def test_packed_request_sampling_isolated_under_reorder_and_workspace_reuse():
    try:
        from hipengine.core.hip import get_hip_runtime
        runtime = get_hip_runtime()
        pointer = runtime.malloc(8)
        runtime.free(pointer)
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host
    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
    from hipengine.generation.sampling import RowSamplingState
    from hipengine.kernels.hip_gfx1100.sampling.sampler import build_sampler
    from hipengine.llm import SamplingParams
    from hipengine.runtime.native_sampler import NativeSamplerWorkspace

    vocab = 257
    logits = np.random.default_rng(133).normal(size=(8, vocab)).astype(np.float32)
    source = malloc(logits.nbytes, runtime=runtime)
    output = malloc(8 * 4, runtime=runtime)
    workspace = NativeSamplerWorkspace(runtime=runtime, vocab_size=vocab, sampler_library=build_sampler(load=True))
    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    owner = SimpleNamespace(
        runtime=runtime, runner=SimpleNamespace(vocab_size=vocab),
        _verify_logits_buf=source, _native_sampler=lambda: workspace,
    )
    rows = tuple(SimpleNamespace(
        request_id=i, native_sampler=True, native_sampled=True,
        sampling_state=RowSamplingState(seed=seed, step_index=step),
        sampling_request=SamplingParams(temperature=temp, top_p=0.95),
    ) for i, seed, step, temp in ((7, 17, 3, 0.7), (2, 29, 19, 1.1)))
    results = tuple(SimpleNamespace(
        request_id=row.request_id, transaction_id=9, row_start=i*4, rows=4,
        target_top1=SimpleNamespace(ptr=output.ptr+i*16),
    ) for i, row in enumerate(rows))
    try:
        copy_host_to_device(source, logits.ctypes.data, logits.nbytes, runtime=runtime)
        expected = []
        for i, row in enumerate(rows):
            for depth in range(4):
                state = row.sampling_state.clone()
                state.step_index += depth
                expected.append(workspace.sample(
                    source.ptr+(i*4+depth)*vocab*4, row.sampling_request, state,
                ).token_id)
        for order in ((0, 1), (1, 0), (0, 1)):
            adapter._sample_packed_target_rows(
                owner, tuple(results[i] for i in order), tuple(rows[i] for i in order),
                transaction_id=9,
            )
            selected = np.empty(8, dtype=np.int32)
            copy_device_to_host(selected.ctypes.data, output, selected.nbytes, runtime=runtime)
            np.testing.assert_array_equal(selected, expected)
        assert [row.sampling_state.step_index for row in rows] == [3, 19]
    finally:
        for chain in getattr(adapter, "_packed_sampler_workspaces", {}).values():
            chain.close()
        workspace.close()
        free(output, runtime=runtime)
        free(source, runtime=runtime)
