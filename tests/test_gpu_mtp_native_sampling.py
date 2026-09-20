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
