"""Public compact DMS generation, interruption, and HTTP ownership checks."""
from pathlib import Path
import pytest

from scripts.gguf_packed_lease_probe import _resident_runner


@pytest.mark.parametrize('prefill_mode,decision_mode', [
    ('dense_pool', 'sidecar'), ('layer_outer', 'sidecar'), ('dense_pool', 'no_evict')])
def test_public_dms_generation_cancel_refill_and_http(hip_test_target_arch, prefill_mode, decision_mode):
    from hipengine import DMSConfig, LLM, SamplingParams
    from hipengine.core.memory import memory_stats
    from hipengine.kernels.backends import HIP_TARGET_ARCH_BACKEND
    from hipengine.server.api import ServerConfig, create_app
    from fastapi.testclient import TestClient

    model = Path('/models/gguf/Qwen3.8-27B-Q4_K_M.gguf')
    metadata = Path('/models/hf/Qwen3.8-27B-Q4_K_M-DMS-W8192/dms_metadata.json')
    if not model.is_file() or not metadata.is_file():
        pytest.skip('local GGUF and external DMS sidecar required')
    baseline = memory_stats()
    config = DMSConfig(metadata, prefill_mode=prefill_mode, decision_mode=decision_mode)
    llm = LLM(str(model), backend=HIP_TARGET_ARCH_BACKEND[hip_test_target_arch],
              max_sequence_length=1024, dms=config)
    params = SamplingParams(max_tokens=4, temperature=0)
    try:
        expected = llm.generate(['The capital of France is'], params)
        runner = _resident_runner(llm)
        assert runner._kv_pool is None
        assert runner.observability_snapshot()['retention']['policy'] == 'dms'
        stream = llm.stream_detailed('Count from one to one hundred.',
                                    SamplingParams(max_tokens=64, temperature=0))
        try:
            next(stream)
            session = runner._session
            assert session._dms_backend is not None
            assert session._dms_backend.has_request(0)
            assert runner._kv_pool is None
        finally:
            stream.close()
        # A following public command serializes behind cancellation on the
        # service thread; no timing-based sleep is needed to prove reuse.
        assert llm.generate(['The capital of France is'], params) == expected
        assert not runner._rows
        sampled = llm.generate(['The capital of France is'],
                               SamplingParams(max_tokens=4, temperature=0.7, seed=7))
        assert sampled
        app = create_app(ServerConfig(model=str(model), dms=config,
            backend=HIP_TARGET_ARCH_BACKEND[hip_test_target_arch],
            max_context_tokens=1024, max_active_requests=1,
            prefix_cache='off', speculative_mtp_serving='off',
            served_model_name='dms-test'), llm=llm)
        with TestClient(app) as client:
            assert client.get('/ready').status_code == 200
            response = client.post('/v1/completions', json={
                'model':'dms-test', 'prompt':'The capital of France is',
                'temperature':0, 'max_tokens':4})
            assert response.status_code == 200, response.text
            assert response.json()['choices'][0]['text'] == expected[0]
            assert client.get('/ready').status_code == 200
    finally:
        llm.close()
    after = memory_stats()
    assert after['current_allocated_bytes'] == baseline['current_allocated_bytes']
    assert after['active_allocations'] == baseline['active_allocations']
