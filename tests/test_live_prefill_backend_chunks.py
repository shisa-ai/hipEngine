"""Public GGUF/PARO chunk fallback and explicit overrides report executed sizes."""
import ctypes
from pathlib import Path

import pytest


def _hip_available():
    try:
        ctypes.CDLL('libamdhip64.so')
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason='requires ROCm')


@pytest.mark.parametrize('kind,model,quant', [
    ('gguf', '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf', None),
    ('paro', '/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16', 'w4_paro'),
])
def test_public_chunk_policy_matches_execution(monkeypatch, kind, model, quant):
    if not Path(model).exists():
        pytest.skip('local model is unavailable')
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime import qwen35_paro_runner as paro
    from hipengine.runtime import qwen35_gguf_runner as gguf

    session_type = (paro.Qwen35ParoResidentSession if kind == 'paro'
                    else gguf.Qwen35GGUFResidentSession)
    real_init = session_type.__init__
    real_size = session_type._linear_prefill_layer_chunk_size
    real_resolve = (paro if kind == 'paro' else gguf).resolve_prefill_config_for_sequence
    sessions = []
    sizes = []
    policies = []
    chosen = [0]
    executed_rows = []
    if kind == 'gguf':
        executor_type = gguf.Qwen35GGUFFullStackRunner
        executor_name = '_run_native_attention_bulk_ffn_layer_rows'
    else:
        from hipengine.runtime.qwen35_paro import Qwen35ParoDecodeState
        executor_type = Qwen35ParoDecodeState
        executor_name = 'run_linear_attention_moe_c1_layer_fp16'
    real_execute = getattr(executor_type, executor_name)

    def execute(self, *args, **kwargs):
        if kind == 'paro' and 'tokens' in kwargs:
            executed_rows.append(int(kwargs['tokens']))
        elif kind == 'gguf' and args[1] == 'linear_attention':
            executed_rows.append(int(kwargs['rows']))
        return real_execute(self, *args, **kwargs)

    monkeypatch.setattr(executor_type, executor_name, execute)

    def init(self, *args, **kwargs):
        kwargs['prefill_config'] = (PrefillConfig(
            linear_chunk_size=chosen[0], moe_chunk_size=chosen[0],
            full_attn_query_chunk_size=4096,
            full_attn_post_chunk_size=1024, full_attn_rope_chunk_size=1024)
                                   if chosen[0] else PrefillConfig())
        real_init(self, *args, **kwargs)
        sessions.append(self)

    def size(self, tokens):
        result = real_size(self, tokens)
        sizes.append((int(tokens), int(result), self.prefill_config))
        return result

    def resolve(config, **kwargs):
        actual, record = real_resolve(config, **kwargs)
        policies.append((actual, record, kwargs))
        return actual, record

    monkeypatch.setattr(session_type, '__init__', init)
    monkeypatch.setattr(session_type, '_linear_prefill_layer_chunk_size', size)
    monkeypatch.setattr(paro if kind == 'paro' else gguf,
                        'resolve_prefill_config_for_sequence', resolve)
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    for manual in (0, 128):
        chosen[0] = manual
        sessions.clear()
        llm = LLM(model, backend='hip_gfx1151', quant=quant,
                  max_active_requests=1, max_sequence_length=2048, prefix_cache='off')
        native_session = None
        try:
            for length in (37, 257, 1537):
                sizes.clear()
                policies.clear()
                output = llm.generate_detailed(
                    [[9707] * length], SamplingParams(max_tokens=2, temperature=0,
                         ignore_eos=True, kv_storage='bf16', speculative_mtp=False))
                assert len(output[0].generated_token_ids) == 2
                assert sessions, 'public generation never constructed a resident session'
                # Public serving uses packed slabs, whose scheduler owns chunk
                # ranges. Its GGUF cache is also deferred to the serving owner;
                # direct native prefill needs a separate cache-owning session.
                # Share weights, not cache ownership, for that diagnostic.
                public_session = sessions[0]
                if kind == 'gguf' and native_session is None:
                    native_session = session_type(
                        model, shared_runner=public_session.runner,
                        backend='hip_gfx1151', max_sequence_length=2048,
                        use_wmma_prefill=True, use_gemv_decode=True)
                active = native_session if native_session is not None else sessions[-1]
                active.reset()
                executed_rows.clear()
                if kind == 'paro':
                    native = active.prefill_native([9707] * length, sample=True)
                else:
                    native = active.prefill([9707] * length, use_bulk=True,
                                            bulk_attention_mode='native')
                assert native is not None
                assert sizes, 'native consumer never consulted its linear chunk plan'
                # GGUF resolves at construction, PARO also resolves per prompt.
                if not policies:
                    policies.append((active.prefill_config, active.prefill_chunk_tuning,
                                     {'backend': active.backend}))
                for config, report, args in policies:
                    assert report['chunk_sizes']['linear'] == config.linear_chunk_size
                    assert report['chunk_sizes']['moe'] == config.moe_chunk_size
                    if manual:
                        assert config.linear_chunk_size == manual
                        assert report['reason'] == 'manual_chunk_sizes'
                    else:
                        assert report['arch_profile_applied'] is True
                        expected = (1024 if kind == 'gguf'
                                    and report['geometry_profile_applied'] else 256)
                        assert config.linear_chunk_size == expected
                        assert config.moe_chunk_size == expected
                    assert args['backend'] == 'hip_gfx1151'
                    if kind == 'paro':
                        assert args.get('geometry_chunk_sizes') is None
                assert executed_rows, 'no native linear layer executed'
                chunk = min(active.prefill_config.linear_chunk_size,
                            active.prefill_config.moe_chunk_size)
                minimum = (active.config.linear_conv_kernel_dim if kind == 'paro'
                           else active.runner.weights.config.ssm_conv_kernel)
                ranges = (paro.Qwen35ParoResidentSession._chunk_ranges if kind == 'paro'
                          else gguf._chunk_ranges)
                expected_rows = [end - start for start, end in
                                 ranges(length, chunk, min_chunk_size=int(minimum))]
                assert len(executed_rows) % len(expected_rows) == 0
                assert executed_rows == expected_rows * (len(executed_rows) // len(expected_rows))
                for total, executed, config in sizes:
                    planned = min(config.linear_chunk_size, config.moe_chunk_size)
                    assert executed == min(total, planned), (total, executed, config)
        finally:
            if native_session is not None:
                native_session.close()
            llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for name in ('current_allocated_bytes', 'active_allocations'):
        assert after[name] == before[name]
