"""Tier-1 real allocation checks for constrained prefill chunk admission."""
import ctypes
import os
from pathlib import Path

import numpy as np
import pytest


def _hip_available():
    try:
        ctypes.CDLL('libamdhip64.so')
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _hip_available(), reason='HIP unavailable')
@pytest.mark.parametrize('rung', [4096, 2048])
def test_real_chunk_admission_under_reported_memory_budget(monkeypatch, rung):
    model = Path(os.environ.get('HIPENGINE_TEST_QWEN4EXP_GGUF',
        '/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL'))
    if not model.exists():
        pytest.skip('Qwen4Exp GGUF fixture unavailable')
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.loading.gguf import GGUFReader, discover_gguf_files, load_gguf_index
    from hipengine.loading.qwen4_exp_gguf import build_qwen4_exp_gguf_tensor_map
    from hipengine.loading.qwen4_exp_materialize import plan_qwen4_exp_residency
    from hipengine.loading.qwen4_exp_context import resolve_qwen4_exp_context
    from hipengine.models import resolve_model
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator

    paths = discover_gguf_files(model)
    readers = tuple(GGUFReader(path) for path in paths)
    model_map = build_qwen4_exp_gguf_tensor_map(tuple(r.info for r in readers))
    plan = plan_qwen4_exp_residency(model_map, staging_token_capacity=rung)
    runtime = get_hip_runtime()
    free, total = runtime.mem_get_info()
    admission = resolve_qwen4_exp_context(
        plan, available_device_bytes=free, requested_context=4352,
        native_context_length=model_map.config.context_length,
        resident_capacity=1, prefill_chunk_size=rung,
    )
    budget = admission.required_bytes
    before = memory_stats()['current_allocated_bytes']
    index = load_gguf_index(paths[0])
    with monkeypatch.context() as scoped:
        scoped.setattr(type(runtime), 'mem_get_info', lambda self: (budget, total))
        generator = Qwen4ExpGGUFTextGenerator(
            model_path=model, weight_index=index,
            model_plugin=resolve_model(index.architecture or ''), backend='hip_gfx1151',
            max_sequence_length=4352, resident_capacity=1, prefill_chunk_size=None,
        )
    try:
        assert generator.admitted_prefill_chunk_size == rung
        assert generator.runner.prefill_chunk_size == rung
        from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk
        from hipengine.loading.qwen4_exp_scratch import qwen4_exp_wide_f16_activation_nbytes
        from hipengine.runtime import qwen4_exp_runner as runner_module
        from hipengine.runtime.gguf_linear import wide_f16_activation_workspace

        cfg = generator.runner.config
        expected_repair = 4 + 4 * rung * cfg.expert_used_count * max(
            cfg.hidden_size, 2 * cfg.expert_feed_forward_length)
        queues = prepare_lazy_group_risk(generator.runner)
        assert len(queues) == 2
        assert all(record['nbytes'] == expected_repair for record in queues)
        expected_staging = qwen4_exp_wide_f16_activation_nbytes(
            cfg, context_tokens=4352, prefill_chunk_size=rung)
        # Provision the optional staging owner explicitly; do not depend on
        # whichever execution profile the surrounding test process last bound.
        with monkeypatch.context() as scoped:
            scoped.setattr(runner_module, '_q8_dense_wide_enabled', lambda: True)
            scoped.setattr(runner_module, 'wide_f16_activation_enabled', lambda default=True: True)
            with generator.runner._wide_f16_activation_context():
                assert wide_f16_activation_workspace().nbytes == expected_staging
        assert wide_f16_activation_workspace() is None
        tokens = generator.tokenizer.encode('Describe how rain forms.')
        output = generator.runner.prefill(tokens)
        assert np.isfinite(output.logits).all()
        step = generator.runner.step(int(output.token_id))
        assert np.isfinite(step.logits).all()
    finally:
        generator.close()
    assert memory_stats()['current_allocated_bytes'] == before
