"""Real arithmetic admission, including below/at/above available-memory bounds."""
from types import SimpleNamespace
import pytest
from hipengine.generation.qwen4_exp_gguf import _select_prefill_chunk
from hipengine.loading.qwen4_exp_context import resolve_qwen4_exp_context
from tests.test_unit_qwen4exp_scratch_plan import config


@pytest.mark.parametrize('context', [2051, 2052, 4352])
def test_chunk_ladder_uses_real_context_admission(context):
    cfg = config()
    residency = SimpleNamespace(config=cfg, device_weight_bytes=1 << 30, staging_bytes=0)
    def plan(model_map, *, staging_token_capacity):
        return residency
    def select(available):
        return _select_prefill_chunk(
            None, ladder=(4096, 2048, 1024), available_device_bytes=available,
            requested_context=context, native_context_length=cfg.context_length,
            resident_capacity=2, plan_fn=plan, admit_fn=resolve_qwen4_exp_context,
        )
    limits = {chunk: resolve_qwen4_exp_context(
        residency, available_device_bytes=128 << 30, requested_context=context,
        resident_capacity=2, prefill_chunk_size=chunk,
    ).required_bytes for chunk in (4096, 2048, 1024)}
    assert select(limits[4096])[0] == 4096
    assert select(limits[4096] + (1 << 30))[0] == 4096
    if limits[4096] > limits[2048]:
        assert select(limits[4096] - 1)[0] in (2048, 1024)
    with pytest.raises(MemoryError):
        select(min(limits.values()) - 1)
