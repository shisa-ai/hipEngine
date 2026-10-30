"""Admission reserves for lazy, shared QSA prefill buffers."""
from types import SimpleNamespace

import pytest

from tests.test_unit_qwen4exp_scratch_plan import config


@pytest.mark.parametrize("context,chunk", [(127, 17), (128, 64), (129, 65), (513, 257), (4352, 4096)])
def test_qsa_wmma_reserve_matches_buffer_recipe(context, chunk):
    from hipengine.loading.qwen4_exp_scratch import qwen4_exp_qsa_wmma_nbytes
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa_wmma import qsa_wmma_mask_words

    cfg = config()
    kv_buffer = context * cfg.attention_kv_head_count * cfg.attention_key_length * 2
    mask = min(context, chunk) * qsa_wmma_mask_words(context) * 4
    assert qwen4_exp_qsa_wmma_nbytes(
        cfg, context_tokens=context, prefill_chunk_size=chunk) == 2 * kv_buffer + mask


def test_admission_counts_shared_qsa_buffers_once_per_runner():
    from hipengine.loading.qwen4_exp_context import resolve_qwen4_exp_context
    from hipengine.loading.qwen4_exp_scratch import (
        qwen4_exp_scratch_breakdown, qwen4_exp_wide_f16_activation_nbytes,
    )

    cfg = config()
    context, chunk = 513, 257
    residency = SimpleNamespace(config=cfg, device_weight_bytes=1 << 30, staging_bytes=0)
    kwargs = dict(requested_context=context, prefill_chunk_size=chunk,
                  resident_capacity=2, scratch_bytes_per_runner=0, reserve_bytes=0)
    base = sum(qwen4_exp_scratch_breakdown(
        cfg, context_tokens=context, prefill_chunk_size=chunk).values())
    cast = qwen4_exp_wide_f16_activation_nbytes(
        cfg, context_tokens=context, prefill_chunk_size=chunk)
    qsa = 4 * context * cfg.attention_kv_head_count * cfg.attention_key_length + 4 * chunk * 5
    plan = resolve_qwen4_exp_context(residency, available_device_bytes=128 << 30, **kwargs)
    assert plan.scratch_bytes == 2 * (base + cast + qsa)
    with pytest.raises(MemoryError):
        resolve_qwen4_exp_context(residency, available_device_bytes=plan.required_bytes - 1, **kwargs)


@pytest.mark.parametrize("context,chunk", [(0, 1), (1, 0), (-1, 2)])
def test_qsa_reserve_rejects_nonpositive_sizes(context, chunk):
    from hipengine.loading.qwen4_exp_scratch import qwen4_exp_qsa_wmma_nbytes

    with pytest.raises(ValueError):
        qwen4_exp_qsa_wmma_nbytes(config(), context_tokens=context, prefill_chunk_size=chunk)
