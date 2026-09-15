import pytest

from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata
from tests.test_live_qwen4_exp_gguf_config import _info


def config():
    return qwen4_exp_gguf_config_from_metadata(_info())


@pytest.mark.parametrize("context,chunk", [(256, 1), (256, 17), (4352, 1024),
                                         (4352, 2048), (4352, 4096), (262144, 4096)])
def test_scratch_plan_matches_real_allocation_recipes(context, chunk):
    from hipengine.loading.qwen4_exp_scratch import qwen4_exp_scratch_breakdown
    from hipengine.loading.qwen4_exp_materialize import (
        _qsa_index_state_bytes, _runtime_state_bytes_per_request,
    )
    from scripts.qwen4exp_allocation_census import census

    cfg = config()
    actual = census(cfg, context=context, chunk=chunk)
    kv = ((context + 255) // 256 * 256) * cfg.bf16_kv_bytes_per_token
    nonscratch = kv + _qsa_index_state_bytes(cfg, context) + _runtime_state_bytes_per_request(cfg)
    planned = qwen4_exp_scratch_breakdown(cfg, context_tokens=context, prefill_chunk_size=chunk)
    assert sum(planned.values()) == actual["prepared_bytes"] - nonscratch
    for name in ("gdn_prefill_scratch", "qsa_prefill_scratch", "ple_prefill_scratch",
                 "qsa_prefill_metadata", "_prefill_buffers", "_buffers"):
        assert planned[name] == actual["owner_bytes"][name]


def test_context_admission_preserves_old_floor_and_reserve_but_accounts_large_chunks():
    from types import SimpleNamespace
    from hipengine.loading.qwen4_exp_context import resolve_qwen4_exp_context
    from hipengine.loading.qwen4_exp_scratch import qwen4_exp_scratch_breakdown

    cfg = config()
    residency = SimpleNamespace(config=cfg, device_weight_bytes=1 << 30, staging_bytes=0)
    kwargs = dict(available_device_bytes=128 << 30, requested_context=4352, resident_capacity=2)
    before = resolve_qwen4_exp_context(residency, **kwargs)
    small = resolve_qwen4_exp_context(residency, prefill_chunk_size=2048, **kwargs)
    large = resolve_qwen4_exp_context(residency, prefill_chunk_size=4096, **kwargs)
    assert before == small
    required = sum(qwen4_exp_scratch_breakdown(
        cfg, context_tokens=4352, prefill_chunk_size=4096).values())
    assert large.scratch_bytes == 2 * required
    assert large.scratch_bytes > before.scratch_bytes
    assert large.reserve_bytes == before.reserve_bytes == 4 << 30
    kwargs["available_device_bytes"] = before.required_bytes
    with pytest.raises(MemoryError):
        resolve_qwen4_exp_context(residency, prefill_chunk_size=4096, **kwargs)


def test_device_margin_does_not_credit_host_staging():
    from scripts.qwen4exp_chunk_memory_probe import allocation_margins, device_allocation_margins

    plan = dict(device_weight_bytes=100, staging_bytes=20, kv_bytes=10, index_bytes=5,
                runtime_state_bytes=5, scratch_bytes=30, required_bytes=200)
    assert allocation_margins(plan, 160)["scratch_margin_bytes"] == 10
    assert device_allocation_margins(plan, 160)["device_scratch_margin_bytes"] == -10


def test_auto_context_search_includes_context_scaled_scratch():
    from types import SimpleNamespace
    from hipengine.loading.qwen4_exp_context import resolve_qwen4_exp_context

    residency = SimpleNamespace(config=config(), device_weight_bytes=64 << 30, staging_bytes=0)
    native = resolve_qwen4_exp_context(
        residency, available_device_bytes=128 << 30, resident_capacity=2, prefill_chunk_size=4096)
    available = native.required_bytes - (1 << 30)
    limited = resolve_qwen4_exp_context(
        residency, available_device_bytes=available, resident_capacity=2, prefill_chunk_size=4096)
    assert limited.passed and limited.context_tokens < native.context_tokens
    with pytest.raises(MemoryError):
        resolve_qwen4_exp_context(
            residency, available_device_bytes=available, resident_capacity=2,
            prefill_chunk_size=4096, requested_context=limited.context_tokens + 1)


def test_schema3_allocation_requires_device_margin():
    from scripts.qwen4exp_q8_repair_depth_gate import validate_chunk_allocation

    with pytest.raises(ValueError, match="host staging"):
        validate_chunk_allocation(
            {"schema": 3, "device_allocation_margins": {"device_scratch_margin_bytes": -1}},
            chunk=4096, context=4352, manifest="", host={}, model={})
