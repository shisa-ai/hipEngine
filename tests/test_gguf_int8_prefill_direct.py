"""Direct (oracle-free) INT8 prefill attention on the GGUF resident route.

2026-09-09 capacity follow-up to the INT8 layer_outer hidden alias: the
pure-INT8 route's remaining full-capacity transients are the shared BF16
K/V oracle pair (4,096 B/token) and, in layer_outer mode, the full-capacity
hidden plane (10,240 B/token). The PARO runner already carries a
``streaming_direct`` INT8 prefill attention path
(``HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION``) that reads the retained
INT8 store directly instead of a temporary BF16 oracle; this gate ports
the same structure to the GGUF resident route as
``HIPENGINE_GGUF_INT8_PREFILL_DIRECT`` (default OFF pending the
production-profile numerics gate).

When enabled, ``_full_attention_prefill_scratch_for_layer`` returns the
retained INT8 caches as the attention K/V source with INT8 append and
prefill spans (write-through, no oracle, no retained double-write), and
the INT8 prefill lifetime plan models zero oracle cost so the route
plans chunk-sized hidden storage (no full-capacity planes).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import qwen35_gguf_runner as gguf_runner

from tests.test_gguf_bulk_prefill_workspace_release import _fake_session

DIRECT_ENV = "HIPENGINE_GGUF_INT8_PREFILL_DIRECT"


def test_int8_prefill_direct_env_gates_plan_to_chunk_outer(monkeypatch) -> None:
    """With the direct gate on, the INT8 lifetime plan drops oracle costs."""

    monkeypatch.setenv(DIRECT_ENV, "1")
    plan = gguf_runner._plan_gguf_int8_prefill_lifetime(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=131_072,
        scratch_rows=1_024,
        hidden_size=5_120,
        head_count_kv=8,
        key_length=128,
        full_attention_layers=17,
        bf16_full_attention_layers=0,
        has_bf16_mirror=False,
        hidden_buffer_count=2,
        direct_int8_prefill=True,
    )
    assert plan.mode == "chunk_outer_direct_int8"
    assert plan.required_hidden_capacity == 1_024
    assert plan.oracle_buffer_count == 0
    assert plan.oracle_pair_bytes == 0


def test_int8_prefill_direct_env_off_keeps_oracle_plan(monkeypatch) -> None:
    """Default OFF keeps the oracle-based layer_outer planning unchanged."""

    monkeypatch.delenv(DIRECT_ENV, raising=False)
    assert gguf_runner._gguf_int8_prefill_direct_enabled() is False
    plan = gguf_runner._plan_gguf_int8_prefill_lifetime(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=131_072,
        scratch_rows=1_024,
        hidden_size=5_120,
        head_count_kv=8,
        key_length=128,
        full_attention_layers=17,
        bf16_full_attention_layers=0,
        has_bf16_mirror=False,
        hidden_buffer_count=2,
    )
    assert plan.mode == "layer_outer_shared_oracle"
    assert plan.required_hidden_capacity == 131_072
    assert plan.oracle_buffer_count == 1


def test_int8_prefill_direct_env_on_flag(monkeypatch) -> None:
    monkeypatch.setenv(DIRECT_ENV, "1")
    assert gguf_runner._gguf_int8_prefill_direct_enabled() is True
    monkeypatch.setenv(DIRECT_ENV, "0")
    assert gguf_runner._gguf_int8_prefill_direct_enabled() is False


def test_direct_route_allocates_chunk_sized_hidden(monkeypatch) -> None:
    """The direct plan's workspace is chunk-sized, not full-capacity."""

    monkeypatch.setenv(DIRECT_ENV, "1")
    session, _ = _fake_session(monkeypatch, capacity=8_192)
    session.__dict__["_int8_prefill_lifetime_plan"] = gguf_runner._plan_gguf_int8_prefill_lifetime(
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        max_positions=8_192,
        scratch_rows=768,
        hidden_size=5_120,
        head_count_kv=8,
        key_length=128,
        full_attention_layers=17,
        bf16_full_attention_layers=0,
        has_bf16_mirror=False,
        hidden_buffer_count=2,
        direct_int8_prefill=True,
    )
    session.__dict__["dms_prefill_mode"] = "dense_pool"
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    expected = 768 * 5120 * DType.BF16.itemsize
    assert session._prefill_hidden_a.nbytes == expected
    assert session._prefill_hidden_b.nbytes == expected


if __name__ == "__main__":
    pytest.main([__file__])
