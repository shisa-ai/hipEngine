from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession


class _FakeRuntime:
    def __init__(self, *, free_gib: float = 48.0, total_gib: float = 48.0) -> None:
        self.free_bytes = int(free_gib * 1024**3)
        self.total_bytes = int(total_gib * 1024**3)

    def mem_get_info(self) -> tuple[int, int]:
        return self.free_bytes, self.total_bytes


def _session_with_prefill_config(
    config: PrefillConfig,
    *,
    storage_dtype: DType = DType.BF16,
    free_gib: float = 48.0,
    total_gib: float = 48.0,
) -> Qwen35ParoResidentSession:
    session = object.__new__(Qwen35ParoResidentSession)
    session.prefill_config = config
    session.kv_storage_dtype = storage_dtype
    session.block_size = 256
    session.config = SimpleNamespace(num_key_value_heads=2, head_dim=256)
    session.runtime = _FakeRuntime(free_gib=free_gib, total_gib=total_gib)
    return session


def test_prefill_workspace_overlap_stays_resident_through_32k() -> None:
    session = _session_with_prefill_config(PrefillConfig(linear_chunk_size=1024, moe_chunk_size=1024))

    assert session._should_minimize_prefill_workspace_overlap(512) is False
    assert session._should_minimize_prefill_workspace_overlap(4096) is False
    assert session._should_minimize_prefill_workspace_overlap(16384) is False
    assert session._should_minimize_prefill_workspace_overlap(32768) is False


def test_prefill_workspace_overlap_is_minimized_above_32k_with_active_chunking() -> None:
    session = _session_with_prefill_config(PrefillConfig(linear_chunk_size=1024, moe_chunk_size=1024))

    assert session._should_minimize_prefill_workspace_overlap(49152) is True
    assert session._should_minimize_prefill_workspace_overlap(65536) is True
    assert session._should_minimize_prefill_workspace_overlap(131072) is True


def test_prefill_workspace_overlap_ignores_non_splitting_chunk_sizes() -> None:
    session = _session_with_prefill_config(PrefillConfig(linear_chunk_size=49152, moe_chunk_size=49152))

    assert session._should_minimize_prefill_workspace_overlap(49152) is False


def test_int8_prefill_attention_auto_requires_very_long_low_memory_pressure(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN35_INT8_PREFILL_STREAMING_MIN_TOKENS", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN35_INT8_PREFILL_LOW_MEMORY_TOTAL_GIB", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN35_INT8_PREFILL_ORACLE_RESERVE_MIB", raising=False)

    high_memory = _session_with_prefill_config(
        PrefillConfig(attn_aotriton_min_tokens=512),
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        free_gib=32.0,
        total_gib=48.0,
    )
    assert high_memory._prefill_int8_attention_path(512) == "oracle_bf16"
    assert high_memory._prefill_int8_attention_path(131072) == "oracle_bf16"
    assert high_memory._prefill_int8_attention_path(262143) == "oracle_bf16"
    assert high_memory._prefill_use_aotriton_attention_resolved(262143) is True

    low_memory = _session_with_prefill_config(
        PrefillConfig(attn_aotriton_min_tokens=512),
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        free_gib=2.0,
        total_gib=24.0,
    )
    assert low_memory._prefill_int8_attention_path(196607) == "oracle_bf16"
    assert low_memory._prefill_int8_attention_path(229376) == "streaming_direct"
    assert low_memory._prefill_use_aotriton_attention_resolved(229376) is False

    fragmented_high_memory = _session_with_prefill_config(
        PrefillConfig(attn_aotriton_min_tokens=512),
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        free_gib=1.4,
        total_gib=48.0,
    )
    assert fragmented_high_memory._prefill_int8_attention_path(262143) == "streaming_direct"


def test_int8_prefill_attention_env_overrides_auto_gate(monkeypatch) -> None:
    session = _session_with_prefill_config(
        PrefillConfig(attn_aotriton_min_tokens=512),
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
    )

    monkeypatch.setenv("HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION", "streaming")
    assert session._prefill_int8_attention_path(512) == "streaming_direct"
    assert session._prefill_use_aotriton_attention_resolved(131072) is False

    monkeypatch.setenv("HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION", "oracle")
    assert session._prefill_int8_attention_path(262143) == "oracle_bf16"
    assert session._prefill_use_aotriton_attention_resolved(262143) is True

    session.runtime = _FakeRuntime(free_gib=2.0, total_gib=24.0)
    monkeypatch.setenv("HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION", "auto")
    monkeypatch.setenv("HIPENGINE_QWEN35_INT8_PREFILL_STREAMING_MIN_TOKENS", "131072")
    assert session._prefill_int8_attention_path(131071) == "oracle_bf16"
    assert session._prefill_int8_attention_path(131072) == "streaming_direct"

    monkeypatch.setenv("HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION", "invalid")
    with pytest.raises(ValueError, match="HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION"):
        session._prefill_int8_attention_path(512)


def test_bf16_prefill_attention_still_uses_aotriton_threshold() -> None:
    session = _session_with_prefill_config(PrefillConfig(attn_aotriton_min_tokens=512))

    assert session._prefill_int8_attention_path(131072) is None
    assert session._prefill_use_aotriton_attention_resolved(511) is False
    assert session._prefill_use_aotriton_attention_resolved(512) is True
