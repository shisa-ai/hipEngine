from __future__ import annotations

import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession


def _session_with_prefill_config(
    config: PrefillConfig,
    *,
    storage_dtype: DType = DType.BF16,
) -> Qwen35ParoResidentSession:
    session = object.__new__(Qwen35ParoResidentSession)
    session.prefill_config = config
    session.kv_storage_dtype = storage_dtype
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


def test_int8_prefill_attention_auto_gates_streaming_to_very_long_prompts(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN35_INT8_PREFILL_STREAMING_MIN_TOKENS", raising=False)
    session = _session_with_prefill_config(
        PrefillConfig(attn_aotriton_min_tokens=512),
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
    )

    assert session._prefill_int8_attention_path(512) == "oracle_bf16"
    assert session._prefill_int8_attention_path(131072) == "oracle_bf16"
    assert session._prefill_use_aotriton_attention_resolved(131072) is True
    assert session._prefill_int8_attention_path(262143) == "streaming_direct"
    assert session._prefill_use_aotriton_attention_resolved(262143) is False


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
