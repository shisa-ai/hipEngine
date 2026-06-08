from __future__ import annotations

from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession


def _session_with_prefill_config(config: PrefillConfig) -> Qwen35ParoResidentSession:
    session = object.__new__(Qwen35ParoResidentSession)
    session.prefill_config = config
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
