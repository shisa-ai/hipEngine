from __future__ import annotations

from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession


def _session_with_prefill_config(config: PrefillConfig) -> Qwen35ParoResidentSession:
    session = object.__new__(Qwen35ParoResidentSession)
    session.prefill_config = config
    return session


def test_prefill_workspace_overlap_stays_resident_for_unchunked_prompts() -> None:
    session = _session_with_prefill_config(PrefillConfig())

    assert session._should_minimize_prefill_workspace_overlap(512) is False


def test_prefill_workspace_overlap_is_minimized_for_active_chunking() -> None:
    session = _session_with_prefill_config(PrefillConfig(linear_chunk_size=1024, moe_chunk_size=1024))

    assert session._should_minimize_prefill_workspace_overlap(4096) is True


def test_prefill_workspace_overlap_ignores_non_splitting_chunk_sizes() -> None:
    session = _session_with_prefill_config(PrefillConfig(linear_chunk_size=1024, moe_chunk_size=1024))

    assert session._should_minimize_prefill_workspace_overlap(1024) is False
