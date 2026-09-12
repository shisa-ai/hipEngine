from __future__ import annotations

import pytest

from hipengine.runtime import qwen35_gguf_runner as runner_module


def test_prefill_router_select_threads_use_scoped_backend_policy() -> None:
    assert runner_module._gguf_prefill_router_select_threads("hip_gfx1100") == 128
    assert runner_module._gguf_prefill_router_select_threads("hip_gfx1151") == 128


def test_prefill_router_select_threads_explicit_candidate_and_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS", "64")
    assert runner_module._gguf_prefill_router_select_threads("hip_gfx1151") == 64
    monkeypatch.setenv("HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS", "512")
    assert runner_module._gguf_prefill_router_select_threads("hip_gfx1151") == 512


def test_prefill_router_select_threads_reject_unsupported_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS", "32")
    with pytest.raises(ValueError, match="prefill router-select threads must be one of"):
        runner_module._gguf_prefill_router_select_threads("hip_gfx1151")
