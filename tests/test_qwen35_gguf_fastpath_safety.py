from __future__ import annotations

from hipengine.runtime.gguf_linear import set_gemv_decode_enabled, set_wmma_prefill_enabled
from hipengine.runtime.qwen35_gguf_runner import (
    _QWEN35MOE_UNSAFE_FASTPATH_ENV,
    resolve_qwen35moe_fastpath_safety,
)


def _reset_sessions() -> None:
    set_wmma_prefill_enabled(None)
    set_gemv_decode_enabled(None)


def test_qwen35moe_fastpath_safety_disables_requested_env_opt_ins(monkeypatch) -> None:
    _reset_sessions()
    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_GEMV_DECODE", "1")
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)
    monkeypatch.delenv(_QWEN35MOE_UNSAFE_FASTPATH_ENV, raising=False)

    safety = resolve_qwen35moe_fastpath_safety(
        is_qwen35moe=True,
        use_wmma_prefill=None,
        use_gemv_decode=None,
    )

    assert safety.requested_wmma_prefill is True
    assert safety.requested_gemv_decode is True
    assert safety.effective_wmma_prefill is False
    assert safety.effective_gemv_decode is False
    assert safety.disabled_wmma_prefill is True
    assert safety.disabled_gemv_decode is True
    assert "P9.E2" in str(safety.reason)


def test_qwen35moe_fastpath_safety_disables_explicit_session_opt_ins(monkeypatch) -> None:
    _reset_sessions()
    monkeypatch.delenv("HIPENGINE_GGUF_WMMA_PREFILL", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_GEMV_DECODE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)
    monkeypatch.delenv(_QWEN35MOE_UNSAFE_FASTPATH_ENV, raising=False)

    safety = resolve_qwen35moe_fastpath_safety(
        is_qwen35moe=True,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    )

    assert safety.requested_wmma_prefill is True
    assert safety.requested_gemv_decode is True
    assert safety.effective_wmma_prefill is False
    assert safety.effective_gemv_decode is False


def test_qwen35moe_fastpath_safety_allows_t16_repack_with_wmma_and_gemv(monkeypatch) -> None:
    _reset_sessions()
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    monkeypatch.delenv(_QWEN35MOE_UNSAFE_FASTPATH_ENV, raising=False)

    safety = resolve_qwen35moe_fastpath_safety(
        is_qwen35moe=True,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    )

    assert safety.requested_wmma_prefill is True
    assert safety.requested_gemv_decode is True
    assert safety.effective_wmma_prefill is True
    assert safety.effective_gemv_decode is True
    assert safety.disabled_wmma_prefill is False
    assert safety.disabled_gemv_decode is False
    assert safety.reason is None



def test_qwen35moe_fastpath_safety_allows_explicit_unsafe_override(monkeypatch) -> None:
    _reset_sessions()
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)
    monkeypatch.setenv(_QWEN35MOE_UNSAFE_FASTPATH_ENV, "1")

    safety = resolve_qwen35moe_fastpath_safety(
        is_qwen35moe=True,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    )

    assert safety.allow_unsafe_qwen35moe_fastpaths is True
    assert safety.effective_wmma_prefill is True
    assert safety.effective_gemv_decode is True
    assert safety.disabled_wmma_prefill is False
    assert safety.disabled_gemv_decode is False
    assert safety.reason is None


def test_qwen35moe_fastpath_safety_blocks_wmma_without_t16_repack(monkeypatch) -> None:
    _reset_sessions()
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)
    monkeypatch.delenv(_QWEN35MOE_UNSAFE_FASTPATH_ENV, raising=False)

    safety = resolve_qwen35moe_fastpath_safety(
        is_qwen35moe=True,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    )

    assert safety.requested_wmma_prefill is True
    assert safety.requested_gemv_decode is True
    assert safety.effective_wmma_prefill is False
    assert safety.effective_gemv_decode is False
    assert safety.disabled_wmma_prefill is True
    assert safety.disabled_gemv_decode is True


def test_qwen35moe_fastpath_safety_allows_wmma_prefill_without_gemv_decode(monkeypatch) -> None:
    """Raw-GGUF WMMA prefill alone is the fastest prefill path (P9.C11)."""
    _reset_sessions()
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)
    monkeypatch.delenv(_QWEN35MOE_UNSAFE_FASTPATH_ENV, raising=False)

    safety = resolve_qwen35moe_fastpath_safety(
        is_qwen35moe=True,
        use_wmma_prefill=True,
        use_gemv_decode=False,
    )

    assert safety.requested_wmma_prefill is True
    assert safety.requested_gemv_decode is False
    assert safety.effective_wmma_prefill is True
    assert safety.effective_gemv_decode is False
    assert safety.disabled_wmma_prefill is False
    assert safety.disabled_gemv_decode is False
    assert safety.reason is None


def test_fastpath_safety_does_not_gate_dense_qwen35(monkeypatch) -> None:
    _reset_sessions()
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)
    monkeypatch.delenv(_QWEN35MOE_UNSAFE_FASTPATH_ENV, raising=False)

    safety = resolve_qwen35moe_fastpath_safety(
        is_qwen35moe=False,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    )

    assert safety.effective_wmma_prefill is True
    assert safety.effective_gemv_decode is True
    assert safety.disabled_wmma_prefill is False
    assert safety.disabled_gemv_decode is False
