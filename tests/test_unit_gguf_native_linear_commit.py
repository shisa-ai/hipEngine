"""Host-only selection contract for GGUF native selected-state commits."""

from types import SimpleNamespace

import pytest

from hipengine.runtime import gguf_native_spec_cycle as native


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
def test_native_graph_always_resolves_chunked_commit(monkeypatch, retired):
    if retired is None:
        monkeypatch.delenv("HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED", retired)
    calls = []
    session = SimpleNamespace(
        position=5, backend="test_backend", runtime=object(),
        scratch=SimpleNamespace(max_positions=256),
        _ensure_verify_block_buffers=lambda *a, **k: None,
        _ensure_verify_lm_head_buffers=lambda *a, **k: None,
        _ensure_verify_linear_state_row_buffers=lambda *a, **k: None,
        _ensure_verify_linear_state_commit_tables=lambda **k: True,
    )
    monkeypatch.setattr(native, "_validate_capture_admission", lambda *a, **k: None)
    monkeypatch.setattr(native, "resolve", lambda **kwargs: calls.append(kwargs) or None)
    with pytest.raises(native.NativeSpecTargetGraphUnsupportedError, match="registered primitives are unavailable"):
        native.capture_qwen35_gguf_native_b2_target_graph(
            session, [11, 12, 13], context_limit=256,
            capture_linear_state_rows=True, defer_linear_state_commit=True,
            device_accept_commit=True,
        )
    assert calls[-1] == {
        "backend": "test_backend", "layer": "linear_state_pair_commit",
        "quant": "w4_paro", "variant": "chunked_i32", "missing": "none",
    }
