"""The NextN draft's batch session must not size a target-class prefill scratch.

The draft session only runs 2-8 row hidden steps and (MoE) packed prompt-KV
priming, which grows its own packed workspace. Its bulk prefill scratch rows
also floor the packed verify scratch, so leaving it at the natural 4,096-row
chunk pins two ~3 GiB workspaces on the dense 27B and OOMs a 24 GB card.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_nextn as nextn


class _StopAtSession(Exception):
    pass


def test_nextn_batch_session_uses_small_prefill_scratch_rows(monkeypatch):
    captured: dict[str, object] = {}

    weights = SimpleNamespace(as_full_stack_weights=lambda: object())
    monkeypatch.setattr(nextn, "_resolve_nextn_executor_backend", lambda backend, fallback: "hip_gfx1100")
    monkeypatch.setattr(nextn, "register_gguf_q3_k_gemv_kernels", lambda: None)
    monkeypatch.setattr(nextn, "materialize_qwen35_gguf_nextn_weights", lambda *a, **k: weights)
    monkeypatch.setattr(
        nextn,
        "Qwen35GGUFFullStackRunner",
        lambda *a, **k: SimpleNamespace(hidden_size=5_120, vocab_size=248_320),
    )

    def session(*args, **kwargs):
        captured.update(kwargs)
        raise _StopAtSession

    monkeypatch.setattr(nextn, "Qwen35GGUFResidentSession", session)

    with pytest.raises(_StopAtSession):
        nextn.Qwen35GGUFNextNExecutor(
            "unused.gguf",
            max_positions=32_768,
            max_requests=4,
            runtime=object(),
        )

    assert captured["max_sequence_length"] == 32_768
    assert captured["prefill_chunk_size"] == nextn._NEXTN_BATCH_PREFILL_SCRATCH_ROWS
    assert 8 <= nextn._NEXTN_BATCH_PREFILL_SCRATCH_ROWS <= 256
