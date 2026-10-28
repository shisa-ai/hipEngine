"""Resident draft chain choice is explicit, not inherited from ambient state."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.speculative import mtp_resident_draft as draft


@pytest.mark.parametrize("stale", [None, "0", "1", "true", "invalid"])
@pytest.mark.parametrize("explicit", [None, False, True])
@pytest.mark.parametrize("prewarm", [False, True])
def test_constructor_ignores_retired_env_and_keeps_explicit_chain(monkeypatch, stale, explicit, prewarm):
    if stale is None:
        monkeypatch.delenv("HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN", stale)
    # No device work is needed to exercise the real construction branch.
    for name in (
        "build_mtp_nextn", "build_gguf_k_gemv", "build_gguf_q4_k_gemv",
        "build_gguf_q6_k_pack8_gemv", "build_cast", "build_qwen35_router",
        "build_lm_head", "build_paro_silu", "build_paro_combine", "build_gather",
    ):
        monkeypatch.setattr(draft, name, lambda **kwargs: object())
    monkeypatch.setattr(draft.Qwen35GGUFResidentMTPDraftRunner, "_check_real_model_qtypes", lambda self: None)
    monkeypatch.setattr(draft.Qwen35GGUFResidentMTPDraftRunner, "_upload_weights", lambda self: None)
    monkeypatch.setattr(draft.Qwen35GGUFResidentMTPDraftRunner, "_allocate_buffers", lambda self: None)
    warmed = []
    monkeypatch.setattr(draft.Qwen35GGUFResidentMTPDraftRunner, "ensure_device_chain_ready", lambda self: warmed.append(self))
    weights = {
        "blk.40.attn_q_norm.weight": (np.zeros(32), None, None),
        "blk.40.ffn_gate_exps.weight": (np.zeros((1, 32)), None, None),
        "output.weight": (np.zeros((8, 32)), None, None),
    }
    runner = draft.Qwen35GGUFResidentMTPDraftRunner(
        weights, np.zeros((8, 32), dtype=np.float32), runtime=SimpleNamespace(),
        upload_cache=object(), device_chain_enabled=explicit, prewarm_device_chain=prewarm,
    )
    assert runner._device_chain_enabled is bool(explicit)
    assert warmed == ([runner] if prewarm and explicit else [])
