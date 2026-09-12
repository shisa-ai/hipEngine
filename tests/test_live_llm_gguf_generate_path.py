from __future__ import annotations

from pathlib import Path

import pytest

import hipengine.generation.qwen35_gguf as qwen35_gguf
from hipengine import LLM, SamplingParams
from hipengine.models import resolve_model

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def test_qwen35_gguf_model_plugin_resolves_architecture() -> None:
    plugin = resolve_model("qwen35")

    assert plugin.name == "qwen3_5_gguf"
    assert plugin.default_quant == "gguf_q4_k_m"
    assert plugin.default_backend == "auto"


def test_qwen35moe_gguf_model_plugin_resolves_architecture() -> None:
    plugin = resolve_model("qwen35moe")

    assert plugin.name == "qwen3_5_moe_gguf"
    assert plugin.default_quant == "gguf_q4_k_m"
    assert plugin.default_backend == "auto"


def _fake_session_type(calls, *, include_step: bool):
    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.closed = False
            self.position = 0
            self._decode_graphs = []
            self._device_kv_graph_handles = {}
            calls.append(("init", str(model_path), dict(kwargs)))

        def resident_slot_view(self, slot_index):
            calls.append(("slot_view", int(slot_index)))
            return self

        def reset(self):
            self.position = 0

        def close(self):
            if not self.closed:
                self.closed = True
                calls.append(("close",))

        def prefill(self, token_ids, *, return_logits=True):
            tokens = tuple(int(token) for token in token_ids)
            self.position = len(tokens)
            calls.append(("prefill", tokens, bool(return_logits)))
            return type("Result", (), {"token_id": 220, "logit": 4.5})()

        def prefill_batch_native(self, prompts, *, sessions, return_logits=True, **kwargs):
            del kwargs
            return [
                session.prefill(prompt, return_logits=return_logits)
                for session, prompt in zip(sessions, prompts, strict=True)
            ]

        if include_step:

            def step(self, token_id, *, return_logits=True):
                self.position += 1
                calls.append(("step", int(token_id), bool(return_logits)))
                return type("Result", (), {"token_id": 16, "logit": 1.0})()

    return FakeSession


def _assert_generation2_session_contract(calls, model: Path) -> None:
    init = calls[0]
    assert init[0:2] == ("init", str(model.resolve()))
    assert init[2]["backend"] == "hip_gfx1100"
    assert init[2]["use_wmma_prefill"] is True
    assert init[2]["use_gemv_decode"] is True
    assert init[2]["defer_kv_allocation"] is True
    assert init[2]["max_batch_size"] >= 2
    assert init[2]["shared_runner"] is not None
    assert init[2]["runtime"] is not None
    assert [call for call in calls if call[0] == "slot_view"] == [
        ("slot_view", index) for index in range(1, init[2]["max_batch_size"])
    ]
    assert calls[-1] == ("close",)


def test_llm_generate_gguf_path_uses_resident_session(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        qwen35_gguf,
        "Qwen35GGUFResidentSession",
        _fake_session_type(calls, include_step=True),
    )

    llm = LLM(str(MODEL), backend="hip_gfx1100", quant="gguf_q4_k_m")
    try:
        assert llm.generate("The answer is", SamplingParams(max_tokens=2)) == [" 1"]
    finally:
        llm.close()

    _assert_generation2_session_contract(calls, MODEL)
    assert ("prefill", (760, 4087, 369), False) in calls
    assert ("step", 220, False) in calls


def test_llm_generate_qwen35moe_gguf_path_uses_resident_session(monkeypatch) -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")

    calls = []
    monkeypatch.setattr(
        qwen35_gguf,
        "Qwen35GGUFResidentSession",
        _fake_session_type(calls, include_step=False),
    )

    llm = LLM(str(MOE_MODEL), backend="hip_gfx1100", quant="gguf_q4_k_m")
    try:
        assert llm.generate("The answer is", SamplingParams(max_tokens=1)) == [" "]
    finally:
        llm.close()

    _assert_generation2_session_contract(calls, MOE_MODEL)
    assert ("prefill", (760, 4087, 369), False) in calls
