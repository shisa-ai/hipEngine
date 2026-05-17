from __future__ import annotations

from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams
from hipengine.models import resolve_model

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def test_qwen35_gguf_model_plugin_resolves_architecture() -> None:
    plugin = resolve_model("qwen35")

    assert plugin.name == "qwen3_5_gguf"
    assert plugin.default_quant == "gguf_q4_k_m"


def test_llm_generate_gguf_path_uses_resident_session(monkeypatch) -> None:
    import hipengine.generation.qwen35_gguf as qwen35_gguf

    calls = []

    class FakeSession:
        def __init__(self, model_path):
            calls.append(("init", str(model_path)))
            self.tokens = iter([220, 16])

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids):
            calls.append(("prefill", tuple(int(token) for token in token_ids)))
            return type("Result", (), {"token_id": next(self.tokens), "logit": 4.5})()

        def step(self, token_id, position=None):
            calls.append(("step", int(token_id), position))
            return type("Result", (), {"token_id": next(self.tokens), "logit": 4.5})()

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    llm = LLM(str(MODEL), backend="hip_gfx1100", quant="gguf_q4_k_m")
    assert llm.generate("The answer is", SamplingParams(max_tokens=2)) == [" 1"]

    assert calls == [
        ("init", str(MODEL.resolve())),
        ("enter",),
        ("prefill", (760, 4087, 369)),
        ("step", 220, None),
        ("exit", True),
    ]
