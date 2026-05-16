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


def test_llm_generate_gguf_path_reaches_native_probe(monkeypatch) -> None:
    import hipengine.generation.qwen35_gguf as qwen35_gguf

    calls = []

    class FakeProbe:
        def __init__(self, model_path, *, layer_id=0):
            calls.append(("init", str(model_path), layer_id))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def sample_next_token(self, token_id: int):
            calls.append(("sample_next_token", int(token_id)))
            return type("Result", (), {"token_id": 123, "logit": 4.5})()

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFOneLayerProbe", FakeProbe)

    llm = LLM(str(MODEL), backend="hip_gfx1100", quant="gguf_q4_k_m")
    with pytest.raises(NotImplementedError, match="tokenizer detokenization"):
        llm.generate("The answer is", SamplingParams(max_tokens=1))

    assert calls == [
        ("init", str(MODEL.resolve()), 0),
        ("enter",),
        ("sample_next_token", 760),
        ("exit", True),
    ]
